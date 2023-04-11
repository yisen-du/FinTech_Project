"""
This module leverages transaction logs to calculate the moving distance and speed for each merchant.
To increase the performance of this computation, we utilize multiprocessing and take advantage of the
high processing power of multiple CPUs available on the cluster.

Author: Yisen Du
Date: April 11, 2023
"""

# Import statements
import os
import warnings
from multiprocessing import Pool, cpu_count
import pandas as pd
from haversine import haversine
from tqdm import tqdm


# Constant definitions and global configurations
warnings.filterwarnings("ignore")
tqdm.pandas()


# Function definitions
def read_data():
    """
    read the transaction logs from our dataset (on the cluster)
    """
    os.chdir('/home/yisendu/xu-g1/data/UP_merchant_transactions/UP_Transaction_Details/actual_UP_data/')

    # columns to read (the total file has so many columns)
    fields = ['usr_id', 'cell_id', 'txn_eff_dte',
              'clok_tme', 'chrg_amt', 'txn_amt',
              'cmssn_amt', 'txn_typ']

    # file paths to read
    path_list = []
    for i in range(10):  # week 0 to week 9
        for j in range(1, 5):  # 1,2,3,4
            if i in [0, 1, 2]:
                file_name = 'berkeley_usr_id_week_' + str(i) + '_part_' + str(j) + '.txt'
            else:
                file_name = 'berkeley_usr_id_week_' + str(i) + '_PART_' + str(j) + '.txt'
            path_list.append(file_name)

    # create an empty dataframe to collect all the logs
    df_aggregate = pd.DataFrame({"usr_id": [], "cell_id": [], 'txn_eff_dte': [],
                                 'clok_tme': [], 'chrg_amt': [], 'txn_amt': [],
                                 'cmssn_amt': [], 'txn_typ': []})

    for file_path in path_list:
        print(file_path)
        df_temp = pd.read_csv(file_path, on_bad_lines='skip', sep='|', usecols=fields)
        df_aggregate = pd.concat([df_aggregate, df_temp], axis=0)
        print("log length:", len(df_aggregate))

    del df_temp  # save the space
    return df_aggregate


def process_data(df):
    """
    clean and preprocess the data
    """
    # part 1: merge txn_eff_dte and clok_tme to create a new columns: datetime
    df.clok_tme = df.clok_tme.apply(lambda x: x[0:-3])
    df['datetime'] = \
        pd.to_datetime(df['txn_eff_dte'] + ' ' + df['clok_tme'])

    # part 2: reformat the cell_id to create two new columns: lat and long
    # note that the format of cell_id are different
    def help_clean(x):
        # some of them are float's
        if type(x) == float:
            return None
        else:
            if ',' in x:
                return x.split(",")
            elif ' ' in x:
                return x.split(" ")
            else:
                return None

    # use progress_apply to see the real-time processing progress
    df['cell_id_split'] = df['cell_id'].progress_apply(help_clean)

    # create two new columns lat and long
    # Drop Rows with Missing Values in Specific Columns
    df = df.dropna(subset=['cell_id_split'])
    df['lat'] = df['cell_id_split'].apply(lambda x: float(x[0]))
    df['long'] = df['cell_id_split'].apply(lambda x: float(x[1]))

    # filter out zero values
    df = df[df['lat'] != 0]
    df = df[df['long'] != 0]

    # drop unused columns and change the column names
    df = df.drop(columns=['cell_id', 'txn_eff_dte', 'clok_tme', 'cell_id_split'])
    df.columns = ['MerchantId', 'charge_amount', 'transaction_amount', 'commision_amount',
                  'transaction_type', 'time', 'lat', 'long']
    return df


def compute_distance(df_temp):
    """
    df_temp contains a set of transaction logs for a single merchant.
    We utilize this dataset to calculate the differences in location and time between consecutive transactions,
    which we then use to compute the moving distance and corresponding speed.
    """
    df_temp = df_temp.sort_values('time')
    # if the merchant has only one transaction, no moving distance and speed to compute
    if len(df_temp) == 1:
        df_return = df_temp.copy()
        df_return['distance'] = None
        df_return['speed'] = None
        return df_return

    # shift lat, long, and time to compute delta change in time and distance
    df_temp['lat_shift'] = df_temp.lat.shift()
    df_temp['long_shift'] = df_temp.long.shift()
    df_temp['time_shift'] = df_temp.time.shift()
    df_shift = df_temp.copy()
    df_shift['time_delta'] = (df_shift['time'] - df_shift['time_shift']).dt.total_seconds()
    # since commission amount has Null as well, specify columns here to dropna
    df_shift = df_shift.dropna(subset=['lat_shift', 'long_shift', 'time_shift', 'time_delta'])

    df_shift['distance'] = \
        df_shift.apply(lambda x: haversine((x.lat, x.long), (x.lat_shift, x.long_shift)), axis=1)
    df_shift['speed'] = df_shift['distance'] / (df_shift['time_delta'] / 3600)
    df_shift = df_shift.drop(columns=['lat_shift', 'long_shift', 'time_shift', 'time_delta'])
    return df_shift


def process_chunk(chunk):
    """
    Perform the operation on the chunk
    Make coding structure clear
    """
    result = compute_distance(chunk)
    return result


# Main program
if __name__ == "__main__":
    # Code to execute when the script is run
    # load the data
    df_transaction = read_data()

    # clean and preprocess the data
    df_preprocessed = process_data(df_transaction)

    # do the multiprocessing to compute moving distance and speed
    # Get the number of CPU cores available
    num_processes = cpu_count()

    func = process_chunk
    data = df_preprocessed.sort_values("MerchantId")
    with Pool(num_processes) as pool:
        # we need a sequence to pass pool.map; this line creates a generator (lazy iterator) of columns
        # seq divides the whole transaction logs based on MerchantId
        df_log_length = data.groupby("MerchantId")['time'].count().reset_index()
        df_log_length['cumulative_sum'] = df_log_length['time'].cumsum()
        log_length_list = df_log_length['time'].tolist()
        start_point_list = df_log_length['cumulative_sum'].tolist()[:-1]
        start_point_list.insert(0, 0)
        seq = [data.iloc[start_point_list[i]:start_point_list[i] + log_length_list[i]] for i in
               range(len(log_length_list))]

        # pool.map returns results as a list
        results_list = list(tqdm(pool.imap(func, seq), total=len(seq)))

        # return list of processed rows, concatenated together as a new dataframe
        df_result = pd.concat(results_list, axis=0)

    # save the result
    os.chdir('/home/yisendu/xu-g1/URAP/Yisen_Spring/data/')
    df_result.to_csv("Merchant_Moving_Distance_Complete.csv")
