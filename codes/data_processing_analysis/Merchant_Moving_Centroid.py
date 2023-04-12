"""
The transaction logs of a merchant's various locations, identified by their latitude and longitude coordinates, suggest
that they relocate frequently in order to operate their business.

This module utilizes these movement patterns to estimate the central locations of their operations
using K-means clustering. The optimal number of clusters (k value) is determined using the Silhouette Method.

Author: Yisen Du
Date: April 7, 2023
"""


# Import statements
import warnings
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Constant definitions and global configurations
warnings.filterwarnings("ignore")


# Function definitions
def preprocess_data():
    """
    read and preprocess the data
    """
    os.chdir('/home/yisendu/xu-g1/URAP/Yisen_Spring/data/')
    data_bet = pd.read_csv("/BET/BET_dataset.csv")
    data_log = pd.read_csv("merchant_multiple_lat_long.csv")

    # filter the log data by merchant id's that are appeared in data_bet
    id_bet = data_bet["User ID"].tolist()
    data_log_filter = data_log[data_log['usr_id'].isin(id_bet)]

    # the lat long has 6 decimals with measure error. Drop duplicates according to the first 3 decimals
    data_log_filter['lat_3_decimal'] = data_log_filter['lat'].apply(lambda x: (str(x)[0:6]))
    data_log_filter['long_3_decimal'] = data_log_filter['long'].apply(lambda x: (str(x)[0:6]))
    data_log_filter['unique_location'] = data_log_filter['lat_3_decimal'] + data_log_filter['long_3_decimal']
    data_result = data_log_filter.drop_duplicates(['usr_id', 'unique_location'])

    return data_result


def kmeans_compute(data):
    """
    Compute the centroids using k-means
    """
    # decide best k value
    sil = []
    k_list = []

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, 5):  # 5 is determined by the domain knowledge
        kmeans = KMeans(n_clusters=k).fit(data)
        labels = kmeans.labels_
        sil.append(silhouette_score(data, labels, metric='euclidean'))
        k_list.append(k)
    k_dict = dict(zip(k_list, sil))
    # choose k with the greatest silhouette score
    k_best_value = max(k_dict, key=k_dict.get)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k_best_value)
    kmeans.fit(X)

    # Get cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    result_x_list = [round(centers[x][0], 6) for x in labels]
    result_y_list = [round(centers[x][1], 6) for x in labels]

    result = df_temp[['usr_id', 'lat', 'long']]
    result['centroid_x'] = result_x_list
    result['centroid_y'] = result_y_list

    return result


# Main program
if __name__ == "__main__":
    # Code to execute when the script is run
    # load the data
    df_lat_long = preprocess_data()

    # compute the centroids for each merchant
    # use for-loop
    id_list = df_lat_long['usr_id'].drop_duplicates().tolist()
    # create a dataframe to store the result
    df_centroid = pd.DataFrame({"usr_id": [],
                                "lat": [],
                                'long': [],
                                'centroid_x': [],
                                'centroid_y': []})

    for index in id_list:
        df_temp = df_lat_long[df_lat_long['usr_id'] == index]
        X = df_temp[['lat', 'long']]  # used for k-mean model
        if len(X) <= 3:
            continue

        df_result = kmeans_compute(X)
        df_centroid = pd.concat([df_centroid, df_result], axis=0)

    # save the result
    os.chdir('/home/yisendu/xu-g1/URAP/Yisen_Spring/data/BET')
    df_centroid.to_csv("BET_centoid.csv")
