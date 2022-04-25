# Reference https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
import math

import pandas as pd
from sklearn.cluster import KMeans

# TODO : Remove hard coded list
ALL_COLUMNS = ["id", "sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "sum_qty",  "count_order",  "p_size",  "ps_min_supplycost", "revenue", "o_totalprice", "o_shippriority"]
CLUSTER_COLS = ["sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "sum_qty",  "count_order",  "p_size",  "ps_min_supplycost", "revenue", "o_totalprice", "o_shippriority"]

def createPartions(df : pd.DataFrame):
    kmeans = KMeans(n_clusters=10)
    # Cluster cols is used so that the column "id" is not used for clustering
    kmeans = kmeans.fit(df[CLUSTER_COLS])
    print("Number of lables: ", len(kmeans.labels_))
    print("Unique elements in labels: ", set(kmeans.labels_))
    df["cluster_label"] = kmeans.labels_
    cluster_df = pd.DataFrame(data=kmeans.cluster_centers_, columns=CLUSTER_COLS)
    # TODO : Remove hard coded list
    cluster_df["cluster_label"] = [0,1,2,3,4,5,6,7,8,9]
    # Write all clusters to csv
    for label in set(kmeans.labels_):
        filepath = "part" + str(label) + ".csv"
        label_partition_df = df[df["cluster_label"] == label]
        label_partition_df.to_csv(path_or_buf=filepath, index=False)
    max_df = df.groupby("cluster_label").max()
    min_df = df.groupby("cluster_label").min()
    return cluster_df, max_df, min_df

# TODO: For testing. Reomve the following portion later
if __name__ == "__main__":
    df = pd.read_csv('../tpch.csv', sep=',')
    size = math.ceil(len(df) * .1)
    small_df = df.head(size)
    print("Number rows in small df: ", len(small_df.index))
    # Below portion goes into the method
    cluster_df, max_df, min_df = createPartions(small_df)
    print("Cluster DF: ")
    print(cluster_df.head())
    print("Max DF: ")
    print(max_df.head())
    print("Min DF: ")
    print(min_df.head())
