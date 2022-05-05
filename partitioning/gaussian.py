# Reference https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
import math

import pandas as pd
from sklearn.cluster import KMeans


CLUSTER_COLS = ["sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "sum_qty",
                "count_order", "p_size", "ps_min_supplycost", "revenue", "o_totalprice", "o_shippriority"]


# Jahnavi Tirunagari

def createPartions(df: pd.DataFrame, cluster_num):
    clusters = cluster_num
    gaussian = KMeans(n_clusters=clusters, random_state=0)
    gaussian = gaussian.fit(df[CLUSTER_COLS])
    df["cluster_label"] = gaussian.labels_
    cluster_df = pd.DataFrame(data=gaussian.cluster_centers_, columns=CLUSTER_COLS)
    print(cluster_df)
    labels_1 = []
    labels_2 = []
    sizes = []
    for i in range(0, clusters):
        labels_1.append(i)
        labels_2.append("rep" + str(i))
        sizes.append(len(df[df.cluster_label == i]))
    cluster_df["cluster_label"] = labels_1
    cluster_df.set_index([pd.Index(labels_2)], inplace=True)
    return cluster_df, sizes, df
