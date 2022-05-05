import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import math
import numpy as np

CLUSTER_COLS = ["sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "sum_qty",
                "count_order", "p_size", "ps_min_supplycost", "revenue", "o_totalprice", "o_shippriority"]


# Jahnavi Tirunagari

def createPartitions(df: pd.DataFrame, cluster_num):
    clusters = cluster_num
    hierarchical = AgglomerativeClustering(n_clusters=clusters)
    hierarchical = hierarchical.fit(df[CLUSTER_COLS])
    df["cluster_label"] = hierarchical.labels_
    dict_with_sizes = df["cluster_label"].value_counts().to_dict()
    sizes = []
    for label in range(clusters):
        sizes.append(dict_with_sizes[label])
    cluster_df = df.groupby("cluster_label", as_index=False).mean()
    return cluster_df, sizes, df


if __name__ == "__main__":
    df = pd.read_csv('../tpch.csv', sep=',')
    size = math.ceil(len(df) * .1)
    df = df.head(size)
    print("Number rows in df: ", len(df.index))
    rep_df = createPartitions(df, 10)
    print(rep_df)
