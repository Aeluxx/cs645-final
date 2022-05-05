# Matthew Gregory: matthewgrego@umass.edu
# Shruti Jasoria: sjasoria@umass.edu

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, DistanceMetric

CLUSTER_COLS = ["sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "sum_qty",  "count_order",  "p_size",  "ps_min_supplycost", "revenue", "o_totalprice", "o_shippriority", "cluster_label"]

# Matthew Gregory

def getRepresentation(data, max_size=1000):
    distance = DistanceMetric.get_metric('minkowski', p=data.shape[1])
    data = data.iloc[:, 1:]
    tree_data, index, tree_nodes, node_bounds = KDTree(data, leaf_size=max_size, metric=distance).get_arrays()

    dt = np.dtype('int,int,int,float')
    step_1 = np.array(tree_nodes, dtype=dt)
    step_2 = np.array(step_1, dtype=dt)
    step_3 = np.array(step_2['f0'])
    first_partition = np.max(np.where(step_3 < 1))
    node_bounds = node_bounds[:, first_partition:, :]

    start = np.array(node_bounds[0])
    end = np.array(node_bounds[1])
    total = np.concatenate((start, end), axis=1)

    group = 0
    current = data

    partition = []
    sizes = []
    for x in total:
        for i in range(0, data.shape[1]):
            minimum = min(x[i], x[i+data.shape[1]])
            maximum = max(x[i], x[i+data.shape[1]])
            indices = (current.iloc[:, i] <= maximum) & (current.iloc[:, i] >= minimum)
            current = current.loc[indices]

        current = current.reset_index()
        sizes.append(current.shape[0])
        current['cluster_label'] = pd.Series([group for x in range(0, current.shape[0])])
        group += 1

        partition.append(current)
        current = data.drop(current.index, axis=0, inplace=False)

    final_partition = pd.concat(partition)

    final_partition = final_partition[CLUSTER_COLS]
    cluster_df = final_partition.groupby("cluster_label", as_index=False).mean()

    return cluster_df, sizes, final_partition
