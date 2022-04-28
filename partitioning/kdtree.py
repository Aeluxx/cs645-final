import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, DistanceMetric

def getRepresentation(data, max_size=35000, verbose=False):
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

    partition = []
    current = data
    summation = 0
    for x in total:
        for i in range(0, data.shape[1]):
            minimum = min(x[i], x[i+data.shape[1]])
            maximum = max(x[i], x[i+data.shape[1]])
            indices = (current.iloc[:, i] <= maximum) & (current.iloc[:, i] >= minimum)
            current = current.loc[indices]
        temp = data
        temp.drop(current.index, axis=0, inplace=True)
        summation += len(current)
        partition.append(current)
        current = temp
        print(current.shape)
    num_partitions = len(partition)
    for label in range(num_partitions):
        df = partition[label]
        df["cluster_label"] = label
        filepath = "part" + str(label) + ".csv"
        df.to_csv(path_or_buf=filepath, index=False)
    final_partition = pd.concat(partition)
    rep_df = final_partition.groupby("cluster_label", as_index=False).mean()
    print(rep_df.head())
    rep_df.set_index([pd.Index(["rep1","rep2","rep3","rep4","rep5","rep6","rep7","rep8","rep9","rep10","rep11","rep12","rep13","rep14","rep15","rep16"])], inplace=True)
    print(rep_df.head())

    # if verbose:
    #     print("Scanned: " + str(summation))

    return rep_df


# TODO: For testing. Reomve the following portion later
if __name__ == "__main__":
    df = pd.read_csv('../tpch.csv', sep=',')
    size = math.ceil(len(df) * .1)
    small_df = df.head(size)
    print("Number rows in small df: ", len(small_df.index))
    # Below portion goes into the method
    cluster_df = getRepresentation(small_df)
    print("Cluster DF: ")
    print(cluster_df.head())
    print("Size: ", len(cluster_df))
    cluster_df.to_csv('kdtree.csv')
