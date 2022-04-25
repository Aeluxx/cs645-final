import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, DistanceMetric

def getRepresentation(data, max_size=1000, verbose=False):
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

    final_partition = pd.concat(partition)

    if verbose:
        print("Scanned: " + str(summation))

    return final_partition
