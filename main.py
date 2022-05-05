from direct_method import directMethod
from sketch_refine import sketch, greedy_refine
from partitioning import kmeans, kdtree, gaussian
import pulp
import pandas as pd
import time
import math
import numpy as np
from queries import get_query

# Matthew Gregory

def testQueries(to_solve, to_read, id, data_range, cluster_range, mode=0):
    total = []

    ilp = to_solve
    csv_df = to_read

    data_min, data_max, data_increment = data_range
    cluster_min, cluster_max, cluster_increment = cluster_range
    query = get_query(id)
    flag, A0, constraints, count_constraint = query

    for data in range(int(data_min / data_increment), int(data_max / data_increment) + 1):
        for cluster in range(int(cluster_min / cluster_increment), int(cluster_max / cluster_increment) + 1):
            print("(" + str(data) + ", " + str(cluster) + ")")
            data_actual = data * data_increment
            cluster_actual = cluster * cluster_increment
            start = time.time()

            size = math.ceil(len(csv_df) * data_actual)
            small_df = csv_df.head(size)

            if mode == 0:
                cluster_df, cluster_size, df = kmeans.createPartions(small_df, cluster_actual)
                sketched = sketch(cluster_df, cluster_size, query)
                if sketched is None:
                    total.append([id, data, cluster, "---", "---"])
                    continue

                df = df.drop('id', axis=1)

                result, junk = greedy_refine(ilp, query, cluster_df, cluster_df, sketched, df)
            elif mode == 1:
                cluster_df, cluster_size, df = kdtree.getRepresentation(small_df, int(0.05 * (len(small_df) / cluster_actual)))
                sketched = sketch(cluster_df, cluster_size, query)
                if sketched is None:
                    total.append([id, data, cluster, "---", "---"])
                    continue

                result, junk = greedy_refine(ilp, query, cluster_df, cluster_df, sketched, df)
            elif mode == 2:
                cluster_df, cluster_size, df = kmeans.createPartions(small_df, cluster_actual)

                df = df.drop('id', axis=1)

                cluster_df = df.groupby("cluster_label", as_index=False).min()

                sketched = sketch(cluster_df, cluster_size, query)
                if sketched is None:
                    total.append([id, data, cluster, "---", "---"])
                    continue

                result, junk = greedy_refine(ilp, query, cluster_df, cluster_df, sketched, df)

            # Gaussian
            elif mode == 3:
                cluster_df, cluster_size, df = gaussian.createPartions(small_df, cluster_actual)
                sketched = sketch(cluster_df, cluster_size, query)
                if sketched is None:
                    total.append([id, data, cluster, "---", "---"])
                    continue

                df = df.drop('id', axis=1)

                result, junk = greedy_refine(ilp, query, cluster_df, cluster_df, sketched, df)
            else:
                result = directMethod(ilp, small_df, query)

            if A0 is None:
                sum = len(result)
            else:
                sum = np.sum(result[A0])
            end = time.time()
            total.append([id, data_actual, cluster_actual, (end - start), sum])

    return pd.DataFrame(total, columns=('query', 'data_percent', 'clusters', 'time', 'objective'))


if __name__ == "__main__":
    total = []
    data_sizes = (0.1, 0.1, 0.1)
    cluster_sizes = (10, 100, 10)
    ilp = pulp.CPLEX_CMD(path="/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex")
    csv_df = pd.read_csv('./tpch.csv', sep=',')

    to_test = [1, 3, 4]
    for i in to_test:
        to_add = testQueries(ilp, csv_df, i, data_sizes, cluster_sizes, 3)
        total.append(to_add)

    result_df = pd.concat(total)
    print("RESULTS")
    print(result_df)
