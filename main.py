# Jahnavi Tirunagari: vtirunagari@umass.edu
# Matthew Gregory: matthewgrego@umass.edu
# Shruti Jasoria: sjasoria@umass.edu

from direct_method import directMethod
from sketch_refine import sketch, greedy_refine
from partitioning import kmeans, kdtree, gaussian, quad_tree
import pulp
import pandas as pd
import time
import math
import sys
import numpy as np
from queries import get_query

def testQueries(to_solve, to_read, id, data_range, cluster_range, mode=0):
    total = []
    results = []

    ilp = to_solve
    csv_df = to_read

    data_min, data_max, data_increment = data_range
    cluster_min, cluster_max, cluster_increment = cluster_range
    query = get_query(id)
    flag, A0, constraints, count_constraint = query

    runs = math.ceil((data_max + data_increment - data_min) / data_increment), math.ceil((cluster_max + cluster_increment - cluster_min) / cluster_increment)

    for data in range(0, runs[0]):
        for cluster in range(0, runs[1]):
            print("(" + str(data) + ", " + str(cluster) + ")")
            data_actual = min(data_max, data_min + (data * data_increment))
            cluster_actual = int(min(cluster_max, cluster_min + (cluster * cluster_increment)))
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

            elif mode == 4:
                cluster_df, cluster_size, df = gaussian.createPartions(small_df, cluster_actual)

                df = df.drop('id', axis=1)

                cluster_df = df.groupby("cluster_label", as_index=False).min()

                sketched = sketch(cluster_df, cluster_size, query)
                if sketched is None:
                    total.append([id, data, cluster, "---", "---"])
                    continue

                result, junk = greedy_refine(ilp, query, cluster_df, cluster_df, sketched, df)

            elif mode == 5:
                cluster_df, cluster_size, df = quad_tree.genRepresenation(small_df, size * 0.1)

                sketched = sketch(cluster_df, cluster_size, query)
                if sketched is None:
                    total.append([id, data, cluster, "---", "---"])
                    continue

                result, junk = greedy_refine(ilp, query, cluster_df, cluster_df, sketched, df)
            else:
                result = directMethod(ilp, small_df, query)

            if A0 is None:
                sum = len(result)
            else:
                sum = np.sum(result[A0])
            end = time.time()
            total.append([id, data_actual, cluster_actual, (end - start), sum])
            results.append(result)

    return pd.DataFrame(total, columns=('query', 'data_percent', 'clusters', 'time', 'objective')), results


def demo_run(arguments):
    data_size, cluster_size, query, mode, file = arguments
    data_size = int(data_size)
    cluster_size = float(cluster_size)
    query = int(query)
    mode = int(mode)
    total = []

    data_sizes = (data_size, data_size, 1)
    cluster_sizes = (cluster_size, cluster_size, 1)
    ilp = pulp.CPLEX_CMD(path="/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex")
    csv_df = pd.read_csv('./' + str(file), sep=',')
    to_test = [query]
    for i in to_test:
        to_add, results = testQueries(ilp, csv_df, i, data_sizes, cluster_sizes, mode)
        total.append(to_add)

    result_df = pd.concat(total)
    print("RESULTS")
    print(result_df)
    print("")

    for result in results:
        print(result)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        total = []
        data_sizes = (0.1, 1, 0.1)
        cluster_sizes = (10, 10, 1)
        ilp = pulp.CPLEX_CMD(path="/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex")
        csv_df = pd.read_csv('./tpch_small.csv', sep=',')

        to_test = [5]
        for i in to_test:
            to_add, results = testQueries(ilp, csv_df, i, data_sizes, cluster_sizes, 0)
            total.append(to_add)

        result_df = pd.concat(total)
        print("RESULTS")
        print(result_df)
    elif len(sys.argv) == 6:
        arguments = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        if sys.argv[1] == 0:
            raise ValueError("Cannot examine 0 data!")
        demo_run(arguments)
    else:
        raise ValueError("Must have 0 or 5 inputs in the form data_size (0.1 to 1) cluster_size (any non-negative "
                         "number up to the number of tuples, query (1-4 from example, 5-6 small queries 1 & 2), "
                         "mode (0 = KDTREE, 1 = KMEANS, 2 = KMEANS (MIN), 3 = GAUSSIAN, 4 = GAUSSIAN (MIN), 5 = "
                         "QUADTREE, 6+ = DIRECT), and file name (csv format)")
