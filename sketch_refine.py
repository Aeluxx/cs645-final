# Matthew Gregory: matthewgrego@umass.edu
# Shruti Jasoria: sjasoria@umass.edu

import numpy as np
import pandas as pd
import pulp

from direct_method import directMethod
from partitioning import kmeans
from queries import get_query

pd.options.mode.chained_assignment = None  # default='warn'

csv_df = None

# Matthew Gregory (matthewgrego@umass.edu)
# Algorithm referenced from PaSQL paper

# Based on direct_method.py by Jahnavi Tirunagari
def sketch(data, sizes, query):
    flag, A0, constraints, count_constraint = query
    if flag:
        problem = pulp.LpProblem("PAQL_Maximise", pulp.LpMaximize)
    else:
        problem = pulp.LpProblem("PAQL_Minimize", pulp.LpMinimize)

    counts = pulp.LpVariable.dicts("Count", range(0, len(sizes)), cat='Integer')
    for i in range(0, len(sizes)):
        problem += counts[i] <= sizes[i]
        problem += counts[i] >= 0

    if A0 is not None:
        numpy_data = data[A0].to_numpy()
        problem += pulp.lpSum([numpy_data[i] * counts[i] for i in range(0, len(sizes))])
    else:
        problem += pulp.lpSum([counts[i] for i in range(0, len(sizes))])

    Lc = count_constraint['LC']
    Uc = count_constraint['UC']

    if Lc is not None:
        problem += pulp.lpSum([counts[i] for i in range(0, len(sizes))]) >= Lc
    if Uc is not None:
        problem += pulp.lpSum([counts[i] for i in range(0, len(sizes))]) <= Uc

    for constraint in constraints:
        numpy_data = data[constraint[0]].to_numpy()
        if constraint[1] is not None:
            problem += pulp.lpSum([numpy_data[i] * counts[i] for i in range(0, len(sizes))]) \
                       >= constraint[2]

        if constraint[2] is not None:
            problem += pulp.lpSum([numpy_data[i] * counts[i] for i in range(0, len(sizes))])\
                       <= constraint[2]

    ilp = pulp.CPLEX_CMD(path="/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex")
    problem.solve(ilp)
    solution = np.array([counts[i].varValue for i in range(0, len(sizes))])
    total = []
    if solution[0] is None:
        return None
    for q in range(0, len(solution)):
        for i in range(0, int(solution[q])):
            total.append(data.iloc[q])
    final_df = pd.DataFrame(total)

    return final_df

def greedy_refine(ilp, query, to_refine, groups, package, df):
    failures = []
    S = to_refine
    priority = []
    if S.empty:
        return package, failures
    flag, A0, constraints, count_constraint = query
    while len(S) != 0 or len(priority) != 0:
        if len(priority) == 0:
            first = S.iloc[0]
            S = S.iloc[1:]
        else:
            first = priority[0]
            priority = priority[1:]
        representative, group = first.drop('cluster_label'), first['cluster_label']

        temp_package = package.loc[package['cluster_label'] != group]
        if temp_package.shape[0] == package.shape[0]:
            if S.empty:
                return package, failures
            continue

        offsets = get_offsets(constraints, temp_package)
        group_data = df.loc[df['cluster_label'] == group]

        direct = directMethod(ilp, group_data, query, offsets=offsets)

        if not direct.empty:
            new_package = pd.concat([temp_package, direct], axis=0)
            rec_package, rec_failure = greedy_refine(ilp, query, S, groups, new_package, df)
            if len(rec_failure) > 0:
                for failure in rec_failure:
                    failures.append(failure)
                for failure in failures:
                    priority.append(failure)
            else:
                return rec_package, failures
        else:
            if not S.equals(groups):
                failures.append(first)
                return None, failures
    return None, failures

def get_offsets(constraints, data):
    to_Return = []
    for i in range(0, len(constraints)):
        to_Return.append(np.sum(np.asarray(data[constraints[i][0]])))
    return to_Return
