import copy
from itertools import count
import math
import random
from traceback import print_tb

import numpy as np
import pandas as pd
import pulp
from pulp.pulp import LpVariable
from partitioning import kmeans

# Need these global variables for greedy-refine
flag = None
A0 = None
constraints = []
count_constraint = {}

def sketch(data, flag, A0, constraints, count_constraint):
	# Initialising the problem
    if flag:
        # Maximise
        problem = pulp.LpProblem("PAQL Maximise", pulp.LpMaximize)
    else:
        problem = pulp.LpProblem("PAQL Minimize", pulp.LpMinimize)

    # Since variables are binary objects (0 or 1)
    indices = LpVariable.dict('x_', data.cluster_label, cat='Binary')
    print(indices)

    variables = indices.values()
    print(variables)
    attr_col = data[A0]
    # print(attr_col) # dim : n, 1
    # print(variables) # dim : 1, n
    if A0 is not None:
        problem += pulp.lpSum(np.transpose(np.asarray(attr_col)) * np.asarray(list(variables)))
    # print(problem)
    else:
        problem += pulp.lpSum(np.asarray(list(variables)))

    # COUNT CONSTRAINT
    Lc = count_constraint['LC']
    Uc = count_constraint['UC']

    if Lc is not None:
        problem += pulp.lpSum(variables) >= Lc
    if Uc is not None:
        problem += pulp.lpSum(variables) <= Uc
    # print(problem)

	# Group Constraints

    # ATTRIBUTE CONSTRAINT
    for constraint in constraints:
        attr_variables = []
        if constraint is not None:
            attr_variables = np.transpose(np.asarray(data[constraint[0]])) * np.asarray(list(variables))

        # Lower bound on attribute
        if constraint[1] is not None:
            problem += pulp.lpSum(attr_variables) >= constraint[1]

        # Upper bound on attribute
        if constraint[2] is not None:
            problem += pulp.lpSum(attr_variables) <= constraint[2]

    # Added the solver
    ilp = pulp.CPLEX_CMD()
    problem.solve(ilp)
    # print("Status:", pulp.LpStatus[problem.status])
    # Get the result tuples
    # print(problem)
    res = []
    # print(problem.variables())
    for var in problem.variables():
        if var.varValue != 0:
            # print(var)
            res.append(data.iloc[int(var.name[3:]), :])

    # Return the result dataframe
    final_df = pd.DataFrame(res)
    return final_df

## partitions is list of GIDs
## refiningPackage is initially result of sketch. In this algo we keep replacing
## rep tuples with actual tuples from the partition
def greedy_refine(partitions, refiningPackage):
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
    if len(partitions) == 0:
        return refiningPackage
    U = copy.deepcopy(partitions)
    print("U: ", U)
    while len(U) != 0:
        gid = U.pop()
        partition_df = pd.read_csv('./part'+ str(int(gid)) + '.csv', sep=',')
        updated_refined_package = refiningPackage[refiningPackage["cluster_label"] != gid]
        problem = constructRefinedQuery(updated_refined_package, partition_df)
        ilp = pulp.CPLEX_CMD()
        problem.solve(ilp)
        if problem.status == 1:
            # problem was solved. replace the rep tuple with result. continue
            # solving greedily 
            # update refining package
            res = []
            for var in problem.variables():
                if var.varValue != 0:
                    # print(var)
                    res.append(partition_df.iloc[int(var.name[3:]), :])
            result = pd.DataFrame(res)
            print("Result: ", result)
            concat_df = pd.concat([updated_refined_package, result], axis = 0, ignore_index=True)
            print("Concat: ", concat_df)
            # remove current gid from partitions
            partitions.remove(gid)
            refined= greedy_refine(partitions, concat_df)
            print("Refined: ", refined)
            if refined is not None:
                return refined    
        else:
            return None
    return None

# This is present on page 9 second column of the paper
def constructRefinedQuery(repRelation, currentPartition):
    # Initialising the problem
    if flag:
        # Maximise
        problem = pulp.LpProblem("PAQL Maximise", pulp.LpMaximize)
    else:
        problem = pulp.LpProblem("PAQL Minimize", pulp.LpMinimize)

    # Since variables are binary objects (0 or 1)
    indices = LpVariable.dict('x', currentPartition.index, cat='Binary')
    # Since variables are binary objects (0 or 1)
    rep_indices = LpVariable.dict('y', repRelation.index, cat='Binary')
    # print(indices)

    variables = list(indices.values())
    # print(type(variables))
    all_variables = variables + list(rep_indices.values())

    attr_col = currentPartition[A0]
    # print(attr_col) # dim : n, 1
    # print(variables) # dim : 1, n
    if A0 is not None:
        problem += pulp.lpSum(np.transpose(np.asarray(attr_col)) * np.asarray(list(variables)))
    # print(problem)
    else:
        problem += pulp.lpSum(np.asarray(list(variables)))

    # COUNT CONSTRAINT
    Lc = count_constraint['LC']
    Uc = count_constraint['UC']

    if Lc is not None:
        problem += pulp.lpSum(all_variables) >= Lc
    if Uc is not None:
        problem += pulp.lpSum(all_variables) <= Uc
    # print(problem)

	# Group Constraints
    problem += pulp.lpSum(indices.values()) == 1

    concat_df = pd.concat([currentPartition,repRelation], axis = 0, ignore_index=True)
    # ATTRIBUTE CONSTRAINT
    for constraint in constraints:
        attr_variables = []
        if constraint is not None:
            attr_variables = np.transpose(np.asarray(concat_df[constraint[0]])) * np.asarray(list(all_variables))

        # Lower bound on attribute
        if constraint[1] is not None:
            problem += pulp.lpSum(attr_variables) >= constraint[1]

        # Upper bound on attribute
        if constraint[2] is not None:
            problem += pulp.lpSum(attr_variables) <= constraint[2]
    return problem
    	


if __name__ == "__main__":
    df = pd.read_csv('./tpch.csv', sep=',')
    size = math.ceil(len(df) * .1)
    small_df = df.head(size)
    cluster_df = kmeans.createPartions(small_df)
    flag = True
    A0 = "count_order"
    constraints = [['sum_base_price', None, 15469853.7043], ['sum_disc_price', None, 45279795.0584],
                    ['sum_charge', None, 95250227.7918], ['avg_qty', None, 50.353948653],
                    ['avg_price', None, 68677.5852459], ['avg_disc', None, 0.110243522496],
                    ['sum_qty', None, 77782.028739]]
    count_constraint = {'LC': 1, "UC": None}
    result =  sketch(cluster_df, flag, A0, constraints, count_constraint)
    P = result["cluster_label"].tolist()
    print("#######################################################################")
    print(P)
    result = greedy_refine(P, result)
    print("#######################################################################")
    print(result)
	