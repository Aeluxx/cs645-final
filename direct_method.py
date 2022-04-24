from pulp.pulp import LpVariable
import math
import pandas as pd
import pulp
import numpy as np
import cplex


def directMethod(data, flag, A0, constraints, count_constraint):
    # Initialising the problem
    if flag:
        # Maximise
        problem = pulp.LpProblem("PAQL Maximise", pulp.LpMaximize)
    else:
        problem = pulp.LpProblem("PAQL Minimize", pulp.LpMinimize)

    # Since variables are binary objects (0 or 1)
    indices = LpVariable.dict('x_', data.index, cat='Binary')
    # print(indices)

    variables = indices.values()
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

    print("Direct")
    # Added the solver
    ilp = pulp.CPLEX_CMD(path='/Users/jahnavitirunagari/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex')
    problem.solve(ilp)
    print("Status:", pulp.LpStatus[problem.status])
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


# Testing
if __name__ == "__main__":
    df = pd.read_csv('tpch.csv', sep=',')
    size = math.ceil(len(df) * .1)
    small_df = df.head(size)

    print(directMethod(small_df, True, "count_order",
                       [['sum_base_price', None, 15469853.7043], ['sum_disc_price', None, 45279795.0584],
                        ['sum_charge', None, 95250227.7918], ['avg_qty', None, 50.353948653],
                        ['avg_price', None, 68677.5852459], ['avg_disc', None, 0.110243522496],
                        ['sum_qty', None, 77782.028739]],
                       {'LC': 1, "UC": None}))
