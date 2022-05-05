# Jahnavi Tirunagari: vtirunagari@umass.edu

import numpy as np
import pandas as pd
import pulp
from pulp.pulp import LpVariable

def directMethod(ilp, data, query, offsets=None):
    flag, A0, constraints, count_constraint = query
    if flag:
        problem = pulp.LpProblem("PAQL_Maximise", pulp.LpMaximize)
    else:
        problem = pulp.LpProblem("PAQL_Minimize", pulp.LpMinimize)

    indices = LpVariable.dict('x_', data.index, cat='Binary')

    variables = indices.values()
    if A0 is not None:
        attr_col = data[A0]
        problem += pulp.lpSum(np.transpose(np.asarray(attr_col)) * np.asarray(list(variables)))
    else:
        problem += pulp.lpSum(np.asarray(list(variables)))

    # COUNT CONSTRAINT
    Lc = count_constraint['LC']
    Uc = count_constraint['UC']

    if Lc is not None:
        problem += pulp.lpSum(variables) >= Lc
    if Uc is not None:
        problem += pulp.lpSum(variables) <= Uc

    # ATTRIBUTE CONSTRAINT
    for i in range(0, len(constraints)):
        offset = 0

        attr_variables = []

        if constraints[i] is not None:
            attr_variables = np.transpose(np.asarray(data[constraints[i][0]])) * np.asarray(list(variables))
            if offsets is not None:
                offset = offsets[i]

        # Lower bound on attribute
        if constraints[i][1] is not None:
            problem += pulp.lpSum(attr_variables) >= constraints[i][1] + offset

        # Upper bound on attribute
        if constraints[i][2] is not None:
            problem += pulp.lpSum(attr_variables) <= constraints[i][2] + offset

    # Added the solver
    problem.solve(ilp)

    # Get the result tuples
    res = []
    for var in problem.variables():
        if var.varValue != 0:
            index = int(var.name[(var.name.rindex('_') + 1):])
            res.append(data.loc[index])

    # Return the result dataframe
    final_df = pd.DataFrame(res)
    return final_df
