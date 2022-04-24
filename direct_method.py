# (1) a table name --> dataframe
# (2) “MIN” or “MAX” for the objective --> flag
# (3) an attribute name Ao for the objective function or None --> A0
# (4) a list of pairs of the form (Lk, Uk), for each attribute Ak of the table for which there is a
# constraint, and --> list of [attribute name, Lk, Uk]
# (5) a pair (Lc, Uc).  ----> {Lc : val, Uc: val}
# The objective will be
# MINIMIZE (or MAXIMIZE) SUM(Ao)
# if Ao is not None, or
# MINIMIXE/MAXIMIZE COUNT(*)
# otherwise.

import math

import numpy as np
import pandas as pd
import pulp
from pulp.constants import LpMaximize
from pulp.pulp import LpVariable


def directMethod(data, flag, A0, constraints, count_constraint):

  #Initialising the problem
  if flag:
    #Maximise
    problem = pulp.LpProblem("PAQL Maximise", pulp.LpMaximize)
  else:
    problem = pulp.LpProblem("PAQL Minimize", pulp.LpMinimize)

  #Since variables are binary objects (0 or 1)
  indices = LpVariable.dict('x_', data.index, cat='Binary')
  # print(indices)

  variables = indices.values()
  attr_col = data[A0]
  # print(attr_col) # dim : n, 1
  # print(variables) # dim : 1, n
  if A0 != None:
    problem += pulp.lpSum(np.transpose(np.asarray(attr_col)) * np.asarray(list(variables)))
  # print(problem)
  else:
    problem += pulp.lpSum(np.asarray(list(variables)))

  #COUNT CONSTRAINT
  Lc = count_constraint['LC']
  Uc = count_constraint['UC']

  if(Lc != None):
    problem += pulp.lpSum(variables) >= Lc
  if(Uc != None):
    problem += pulp.lpSum(variables) <= Uc
  # print(problem)

  #ATTRIBUTE CONSTRAINT
  for constraint in constraints:
    attr_variables = []
    if constraint != None:
      attr_variables = np.transpose(np.asarray(data[constraint[0]])) * np.asarray(list(variables))

    #Lower bound on attribute
    if(constraint[1] != None):
      problem += pulp.lpSum(attr_variables) >= constraint[1]

    #Upper bound on attribute
    if(constraint[2] != None):
      problem += pulp.lpSum(attr_variables) <= constraint[2]

    # print(problem)
    
  ilp = pulp.CPLEX_CMD(keepFiles=True)
  result = problem.solve(ilp)

  #TODO : Add the solver
  # Get the result tuples 
  # Return the result dataframe

if __name__ == "__main__":
  print("Direct Running")
  df = pd.read_csv('tpch.csv', sep=',')
  size = math.ceil(len(df) * .001)
  small_df = df.head(size)
  directMethod(small_df, True, "count_order", [['sum_base_price', None, 15469853]], {'LC' : 1, "UC" : None})
