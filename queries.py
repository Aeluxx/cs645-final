# Matthew Gregory: matthewgrego@umass.edu

def get_query(id):
    if id == 1:
        flag = True
        A0 = "count_order"
        constraints = [['sum_base_price', None, 15469853.7043], ['sum_disc_price', None, 45279795.0584],
                       ['sum_charge', None, 95250227.7918], ['avg_qty', None, 50.353948653],
                       ['avg_price', None, 68677.5852459], ['avg_disc', None, 0.110243522496],
                       ['sum_qty', None, 77782.028739]]
        count_constraint = {'LC': 1, "UC": None}
    elif id == 2:
        flag = False
        A0 = "ps_min_supplycost"
        constraints = [['p_size', None, 8]]
        count_constraint = {'LC': 1, "UC": None}
    elif id == 3:
        flag = False
        A0 = None
        constraints = [['revenue', 413930.849506, None]]
        count_constraint = {'LC': 1, "UC": None}
    else:
        # Assumed to be id = 4
        flag = False
        A0 = None
        constraints = [['o_totalprice', None, 453998.242103], ['o_shippriority', 3, None]]
        count_constraint = {'LC': 1, "UC": None}

    return flag, A0, constraints, count_constraint