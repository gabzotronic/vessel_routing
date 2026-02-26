import numpy as np
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
%matplotlib qt
plt.close('all')

#__________ SETUP __________#
VESSEL_SPEED = 1 # distance/time
SIM_PERIOD = 500 # time units
RATE_DISCHARGE = 2 # charge/distance
N_VESSELS = 2
N_DUPLICATE_CHARGE_NODES = 3
AREA_LEN = 20
N_CUSTOMERS = 5
MAX_DEMAND_PER_CUSTOMER = 5

rng = np.random.default_rng(seed=42)
customers = []
for c in range(N_CUSTOMERS):
    start_time = rng.integers(0, SIM_PERIOD - 10)
    customers.append({
        'name': f'cust{c}',
        'location_start': np.array([rng.integers(0, AREA_LEN), rng.integers(0, AREA_LEN)]),
        'location_end': np.array([rng.integers(0, AREA_LEN), rng.integers(0, AREA_LEN)]),
        'start_time': start_time,
        'start_window': np.array([start_time, start_time + 10]),
        'demand': rng.integers(1, MAX_DEMAND_PER_CUSTOMER + 1),
    })

charging_stations = [
    {
        'name': 'charger0',
        'location': np.array([0,0]),
        'rate_charge': 5 # capacity/time_unit
    },
    {
        'name': 'charger1',
        'location': np.array([AREA_LEN, AREA_LEN]),
        'rate_charge': 5 # capacity/time_unit
    },
]

vessels = [
    {
        'name': f'boat{i}',
        'capacity_cargo': 5,
        'capacity_charge': 100,
        'rate_discharge': RATE_DISCHARGE, # capacity/distance
        'speed': VESSEL_SPEED, # distance/time
    }
    for i in range(N_VESSELS)
]

# # visualise scenario
# plt.figure()
# plt.grid(True)
# for cust in customers:
#     plt.plot(cust['location_start'][0], cust['location_start'][1], 'go', label='cust_src')
#     plt.plot(cust['location_end'][0], cust['location_end'][1], 'rx', label='cust_dest')
#     plt.plot(
#         [cust['location_start'][0], cust['location_end'][0]],
#         [cust['location_start'][1], cust['location_end'][1]], 'k'
#     )
# for charger in charging_stations:
#     plt.plot(charger['location'][0], charger['location'][1], 'ms', label=charger['name'])
# plt.legend()


#__________ BUILD GRAPH __________#

nodes, edges = [], []

# source depot node
node = {
    'node_id': len(nodes),
    'location_type': 'depot',
    'location': charging_stations[0]['location'], # assume start from 0-th charging station
    'demand': 0, # non cust
    'start_time': 0, # no limit
    'start_window': np.array([0,SIM_PERIOD]), # no limit
}
nodes.append(node)


# customer nodes
for cust in customers:
    # pickup node
    node = {
        'node_id': len(nodes),
        'location_type': 'start',
        'location': cust['location_start'],
        'demand': cust['demand'],
        'start_time': cust['start_time'],
        'start_window': cust['start_window'],
    }
    nodes.append(node)
    # delivery node
    travel_time = np.linalg.norm(cust['location_end'] - cust['location_start'])/VESSEL_SPEED
    node = {
        'node_id': len(nodes),
        'location_type': 'end',
        'location': cust['location_end'],
        'demand': -cust['demand'],
        'start_time': cust['start_time'] ,
        'start_window': np.array([cust['start_time'], SIM_PERIOD])
    }
    nodes.append(node)

# charger nodes
for charger in charging_stations:
    for n in range(N_DUPLICATE_CHARGE_NODES):
        node = {
            'node_id': len(nodes),
            'location_type': 'charger',
            'location': np.round(charger['location'] + rng.uniform(-0.5, 0.5, size=2), 3),
            'rate_charge': charger['rate_charge'], # capacity/time_unit
            'demand': 0,
            'start_time': 0, # no limit
            'start_window': np.array([0,SIM_PERIOD]), # no limit
        }
        nodes.append(node)

# sink depot node
node = {
    'node_id': len(nodes),
    'location_type': 'depot',
    'location': charging_stations[0]['location'], # assume start from 0-th charging station
    'demand': 0, # non cust
    'start_time': 0,
    'start_window': np.array([0,SIM_PERIOD]), # no limit
}
nodes.append(node)

# edges
for idx in range(len(nodes)):
    for jdx in range(len(nodes)):
        if idx == jdx: continue
        
        # travel distance
        d_ij = np.linalg.norm(nodes[idx]['location'] - nodes[jdx]['location'])

        # travel time
        t_ij = d_ij/VESSEL_SPEED

        # charge consumed
        rd_ij = RATE_DISCHARGE * d_ij

        edge = {
            'edge_id': len(edges),
            'd_ij': d_ij,
            't_ij': t_ij,
            'rd_ij': rd_ij,
        }
        edges.append(edge)



#__________ BUILD CP-SAT MODEL __________#
print('building problem')
model = cp_model.CpModel()
N = len(nodes)


# decision variables
x = [[[model.NewBoolVar(f'x_{i}_{j}_{k}') for k in range(N_VESSELS)] for i in range(N)] for j in range(N)]
SCALE = 1000  # CP-SAT needs integers, so scale floats
MAX_CARGO = max(v['capacity_cargo'] for v in vessels)
MAX_CHARGE = max(v['capacity_charge'] for v in vessels)

time_var = [[model.NewIntVar(0, SIM_PERIOD * SCALE, f'time_{i}_{k}') for k in range(N_VESSELS)] for i in range(N)]
cargo_var = [[model.NewIntVar(0, MAX_CARGO, f'cargo_{i}_{k}') for k in range(N_VESSELS)] for i in range(N)]
charge_var = [[model.NewIntVar(0, MAX_CHARGE * SCALE, f'charge_{i}_{k}') for k in range(N_VESSELS)] for i in range(N)]
# charge_time_var: how long vessel k spends charging at node i (0 for non-charger nodes)
charge_time_var = [[model.NewIntVar(0, SIM_PERIOD * SCALE, f'charge_time_{i}_{k}') for k in range(N_VESSELS)] for i in range(N)]

# constraints
# all customer nodes need to be visited
cust_node_index = [idx for idx in range(N) if nodes[idx]['location_type'] in ['start', 'end']]
for i in cust_node_index:
    model.Add(sum(x[i][j][k] for j in range(1,N) for k in range(N_VESSELS) if i != j) == 1)

# charging station visits
charge_node_index = [idx for idx in range(N) if nodes[idx]['location_type'] == "charger"]
for i in charge_node_index:
    model.Add(sum(x[i][j][k] for j in range(1,N) for k in range(N_VESSELS) if i != j) <= 1)

# flow constraint: for each vessel k at each intermediate node i, inflow == outflow
source = 0
sink = N - 1
for k in range(N_VESSELS):
    for i in range(1, N - 1):  # skip source and sink
        model.Add(
            sum(x[j][i][k] for j in range(N) if j != i)  # inflow
            == sum(x[i][j][k] for j in range(N) if j != i)  # outflow
        )
    # each vessel leaves the source exactly once
    model.Add(sum(x[source][j][k] for j in range(1, N)) == 1)
    # each vessel arrives at the sink exactly once
    model.Add(sum(x[i][sink][k] for i in range(0, N - 1)) == 1)

# prevent arcs between depot and charger nodes at the same location (avoids free loops)
for k in range(N_VESSELS):
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if (nodes[i]['location_type'] in ['depot', 'charger'] and
                nodes[j]['location_type'] in ['depot', 'charger'] and
                np.array_equal(nodes[i]['location'], nodes[j]['location'])):
                # only allow source -> sink directly
                if not (i == source and j == sink):
                    model.Add(x[i][j][k] == 0)


# time constraint: if vessel k travels arc i->j, then time[j][k] >= time[i][k] + travel_time + charging_time at i
for k in range(N_VESSELS):
    for i in range(N):
        for j in range(N):
            if i == j: continue
            t_ij = int(np.ceil(np.linalg.norm(nodes[i]['location'] - nodes[j]['location']) / VESSEL_SPEED * SCALE))
            model.Add(time_var[j][k] >= time_var[i][k] + charge_time_var[i][k] + t_ij).OnlyEnforceIf(x[i][j][k])

# helper variable: visits[i][k] = 1 if vessel k visits node i
visits = [[model.NewBoolVar(f'visits_{i}_{k}') for k in range(N_VESSELS)] for i in range(N)]
for k in range(N_VESSELS):
    for i in range(N):
        # visits[i][k] == 1 iff any arc j->i is used by vessel k
        model.Add(sum(x[j][i][k] for j in range(N) if j != i) == 1).OnlyEnforceIf(visits[i][k])
        model.Add(sum(x[j][i][k] for j in range(N) if j != i) == 0).OnlyEnforceIf(visits[i][k].Not())

# time window constraint: arrival at node i must be within its window (only if vessel k visits node i)
for k in range(N_VESSELS):
    for i in range(N):
        tw_lo = int(nodes[i]['start_window'][0] * SCALE)
        tw_hi = int(nodes[i]['start_window'][1] * SCALE)
        model.Add(time_var[i][k] >= tw_lo).OnlyEnforceIf(visits[i][k])
        model.Add(time_var[i][k] <= tw_hi).OnlyEnforceIf(visits[i][k])

# pickup-delivery pairing: same vessel must serve both pickup and delivery for each customer
# customer nodes are at indices 1,2 (cust 0), 3,4 (cust 1), etc. (pickup=odd, delivery=even)
n_customers = len(customers)
for c in range(n_customers):
    pickup = 1 + 2 * c   # pickup node index
    delivery = 2 + 2 * c  # delivery node index
    for k in range(N_VESSELS):
        # if vessel k visits pickup, it must also visit delivery (and vice versa)
        model.Add(visits[pickup][k] == visits[delivery][k])
        # pickup must happen before delivery
        model.Add(time_var[pickup][k] <= time_var[delivery][k]).OnlyEnforceIf(visits[pickup][k])

# cargo constraint: if vessel k travels arc i->j, cargo[j][k] = cargo[i][k] + demand[j]
for k in range(N_VESSELS):
    # start at depot with 0 cargo
    model.Add(cargo_var[source][k] == 0)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            demand_j = nodes[j]['demand']
            model.Add(cargo_var[j][k] == cargo_var[i][k] + demand_j).OnlyEnforceIf(x[i][j][k])

# charging time is 0 at non-charger nodes
for k in range(N_VESSELS):
    for i in range(N):
        if nodes[i]['location_type'] != 'charger':
            model.Add(charge_time_var[i][k] == 0)

# charge constraint: if vessel k travels arc i->j,
#   charge[j][k] = charge[i][k] + energy_gained_at_i - energy_consumed_on_arc_ij
# energy_gained_at_i = rate_charge * charge_time (only at charger nodes, 0 elsewhere)
for k in range(N_VESSELS):
    # start at depot fully charged
    model.Add(charge_var[source][k] == MAX_CHARGE * SCALE)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            rd_ij = int(np.ceil(RATE_DISCHARGE * np.linalg.norm(nodes[i]['location'] - nodes[j]['location']) * SCALE))
            if nodes[i]['location_type'] == 'charger':
                # charge gained = rate_charge * charge_time
                # since both are scaled, we need: rate_charge * (charge_time / SCALE) * SCALE = rate_charge * charge_time
                # but rate_charge * charge_time would overflow the linear range, so we use an intermediate variable
                charge_gained = model.NewIntVar(0, MAX_CHARGE * SCALE, f'charge_gained_{i}_{k}')
                # charge_gained is in charge * SCALE units
                # rate_charge [charge/time] * charge_time_var [time * SCALE] = charge * SCALE
                rate_c = nodes[i]['rate_charge']  # capacity/time_unit
                model.Add(charge_gained == rate_c * charge_time_var[i][k])
                model.Add(charge_var[j][k] == charge_var[i][k] + charge_gained - rd_ij).OnlyEnforceIf(x[i][j][k])
                # can't exceed max charge
                model.Add(charge_var[i][k] + charge_gained <= MAX_CHARGE * SCALE)
            else:
                model.Add(charge_var[j][k] == charge_var[i][k] - rd_ij).OnlyEnforceIf(x[i][j][k])

#__________ OBJECTIVE __________#

# minimise total travel distance
total_distance = model.NewIntVar(0, SIM_PERIOD * N_VESSELS * SCALE, 'total_distance')
arc_distances = []
for k in range(N_VESSELS):
    for i in range(N):
        for j in range(N):
            if i == j: continue
            d_ij = int(np.ceil(np.linalg.norm(nodes[i]['location'] - nodes[j]['location']) * SCALE))
            arc_dist = model.NewIntVar(0, d_ij, f'arc_dist_{i}_{j}_{k}')
            model.Add(arc_dist == d_ij).OnlyEnforceIf(x[i][j][k])
            model.Add(arc_dist == 0).OnlyEnforceIf(x[i][j][k].Not())
            arc_distances.append(arc_dist)
model.Add(total_distance == sum(arc_distances))
model.Minimize(total_distance)

#__________ SOLVE __________#
print('solving problem')
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
solver.parameters.num_workers = 4
status = solver.Solve(model)

#__________ RESULTS __________#

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'Status: {"OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"}')
    print(f'Total distance: {solver.ObjectiveValue() / SCALE:.3f}')
    for k in range(N_VESSELS):
        print(f'\nVessel {k} ({vessels[k]["name"]}):')
        header = f'  {"From":<15} {"To":<15} {"Dist":>6} {"T_dep":>8} {"T_arr":>8} {"Cargo_i":>8} {"Cargo_j":>8} {"Chg_i":>8} {"Chg_j":>8}'
        print(header)
        print('  ' + '-' * (len(header) - 2))
        current = source
        while current != sink:
            next_node = None
            for j in range(N):
                if current != j and solver.Value(x[current][j][k]) == 1:
                    next_node = j
                    break
            d_ij = np.linalg.norm(nodes[current]['location'] - nodes[next_node]['location'])
            from_str = f'{current}({nodes[current]["location_type"]})'
            to_str = f'{next_node}({nodes[next_node]["location_type"]})'
            print(f'  {from_str:<15} {to_str:<15} {d_ij:>6.2f} '
                  f'{solver.Value(time_var[current][k])/SCALE:>8.2f} '
                  f'{solver.Value(time_var[next_node][k])/SCALE:>8.2f} '
                  f'{solver.Value(cargo_var[current][k]):>8} '
                  f'{solver.Value(cargo_var[next_node][k]):>8} '
                  f'{solver.Value(charge_var[current][k])/SCALE:>8.1f} '
                  f'{solver.Value(charge_var[next_node][k])/SCALE:>8.1f}')
            current = next_node

    # customer service times
    print(f'\nCustomer Service Times:')
    print(f'  {"Customer":<10} {"Vessel":<10} {"T_pickup":>10} {"T_delivery":>10} {"T_service":>10}')
    print('  ' + '-' * 52)
    for c in range(n_customers):
        pickup = 1 + 2 * c
        delivery = 2 + 2 * c
        for k in range(N_VESSELS):
            if solver.Value(visits[pickup][k]) == 1:
                t_pickup = solver.Value(time_var[pickup][k]) / SCALE
                t_delivery = solver.Value(time_var[delivery][k]) / SCALE
                print(f'  {"cust" + str(c):<10} {vessels[k]["name"]:<10} {t_pickup:>10.2f} {t_delivery:>10.2f} {t_delivery - t_pickup:>10.2f}')
                break

    #__________ PLOT ROUTES __________#

    vessel_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    fig, ax = plt.subplots()
    ax.set_title('Vehicle Routes')
    ax.grid(True)

    # plot nodes
    for node in nodes:
        loc = node['location']
        if node['location_type'] == 'depot':
            ax.plot(loc[0], loc[1], 'ks', markersize=10, zorder=5)
            ax.annotate('depot', (loc[0], loc[1]), textcoords='offset points', xytext=(5, 5))
        elif node['location_type'] == 'charger':
            ax.plot(loc[0], loc[1], 'm^', markersize=9, zorder=5)
        elif node['location_type'] == 'start':
            cust_id = (node['node_id'] - 1) // 2
            ax.plot(loc[0], loc[1], 'go', markersize=8, zorder=5)
            ax.annotate(f'P{cust_id}', (loc[0], loc[1]), textcoords='offset points', xytext=(5, 5))
        elif node['location_type'] == 'end':
            cust_id = (node['node_id'] - 1) // 2
            ax.plot(loc[0], loc[1], 'rx', markersize=8, zorder=5)
            ax.annotate(f'D{cust_id}', (loc[0], loc[1]), textcoords='offset points', xytext=(5, 5))

    # plot vessel routes using chain traversal for correct ordering
    for k in range(N_VESSELS):
        color = vessel_colors[k % len(vessel_colors)]
        current = source
        step = 1
        while current != sink:
            next_node = None
            for j in range(N):
                if current != j and solver.Value(x[current][j][k]) == 1:
                    next_node = j
                    break
            loc_i = nodes[current]['location']
            loc_j = nodes[next_node]['location']
            ax.annotate('', xy=(loc_j[0], loc_j[1]), xytext=(loc_i[0], loc_i[1]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            mid_x = (loc_i[0] + loc_j[0]) / 2
            mid_y = (loc_i[1] + loc_j[1]) / 2
            ax.annotate(f'{step}', (mid_x, mid_y), fontsize=8, fontweight='bold',
                        color=color, backgroundcolor='white', ha='center', va='center')
            current = next_node
            step += 1
        # legend entry
        ax.plot([], [], '-', color=color, lw=2, label=vessels[k]['name'])

    ax.legend()
    plt.show()

else:
    print(f'No solution found. Status: {solver.StatusName(status)}')
