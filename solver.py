import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, field
from ortools.sat.python import cp_model

SCALE = 1000


@dataclass
class SolverResult:
    status: str           # "OPTIMAL" / "FEASIBLE" / "INFEASIBLE"
    total_distance: float
    route_rows: dict      # dict[int, list[dict]] — per vessel
    service_rows: list    # list[dict]
    fig: object           # plotly Figure or None


def build_scenario(params):
    """Generate customers, charging stations, and vessels from params."""
    rng = np.random.default_rng(seed=params['SEED'])

    VESSEL_SPEED = params['VESSEL_SPEED']
    SIM_PERIOD = params['SIM_PERIOD']
    RATE_DISCHARGE = params['RATE_DISCHARGE']
    N_VESSELS = params['N_VESSELS']
    AREA_LEN = params['AREA_LEN']
    N_CUSTOMERS = params['N_CUSTOMERS']
    MAX_DEMAND = params['MAX_DEMAND_PER_CUSTOMER']

    customers = []
    for c in range(N_CUSTOMERS):
        start_time = int(rng.integers(0, SIM_PERIOD - 10))
        customers.append({
            'name': f'cust{c}',
            'location_start': np.array([int(rng.integers(0, AREA_LEN)), int(rng.integers(0, AREA_LEN))]),
            'location_end': np.array([int(rng.integers(0, AREA_LEN)), int(rng.integers(0, AREA_LEN))]),
            'start_time': start_time,
            'start_window': np.array([start_time, start_time + 10]),
            'demand': int(rng.integers(1, MAX_DEMAND + 1)),
        })

    charging_stations = [
        {'name': 'charger0', 'location': np.array([0, 0]),         'rate_charge': 5},
        {'name': 'charger1', 'location': np.array([AREA_LEN, AREA_LEN]), 'rate_charge': 5},
    ]

    vessels = [
        {
            'name': f'boat{i}',
            'capacity_cargo': params['VESSEL_CARGO_CAPACITY'],
            'capacity_charge': params['VESSEL_MAX_CHARGE'],
            'rate_discharge': RATE_DISCHARGE,
            'speed': VESSEL_SPEED,
        }
        for i in range(N_VESSELS)
    ]

    return customers, charging_stations, vessels


def build_graph(customers, charging_stations, vessels, params):
    """Build nodes and edges from scenario data."""
    VESSEL_SPEED = params['VESSEL_SPEED']
    SIM_PERIOD = params['SIM_PERIOD']
    RATE_DISCHARGE = params['RATE_DISCHARGE']
    N_DUPLICATE_CHARGE_NODES = params['N_DUPLICATE_CHARGE_NODES']
    AREA_LEN = params['AREA_LEN']
    MAX_DEMAND = params['MAX_DEMAND_PER_CUSTOMER']

    # Recreate rng and advance past customer generation to reach charger-jitter state
    rng = np.random.default_rng(seed=params['SEED'])
    for _ in range(len(customers)):
        rng.integers(0, SIM_PERIOD - 10)   # start_time
        rng.integers(0, AREA_LEN)           # location_start[0]
        rng.integers(0, AREA_LEN)           # location_start[1]
        rng.integers(0, AREA_LEN)           # location_end[0]
        rng.integers(0, AREA_LEN)           # location_end[1]
        rng.integers(1, MAX_DEMAND + 1)     # demand

    nodes = []

    # Source depot
    nodes.append({
        'node_id': len(nodes),
        'location_type': 'depot',
        'location': charging_stations[0]['location'].copy(),
        'demand': 0,
        'start_time': 0,
        'start_window': np.array([0, SIM_PERIOD]),
    })

    # Customer pickup / delivery nodes
    for cust in customers:
        nodes.append({
            'node_id': len(nodes),
            'location_type': 'start',
            'location': cust['location_start'],
            'demand': cust['demand'],
            'start_time': cust['start_time'],
            'start_window': cust['start_window'],
        })
        nodes.append({
            'node_id': len(nodes),
            'location_type': 'end',
            'location': cust['location_end'],
            'demand': -cust['demand'],
            'start_time': cust['start_time'],
            'start_window': np.array([cust['start_time'], SIM_PERIOD]),
        })

    # Charger nodes (with same jitter as original)
    for charger in charging_stations:
        for _ in range(N_DUPLICATE_CHARGE_NODES):
            nodes.append({
                'node_id': len(nodes),
                'location_type': 'charger',
                'location': np.round(charger['location'] + rng.uniform(-0.5, 0.5, size=2), 3),
                'rate_charge': charger['rate_charge'],
                'demand': 0,
                'start_time': 0,
                'start_window': np.array([0, SIM_PERIOD]),
            })

    # Sink depot
    nodes.append({
        'node_id': len(nodes),
        'location_type': 'depot',
        'location': charging_stations[0]['location'].copy(),
        'demand': 0,
        'start_time': 0,
        'start_window': np.array([0, SIM_PERIOD]),
    })

    # All-to-all edges
    edges = []
    for idx in range(len(nodes)):
        for jdx in range(len(nodes)):
            if idx == jdx:
                continue
            d_ij = np.linalg.norm(nodes[idx]['location'] - nodes[jdx]['location'])
            edges.append({
                'edge_id': len(edges),
                'd_ij': d_ij,
                't_ij': d_ij / VESSEL_SPEED,
                'rd_ij': RATE_DISCHARGE * d_ij,
            })

    return nodes, edges


def build_model(nodes, edges, customers, vessels, params):
    """Build CP-SAT model and return (model, variables)."""
    SIM_PERIOD = params['SIM_PERIOD']
    VESSEL_SPEED = params['VESSEL_SPEED']
    RATE_DISCHARGE = params['RATE_DISCHARGE']
    N_VESSELS = params['N_VESSELS']

    model = cp_model.CpModel()
    N = len(nodes)
    source = 0
    sink = N - 1

    MAX_CARGO = max(v['capacity_cargo'] for v in vessels)
    MAX_CHARGE = max(v['capacity_charge'] for v in vessels)

    x = [[[model.NewBoolVar(f'x_{i}_{j}_{k}') for k in range(N_VESSELS)] for i in range(N)] for j in range(N)]
    time_var       = [[model.NewIntVar(0, SIM_PERIOD * SCALE,  f'time_{i}_{k}')        for k in range(N_VESSELS)] for i in range(N)]
    cargo_var      = [[model.NewIntVar(0, MAX_CARGO,            f'cargo_{i}_{k}')       for k in range(N_VESSELS)] for i in range(N)]
    charge_var     = [[model.NewIntVar(0, MAX_CHARGE * SCALE,  f'charge_{i}_{k}')      for k in range(N_VESSELS)] for i in range(N)]
    charge_time_var= [[model.NewIntVar(0, SIM_PERIOD * SCALE,  f'charge_time_{i}_{k}') for k in range(N_VESSELS)] for i in range(N)]

    cust_node_index   = [i for i in range(N) if nodes[i]['location_type'] in ('start', 'end')]
    charge_node_index = [i for i in range(N) if nodes[i]['location_type'] == 'charger']

    # Every customer node must be visited exactly once
    for i in cust_node_index:
        model.Add(sum(x[i][j][k] for j in range(1, N) for k in range(N_VESSELS) if i != j) == 1)

    # Each charger node visited at most once
    for i in charge_node_index:
        model.Add(sum(x[i][j][k] for j in range(1, N) for k in range(N_VESSELS) if i != j) <= 1)

    # Flow conservation + depot departure/arrival
    for k in range(N_VESSELS):
        for i in range(1, N - 1):
            model.Add(
                sum(x[j][i][k] for j in range(N) if j != i) ==
                sum(x[i][j][k] for j in range(N) if j != i)
            )
        model.Add(sum(x[source][j][k] for j in range(1, N)) == 1)
        model.Add(sum(x[i][sink][k]   for i in range(0, N - 1)) == 1)

    # Prevent arcs between depot/charger nodes at the same location (no free loops)
    for k in range(N_VESSELS):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if (nodes[i]['location_type'] in ('depot', 'charger') and
                        nodes[j]['location_type'] in ('depot', 'charger') and
                        np.array_equal(nodes[i]['location'], nodes[j]['location'])):
                    if not (i == source and j == sink):
                        model.Add(x[i][j][k] == 0)

    # Time propagation along arcs
    for k in range(N_VESSELS):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                t_ij = int(np.ceil(np.linalg.norm(nodes[i]['location'] - nodes[j]['location']) / VESSEL_SPEED * SCALE))
                model.Add(time_var[j][k] >= time_var[i][k] + charge_time_var[i][k] + t_ij).OnlyEnforceIf(x[i][j][k])

    # Helper: visits[i][k] = 1 iff vessel k visits node i
    visits = [[model.NewBoolVar(f'visits_{i}_{k}') for k in range(N_VESSELS)] for i in range(N)]
    for k in range(N_VESSELS):
        for i in range(N):
            model.Add(sum(x[j][i][k] for j in range(N) if j != i) == 1).OnlyEnforceIf(visits[i][k])
            model.Add(sum(x[j][i][k] for j in range(N) if j != i) == 0).OnlyEnforceIf(visits[i][k].Not())

    # Time windows
    for k in range(N_VESSELS):
        for i in range(N):
            tw_lo = int(nodes[i]['start_window'][0] * SCALE)
            tw_hi = int(nodes[i]['start_window'][1] * SCALE)
            model.Add(time_var[i][k] >= tw_lo).OnlyEnforceIf(visits[i][k])
            model.Add(time_var[i][k] <= tw_hi).OnlyEnforceIf(visits[i][k])

    # Pickup-delivery pairing (same vessel, pickup before delivery)
    for c in range(len(customers)):
        pickup   = 1 + 2 * c
        delivery = 2 + 2 * c
        for k in range(N_VESSELS):
            model.Add(visits[pickup][k] == visits[delivery][k])
            model.Add(time_var[pickup][k] <= time_var[delivery][k]).OnlyEnforceIf(visits[pickup][k])

    # Cargo must be 0 when entering a charger
    for k in range(N_VESSELS):
        for i in charge_node_index:
            model.Add(cargo_var[i][k] == 0).OnlyEnforceIf(visits[i][k])

    # Cargo flow
    for k in range(N_VESSELS):
        model.Add(cargo_var[source][k] == 0)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                model.Add(cargo_var[j][k] == cargo_var[i][k] + nodes[j]['demand']).OnlyEnforceIf(x[i][j][k])

    # Charge time is 0 at non-charger nodes
    for k in range(N_VESSELS):
        for i in range(N):
            if nodes[i]['location_type'] != 'charger':
                model.Add(charge_time_var[i][k] == 0)

    # Charge flow
    for k in range(N_VESSELS):
        model.Add(charge_var[source][k] == MAX_CHARGE * SCALE)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                rd_ij = int(np.ceil(RATE_DISCHARGE * np.linalg.norm(nodes[i]['location'] - nodes[j]['location']) * SCALE))
                if nodes[i]['location_type'] == 'charger':
                    charge_gained = model.NewIntVar(0, MAX_CHARGE * SCALE, f'charge_gained_{i}_{j}_{k}')
                    rate_c = nodes[i]['rate_charge']
                    model.Add(charge_gained == rate_c * charge_time_var[i][k])
                    model.Add(charge_var[j][k] == charge_var[i][k] + charge_gained - rd_ij).OnlyEnforceIf(x[i][j][k])
                    model.Add(charge_var[i][k] + charge_gained <= MAX_CHARGE * SCALE)
                else:
                    model.Add(charge_var[j][k] == charge_var[i][k] - rd_ij).OnlyEnforceIf(x[i][j][k])

    # Objective: minimise total travel distance
    total_distance = model.NewIntVar(0, SIM_PERIOD * N_VESSELS * SCALE, 'total_distance')
    arc_distances = []
    for k in range(N_VESSELS):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d_ij = int(np.ceil(np.linalg.norm(nodes[i]['location'] - nodes[j]['location']) * SCALE))
                arc_dist = model.NewIntVar(0, max(d_ij, 0), f'arc_dist_{i}_{j}_{k}')
                model.Add(arc_dist == d_ij).OnlyEnforceIf(x[i][j][k])
                model.Add(arc_dist == 0).OnlyEnforceIf(x[i][j][k].Not())
                arc_distances.append(arc_dist)
    model.Add(total_distance == sum(arc_distances))
    model.Minimize(total_distance)

    variables = {
        'x': x,
        'time_var': time_var,
        'cargo_var': cargo_var,
        'charge_var': charge_var,
        'charge_time_var': charge_time_var,
        'visits': visits,
        'total_distance': total_distance,
        'source': source,
        'sink': sink,
        'N': N,
        'cust_node_index': cust_node_index,
        'charge_node_index': charge_node_index,
    }

    return model, variables


def solve(model, variables, nodes, customers, vessels, params):
    """Run CP-SAT solver and return a SolverResult."""
    N_VESSELS = params['N_VESSELS']

    x           = variables['x']
    time_var    = variables['time_var']
    cargo_var   = variables['cargo_var']
    charge_var  = variables['charge_var']
    visits      = variables['visits']
    source      = variables['source']
    sink        = variables['sink']
    N           = variables['N']

    cp_solver = cp_model.CpSolver()
    cp_solver.parameters.max_time_in_seconds = 30.0
    cp_solver.parameters.num_workers = 4
    status_code = cp_solver.Solve(model)

    if status_code == cp_model.OPTIMAL:
        status_str = 'OPTIMAL'
    elif status_code == cp_model.FEASIBLE:
        status_str = 'FEASIBLE'
    else:
        return SolverResult(status='INFEASIBLE', total_distance=0.0,
                            route_rows={}, service_rows=[], fig=None)

    total_dist = cp_solver.ObjectiveValue() / SCALE

    # Route rows per vessel
    route_rows = {}
    for k in range(N_VESSELS):
        rows = []
        current = source
        while current != sink:
            next_node = next(
                j for j in range(N)
                if j != current and cp_solver.Value(x[current][j][k]) == 1
            )
            d_ij = np.linalg.norm(nodes[current]['location'] - nodes[next_node]['location'])
            rows.append({
                'From':    f'{current}({nodes[current]["location_type"]})',
                'To':      f'{next_node}({nodes[next_node]["location_type"]})',
                'Dist':    round(d_ij, 3),
                'T_dep':   round(cp_solver.Value(time_var[current][k]) / SCALE, 2),
                'T_arr':   round(cp_solver.Value(time_var[next_node][k]) / SCALE, 2),
                'Cargo_i': cp_solver.Value(cargo_var[current][k]),
                'Cargo_j': cp_solver.Value(cargo_var[next_node][k]),
                'Chg_i':   round(cp_solver.Value(charge_var[current][k]) / SCALE, 1),
                'Chg_j':   round(cp_solver.Value(charge_var[next_node][k]) / SCALE, 1),
            })
            current = next_node
        route_rows[k] = rows

    # Customer service times
    service_rows = []
    for c in range(len(customers)):
        pickup   = 1 + 2 * c
        delivery = 2 + 2 * c
        for k in range(N_VESSELS):
            if cp_solver.Value(visits[pickup][k]) == 1:
                t_pick = cp_solver.Value(time_var[pickup][k]) / SCALE
                t_del  = cp_solver.Value(time_var[delivery][k]) / SCALE
                t_wait = t_pick - nodes[pickup]['start_window'][0]
                service_rows.append({
                    'Customer':   f'cust{c}',
                    'Vessel':     vessels[k]['name'],
                    'T_pickup':   round(t_pick, 2),
                    'T_delivery': round(t_del, 2),
                    'T_service':  round(t_del - t_pick, 2),
                    'T_wait':     round(t_wait, 2),
                })
                break

    # Route plot with plotly (interactive)
    vessel_colors = ['#0099FF', '#FF7700', '#00AA00', '#FF0000', '#9933FF', '#00CCFF']

    fig = go.Figure()

    # Add vessel routes as arrows
    for k in range(N_VESSELS):
        color   = vessel_colors[k % len(vessel_colors)]
        current = source
        step    = 0
        while current != sink:
            next_node = next(
                j for j in range(N)
                if j != current and cp_solver.Value(x[current][j][k]) == 1
            )
            loc_i = nodes[current]['location']
            loc_j = nodes[next_node]['location']

            # Arrow line
            fig.add_trace(go.Scatter(
                x=[loc_i[0], loc_j[0]],
                y=[loc_i[1], loc_j[1]],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo='skip',
                name=f'{vessels[k]["name"]} step {step}'
            ))

            # Step number at midpoint
            mid_x = (loc_i[0] + loc_j[0]) / 2
            mid_y = (loc_i[1] + loc_j[1]) / 2
            fig.add_annotation(
                x=mid_x, y=mid_y,
                text=str(step),
                showarrow=False,
                font=dict(size=10, color=color),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor=color,
                borderwidth=1
            )

            current = next_node
            step   += 1

    # Add depot nodes
    depot_nodes = [n for n in nodes if n['location_type'] == 'depot']
    if depot_nodes:
        depot_x = [n['location'][0] for n in depot_nodes]
        depot_y = [n['location'][1] for n in depot_nodes]
        fig.add_trace(go.Scatter(
            x=depot_x, y=depot_y,
            mode='markers',
            marker=dict(symbol='square', size=12, color='gray'),
            name='Depot',
            text=['Depot (start/end)'] * len(depot_x),
            hovertemplate='<b>Depot</b><br>Location: (%{x:.1f}, %{y:.1f})<extra></extra>'
        ))

    # Add pickup nodes
    pickup_nodes = [n for n in nodes if n['location_type'] == 'start']
    if pickup_nodes:
        pickup_x = [n['location'][0] for n in pickup_nodes]
        pickup_y = [n['location'][1] for n in pickup_nodes]
        cust_ids = [(n['node_id'] - 1) // 2 for n in pickup_nodes]
        fig.add_trace(go.Scatter(
            x=pickup_x, y=pickup_y,
            mode='markers+text',
            marker=dict(symbol='circle', size=10, color='green'),
            text=[f'P{c}' for c in cust_ids],
            textposition='top center',
            customdata=[c for c in cust_ids],
            name='Pickup (P)',
            hovertemplate='<b>Pickup</b><br>Customer: cust%{customdata}<br>Location: (%{x:.1f}, %{y:.1f})<extra></extra>',
            textfont=dict(size=9)
        ))

    # Add delivery nodes
    delivery_nodes = [n for n in nodes if n['location_type'] == 'end']
    if delivery_nodes:
        delivery_x = [n['location'][0] for n in delivery_nodes]
        delivery_y = [n['location'][1] for n in delivery_nodes]
        cust_ids = [(n['node_id'] - 1) // 2 for n in delivery_nodes]
        fig.add_trace(go.Scatter(
            x=delivery_x, y=delivery_y,
            mode='markers+text',
            marker=dict(symbol='x', size=12, color='red'),
            text=[f'D{c}' for c in cust_ids],
            textposition='top center',
            customdata=[c for c in cust_ids],
            name='Delivery (D)',
            hovertemplate='<b>Delivery</b><br>Customer: cust%{customdata}<br>Location: (%{x:.1f}, %{y:.1f})<extra></extra>',
            textfont=dict(size=9)
        ))

    # Add charger nodes
    charger_nodes = [n for n in nodes if n['location_type'] == 'charger']
    if charger_nodes:
        charger_x = [n['location'][0] for n in charger_nodes]
        charger_y = [n['location'][1] for n in charger_nodes]
        fig.add_trace(go.Scatter(
            x=charger_x, y=charger_y,
            mode='markers',
            marker=dict(symbol='triangle-up', size=11, color='magenta'),
            name='Charger',
            text=['Charger node'] * len(charger_x),
            hovertemplate='<b>Charger</b><br>Location: (%{x:.1f}, %{y:.1f})<extra></extra>'
        ))

    # Add vessel route legend
    for k in range(N_VESSELS):
        color = vessel_colors[k % len(vessel_colors)]
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color, width=3),
            name=f'{vessels[k]["name"]} route',
            hoverinfo='skip'
        ))

    fig.update_layout(
        title='Vehicle Routes (Interactive)',
        xaxis=dict(title='X coordinate', showgrid=True),
        yaxis=dict(title='Y coordinate', showgrid=True),
        hovermode='closest',
        height=1000,
        width=800,
        legend=dict(
            title='<b>Legend</b><br><sub>■ Depot (start/end)<br>● Pickup (P)<br>✕ Delivery (D)<br>▲ Charger<br>―→ Vessel routes</sub>',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        )
    )

    return SolverResult(
        status=status_str,
        total_distance=total_dist,
        route_rows=route_rows,
        service_rows=service_rows,
        fig=fig,
    )
