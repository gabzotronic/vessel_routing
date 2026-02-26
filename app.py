import streamlit as st
import pandas as pd
from solver import build_scenario, build_graph, build_model, solve

st.set_page_config(page_title="Electric Vessel Routing Demo", layout="wide")
st.title("⚓ Electric Vessel Routing Demo")

# ── Sidebar parameters ──────────────────────────────────────────────────────
with st.sidebar:

    run_clicked = st.button(
        "▶  Run Solver",
        disabled=st.session_state.get("is_solving", False),
        use_container_width=True,
    )

    with st.container(border=True):
        st.header("SCENARIO PARAMETERS", divider="blue")
        AREA_LEN = st.slider(
            "Area-of-Ops Length", min_value=10, max_value=25, value=15, step=5,
            help='In distance units. AO is assumed square'
        )
        SEED = st.slider(
            "RNG Seed", min_value=0, max_value=100, value=42, step=1,
            help="Customer Pickup/Dropoff Locations are determined randomly. Change seed value to influence locations."
        )
        SIM_PERIOD = st.slider(
            "Simulation Period", min_value=50, max_value=1000, value=300, step=10,
            help='Shorter simulation forces more vessels to be used, but could result in infeasible solution'
        )
        N_DUPLICATE_CHARGE_NODES = st.slider(
            "Number of Charging Ports (SEE ? FOR DETAILS)", min_value=1, max_value=5, value=3, step=1,
            help="Two charging stations fixed at opposite corners of AO. \nThis parameter is an approximation for number of charging ports. \nOptimiser assumes charging ports can only be used ONCE per simulation period, so be generous here ;)"
        )

    with st.container(border=True):
        st.header("CUSTOMER PARAMETERS", divider="orange")
        N_CUSTOMERS = st.slider(
            "Num Pickup/Delivery Orders", min_value=5, max_value=10, value=5, step=1
        )
        MAX_DEMAND_PER_CUSTOMER = st.slider(
            "Max Cargo per Order", min_value=1, max_value=5, value=5, step=1,
            help="Max number of cargo/personnel to transport per Order. Actual number is randomly generated per order"
        )

    with st.container(border=True):
        st.header("VESSEL PARAMETERS", divider="green")
        N_VESSELS = st.slider(
            "Number of Vessels", min_value=1, max_value=3, value=2, step=1,
            help="Number of Vessels"
        )
        VESSEL_SPEED = st.slider(
            "Vessel Speed", min_value=1, max_value=10, value=1, step=1,
            help="Distance units travelled per time unit"
        )

        RATE_DISCHARGE = st.slider(
            "Discharge Rate", min_value=1, max_value=10, value=2, step=1,
            help="Battery discharge rate per distance unit"
        )

        VESSEL_CARGO_CAPACITY = st.slider(
            "Vessel Cargo Capacity", min_value=1, max_value=20, value=5, step=1,
            help="Max cargo/personnel each vessel can carry at any point in time"
        )
        VESSEL_MAX_CHARGE = st.slider(
            "Vessel Battery Capacity", min_value=50, max_value=500, value=100, step=50,
            help="Battery capacity (energy units) per vessel"
        )

    

params = {
    "VESSEL_SPEED":             VESSEL_SPEED,
    "SIM_PERIOD":               SIM_PERIOD,
    "RATE_DISCHARGE":           RATE_DISCHARGE,
    "N_VESSELS":                N_VESSELS,
    "N_DUPLICATE_CHARGE_NODES": N_DUPLICATE_CHARGE_NODES,
    "AREA_LEN":                 AREA_LEN,
    "N_CUSTOMERS":              N_CUSTOMERS,
    "MAX_DEMAND_PER_CUSTOMER":  MAX_DEMAND_PER_CUSTOMER,
    "VESSEL_CARGO_CAPACITY":    VESSEL_CARGO_CAPACITY,
    "VESSEL_MAX_CHARGE":        VESSEL_MAX_CHARGE,
    "SEED":                     int(SEED),
}

# ── Solve on button click ────────────────────────────────────────────────────
if run_clicked:
    st.session_state["is_solving"] = True

    with st.status("Solving…", expanded=True) as status_ctx:
        st.write("Building scenario…")
        customers, charging_stations, vessels = build_scenario(params)

        st.write("Building graph…")
        nodes, edges = build_graph(customers, charging_stations, vessels, params)

        st.write("Building CP-SAT model…")
        model, variables = build_model(nodes, edges, customers, vessels, params)

        st.write("Solving (up to 30 s)…")
        result = solve(model, variables, nodes, customers, vessels, params)

        if result.status in ("OPTIMAL", "FEASIBLE"):
            status_ctx.update(label="Done!", state="complete")
        else:
            status_ctx.update(label="No solution found.", state="error")

    st.session_state["result"]     = result
    st.session_state["vessels"]    = vessels
    st.session_state["is_solving"] = False

# ── Render results (persists across widget interactions) ─────────────────────
if "result" in st.session_state:
    result  = st.session_state["result"]
    vessels = st.session_state.get("vessels", [])

    if result.status in ("OPTIMAL", "FEASIBLE"):

        # Route map
        st.subheader("ROUTE MAP")
        st.plotly_chart(result.fig, use_container_width=True)

        # Status bar
        msg = f"{result.status} — Total distance: {result.total_distance:.3f}"
        if result.status == "OPTIMAL":
            st.success(msg)
        else:
            st.warning(msg)

        # Results tables
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("ROUTE DETAILS")
            n_v = len(result.route_rows)
            if n_v > 3:
                tab_labels = [f"Vessel {k} ({vessels[k]['name'] if k < len(vessels) else f'boat{k}'})"
                              for k in result.route_rows]
                tabs = st.tabs(tab_labels)
                for tab, k in zip(tabs, result.route_rows):
                    with tab:
                        st.dataframe(pd.DataFrame(result.route_rows[k]), use_container_width=True)
            else:
                for k, rows in result.route_rows.items():
                    vessel_name = vessels[k]['name'] if k < len(vessels) else f'boat{k}'
                    st.markdown(f"**Vessel {k} ({vessel_name})**")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with col_right:
            st.subheader("CUSTOMER SERVICE TIMES")
            st.dataframe(pd.DataFrame(result.service_rows), use_container_width=True)

    else:
        st.error("No solution found.")
