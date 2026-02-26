# Streamlit Demo App — Tech Spec: Electric Vessel Routing

## 1. Overview

A single-page Streamlit app that lets users interactively configure the VRPTW
scenario parameters, trigger the solver, and inspect the solution through a
route plot and tabular results.

---

## 2. Layout Wireframe

**Idle / pre-solve:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚓ Electric Vessel Routing Demo                                             │
├───────────────┬─────────────────────────────────────────────────────────────┤
│  PARAMETERS   │                                                             │
│  ─────────── │                    ROUTE MAP                                │
│  Vessel Speed │                                                             │
│  [────●────]  │   ┌─────────────────────────────────────────────────────┐  │
│               │   │                                                     │  │
│  Sim Period   │   │      [matplotlib figure: nodes + vessel routes]     │  │
│  [──●──────]  │   │                                                     │  │
│               │   │                                                     │  │
│  Discharge    │   └─────────────────────────────────────────────────────┘  │
│  Rate         │                                                             │
│  [────●────]  ├─────────────────────────────────────────────────────────────┤
│               │                                                             │
│  Num Vessels  │   ROUTE DETAILS            CUSTOMER SERVICE TIMES           │
│  [──●──────]  │   ┌──────────────────────┐ ┌───────────────────────────┐   │
│               │   │ Vessel 0 (boat0):    │ │ Cust  Vessel  T_pick ...  │   │
│  Charge Nodes │   │ From  To  Dist  ...  │ │ cust0 boat0   12.00  ...  │   │
│  [───●─────]  │   │ ─────────────────── │ │ cust1 boat1   45.00  ...  │   │
│               │   │  0(depot) 2(start).. │ │ ...                       │   │
│  Area Length  │   ├──────────────────────┤ └───────────────────────────┘   │
│  [─────●───]  │   │ Vessel 1 (boat1):    │                                 │
│               │   │ From  To  Dist  ...  │                                 │
│  Num Customers│   │ ─────────────────── │                                 │
│  [──●──────]  │   │  0(depot) 4(start).. │                                 │
│               │   └──────────────────────┘                                 │
│  Max Demand   │                                                             │
│  [───●─────]  │   Status: OPTIMAL   Total Distance: 87.432                 │
│               │                                                             │
│  ┌──────────┐ │                                                             │
│  │ ▶  Run   │ │                                                             │
│  └──────────┘ │                                                             │
└───────────────┴─────────────────────────────────────────────────────────────┘
```

**While solving (button disabled, `st.status` expanded in main area):**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚓ Electric Vessel Routing Demo                                             │
├───────────────┬─────────────────────────────────────────────────────────────┤
│  PARAMETERS   │                                                             │
│  ─────────── │  ┌─ ⟳ Solving… ──────────────────────────────────────────┐ │
│  Vessel Speed │  │  ✔ Building scenario                                  │ │
│  [────●────]  │  │  ✔ Building graph                                     │ │
│               │  │  ✔ Building CP-SAT model                              │ │
│  Sim Period   │  │  ⟳ Solving (up to 30s)…                               │ │
│  [──●──────]  │  └───────────────────────────────────────────────────────┘ │
│               │                                                             │
│  ...          │   [previous results remain visible below, greyed out]       │
│               │                                                             │
│  ┌──────────┐ │                                                             │
│  │ ▶  Run   │ │  (button disabled while solving)                           │
│  │ (dimmed) │ │                                                             │
│  └──────────┘ │                                                             │
└───────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 3. Panels

### 3.1 Parameter Panel (Sidebar)

Displayed in `st.sidebar`. Each parameter maps to a widget. A **Run Solver**
button at the bottom triggers computation.

| Parameter                 | Widget        | Default | Min | Max  | Step |
|---------------------------|---------------|---------|-----|------|------|
| `VESSEL_SPEED`            | `number_input`| 1       | 0.1 | 10.0 | 0.1  |
| `SIM_PERIOD`              | `slider`      | 300     | 50  | 1000 | 10   |
| `RATE_DISCHARGE`          | `number_input`| 2       | 0.1 | 20.0 | 0.1  |
| `N_VESSELS`               | `slider`      | 2       | 1   | 6    | 1    |
| `N_DUPLICATE_CHARGE_NODES`| `slider`      | 3       | 1   | 6    | 1    |
| `AREA_LEN`                | `slider`      | 20      | 5   | 100  | 5    |
| `N_CUSTOMERS`             | `slider`      | 5       | 1   | 12   | 1    |
| `MAX_DEMAND_PER_CUSTOMER` | `slider`      | 5       | 1   | 20   | 1    |

The RNG seed is fixed at 43 and not exposed (reproducibility).

### 3.2 Route Map Panel

- Occupies the full width of the main column above the results.
- Rendered with `st.pyplot(fig)` using the matplotlib figure produced by the
  existing plot logic.
- Shows: depot (black square), pickup nodes (green circle, P_n label),
  delivery nodes (red cross, D_n label), charger nodes (magenta triangle),
  and per-vessel colored arrows with step numbers.
- Displayed only after a successful solve.

### 3.3 Results Panel

Split into two `st.columns` side by side, below the route map.

**Left — Route Details:**
- One `st.dataframe` per vessel (or tabs if N_VESSELS > 3).
- Columns: `From`, `To`, `Dist`, `T_dep`, `T_arr`, `Cargo_i`, `Cargo_j`,
  `Chg_i`, `Chg_j`.
- Vessel name shown as a subheader above each table.

**Right — Customer Service Times:**
- Single `st.dataframe`.
- Columns: `Customer`, `Vessel`, `T_pickup`, `T_delivery`, `T_service`,
  `T_wait`.

**Status bar (above both tables):**
- `st.success("OPTIMAL — Total distance: 87.432")` or
  `st.warning("FEASIBLE — ...")` or `st.error("No solution found.")`.

---

## 4. App State & Control Flow

```
User adjusts parameters
        │
        ▼
[ Run Solver ] clicked
        │
        ▼
  st.session_state.is_solving = True   ← disables the Run button
        │
        ▼
  with st.status("Solving…", expanded=True) as status:
        │
        ├── st.write("Building scenario…")
        │   build_scenario(params)
        │
        ├── st.write("Building graph…")
        │   build_graph(...)
        │
        ├── st.write("Building CP-SAT model…")
        │   build_model(...)
        │
        ├── st.write("Solving (up to 30s)…")
        │   solver.Solve()                  ← blocking; spinner visible here
        │
        └── status.update(label="Done!", state="complete")
                  │
            ┌─────┴──────┐
          OPTIMAL/    INFEASIBLE
          FEASIBLE        │
            │         status.update(state="error")
            │         st.error("No solution found.")
            ▼
    st.session_state.result = SolverResult(...)
    st.session_state.is_solving = False
            │
            ▼
    render Route Map
    render Results tables
```

**Session state keys:**

| Key          | Type          | Purpose                                      |
|--------------|---------------|----------------------------------------------|
| `is_solving` | `bool`        | Disables Run button while solver is running  |
| `result`     | `SolverResult`| Persists last solution across widget changes |

- All solver output is captured in Python (not `print`), stored in
  `st.session_state` so results persist across widget interactions without
  re-solving.
- The **Run Solver** button is the only trigger for re-solving. Changing a
  slider does not auto-rerun.
- Previous results remain visible below the `st.status` box while a new solve
  is in progress, so the user is not left with a blank screen.

---

## 5. File Structure

```
code/
├── main_try.py              # existing script (unchanged)
├── app.py                   # new Streamlit entry point
├── solver.py                # extracted solver logic (pure functions, no prints)
└── STREAMLIT_TECH_SPEC.md   # this file
```

### `solver.py` responsibilities
- `build_scenario(params) -> (customers, charging_stations, vessels)`
- `build_graph(scenario, params) -> (nodes, edges)`
- `build_model(nodes, edges, scenario, params) -> (model, variables)`
- `solve(model, variables, nodes, params) -> SolverResult` (dataclass)

`SolverResult` holds:
- `status: str` — `"OPTIMAL"` / `"FEASIBLE"` / `"INFEASIBLE"`
- `total_distance: float`
- `route_rows: dict[int, list[dict]]` — per vessel, list of row dicts
- `service_rows: list[dict]` — customer service time rows
- `fig: matplotlib.figure.Figure`

### `app.py` responsibilities
- Sidebar parameter widgets
- Run button + spinner
- Calls `solver.py` functions
- Renders map and tables from `SolverResult`

---

## 6. Dependencies

```
streamlit
ortools
numpy
matplotlib
```

No additional libraries required. Pin versions in `requirements.txt`.
