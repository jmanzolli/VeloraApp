# Converted from opt.ipynb
# This script preserves notebook cells as Python sections for maintainability.

# %% Cell 1
import pyomo.environ as pyo
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from deap import base, creator, tools
import random
import osmnx as ox
import networkx as nx

# %% Cell 2
# ============================================================
# 1) INPUTS / DATA (put this in the first notebook cell)
# ============================================================

# This expects an export from dataset_manipulation.ipynb
# File should contain: station, outbound, inbound, total_trips, estimated_docks, lat, lon
DATA_FILE = Path("busiest_40.csv")

rng = np.random.default_rng(42)
ox.settings.use_cache = True

def build_data_from_bixi(csv_path=DATA_FILE):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Export busiest_40.csv from dataset_manipulation.ipynb."
        )

    df = pd.read_csv(csv_path)
    required_cols = {"station", "outbound", "inbound", "total_trips", "estimated_docks", "lat", "lon"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Nodes
    N = df["station"].tolist()

    # Coordinates
    if df[["lat", "lon"]].isna().any().any():
        bad = df[df[["lat", "lon"]].isna().any(axis=1)]["station"].tolist()
        raise ValueError(f"Missing lat/lon for stations: {bad}")
    LAT = {row.station: float(row.lat) for row in df.itertuples(index=False)}
    LON = {row.station: float(row.lon) for row in df.itertuples(index=False)}

    # Demand and peak demand
    D = {row.station: float(row.total_trips) for row in df.itertuples(index=False)}

    # Use estimated docks as peak demand proxy (fallback to 10 if missing)
    D_peak = {
        row.station: float(row.estimated_docks) if pd.notna(row.estimated_docks) else 10.0
        for row in df.itertuples(index=False)
    }

    # Street network for shortest paths (OSM)
    G_streets = ox.graph_from_place("Montreal, Quebec, Canada", network_type="bike", simplify=True)
    station_nodes_list = ox.distance.nearest_nodes(
        G_streets,
        [LON[s] for s in N],
        [LAT[s] for s in N],
    )
    station_nodes = {s: int(n) for s, n in zip(N, station_nodes_list)}

    # Candidate links: complete undirected graph
    candidate_pairs = [(i, j) for idx, i in enumerate(N) for j in N[idx + 1 :]]

    # Shortest path lengths (km) over the street graph
    L = []
    path_km = {}
    for (i, j) in candidate_pairs:
        u = station_nodes[i]
        v = station_nodes[j]
        try:
            length_m = nx.shortest_path_length(G_streets, u, v, weight="length")
        except Exception:
            continue
        L.append((i, j))
        path_km[(i, j)] = float(length_m) / 1000.0

    # LTS (1–4 integer) and upgrade cost (randomized for this exercise)
    LTS = {(i, j): int(rng.integers(1, 5)) for (i, j) in L}
    C_up = {
        (i, j): (0.0 if LTS[(i, j)] == 1 else float(rng.integers(40_000, 60_001)))
        for (i, j) in L
    }

    # Dock cost (CAD per dock)
    C_dock = 900.0

    # Minimums (heuristic defaults)
    S_min = max(5, len(N) // 4)
    M_min = max(5, len(N) // 3)
    D_min = 0.3 * sum(D.values())

    return dict(
        N=N,
        L=L,
        D=D,
        D_peak=D_peak,
        LTS=LTS,
        C_up=C_up,
        C_dock=C_dock,
        S_min=S_min,
        M_min=M_min,
        D_min=D_min,
        lat=LAT,
        lon=LON,
        street_graph=G_streets,
        station_nodes=station_nodes,
        path_km=path_km,
    )

# Build data
# Export the file from dataset_manipulation.ipynb as "busiest_40.csv" then run this cell.
data = build_data_from_bixi()

# %% Cell 3
# Quick input check
print("Stations:", len(data["N"]))
print("Links:", len(data["L"]))
print("Demand total:", sum(data["D"].values()))
print("Peak demand total:", sum(data["D_peak"].values()))
print("LTS range:", (min(data["LTS"].values()), max(data["LTS"].values())))
print("C_up range:", (min(data["C_up"].values()), max(data["C_up"].values())))
print("C_dock:", data["C_dock"])
print("S_min:", data["S_min"])
print("M_min:", data["M_min"])
print("D_min:", data["D_min"])

# %% Cell 4
# ============================================================
# 2) OPTIMIZATION MODEL
# ============================================================
import pyomo.environ as pyo

def create_bikeshare_network_model(data, alpha, beta, gamma):
    """
    Weighted-sum scalarization:
      max alpha*Z1 - beta*Z2 - gamma*Z3
    where:
      Z1 = sum_i D[i]*y[i]
      Z2 = sum_(i,j) LTS[i,j]*x[i,j]
      Z3 = sum_(i,j) C_up[i,j]*x[i,j] + sum_i C_dock*z[i]

    Notes:
      - Ensure your weights satisfy alpha+beta+gamma=1 outside this function.
      - This model matches the math as written (no extra budgets, no explicit connectivity beyond "no isolated active station").
    """

    m = pyo.ConcreteModel("Bikeway_Station_Planning")

    # ----------------------------
    # Sets
    # ----------------------------
    m.N = pyo.Set(initialize=list(data["N"]), ordered=True)
    m.L = pyo.Set(within=m.N * m.N, initialize=list(data["L"]), ordered=True)

    # ----------------------------
    # Parameters
    # ----------------------------
    D = data["D"]
    Dp = data["D_peak"]
    LTS = data["LTS"]
    Cup = data["C_up"]
    LAT = data["lat"]
    LON = data["lon"]

    m.D = pyo.Param(m.N, initialize=lambda _, i: float(D[i]))
    m.D_peak = pyo.Param(m.N, initialize=lambda _, i: float(Dp[i]))
    m.lat = pyo.Param(m.N, initialize=lambda _, i: float(LAT[i]))
    m.lon = pyo.Param(m.N, initialize=lambda _, i: float(LON[i]))

    m.LTS = pyo.Param(m.L, initialize=lambda _, i, j: float(LTS[(i, j)]))
    m.C_up = pyo.Param(m.L, initialize=lambda _, i, j: float(Cup[(i, j)]))

    m.C_dock = pyo.Param(initialize=float(data["C_dock"]))

    m.S_min = pyo.Param(initialize=int(data["S_min"]))
    m.M_min = pyo.Param(initialize=int(data["M_min"]))
    m.D_min = pyo.Param(initialize=float(data["D_min"]))

    m.alpha = pyo.Param(initialize=float(alpha))
    m.beta = pyo.Param(initialize=float(beta))
    m.gamma = pyo.Param(initialize=float(gamma))

    # ----------------------------
    # Decision variables
    # ----------------------------
    m.x = pyo.Var(m.L, within=pyo.Binary)               # link selected
    m.y = pyo.Var(m.N, within=pyo.Binary)               # station activated
    m.z = pyo.Var(m.N, within=pyo.NonNegativeReals)     # docks/capacity

    # ----------------------------
    # Objective components (expressions)
    # ----------------------------
    m.Z1 = pyo.Expression(expr=sum(m.D[i] * m.y[i] for i in m.N))
    m.Z2 = pyo.Expression(expr=sum(m.LTS[i, j] * m.x[i, j] for (i, j) in m.L))
    m.Z3 = pyo.Expression(
        expr=sum(m.C_up[i, j] * m.x[i, j] for (i, j) in m.L)
        + sum(m.C_dock * m.z[i] for i in m.N)
    )

    # Scalarized objective (maximize)
    m.OBJ = pyo.Objective(
        expr=m.alpha * m.Z1 - m.beta * m.Z2 - m.gamma * m.Z3,
        sense=pyo.maximize
    )

    # ----------------------------
    # Constraints
    # ----------------------------
    # (2) minimum stations
    m.min_stations = pyo.Constraint(expr=sum(m.y[i] for i in m.N) >= m.S_min)

    # (3) minimum links
    m.min_links = pyo.Constraint(expr=sum(m.x[i, j] for (i, j) in m.L) >= m.M_min)

    # (4) minimum covered demand
    m.min_demand = pyo.Constraint(expr=sum(m.D[i] * m.y[i] for i in m.N) >= m.D_min)

    # (cap) capacity requirement when active
    m.cap_req = pyo.Constraint(m.N, rule=lambda m, i: m.z[i] >= m.D_peak[i] * m.y[i])

    # (5) link implies station i active
    m.link_station_i = pyo.Constraint(m.L, rule=lambda m, i, j: m.x[i, j] <= m.y[i])

    # (6) link implies station j active
    m.link_station_j = pyo.Constraint(m.L, rule=lambda m, i, j: m.x[i, j] <= m.y[j])

    # (7) no isolated active stations:
    # y[k] <= sum_{(i,j) in L: i=k or j=k} x[i,j]
    def incident_links_sum(m, k):
        return sum(m.x[i, j] for (i, j) in m.L if i == k or j == k)

    m.no_isolated = pyo.Constraint(m.N, rule=lambda m, k: m.y[k] <= incident_links_sum(m, k))

    return m

# %% Cell 5
# ============================================================
# 3) SOLUTION / PARETO GRID (weighted sum)
# ============================================================

'''
import logging

# ---- USER PARAMETERS (edit these) ----
SOLVER_NAME = "gurobi"      # or "cbc"
TEE = False
TIME_LIMIT_S = None
MIP_GAP = 0.001

# Weighted-sum grid (tighter steps)
WEIGHT_STEP = 0.02   # smaller = tighter grid (e.g., 0.02 or 0.01)
USE_MONTE_CARLO = True
MC_SAMPLES = 1000     # number of random weight triplets
RNG_SEED = 42

# ------------------------------------------------------------

def solve_one(m, solver_name=SOLVER_NAME, tee=TEE, time_limit_s=TIME_LIMIT_S, mip_gap=MIP_GAP):
    """Solve a single model instance and return a dict of results."""
    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        raise RuntimeError(f"Solver '{solver_name}' is not available in this environment.")

    # Quiet Pyomo warnings (e.g., infeasible solves)
    logging.getLogger("pyomo").setLevel(logging.ERROR)
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)

    # Optional solver settings (common ones)
    if solver_name.lower() in {"gurobi", "gurobi_persistent"}:
        if time_limit_s is not None:
            solver.options["TimeLimit"] = float(time_limit_s)
        if mip_gap is not None:
            solver.options["MIPGap"] = float(mip_gap)
    elif solver_name.lower() in {"cbc"}:
        if time_limit_s is not None:
            solver.options["seconds"] = float(time_limit_s)
        if mip_gap is not None:
            solver.options["ratio"] = float(mip_gap)

    # Avoid loading infeasible solutions to suppress warnings
    res = solver.solve(m, tee=tee, load_solutions=False)

    # Basic status checks
    term = res.solver.termination_condition
    ok = term in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)

    if not ok:
        return dict(ok=False, termination=str(term), scalar_obj=None, Z1=None, Z2=None, Z3=None, y=None, x=None, z=None, alpha=None, beta=None, gamma=None)

    # Load solution only when feasible/optimal
    m.solutions.load_from(res)

    # Extract decisions
    y_sol = {i: int(round(pyo.value(m.y[i]))) for i in m.N}
    x_sol = {(i, j): int(round(pyo.value(m.x[i, j]))) for (i, j) in m.L}
    z_sol = {i: float(pyo.value(m.z[i])) for i in m.N}

    # Extract objective values (original components)
    Z1 = float(pyo.value(m.Z1))
    Z2 = float(pyo.value(m.Z2))
    Z3 = float(pyo.value(m.Z3))
    scalar = float(pyo.value(m.OBJ))

    return dict(
        ok=ok, termination=str(term),
        scalar_obj=scalar, Z1=Z1, Z2=Z2, Z3=Z3,
        y=y_sol, x=x_sol, z=z_sol,
        alpha=float(pyo.value(m.alpha)), beta=float(pyo.value(m.beta)), gamma=float(pyo.value(m.gamma)),
    )


def weight_grid(step=WEIGHT_STEP):
    """Deterministic grid over alpha,beta,gamma with alpha+beta+gamma=1."""
    vals = np.arange(0, 1 + 1e-9, step)
    grid = []
    for a in vals:
        for b in vals:
            g = 1.0 - a - b
            if g < -1e-9:
                continue
            if g < 0:
                g = 0.0
            grid.append((float(a), float(b), float(g)))
    return grid


def monte_carlo_weights(n=MC_SAMPLES, seed=RNG_SEED):
    """Random weights on simplex via Dirichlet(1,1,1)."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    return [(float(a), float(b), float(g)) for a, b, g in w]


def pareto_sweep_weighted(data):
    """Weighted-sum sweep: maximize alpha*Z1 - beta*Z2 - gamma*Z3."""
    sols = []
    weights = weight_grid()
    if USE_MONTE_CARLO:
        weights = weights + monte_carlo_weights()

    start_t = time.time()
    with tqdm(total=len(weights), desc="Solving weighted-sum grid", unit="model") as pbar:
        for alpha, beta, gamma in weights:
            m = create_bikeshare_network_model(data, alpha=alpha, beta=beta, gamma=gamma)
            sol = solve_one(m)
            sols.append(sol)
            elapsed = time.time() - start_t
            pbar.set_postfix({"elapsed_s": f"{elapsed:.1f}"})
            pbar.update(1)

    return sols

# ---- Run the sweep ----
solutions = pareto_sweep_weighted(data)

# ---- Quick summary table ----
import pandas as pd

df = pd.DataFrame([{
    "ok": s["ok"], "term": s["termination"],
    "alpha": s.get("alpha"), "beta": s.get("beta"), "gamma": s.get("gamma"),
    "Z1_demand": s["Z1"], "Z2_stress": s["Z2"], "Z3_cost": s["Z3"],
    "stations": sum(s["y"].values()) if s["y"] else None,
    "links": sum(s["x"].values()) if s["x"] else None,
} for s in solutions])

df.sort_values(["ok", "Z1_demand"], ascending=[False, False]).head(10)
'''

# %% Cell 6
# ============================================================
# 3b) SOLUTION / PARETO GRID (epsilon-constraint)
# ============================================================

'''
EPS_Z2_MIN = 0
EPS_Z2_MAX = 10_000_000
EPS_Z3_MIN = 0
EPS_Z3_MAX = 50_000_000
EPS_N_Z2 = 100
EPS_N_Z3 = 100

def pareto_sweep_epsilon(data):
    """Epsilon-constraint sweep: maximize Z1 subject to Z2<=eps2, Z3<=eps3."""
    sols = []
    eps2_vals = np.linspace(EPS_Z2_MIN, EPS_Z2_MAX, EPS_N_Z2)
    eps3_vals = np.linspace(EPS_Z3_MIN, EPS_Z3_MAX, EPS_N_Z3)

    start_t = time.time()
    with tqdm(total=len(eps2_vals) * len(eps3_vals), desc="Solving epsilon grid", unit="model") as pbar:
        for eps2 in eps2_vals:
            for eps3 in eps3_vals:
                m = create_bikeshare_network_model(data, alpha=1.0, beta=0.0, gamma=0.0)
                # Replace objective to maximize Z1 only
                m.del_component(m.OBJ)
                m.OBJ = pyo.Objective(expr=m.Z1, sense=pyo.maximize)
                # Epsilon constraints
                m.eps_Z2 = pyo.Constraint(expr=m.Z2 <= float(eps2))
                m.eps_Z3 = pyo.Constraint(expr=m.Z3 <= float(eps3))
                sol = solve_one(m)
                sol.update({"eps_Z2": float(eps2), "eps_Z3": float(eps3)})
                sols.append(sol)
                elapsed = time.time() - start_t
                pbar.set_postfix({"elapsed_s": f"{elapsed:.1f}"})
                pbar.update(1)

    return sols

# ---- Run epsilon sweep (optional) ----
solutions_eps = pareto_sweep_epsilon(data)

# ---- Summary table ----
df_eps = pd.DataFrame([{
    "ok": s["ok"], "term": s["termination"],
    "eps_Z2": s.get("eps_Z2"), "eps_Z3": s.get("eps_Z3"),
    "Z1_demand": s["Z1"], "Z2_stress": s["Z2"], "Z3_cost": s["Z3"],
    "stations": sum(s["y"].values()) if s["y"] else None,
    "links": sum(s["x"].values()) if s["x"] else None,
} for s in solutions_eps])

df_eps.sort_values(["ok", "Z1_demand"], ascending=[False, False]).head(10)
'''

# %% Cell 7
# ============================================================
# 3c) NSGA-II
# ============================================================

DEAP_POP = 2000
DEAP_GEN = 100
DEAP_CXPB = 0.9
DEAP_MUTPB = 0.2
DEAP_SEED = RNG_SEED if "RNG_SEED" in globals() else 42
random.seed(DEAP_SEED)
np.random.seed(DEAP_SEED)

# Gene sizes
_N = data["N"]
_L = data["L"]
_D = data["D"]
_Dp = data["D_peak"]
_LTS = data["LTS"]
_Cup = data["C_up"]
_C_dock = data["C_dock"]
_S_min = data["S_min"]
_M_min = data["M_min"]
_D_min = data["D_min"]

n_y = len(_N)
L_list = list(_L)
n_x = len(L_list)
n_z = len(_N)
_z_max = max(_Dp.values()) if len(_Dp) else 0.0

def _decode(ind):
    y_vals = np.array(ind[:n_y])
    x_vals = np.array(ind[n_y:n_y + n_x])
    z_vals = np.array(ind[n_y + n_x:])
    y = {i: int(v) for i, v in zip(_N, y_vals)}
    x = {L_list[idx]: int(x_vals[idx]) for idx in range(n_x)}
    z = {i: float(v) for i, v in zip(_N, z_vals)}
    return y, x, z

def _encode(y, x, z):
    return [
        *[int(y[i]) for i in _N],
        *[int(x[p]) for p in L_list],
        *[float(z[i]) for i in _N],
    ]

def _repair(ind):
    """Repair an individual to better satisfy constraints."""
    y, x, z = _decode(ind)
    # binarize
    y = {i: 1 if y[i] >= 1 else 0 for i in _N}
    x = {p: 1 if x[p] >= 1 else 0 for p in L_list}

    # ensure minimum stations and demand
    total_stations = sum(y.values())
    total_demand = sum(_D[i] * y[i] for i in _N)
    if total_stations < _S_min or total_demand < _D_min:
        candidates = sorted(_N, key=lambda i: _D[i], reverse=True)
        for i in candidates:
            if y[i] == 0:
                y[i] = 1
                total_stations += 1
                total_demand += _D[i]
            if total_stations >= _S_min and total_demand >= _D_min:
                break

    # enforce link implies station + only among active nodes
    active = [i for i in _N if y[i] == 1]
    active_set = set(active)
    x = {p: (1 if (p[0] in active_set and p[1] in active_set and x[p]) else 0) for p in L_list}

    # ensure minimum links among active nodes
    if len(active) >= 2:
        current_links = sum(x.values())
        if current_links < _M_min:
            remaining = [p for p in L_list if p[0] in active_set and p[1] in active_set and x[p] == 0]
            random.shuffle(remaining)
            needed = _M_min - current_links
            for p in remaining[:needed]:
                x[p] = 1

    # ensure no isolated active stations
    if len(active) >= 2:
        incident = {k: 0 for k in _N}
        for (i, j), v in x.items():
            if v:
                incident[i] += 1
                incident[j] += 1
        for k in active:
            if incident[k] == 0:
                # connect to another active node
                other = random.choice([a for a in active if a != k])
                p = (k, other) if (k, other) in x else (other, k)
                x[p] = 1
                incident[k] += 1
                incident[other] += 1

    # capacity: z >= Dp * y; zero when inactive
    z = {i: (0.0 if y[i] == 0 else max(float(_Dp[i]), float(z[i]))) for i in _N}
    # clip to [0, z_max]
    z = {i: min(max(0.0, z[i]), float(_z_max)) for i in _N}

    # write back
    ind[:] = _encode(y, x, z)
    return ind

def _constraint_violation(y, x, z):
    viol = 0.0
    viol += max(0.0, _S_min - sum(y.values()))
    viol += max(0.0, _M_min - sum(x.values()))
    viol += max(0.0, _D_min - sum(_D[i] * y[i] for i in _N))
    for i in _N:
        viol += max(0.0, _Dp[i] * y[i] - z[i])
    # no isolated stations
    incident = {k: 0 for k in _N}
    for (i, j), v in x.items():
        if v:
            incident[i] += 1
            incident[j] += 1
    for k in _N:
        viol += max(0.0, y[k] - incident[k])
    # link implies station
    for (i, j), v in x.items():
        if v:
            viol += max(0.0, v - y[i])
            viol += max(0.0, v - y[j])
    return viol

# Fitness: maximize Z1, minimize Z2 and Z3
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bin", random.randint, 0, 1)
toolbox.register("attr_z", random.uniform, 0.0, float(_z_max))

toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    ([toolbox.attr_bin] * (n_y + n_x) + [toolbox.attr_z] * n_z),
    n=1,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    _repair(ind)
    y, x, z = _decode(ind)
    Z1 = sum(_D[i] * y[i] for i in _N)
    Z2 = sum(_LTS[(i, j)] * x[(i, j)] for (i, j) in _L)
    Z3 = sum(_Cup[(i, j)] * x[(i, j)] for (i, j) in _L) + _C_dock * sum(z[i] for i in _N)
    cv = _constraint_violation(y, x, z)
    penalty = 1e6 * cv
    return (Z1 - penalty, Z2 + penalty, Z3 + penalty)

def mutate(ind, indpb=0.02, z_sigma=0.1):
    # flip bits on y/x; gaussian on z
    for i in range(n_y + n_x):
        if random.random() < indpb:
            ind[i] = 1 - int(ind[i])
    for i in range(n_y + n_x, n_y + n_x + n_z):
        if random.random() < indpb:
            ind[i] = float(ind[i]) + random.gauss(0.0, z_sigma * float(_z_max))
            ind[i] = min(max(0.0, float(ind[i])), float(_z_max))
    _repair(ind)
    return (ind,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate, indpb=0.02, z_sigma=0.1)
toolbox.register("select", tools.selNSGA2)

pop = toolbox.population(n=DEAP_POP)
for ind in pop:
    _repair(ind)
invalid = [ind for ind in pop if not ind.fitness.valid]
for ind in invalid:
    ind.fitness.values = toolbox.evaluate(ind)

pop = toolbox.select(pop, len(pop))
for gen in tqdm(range(DEAP_GEN), desc="DEAP NSGA-II", unit="gen"):
    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = [toolbox.clone(ind) for ind in offspring]

    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < DEAP_CXPB:
            toolbox.mate(ind1, ind2)
            del ind1.fitness.values, ind2.fitness.values

    for ind in offspring:
        if random.random() < DEAP_MUTPB:
            toolbox.mutate(ind)
            del ind.fitness.values

    invalid = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    pop = toolbox.select(pop + offspring, DEAP_POP)

# Build DEAP results table
rows = []
for ind in pop:
    y, x, z = _decode(ind)
    Z1 = sum(_D[i] * y[i] for i in _N)
    Z2 = sum(_LTS[(i, j)] * x[(i, j)] for (i, j) in _L)
    Z3 = sum(_Cup[(i, j)] * x[(i, j)] for (i, j) in _L) + _C_dock * sum(z[i] for i in _N)
    cv = _constraint_violation(y, x, z)
    rows.append({"ok": cv == 0, "Z1_demand": Z1, "Z2_stress": Z2, "Z3_cost": Z3, "cv": cv})

df_nsga_deap = pd.DataFrame(rows)
print("DEAP NSGA-II results:", len(df_nsga_deap), "rows")
print("Feasible:", int(df_nsga_deap["ok"].sum()))
df_nsga_deap.head(10)

# %% Cell 8
# ============================================================
# 4) PLOTS (Pareto front)
# ============================================================

import matplotlib.tri as mtri
from matplotlib.ticker import StrMethodFormatter

PLOT_SOURCE = "deap"  # "weighted", "epsilon", or "deap"

if PLOT_SOURCE == "epsilon":
    df_plot = df_eps.copy()
    title_suffix = " (epsilon)"
elif PLOT_SOURCE == "deap":
    df_plot = df_nsga_deap.copy()
    title_suffix = " (NSGA-II)"
else:
    df_plot = df.copy()
    title_suffix = " (weighted)"

# Keep only feasible solutions
plot_df = df_plot[df_plot["ok"]].copy()
plot_df = plot_df.dropna(subset=["Z1_demand", "Z2_stress", "Z3_cost"])

# 2D projections + approximate Pareto curves (quadratic fit)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

def _plot_with_fit(ax, x, y, xlabel, ylabel, title):
    ax.scatter(x, y, s=20, alpha=0.7)
    if len(x) >= 5:
        order = np.argsort(x)
        xs = np.asarray(x)[order]
        ys = np.asarray(y)[order]
        deg = 2
        coeff = np.polyfit(xs, ys, deg=deg)
        xs_fit = np.linspace(xs.min(), xs.max(), 200)
        ys_fit = np.polyval(coeff, xs_fit)
        ax.plot(xs_fit, ys_fit, color="crimson", linewidth=2, label="Pareto approx")
        ax.legend(loc="best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

_plot_with_fit(
    axes[0],
    plot_df["Z1_demand"],
    plot_df["Z2_stress"],
    "Z1 (demand)",
    "Z2 (stress)",
    "Z1 vs Z2" + title_suffix,
)
_plot_with_fit(
    axes[1],
    plot_df["Z1_demand"],
    plot_df["Z3_cost"],
    "Z1 (demand)",
    "Z3 (cost)",
    "Z1 vs Z3" + title_suffix,
)
_plot_with_fit(
    axes[2],
    plot_df["Z2_stress"],
    plot_df["Z3_cost"],
    "Z2 (stress)",
    "Z3 (cost)",
    "Z2 vs Z3" + title_suffix,
)

plt.tight_layout()
plt.show()

# 3D Pareto front with surface approximation
fig = plt.figure(figsize=(6.8, 5.8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    plot_df["Z1_demand"],
    plot_df["Z2_stress"],
    plot_df["Z3_cost"],
    s=20,
    alpha=0.7,
    c=plot_df["Z1_demand"],
    cmap="viridis",
)

if len(plot_df) >= 10:
    xs = plot_df["Z1_demand"].to_numpy()
    ys = plot_df["Z2_stress"].to_numpy()
    zs = plot_df["Z3_cost"].to_numpy()
    tri = mtri.Triangulation(xs, ys)
    ax.plot_trisurf(tri, zs, cmap="viridis", alpha=0.35, linewidth=0.2)

# Padding and formatting so Z3 labels are not cramped
if len(plot_df):
    zmin = float(plot_df["Z3_cost"].min())
    zmax = float(plot_df["Z3_cost"].max())
    zpad = max(1.0, 0.08 * (zmax - zmin))
    ax.set_zlim(zmin - zpad, zmax + zpad)

ax.set_xlabel("Z1 (demand)", labelpad=8)
ax.set_ylabel("Z2 (stress)", labelpad=8)
ax.set_zlabel("Z3 (cost)", labelpad=10)
ax.zaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.tick_params(axis="z", pad=6)
ax.set_title("Pareto front (3D)" + title_suffix)

# Adjust 3D view and aspect for better separation
ax.view_init(elev=26, azim=140)
ax.set_box_aspect((1.2, 1.0, 0.8))
plt.tight_layout()
plt.show()

# %% Cell 9
# ============================================================
# 5) MAP VIEW OF ONE BALANCED SOLUTION (basemap)
# ============================================================

import math

def _norm01(arr):
    arr = np.asarray(arr, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

def _pick_balanced(z1, z2, z3):
    # Balanced toward high Z1 and low Z2/Z3
    n1 = _norm01(z1)
    n2 = _norm01(z2)
    n3 = _norm01(z3)
    scores = (1 - n1) ** 2 + n2 ** 2 + n3 ** 2
    return int(np.argmin(scores))

def _balanced_solution_from_weighted():
    if "solutions" not in globals():
        return None
    sols = [s for s in solutions if s.get("ok")]
    if not sols:
        return None
    z1 = [s["Z1"] for s in sols]
    z2 = [s["Z2"] for s in sols]
    z3 = [s["Z3"] for s in sols]
    idx = _pick_balanced(z1, z2, z3)
    return sols[idx]

def _balanced_solution_from_epsilon():
    if "solutions_eps" not in globals():
        return None
    sols = [s for s in solutions_eps if s.get("ok")]
    if not sols:
        return None
    z1 = [s["Z1"] for s in sols]
    z2 = [s["Z2"] for s in sols]
    z3 = [s["Z3"] for s in sols]
    idx = _pick_balanced(z1, z2, z3)
    return sols[idx]

def _balanced_solution_from_deap():
    if "pop" not in globals():
        return None
    feasible = []
    for ind in pop:
        y, x, z = _decode(ind)
        if _constraint_violation(y, x, z) == 0:
            Z1 = sum(_D[i] * y[i] for i in _N)
            Z2 = sum(_LTS[(i, j)] * x[(i, j)] for (i, j) in _L)
            Z3 = sum(_Cup[(i, j)] * x[(i, j)] for (i, j) in _L) + _C_dock * sum(z[i] for i in _N)
            feasible.append((Z1, Z2, Z3, y, x, z))
    if not feasible:
        return None
    z1 = [t[0] for t in feasible]
    z2 = [t[1] for t in feasible]
    z3 = [t[2] for t in feasible]
    idx = _pick_balanced(z1, z2, z3)
    Z1, Z2, Z3, y, x, z = feasible[idx]
    return {"ok": True, "Z1": Z1, "Z2": Z2, "Z3": Z3, "y": y, "x": x, "z": z}

def select_balanced_solution():
    if PLOT_SOURCE == "epsilon":
        sol = _balanced_solution_from_epsilon()
        return sol or _balanced_solution_from_weighted() or _balanced_solution_from_deap()
    if PLOT_SOURCE == "deap":
        sol = _balanced_solution_from_deap()
        return sol or _balanced_solution_from_weighted() or _balanced_solution_from_epsilon()
    sol = _balanced_solution_from_weighted()
    return sol or _balanced_solution_from_deap() or _balanced_solution_from_epsilon()

# Diagnostics
if "df" in globals():
    print("Weighted feasible:", int(df["ok"].sum()))
if "df_eps" in globals():
    print("Epsilon feasible:", int(df_eps["ok"].sum()))
if "df_nsga_deap" in globals():
    print("DEAP feasible:", int(df_nsga_deap["ok"].sum()))

sol = select_balanced_solution()
if sol is None:
    print("No feasible solution found to plot.")
else:
    print("Balanced solution:", {"Z1": sol["Z1"], "Z2": sol["Z2"], "Z3": sol["Z3"]})
    y = sol["y"]
    x = sol["x"]
    z = sol["z"]
    lat = data["lat"]
    lon = data["lon"]
    stations = list(data["N"])
    lats = [lat[i] for i in stations]
    lons = [lon[i] for i in stations]
    max_dock = max(1.0, max(z.values()))

    def _haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0088
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    # Shortest-path lengths among active stations (based on selected links)
    active_nodes = [i for i in stations if y.get(i, 0) == 1]
    G = nx.Graph()
    G.add_nodes_from(active_nodes)
    for (i, j), v in x.items():
        if v and i in G and j in G:
            dist_km = _haversine_km(lat[i], lon[i], lat[j], lon[j])
            G.add_edge(i, j, weight=dist_km)

    shortest_paths = {
        i: nx.single_source_dijkstra_path_length(G, i, weight="weight")
        for i in active_nodes
    }

    path_stats = []
    for i, lengths in shortest_paths.items():
        reachable = len(lengths) - 1
        missing = max(0, len(active_nodes) - 1 - reachable)
        dist_vals = [d for node, d in lengths.items() if node != i]
        avg_km = float(np.mean(dist_vals)) if dist_vals else float("nan")
        max_km = float(np.max(dist_vals)) if dist_vals else float("nan")
        path_stats.append({
            "station": i,
            "reachable": reachable,
            "missing": missing,
            "avg_shortest_km": avg_km,
            "max_shortest_km": max_km,
        })
    path_stats_df = pd.DataFrame(path_stats).sort_values(["missing", "avg_shortest_km"], ascending=[False, False])
    print("Shortest-path summary (active stations):")
    display(path_stats_df.head(10))

    # Basemap visualization (falls back to plain scatter if libs missing)
    try:
        import geopandas as gpd
        import contextily as ctx
        from shapely.geometry import Point, LineString

        station_rows = []
        for i in stations:
            station_rows.append({
                "station": i,
                "active": int(y.get(i, 0) == 1),
                "docks": float(z[i]) if y.get(i, 0) == 1 else 0.0,
                "geometry": Point(lon[i], lat[i]),
            })
        stations_gdf = gpd.GeoDataFrame(station_rows, crs="EPSG:4326").to_crs(epsg=3857)

        G_streets = data.get("street_graph")
        station_nodes = data.get("station_nodes")

        link_rows = []
        for (i, j), v in x.items():
            if not v:
                continue
            if G_streets is not None and station_nodes is not None:
                try:
                    path_nodes = nx.shortest_path(
                        G_streets, station_nodes[i], station_nodes[j], weight="length"
                    )
                    coords = [(G_streets.nodes[n]["x"], G_streets.nodes[n]["y"]) for n in path_nodes]
                    link_rows.append({"geometry": LineString(coords)})
                    continue
                except Exception:
                    pass
            link_rows.append({"geometry": LineString([(lon[i], lat[i]), (lon[j], lat[j])])})

        links_gdf = gpd.GeoDataFrame(link_rows, crs="EPSG:4326").to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(7, 7))
        if not links_gdf.empty:
            links_gdf.plot(ax=ax, color="#2c7fb8", linewidth=1.5, alpha=0.6)

        active = stations_gdf[stations_gdf["active"] == 1]
        inactive = stations_gdf[stations_gdf["active"] == 0]

        if not inactive.empty:
            inactive.plot(ax=ax, color="#999999", markersize=20, alpha=0.6)
        if not active.empty:
            sizes = 30 + 200 * (active["docks"] / max_dock)
            active.plot(ax=ax, color="#d7191c", markersize=sizes, alpha=0.8)

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        ax.set_axis_off()
        ax.set_title("Balanced solution map (basemap)")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Basemap unavailable, using plain scatter. Reason: {exc}")
        fig, ax = plt.subplots(figsize=(7, 7))
        for (i, j), v in x.items():
            if v:
                ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="#2c7fb8", alpha=0.5, linewidth=1)

        sizes = []
        colors = []
        for i in stations:
            if y.get(i, 0) == 1:
                sizes.append(30 + 200 * (float(z[i]) / max_dock))
                colors.append("#d7191c")
            else:
                sizes.append(15)
                colors.append("#999999")
        ax.scatter(lons, lats, s=sizes, c=colors, alpha=0.8)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Balanced solution map")
        ax.set_aspect("equal", adjustable="box")
        plt.show()

