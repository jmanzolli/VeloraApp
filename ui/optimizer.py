from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from deap import base, creator, tools
from shapely.geometry import LineString, MultiLineString, Point


REQUIRED_STATION_COLUMNS = {"Station_Name", "Latitude", "Longitude", "Trips"}
SUPPORTED_VECTOR_SUFFIXES = {".gpkg", ".geojson", ".json", ".shp", ".parquet"}
STATION_COLUMN_ALIASES = {
    "station": "Station_Name",
    "name": "Station_Name",
    "station_name": "Station_Name",
    "lat": "Latitude",
    "latitude": "Latitude",
    "lon": "Longitude",
    "lng": "Longitude",
    "longitude": "Longitude",
    "trips": "Trips",
    "total_trips": "Trips",
    "demand": "Trips",
    "estimated_docks": "estimated_docks",
    "docks": "estimated_docks",
}


@dataclass
class UIConfig:
    candidate_points_per_station: int = 2
    station_buffer_meters: float = 150.0
    area_of_interest_buffer_meters: float = 2000.0
    station_minimum: int = 15
    link_minimum: int = 50
    demand_minimum: float = 0.0
    population_size: int = 120
    generations: int = 16
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    seed: int = 42
    dock_unit_cost: float = 900.0
    station_fixed_cost: float = 5000.0
    link_cost_lts1_per_km: float = 0.0
    link_cost_lts2_per_km: float = 2_000_000.0
    link_cost_lts3_per_km: float = 4_000_000.0
    link_cost_lts4_per_km: float = 8_000_000.0
    mode_shift_rate: float = 0.15
    average_trip_distance_km: float = 2.5
    car_emission_factor_g_per_km: float = 190.0


@dataclass
class OptimizationArtifacts:
    stations_gdf: gpd.GeoDataFrame
    network_gdf: gpd.GeoDataFrame
    candidates_gdf: gpd.GeoDataFrame
    graph: nx.Graph
    links_df: pd.DataFrame
    nodes_df: pd.DataFrame
    pareto_df: pd.DataFrame
    solutions: dict[str, dict[str, Any]]


def _notify(progress_callback, stage: str, message: str, progress: float) -> None:
    if progress_callback is not None:
        progress_callback(stage=stage, message=message, progress=progress)


def validate_station_table(df: pd.DataFrame) -> None:
    missing = REQUIRED_STATION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Station file is missing required columns: "
            + ", ".join(sorted(missing))
        )


def load_station_table(uploaded_path: str | Path) -> pd.DataFrame:
    path = Path(uploaded_path)
    if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("Station file must be CSV or Excel.")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    rename_map = {}
    for column in df.columns:
        normalized = str(column).strip()
        alias = STATION_COLUMN_ALIASES.get(normalized)
        if alias:
            rename_map[column] = alias
    df = df.rename(columns=rename_map)
    validate_station_table(df)
    df = df.copy()
    df["Station_Name"] = df["Station_Name"].astype(str).str.strip()
    for col in ["Latitude", "Longitude", "Trips"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude", "Trips"]).reset_index(drop=True)
    if "estimated_docks" not in df.columns:
        df["estimated_docks"] = np.maximum(10, np.ceil(df["Trips"] / 4000.0)).astype(float)
    else:
        df["estimated_docks"] = pd.to_numeric(df["estimated_docks"], errors="coerce")
        fallback = np.maximum(10, np.ceil(df["Trips"] / 4000.0)).astype(float)
        df["estimated_docks"] = df["estimated_docks"].fillna(fallback)
    return df


def load_network(uploaded_path: str | Path) -> gpd.GeoDataFrame:
    path = Path(uploaded_path)
    if path.suffix.lower() not in SUPPORTED_VECTOR_SUFFIXES:
        raise ValueError(
            "Network file must be one of: " + ", ".join(sorted(SUPPORTED_VECTOR_SUFFIXES))
        )
    network_gdf = gpd.read_file(path)
    required = {"geometry", "lts"}
    missing = required - set(network_gdf.columns)
    if missing:
        raise ValueError(
            "Network file is missing required columns: " + ", ".join(sorted(missing))
        )
    network_gdf = network_gdf.copy()
    network_gdf["lts"] = pd.to_numeric(network_gdf["lts"], errors="coerce")
    network_gdf["lts"] = network_gdf["lts"].fillna(4).clip(1, 4)
    if "length" not in network_gdf.columns:
        network_gdf["length"] = network_gdf.geometry.length
    else:
        network_gdf["length"] = pd.to_numeric(network_gdf["length"], errors="coerce")
        network_gdf["length"] = network_gdf["length"].fillna(network_gdf.geometry.length)
    network_gdf = network_gdf.dropna(subset=["geometry"]).reset_index(drop=True)
    return network_gdf


def build_station_geodataframe(stations_df: pd.DataFrame) -> gpd.GeoDataFrame:
    stations_gdf = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df["Longitude"], stations_df["Latitude"]),
        crs="EPSG:4326",
    )
    return stations_gdf


def generate_candidates(
    stations_gdf: gpd.GeoDataFrame,
    buffer_meters: float,
    candidate_points_per_station: int,
    seed: int,
) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    metric = stations_gdf.to_crs(2950).copy()
    metric["station_geometry"] = metric.geometry
    rows: list[dict[str, Any]] = []

    for _, row in metric.iterrows():
        rows.append(
            {
                "Station_Name": row["Station_Name"],
                "Trips": float(row["Trips"]),
                "estimated_docks": float(row["estimated_docks"]),
                "type": "original_station",
                "geometry": row["station_geometry"],
            }
        )
        buffer_geom = row["station_geometry"].buffer(buffer_meters)
        minx, miny, maxx, maxy = buffer_geom.bounds
        created = 0
        tries = 0
        while created < candidate_points_per_station and tries < 5000:
            candidate = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if buffer_geom.contains(candidate):
                rows.append(
                    {
                        "Station_Name": row["Station_Name"],
                        "Trips": float(row["Trips"]),
                        "estimated_docks": float(row["estimated_docks"]),
                        "type": "candidate_station",
                        "geometry": candidate,
                    }
                )
                created += 1
            tries += 1

    return gpd.GeoDataFrame(rows, crs=metric.crs)


def build_graph(network_gdf: gpd.GeoDataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in network_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if isinstance(geom, LineString) else list(geom.geoms) if isinstance(geom, MultiLineString) else []
        for part in parts:
            coords = list(part.coords)
            for u, v in zip(coords[:-1], coords[1:]):
                segment_length = LineString([u, v]).length
                graph.add_edge(
                    u,
                    v,
                    lts=float(row["lts"]),
                    length=float(segment_length),
                )
    return graph


def nearest_graph_node(point: Point, graph: nx.Graph) -> tuple[float, float]:
    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("Graph has no nodes after clipping the uploaded network.")
    distances = np.array([point.distance(Point(node)) for node in nodes], dtype=float)
    return nodes[int(np.argmin(distances))]


def build_links_dataframe(candidates_gdf: gpd.GeoDataFrame, graph: nx.Graph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unique_candidates = (
        candidates_gdf[["graph_node", "Station_Name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    for (_, left), (_, right) in itertools.combinations(unique_candidates.iterrows(), 2):
        if left["graph_node"] == right["graph_node"]:
            continue
        try:
            path = nx.shortest_path(graph, left["graph_node"], right["graph_node"], weight="length")
        except nx.NetworkXNoPath:
            continue
        edges = list(zip(path[:-1], path[1:]))
        edge_data = [graph.get_edge_data(u, v) for u, v in edges if graph.has_edge(u, v)]
        if not edge_data:
            continue
        rows.append(
            {
                "from_station": left["Station_Name"],
                "to_station": right["Station_Name"],
                "from_node": left["graph_node"],
                "to_node": right["graph_node"],
                "total_length": float(sum(edge["length"] for edge in edge_data)),
                "mean_lts": float(np.mean([edge["lts"] for edge in edge_data])),
            }
        )
    links_df = pd.DataFrame(rows).drop_duplicates(subset=["from_node", "to_node"])
    if links_df.empty:
        raise ValueError(
            "No feasible links were generated from the uploaded files. "
            "Try increasing the network coverage or reducing the study area."
        )
    return links_df


def upgrade_cost_per_meter_from_lts(value: float, config: UIConfig) -> float:
    level = int(np.clip(np.rint(float(value)), 1, 4))
    per_km = {
        1: float(config.link_cost_lts1_per_km),
        2: float(config.link_cost_lts2_per_km),
        3: float(config.link_cost_lts3_per_km),
        4: float(config.link_cost_lts4_per_km),
    }[level]
    return per_km / 1000.0


def prepare_problem(
    stations_path: str | Path,
    network_path: str | Path,
    config: UIConfig,
    progress_callback=None,
) -> tuple[dict[str, Any], gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, nx.Graph, pd.DataFrame, pd.DataFrame]:
    _notify(progress_callback, "loading", "Loading station and network files", 0.05)
    stations_df = load_station_table(stations_path)
    stations_gdf = build_station_geodataframe(stations_df)
    network_gdf = load_network(network_path)

    _notify(progress_callback, "candidates", "Generating candidate station locations", 0.12)
    candidates_gdf = generate_candidates(
        stations_gdf,
        buffer_meters=config.station_buffer_meters,
        candidate_points_per_station=config.candidate_points_per_station,
        seed=config.seed,
    )

    if candidates_gdf.crs != network_gdf.crs:
        network_metric = network_gdf.to_crs(candidates_gdf.crs)
    else:
        network_metric = network_gdf.copy()

    _notify(progress_callback, "clip", "Clipping network to the study area", 0.2)
    area_of_interest = candidates_gdf.unary_union.buffer(config.area_of_interest_buffer_meters)
    network_aoi = network_metric[network_metric.intersects(area_of_interest)].copy()
    if network_aoi.empty:
        raise ValueError("The clipped network is empty. Check the network CRS or uploaded coverage.")

    _notify(progress_callback, "graph", "Building the network graph", 0.28)
    graph = build_graph(network_aoi)
    candidates_gdf = candidates_gdf.copy()
    _notify(progress_callback, "snap", "Snapping candidate stations to graph nodes", 0.36)
    candidates_gdf["graph_node"] = candidates_gdf.geometry.apply(lambda point: nearest_graph_node(point, graph))

    _notify(progress_callback, "links", "Computing shortest-path candidate links", 0.48)
    links_df = build_links_dataframe(candidates_gdf, graph)
    nodes_df = (
        candidates_gdf[["graph_node", "Station_Name", "Trips", "estimated_docks", "geometry", "type"]]
        .rename(columns={"graph_node": "node", "Trips": "demand"})
        .drop_duplicates(subset=["node"])
        .reset_index(drop=True)
    )
    nodes_df["x"] = nodes_df.geometry.x
    nodes_df["y"] = nodes_df.geometry.y

    lts: dict[tuple[Any, Any], float] = {}
    costs: dict[tuple[Any, Any], float] = {}
    for _, row in links_df.iterrows():
        i, j = row["from_node"], row["to_node"]
        lts_val = float(row["mean_lts"])
        cost = float(row["total_length"]) * upgrade_cost_per_meter_from_lts(lts_val, config)
        lts[(i, j)] = lts_val
        lts[(j, i)] = lts_val
        costs[(i, j)] = cost
        costs[(j, i)] = cost

    data = {
        "N": nodes_df["node"].tolist(),
        "L": list(lts.keys()),
        "D": dict(zip(nodes_df["node"], nodes_df["demand"])),
        "D_peak": dict(zip(nodes_df["node"], nodes_df["estimated_docks"])),
        "LTS": lts,
        "C_up": costs,
        "C_dock": float(config.dock_unit_cost),
        "C_station": float(config.station_fixed_cost),
        "S_min": int(config.station_minimum),
        "M_min": int(config.link_minimum),
        "D_min": float(config.demand_minimum),
    }
    return data, stations_gdf, network_aoi, candidates_gdf, graph, links_df, nodes_df


class NSGARunner:
    def __init__(self, data: dict[str, Any], config: UIConfig, progress_callback=None) -> None:
        self.data = data
        self.config = config
        self.progress_callback = progress_callback
        self._nodes = list(data["N"])
        self._links = list(data["L"])
        self._demand = dict(data["D"])
        self._peak = dict(data["D_peak"])
        self._lts = dict(data["LTS"])
        self._upgrade_cost = dict(data["C_up"])
        self._dock_cost = float(data["C_dock"])
        self._station_cost = float(data.get("C_station", 0.0))
        self._station_minimum = int(data["S_min"])
        self._link_minimum = int(data["M_min"])
        self._demand_minimum = float(data["D_min"])
        self._capacity_max = max(self._peak.values()) if self._peak else 50.0
        self._link_index = list(self._links)
        self.n_y = len(self._nodes)
        self.n_x = len(self._links)
        self.n_z = len(self._nodes)

        random.seed(config.seed)
        np.random.seed(config.seed)

        if "BixiDatasetFitnessMulti" not in creator.__dict__:
            creator.create("BixiDatasetFitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
        if "BixiDatasetIndividual" not in creator.__dict__:
            creator.create("BixiDatasetIndividual", list, fitness=creator.BixiDatasetFitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bin", random.randint, 0, 1)
        self.toolbox.register("attr_z", random.uniform, 0.0, float(self._capacity_max))
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.BixiDatasetIndividual,
            ([self.toolbox.attr_bin] * (self.n_y + self.n_x) + [self.toolbox.attr_z] * self.n_z),
            n=1,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate, indpb=0.02, z_sigma=0.1)
        self.toolbox.register("select", tools.selNSGA2)

    def get_lts(self, i: Any, j: Any) -> float:
        return float(self._lts.get((i, j), self._lts.get((j, i), 0.0)))

    def get_upgrade_cost(self, i: Any, j: Any) -> float:
        return float(self._upgrade_cost.get((i, j), self._upgrade_cost.get((j, i), 0.0)))

    def decode(self, individual: list[float]) -> tuple[dict[Any, int], dict[tuple[Any, Any], int], dict[Any, float]]:
        y_vals = np.array(individual[: self.n_y], dtype=float)
        x_vals = np.array(individual[self.n_y : self.n_y + self.n_x], dtype=float)
        z_vals = np.array(individual[self.n_y + self.n_x :], dtype=float)
        y = {node: int(v) for node, v in zip(self._nodes, y_vals)}
        x = {self._link_index[idx]: int(x_vals[idx]) for idx in range(self.n_x)}
        z = {node: float(v) for node, v in zip(self._nodes, z_vals)}
        return y, x, z

    def encode(
        self,
        y: dict[Any, int],
        x: dict[tuple[Any, Any], int],
        z: dict[Any, float],
    ) -> list[float]:
        return [
            *[int(y[node]) for node in self._nodes],
            *[int(x[edge]) for edge in self._link_index],
            *[float(z[node]) for node in self._nodes],
        ]

    def constraint_violation(
        self,
        y: dict[Any, int],
        x: dict[tuple[Any, Any], int],
        z: dict[Any, float],
    ) -> float:
        violation = 0.0
        violation += max(0.0, self._station_minimum - sum(y.values()))
        violation += max(0.0, self._link_minimum - sum(x.values()))
        violation += max(0.0, self._demand_minimum - sum(self._demand[node] * y[node] for node in self._nodes))
        for node in self._nodes:
            violation += max(0.0, float(self._peak.get(node, 0.0)) * y[node] - z[node])
        incident = {node: 0 for node in self._nodes}
        for (i, j), selected in x.items():
            if selected:
                violation += max(0.0, selected - y[i])
                violation += max(0.0, selected - y[j])
                incident[i] += 1
                incident[j] += 1
        for node in self._nodes:
            violation += max(0.0, y[node] - incident[node])
        return violation

    def repair(self, individual: list[float]) -> list[float]:
        y, x, z = self.decode(individual)
        y = {node: 1 if y[node] >= 1 else 0 for node in self._nodes}
        x = {edge: 1 if x[edge] >= 1 else 0 for edge in self._link_index}

        total_stations = sum(y.values())
        total_demand = sum(self._demand[node] * y[node] for node in self._nodes)
        if total_stations < self._station_minimum or total_demand < self._demand_minimum:
            for node in sorted(self._nodes, key=lambda item: self._demand[item], reverse=True):
                if y[node] == 0:
                    y[node] = 1
                    total_stations += 1
                    total_demand += self._demand[node]
                if total_stations >= self._station_minimum and total_demand >= self._demand_minimum:
                    break

        active_nodes = {node for node in self._nodes if y[node] == 1}
        x = {
            edge: 1 if edge[0] in active_nodes and edge[1] in active_nodes and x[edge] else 0
            for edge in self._link_index
        }

        if len(active_nodes) >= 2 and sum(x.values()) < self._link_minimum:
            remaining = [edge for edge in self._link_index if edge[0] in active_nodes and edge[1] in active_nodes and x[edge] == 0]
            random.shuffle(remaining)
            needed = self._link_minimum - sum(x.values())
            for edge in remaining[:needed]:
                x[edge] = 1

        if len(active_nodes) >= 2:
            incident = {node: 0 for node in self._nodes}
            for (i, j), selected in x.items():
                if selected:
                    incident[i] += 1
                    incident[j] += 1
            active_list = list(active_nodes)
            for node in active_list:
                if incident[node] == 0:
                    partner = random.choice([other for other in active_list if other != node])
                    edge = (node, partner) if (node, partner) in x else (partner, node)
                    x[edge] = 1

        z = {
            node: 0.0 if y[node] == 0 else min(max(float(self._peak.get(node, 0.0)), float(z[node])), float(self._capacity_max))
            for node in self._nodes
        }
        individual[:] = self.encode(y, x, z)
        return individual

    def evaluate(self, individual: list[float]) -> tuple[float, float, float]:
        self.repair(individual)
        y, x, z = self.decode(individual)
        z1 = sum(self._demand[node] * y[node] for node in self._nodes)
        z2 = sum(self.get_lts(i, j) * x[(i, j)] for (i, j) in self._links)
        z3 = (
            sum(self.get_upgrade_cost(i, j) * x[(i, j)] for (i, j) in self._links)
            + self._dock_cost * sum(z[node] for node in self._nodes)
            + self._station_cost * sum(y.values())
        )
        penalty = 1e6 * self.constraint_violation(y, x, z)
        return z1 - penalty, z2 + penalty, z3 + penalty

    def mutate(self, individual: list[float], indpb: float = 0.02, z_sigma: float = 0.1) -> tuple[list[float]]:
        for idx in range(self.n_y + self.n_x):
            if random.random() < indpb:
                individual[idx] = 1 - int(individual[idx])
        for idx in range(self.n_y + self.n_x, self.n_y + self.n_x + self.n_z):
            if random.random() < indpb:
                individual[idx] = float(individual[idx]) + random.gauss(0.0, z_sigma * float(self._capacity_max))
                individual[idx] = min(max(0.0, float(individual[idx])), float(self._capacity_max))
        self.repair(individual)
        return (individual,)

    def compute_objectives(
        self,
        y: dict[Any, int],
        x: dict[tuple[Any, Any], int],
        z: dict[Any, float],
    ) -> tuple[float, float, float, float]:
        z1 = sum(self._demand[node] * y[node] for node in self._nodes)
        z2 = sum(self.get_lts(i, j) * x[(i, j)] for (i, j) in self._links)
        z3 = (
            sum(self.get_upgrade_cost(i, j) * x[(i, j)] for (i, j) in self._links)
            + self._dock_cost * sum(z[node] for node in self._nodes)
            + self._station_cost * sum(y.values())
        )
        return z1, z2, z3, self.constraint_violation(y, x, z)

    def run(self) -> tuple[pd.DataFrame, list[list[float]]]:
        _notify(self.progress_callback, "optimization", "Initializing NSGA-II population", 0.58)
        population = self.toolbox.population(n=self.config.population_size)
        for individual in population:
            self.repair(individual)
        invalid = [individual for individual in population if not individual.fitness.valid]
        for individual in invalid:
            individual.fitness.values = self.toolbox.evaluate(individual)
        population = self.toolbox.select(population, len(population))

        for generation in range(self.config.generations):
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(individual) for individual in offspring]
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config.crossover_probability:
                    self.toolbox.mate(ind1, ind2)
                    del ind1.fitness.values, ind2.fitness.values
            for individual in offspring:
                if random.random() < self.config.mutation_probability:
                    self.toolbox.mutate(individual)
                    del individual.fitness.values
            invalid = [individual for individual in offspring if not individual.fitness.valid]
            for individual in invalid:
                individual.fitness.values = self.toolbox.evaluate(individual)
            population = self.toolbox.select(population + offspring, self.config.population_size)
            progress = 0.6 + 0.26 * ((generation + 1) / max(1, self.config.generations))
            _notify(
                self.progress_callback,
                "optimization",
                f"Running NSGA-II generation {generation + 1} of {self.config.generations}",
                progress,
            )

        rows: list[dict[str, Any]] = []
        _notify(self.progress_callback, "postprocess", "Scoring final population and selecting representative solutions", 0.9)
        for individual in population:
            y, x, z = self.decode(individual)
            z1, z2, z3, violation = self.compute_objectives(y, x, z)
            selected_stations = [node for node in self._nodes if y[node] == 1]
            selected_links = [edge for edge in self._links if x[edge] == 1]
            rows.append(
                {
                    "ok": violation == 0,
                    "Z1_demand": z1,
                    "Z2_stress": z2,
                    "Z3_cost": z3,
                    "cv": violation,
                    "selected_stations": len(selected_stations),
                    "selected_links": len(selected_links),
                    "individual": individual,
                }
            )
        return pd.DataFrame(rows), population


def build_solution_catalog(
    pareto_df: pd.DataFrame,
    runner: NSGARunner,
) -> dict[str, dict[str, Any]]:
    feasible = pareto_df[pareto_df["ok"]].copy()
    if feasible.empty:
        feasible = pareto_df.sort_values(["cv", "Z1_demand"], ascending=[True, False]).head(20).copy()

    def unpack(row: pd.Series) -> dict[str, Any]:
        y, x, z = runner.decode(row["individual"])
        stations = [node for node in runner._nodes if y[node] == 1]
        links = [edge for edge in runner._links if x[edge] == 1]
        avg_lts = float(row["Z2_stress"]) / max(1, len(links))
        return {
            "metrics": {
                "demand": float(row["Z1_demand"]),
                "stress": float(row["Z2_stress"]),
                "avg_lts": avg_lts,
                "cost": float(row["Z3_cost"]),
                "selected_stations": len(stations),
                "selected_links": len(links),
                "constraint_violation": float(row["cv"]),
            },
            "selected_stations": stations,
            "selected_links": links,
        }

    catalog: dict[str, dict[str, Any]] = {}
    catalog["Best Demand"] = unpack(feasible.sort_values("Z1_demand", ascending=False).iloc[0])
    catalog["Best Stress"] = unpack(feasible.sort_values("Z2_stress", ascending=True).iloc[0])
    catalog["Best Cost"] = unpack(feasible.sort_values("Z3_cost", ascending=True).iloc[0])

    normalized = feasible[["Z1_demand", "Z2_stress", "Z3_cost"]].copy()
    normalized["Z1_demand"] = 1 - (
        (normalized["Z1_demand"] - normalized["Z1_demand"].min())
        / max(1e-9, normalized["Z1_demand"].max() - normalized["Z1_demand"].min())
    )
    normalized["Z2_stress"] = (
        normalized["Z2_stress"] - normalized["Z2_stress"].min()
    ) / max(1e-9, normalized["Z2_stress"].max() - normalized["Z2_stress"].min())
    normalized["Z3_cost"] = (
        normalized["Z3_cost"] - normalized["Z3_cost"].min()
    ) / max(1e-9, normalized["Z3_cost"].max() - normalized["Z3_cost"].min())
    balanced_idx = (normalized.pow(2).sum(axis=1)).sort_values().index[0]
    catalog["Balanced"] = unpack(feasible.loc[balanced_idx])
    return catalog


def create_pareto_figure(pareto_df: pd.DataFrame) -> go.Figure:
    plot_df = pareto_df.copy()
    color = np.where(plot_df["ok"], "Feasible", "Infeasible")
    fig = go.Figure()
    feasible_mask = color == "Feasible"
    if feasible_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=plot_df.loc[feasible_mask, "Z1_demand"],
                y=plot_df.loc[feasible_mask, "Z2_stress"],
                z=plot_df.loc[feasible_mask, "Z3_cost"],
                mode="markers",
                name="Feasible",
                marker=dict(
                    size=6,
                    opacity=0.88,
                    color=plot_df.loc[feasible_mask, "Z1_demand"],
                    colorscale=[
                        [0.0, "#8a9aa9"],
                        [0.45, "#2c7fb8"],
                        [1.0, "#1f7a8c"],
                    ],
                    line=dict(width=0.6, color="rgba(255,255,255,0.45)"),
                    colorbar=dict(title="Demand"),
                ),
                customdata=np.stack(
                    [
                        plot_df.loc[feasible_mask, "selected_stations"],
                        plot_df.loc[feasible_mask, "selected_links"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>Feasible solution</b><br>"
                    "Demand coverage: %{x:,.0f}<br>"
                    "Stress: %{y:,.1f}<br>"
                    "Cost: %{z:,.0f}<br>"
                    "Stations: %{customdata[0]}<br>"
                    "Links: %{customdata[1]}<extra></extra>"
                ),
            )
        )
    infeasible_mask = color == "Infeasible"
    if infeasible_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=plot_df.loc[infeasible_mask, "Z1_demand"],
                y=plot_df.loc[infeasible_mask, "Z2_stress"],
                z=plot_df.loc[infeasible_mask, "Z3_cost"],
                mode="markers",
                name="Infeasible",
                marker=dict(
                    size=4,
                    opacity=0.3,
                    color="#c44e52",
                    line=dict(width=0),
                ),
                customdata=np.stack(
                    [
                        plot_df.loc[infeasible_mask, "selected_stations"],
                        plot_df.loc[infeasible_mask, "selected_links"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>Infeasible solution</b><br>"
                    "Demand coverage: %{x:,.0f}<br>"
                    "Stress: %{y:,.1f}<br>"
                    "Cost: %{z:,.0f}<br>"
                    "Stations: %{customdata[0]}<br>"
                    "Links: %{customdata[1]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Demand coverage",
                showbackground=True,
                backgroundcolor="rgba(232,247,241,0.68)",
                gridcolor="rgba(31,122,140,0.14)",
                zerolinecolor="rgba(31,122,140,0.12)",
            ),
            yaxis=dict(
                title="Stress",
                showbackground=True,
                backgroundcolor="rgba(249,241,240,0.7)",
                gridcolor="rgba(196,78,82,0.14)",
                zerolinecolor="rgba(196,78,82,0.12)",
            ),
            zaxis=dict(
                title="Cost",
                showbackground=True,
                backgroundcolor="rgba(239,244,247,0.8)",
                gridcolor="rgba(44,127,184,0.14)",
                zerolinecolor="rgba(44,127,184,0.12)",
                tickformat=",.0f",
            ),
        ),
        margin=dict(l=0, r=0, t=46, b=0),
        title="Optimization trade-off surface",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.02,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    return fig


def _path_to_latlon(graph: nx.Graph, from_node: Any, to_node: Any, crs: Any) -> list[tuple[float, float]]:
    path = nx.shortest_path(graph, from_node, to_node, weight="length")
    segment = gpd.GeoSeries([LineString(path)], crs=crs).to_crs(4326).iloc[0]
    return [(lat, lon) for lon, lat in segment.coords]


def _reduce_station_layer(layer_df: gpd.GeoDataFrame, limit: int) -> gpd.GeoDataFrame:
    """Keep map payloads manageable by prioritizing the highest-demand points."""
    if len(layer_df) <= limit:
        return layer_df
    sort_columns = ["Trips"] if "Trips" in layer_df.columns else None
    if sort_columns:
        return layer_df.sort_values(sort_columns, ascending=False).head(limit).copy()
    return layer_df.head(limit).copy()


def create_solution_map(
    solution: dict[str, Any],
    graph: nx.Graph,
    candidates_gdf: gpd.GeoDataFrame,
    links_df: pd.DataFrame,
) -> go.Figure:
    MAX_CURRENT_STATIONS_ON_MAP = 450
    MAX_CANDIDATE_STATIONS_ON_MAP = 650

    candidates_4326 = candidates_gdf.to_crs(4326).copy()
    center_series = candidates_4326
    center_lat = float(center_series.geometry.y.mean())
    center_lon = float(center_series.geometry.x.mean())

    candidate_rows = (
        candidates_gdf.sort_values(["type", "Station_Name"])
        .drop_duplicates(subset=["graph_node"], keep="first")
        .copy()
    )
    candidate_rows_4326 = candidate_rows.to_crs(4326)
    candidate_rows_4326["node_key"] = candidate_rows_4326["graph_node"].map(repr)

    station_name_by_node = {row["node_key"]: str(row["Station_Name"]) for _, row in candidate_rows_4326.iterrows()}
    station_docks_by_node = {
        row["node_key"]: float(row.get("estimated_docks", 0.0))
        for _, row in candidate_rows_4326.iterrows()
    }
    selected_node_keys = {repr(node) for node in solution["selected_stations"]}
    candidate_rows_4326["is_selected"] = candidate_rows_4326["node_key"].isin(selected_node_keys)

    link_lookup = links_df.copy()
    link_lookup["edge_key"] = link_lookup.apply(
        lambda row: tuple(sorted((repr(row["from_node"]), repr(row["to_node"])))),
        axis=1,
    )
    selected_link_details: list[dict[str, Any]] = []
    for edge in solution["selected_links"]:
        matches = link_lookup[
            link_lookup["edge_key"] == tuple(sorted((repr(edge[0]), repr(edge[1]))))
        ]
        if matches.empty:
            continue
        detail = matches.iloc[0].to_dict()
        detail["edge"] = edge
        detail["lts_level"] = int(np.clip(np.rint(float(detail["mean_lts"])), 1, 4))
        selected_link_details.append(detail)

    station_points = gpd.GeoSeries(
        [Point(node) for node in solution["selected_stations"]],
        crs=candidates_gdf.crs,
    ).to_crs(4326)

    fig = go.Figure()
    current_mask = candidate_rows_4326["type"].astype(str).eq("original_station")
    potential_mask = ~current_mask

    if current_mask.any():
        current = _reduce_station_layer(candidate_rows_4326.loc[current_mask], MAX_CURRENT_STATIONS_ON_MAP)
        fig.add_trace(
            go.Scattermapbox(
                lat=current.geometry.y,
                lon=current.geometry.x,
                mode="markers",
                marker=dict(size=7, color="#8a9aa9", opacity=0.42),
                customdata=np.stack(
                    [
                        current["Station_Name"],
                        current["Trips"].astype(float),
                        current["estimated_docks"].astype(float),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Layer: Existing station<br>"
                    "Trips: %{customdata[1]:,.0f}<br>"
                    "Estimated docks: %{customdata[2]:,.0f}<extra></extra>"
                ),
                name="Current stations",
                meta={
                    "trace_type": "station_layer",
                    "layer_name": "current",
                    "rendered_points": int(len(current)),
                    "sampled": bool(current_mask.sum() > len(current)),
                },
            )
        )

    if potential_mask.any():
        potential = _reduce_station_layer(candidate_rows_4326.loc[potential_mask], MAX_CANDIDATE_STATIONS_ON_MAP)
        fig.add_trace(
            go.Scattermapbox(
                lat=potential.geometry.y,
                lon=potential.geometry.x,
                mode="markers",
                marker=dict(size=5, color="#2c7fb8", opacity=0.18),
                customdata=np.stack(
                    [
                        potential["Station_Name"],
                        potential["Trips"].astype(float),
                        potential["estimated_docks"].astype(float),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Layer: Candidate option<br>"
                    "Trips: %{customdata[1]:,.0f}<br>"
                    "Estimated docks: %{customdata[2]:,.0f}<extra></extra>"
                ),
                name="Candidate options",
                meta={
                    "trace_type": "station_layer",
                    "layer_name": "candidate",
                    "rendered_points": int(len(potential)),
                    "sampled": bool(potential_mask.sum() > len(potential)),
                },
            )
        )

    lts_colors = {
        1: "#22c55e",
        2: "#f59e0b",
        3: "#f97316",
        4: "#dc2626",
    }
    shown_lts: set[int] = set()
    for detail in selected_link_details:
        edge = detail["edge"]
        try:
            coords = _path_to_latlon(graph, edge[0], edge[1], candidates_gdf.crs)
        except Exception:
            continue
        lats = [lat for lat, _ in coords]
        lons = [lon for _, lon in coords]
        lts_level = int(detail["lts_level"])
        from_station = station_name_by_node.get(repr(edge[0]), str(edge[0]))
        to_station = station_name_by_node.get(repr(edge[1]), str(edge[1]))
        fig.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=6, color=lts_colors.get(lts_level, "#4169E1")),
                hovertemplate=(
                    "<b>Recommended lane upgrade</b><br>"
                    f"From: {from_station}<br>"
                    f"To: {to_station}<br>"
                    f"Distance: {float(detail['total_length']):,.0f} m<br>"
                    f"Distance: {float(detail['total_length'])/1000.0:.2f} km<br>"
                    f"Previous LTS: {float(detail['mean_lts']):.2f}"
                    "<extra></extra>"
                ),
                name=f"Routes with LTS {lts_level}",
                legendgroup=f"lts_{lts_level}",
                showlegend=lts_level not in shown_lts,
                meta={
                    "trace_type": "selected_links",
                    "lts_level": lts_level,
                    "distance_km": float(detail["total_length"]) / 1000.0,
                },
            )
        )
        shown_lts.add(lts_level)

    hover_text = []
    for node in solution["selected_stations"]:
        station_name = station_name_by_node.get(repr(node), str(node))
        hover_text.append(
            f"{station_name}<br>Selected station<br>Docks: {station_docks_by_node.get(repr(node), 0.0):,.0f}"
        )

    fig.add_trace(
        go.Scattermapbox(
            lat=station_points.y,
            lon=station_points.x,
            mode="markers",
            marker=dict(
                size=15,
                color="#84CC16",
                opacity=0.98,
            ),
            customdata=np.stack(
                [
                    [station_name_by_node.get(repr(node), str(node)) for node in solution["selected_stations"]],
                    [
                        float(
                            candidate_rows_4326.loc[
                                candidate_rows_4326["node_key"].eq(repr(node)),
                                "Trips",
                            ].iloc[0]
                        )
                        if not candidate_rows_4326.loc[
                            candidate_rows_4326["node_key"].eq(repr(node)),
                            "Trips",
                        ].empty
                        else 0.0
                        for node in solution["selected_stations"]
                    ],
                    [station_docks_by_node.get(repr(node), 0.0) for node in solution["selected_stations"]],
                ],
                axis=1,
            ),
            text=hover_text,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Layer: Selected station<br>"
                "Trips: %{customdata[1]:,.0f}<br>"
                "Estimated docks: %{customdata[2]:,.0f}<extra></extra>"
            ),
            name="Selected stations",
            meta={"trace_type": "station_layer", "layer_name": "selected"},
        )
    )

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11.7,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="Optimized network map",
        legend=dict(
            orientation="h",
            y=0.02,
            x=0.01,
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="rgba(17,24,39,0.14)",
            borderwidth=1,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="rgba(11,53,82,0.94)",
            bordercolor="rgba(67,184,163,0.9)",
            font=dict(color="#f8fbfd", size=13),
        ),
    )
    return fig


def run_pipeline(
    stations_path: str | Path,
    network_path: str | Path,
    config: UIConfig,
    progress_callback=None,
) -> OptimizationArtifacts:
    data, stations_gdf, network_gdf, candidates_gdf, graph, links_df, nodes_df = prepare_problem(
        stations_path,
        network_path,
        config,
        progress_callback=progress_callback,
    )
    runner = NSGARunner(data, config, progress_callback=progress_callback)
    pareto_df, _ = runner.run()
    solutions = build_solution_catalog(pareto_df, runner)
    _notify(progress_callback, "complete", "Optimization complete", 1.0)
    return OptimizationArtifacts(
        stations_gdf=stations_gdf,
        network_gdf=network_gdf,
        candidates_gdf=candidates_gdf,
        graph=graph,
        links_df=links_df,
        nodes_df=nodes_df,
        pareto_df=pareto_df,
        solutions=solutions,
    )
