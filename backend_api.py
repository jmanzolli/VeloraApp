from __future__ import annotations

import traceback
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from plotly.utils import PlotlyJSONEncoder

from ui.optimizer import UIConfig, create_pareto_figure, create_solution_map, run_pipeline
from ui.storage import figure_to_json, list_runs, load_run, new_run_id, save_run_payload


app = FastAPI(title="BIXI Optimization Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
executor = ThreadPoolExecutor(max_workers=2)
jobs: dict[str, dict[str, Any]] = {}


def save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(upload.file.read())
        return handle.name


def build_payload(artifacts, config: UIConfig, run_id: str) -> dict[str, Any]:
    pareto_figure = create_pareto_figure(artifacts.pareto_df)
    node_summary = (
        artifacts.candidates_gdf.sort_values(["type", "Trips"], ascending=[True, False])
        .groupby("graph_node", as_index=False)
        .first()[["graph_node", "Station_Name", "Trips", "estimated_docks", "type"]]
        .copy()
    )
    node_summary["node_key"] = node_summary["graph_node"].map(repr)

    link_lookup = artifacts.links_df.copy()
    link_lookup["edge_key"] = link_lookup.apply(
        lambda row: tuple(sorted((repr(row["from_node"]), repr(row["to_node"])))),
        axis=1,
    )
    solutions: dict[str, Any] = {}
    for solution_name, solution in artifacts.solutions.items():
        map_figure = create_solution_map(solution, artifacts.graph, artifacts.candidates_gdf, artifacts.links_df)
        selected_station_keys = {repr(node) for node in solution["selected_stations"]}
        selected_station_rows = node_summary[node_summary["node_key"].isin(selected_station_keys)].copy()
        selected_link_rows = []
        for edge in solution["selected_links"]:
            edge_key = tuple(sorted((repr(edge[0]), repr(edge[1]))))
            matches = link_lookup[link_lookup["edge_key"] == edge_key]
            if matches.empty:
                continue
            selected_link_rows.append(matches.iloc[0].to_dict())
        lts_levels = sorted(
            {
                int(float(row["mean_lts"]) + 0.5)
                for row in selected_link_rows
                if pd.notna(row.get("mean_lts"))
            }
        )
        solutions[solution_name] = {
            "metrics": solution["metrics"],
            "map_figure": figure_to_json(map_figure),
            "available_lts_levels": lts_levels,
            "selected_stations": [
                {
                    "node": repr(row["graph_node"]),
                    "station_name": str(row["Station_Name"]),
                    "trips": float(row["Trips"]),
                    "estimated_docks": float(row["estimated_docks"]),
                    "type": str(row["type"]),
                }
                for _, row in selected_station_rows.iterrows()
            ],
            "selected_links": [
                {
                    "from_node": repr(row["from_node"]),
                    "to_node": repr(row["to_node"]),
                    "from_station": str(row["from_station"]),
                    "to_station": str(row["to_station"]),
                    "total_length": float(row["total_length"]),
                    "mean_lts": float(row["mean_lts"]),
                }
                for row in selected_link_rows
            ],
        }

    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config.__dict__,
        "station_count": int(len(artifacts.stations_gdf)),
        "candidate_count": int(len(artifacts.candidates_gdf)),
        "link_pair_count": int(len(artifacts.links_df)),
        "population_rows": int(len(artifacts.pareto_df)),
        "pareto_rows": artifacts.pareto_df.drop(columns=["individual"], errors="ignore").to_dict(orient="records"),
        "pareto_figure": figure_to_json(pareto_figure),
        "solutions": solutions,
    }


def update_job(job_id: str, *, status: str | None = None, stage: str | None = None, message: str | None = None, progress: float | None = None, result: dict[str, Any] | None = None, error: str | None = None) -> None:
    job = jobs.setdefault(job_id, {})
    if status is not None:
        job["status"] = status
    if stage is not None:
        job["stage"] = stage
    if message is not None:
        job["message"] = message
    if progress is not None:
        job["progress"] = progress
    if result is not None:
        job["result"] = json.loads(json.dumps(result, cls=PlotlyJSONEncoder))
    if error is not None:
        job["error"] = error
    job["updated_at"] = datetime.now().isoformat(timespec="seconds")


def run_job(job_id: str, stations_path: str, network_path: str, config: UIConfig) -> None:
    def callback(stage: str, message: str, progress: float) -> None:
        update_job(job_id, status="running", stage=stage, message=message, progress=progress)

    try:
        artifacts = run_pipeline(stations_path, network_path, config, progress_callback=callback)
        payload = build_payload(artifacts, config, job_id)
        save_run_payload(job_id, payload)
        update_job(job_id, status="completed", stage="complete", message="Optimization complete", progress=1.0, result=payload)
    except Exception as exc:
        error_text = str(exc).strip() or repr(exc)
        update_job(job_id, status="failed", stage="failed", message="Optimization failed", progress=1.0, error=error_text)
        traceback.print_exc()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs")
def get_runs() -> list[dict[str, Any]]:
    return list_runs()


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found") from exc


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return jobs[job_id]


@app.post("/optimize")
def optimize(
    station_file: UploadFile = File(...),
    network_file: UploadFile = File(...),
    candidate_points_per_station: int = Form(2),
    station_buffer_meters: float = Form(150.0),
    area_of_interest_buffer_meters: float = Form(2000.0),
    station_minimum: int = Form(15),
    link_minimum: int = Form(50),
    population_size: int = Form(120),
    generations: int = Form(16),
    seed: int = Form(42),
    dock_unit_cost: float = Form(900.0),
    station_fixed_cost: float = Form(5000.0),
    link_cost_lts1_per_km: float = Form(0.0),
    link_cost_lts2_per_km: float = Form(2000000.0),
    link_cost_lts3_per_km: float = Form(4000000.0),
    link_cost_lts4_per_km: float = Form(8000000.0),
    mode_shift_rate: float = Form(0.15),
    average_trip_distance_km: float = Form(2.5),
    car_emission_factor_g_per_km: float = Form(190.0),
) -> dict[str, Any]:
    try:
        config = UIConfig(
            candidate_points_per_station=int(candidate_points_per_station),
            station_buffer_meters=float(station_buffer_meters),
            area_of_interest_buffer_meters=float(area_of_interest_buffer_meters),
            station_minimum=int(station_minimum),
            link_minimum=int(link_minimum),
            population_size=int(population_size),
            generations=int(generations),
            seed=int(seed),
            dock_unit_cost=float(dock_unit_cost),
            station_fixed_cost=float(station_fixed_cost),
            link_cost_lts1_per_km=float(link_cost_lts1_per_km),
            link_cost_lts2_per_km=float(link_cost_lts2_per_km),
            link_cost_lts3_per_km=float(link_cost_lts3_per_km),
            link_cost_lts4_per_km=float(link_cost_lts4_per_km),
            mode_shift_rate=float(mode_shift_rate),
            average_trip_distance_km=float(average_trip_distance_km),
            car_emission_factor_g_per_km=float(car_emission_factor_g_per_km),
        )
        stations_path = save_upload(station_file)
        network_path = save_upload(network_file)
        job_id = new_run_id()
        update_job(
            job_id,
            status="queued",
            stage="queued",
            message="Optimization job queued",
            progress=0.0,
        )
        executor.submit(run_job, job_id, stations_path, network_path, config)
        return {"job_id": job_id, "status": "queued"}
    except Exception as exc:
        error_text = str(exc).strip() or repr(exc)
        raise HTTPException(status_code=400, detail=error_text) from exc
