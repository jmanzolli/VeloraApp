from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


APP_DATA_DIR = Path(__file__).resolve().parents[1] / "app_data"
RUNS_DIR = APP_DATA_DIR / "runs"
ARTIFACTS_DIR = APP_DATA_DIR / "artifacts"
DEMAND_ARTIFACTS_DIR = ARTIFACTS_DIR / "demand"
LTS_ARTIFACTS_DIR = ARTIFACTS_DIR / "lts"


def ensure_storage() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DEMAND_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LTS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def new_artifact_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _scenario_snapshots_path(run_id: str) -> Path:
    return _run_dir(run_id) / "scenario_snapshots.json"


def _artifact_dir(kind: str) -> Path:
    if kind == "demand":
        return DEMAND_ARTIFACTS_DIR
    if kind == "lts":
        return LTS_ARTIFACTS_DIR
    raise ValueError(f"Unsupported artifact kind: {kind}")


def _artifact_path(kind: str, artifact_id: str) -> Path:
    return _artifact_dir(kind) / artifact_id


def save_run_payload(run_id: str, payload: dict[str, Any]) -> None:
    ensure_storage()
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    serialized_solutions = json.loads(json.dumps(payload["solutions"], cls=PlotlyJSONEncoder))
    summary = {
        "run_id": run_id,
        "created_at": payload["created_at"],
        "config": payload["config"],
        "solutions": serialized_solutions,
        "station_count": payload["station_count"],
        "candidate_count": payload["candidate_count"],
        "link_pair_count": payload["link_pair_count"],
        "population_rows": payload["population_rows"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, cls=PlotlyJSONEncoder))
    (run_dir / "pareto.json").write_text(json.dumps(payload["pareto_figure"], cls=PlotlyJSONEncoder))

    pd.DataFrame(payload["pareto_rows"]).to_csv(run_dir / "pareto_rows.csv", index=False)

    for solution_name, solution_payload in payload["solutions"].items():
        slug = slugify(solution_name)
        (run_dir / f"map_{slug}.json").write_text(
            json.dumps(solution_payload["map_figure"], cls=PlotlyJSONEncoder)
        )
        pd.DataFrame(solution_payload["selected_stations"]).to_csv(
            run_dir / f"stations_{slug}.csv",
            index=False,
        )
        pd.DataFrame(solution_payload["selected_links"]).to_csv(
            run_dir / f"links_{slug}.csv",
            index=False,
        )


def list_runs() -> list[dict[str, Any]]:
    ensure_storage()
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        runs.append(summary)
    return runs


def delete_run(run_id: str) -> None:
    run_dir = _run_dir(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)


def delete_all_runs() -> None:
    ensure_storage()
    for run_dir in RUNS_DIR.iterdir():
        if run_dir.is_dir():
            shutil.rmtree(run_dir)


def save_demand_artifact(label: str, station_table: pd.DataFrame, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_storage()
    artifact_id = new_artifact_id("demand")
    artifact_dir = _artifact_path("demand", artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    file_path = artifact_dir / "station_demand.csv"
    station_table.to_csv(file_path, index=False)
    summary = {
        "artifact_id": artifact_id,
        "kind": "demand",
        "label": label or artifact_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "file_path": str(file_path),
        "row_count": int(len(station_table)),
        "total_trips": float(station_table.get("Trips", pd.Series(dtype=float)).sum()),
        "metadata": metadata or {},
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(summary, indent=2, cls=PlotlyJSONEncoder))
    return summary


def save_lts_artifact(label: str, network_gdf: gpd.GeoDataFrame, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_storage()
    artifact_id = new_artifact_id("lts")
    artifact_dir = _artifact_path("lts", artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    file_path = artifact_dir / "lts_network.geojson"
    network_gdf.to_file(file_path, driver="GeoJSON")
    summary = {
        "artifact_id": artifact_id,
        "kind": "lts",
        "label": label or artifact_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "file_path": str(file_path),
        "row_count": int(len(network_gdf)),
        "mean_lts": float(network_gdf["lts"].mean()) if "lts" in network_gdf and not network_gdf.empty else None,
        "mean_effort": float(network_gdf["effort_exposure"].mean()) if "effort_exposure" in network_gdf and network_gdf["effort_exposure"].notna().any() else None,
        "metadata": metadata or {},
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(summary, indent=2, cls=PlotlyJSONEncoder))
    return summary


def list_artifacts(kind: str) -> list[dict[str, Any]]:
    ensure_storage()
    artifacts: list[dict[str, Any]] = []
    for artifact_dir in sorted(_artifact_dir(kind).iterdir(), reverse=True):
        metadata_path = artifact_dir / "metadata.json"
        if metadata_path.exists():
            artifacts.append(json.loads(metadata_path.read_text()))
    return artifacts


def list_demand_artifacts() -> list[dict[str, Any]]:
    return list_artifacts("demand")


def list_lts_artifacts() -> list[dict[str, Any]]:
    return list_artifacts("lts")


def delete_artifact(kind: str, artifact_id: str) -> None:
    artifact_dir = _artifact_path(kind, artifact_id)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


def load_run(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    summary = json.loads((run_dir / "summary.json").read_text())
    pareto_figure = json.loads((run_dir / "pareto.json").read_text())
    pareto_rows = pd.read_csv(run_dir / "pareto_rows.csv").to_dict(orient="records")

    solutions: dict[str, Any] = {}
    for solution_name, solution_payload in summary["solutions"].items():
        slug = slugify(solution_name)
        map_figure = json.loads((run_dir / f"map_{slug}.json").read_text())
        stations_rows = solution_payload.get("selected_stations")
        links_rows = solution_payload.get("selected_links")
        if not stations_rows and (run_dir / f"stations_{slug}.csv").exists():
            stations_rows = pd.read_csv(run_dir / f"stations_{slug}.csv").to_dict(orient="records")
        if not links_rows and (run_dir / f"links_{slug}.csv").exists():
            links_rows = pd.read_csv(run_dir / f"links_{slug}.csv").to_dict(orient="records")
        solutions[solution_name] = {
            **solution_payload,
            "map_figure": map_figure,
            "selected_stations": stations_rows or [],
            "selected_links": links_rows or [],
        }

    return {
        **summary,
        "pareto_figure": pareto_figure,
        "pareto_rows": pareto_rows,
        "solutions": solutions,
    }


def list_scenario_snapshots(run_id: str) -> list[dict[str, Any]]:
    path = _scenario_snapshots_path(run_id)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else []


def save_scenario_snapshot(run_id: str, snapshot: dict[str, Any]) -> None:
    ensure_storage()
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots = list_scenario_snapshots(run_id)
    serialized_snapshot = json.loads(json.dumps(snapshot, cls=PlotlyJSONEncoder))
    updated = [item for item in snapshots if item.get("snapshot_id") != serialized_snapshot.get("snapshot_id")]
    updated.append(serialized_snapshot)
    updated.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    _scenario_snapshots_path(run_id).write_text(json.dumps(updated, indent=2, cls=PlotlyJSONEncoder))


def delete_scenario_snapshot(run_id: str, snapshot_id: str) -> None:
    path = _scenario_snapshots_path(run_id)
    if not path.exists():
        return
    snapshots = [item for item in list_scenario_snapshots(run_id) if item.get("snapshot_id") != snapshot_id]
    path.write_text(json.dumps(snapshots, indent=2, cls=PlotlyJSONEncoder))


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_")


def figure_to_json(fig: go.Figure) -> dict[str, Any]:
    return fig.to_plotly_json()
