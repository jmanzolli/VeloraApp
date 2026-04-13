# Velora

Velora is a web-based decision-support platform for bike-share and cycling-infrastructure planning. It combines a Streamlit front end, a FastAPI backend, and a multi-objective optimization pipeline to help users evaluate trade-offs between:

- demand coverage
- network stress / LTS
- intervention cost
- station and corridor selection

The current application is map-centric: users upload a station dataset and a street-network file, run the optimizer, and explore representative solutions such as `Balanced`, `Best Demand`, `Best Cost`, and `Best Stress`.

## What The Platform Does

Velora turns a station table and a cycling-network file into an interactive planning workflow:

1. Load station demand data and a network with LTS values
2. Generate candidate station locations around existing stations
3. Build a graph representation of the uploaded network
4. Run an NSGA-II search over station and link decisions
5. Extract representative solutions from the Pareto frontier
6. Visualize them through:
   - KPIs
   - a trade-off explorer
   - an interactive map
   - saved runs and downloadable outputs

## Repository Structure

- [streamlit_app.py](/Users/natomanzolli/Documents/GitHub/BIXIdataset/streamlit_app.py): Streamlit UI and interaction logic
- [backend_api.py](/Users/natomanzolli/Documents/GitHub/BIXIdataset/backend_api.py): FastAPI backend for optimization jobs, run storage, and saved-run retrieval
- [ui/optimizer.py](/Users/natomanzolli/Documents/GitHub/BIXIdataset/ui/optimizer.py): optimization pipeline, graph building, map generation, and representative-solution extraction
- [ui/storage.py](/Users/natomanzolli/Documents/GitHub/BIXIdataset/ui/storage.py): run persistence utilities
- [requirements-ui.txt](/Users/natomanzolli/Documents/GitHub/BIXIdataset/requirements-ui.txt): Python dependencies for the app
- [UI_README.md](/Users/natomanzolli/Documents/GitHub/BIXIdataset/UI_README.md): shorter UI-specific notes
- [app_data/](/Users/natomanzolli/Documents/GitHub/BIXIdataset/app_data): persisted optimization runs
- [verification_stations.csv](/Users/natomanzolli/Documents/GitHub/BIXIdataset/verification_stations.csv): tiny verification dataset
- [verification_network.geojson](/Users/natomanzolli/Documents/GitHub/BIXIdataset/verification_network.geojson): tiny verification network

## Architecture

Velora currently runs as two Python services:

- Front end: Streamlit
- Backend: FastAPI

High-level flow:

```text
User uploads files in Streamlit
    ->
Streamlit posts files + config to FastAPI /optimize
    ->
FastAPI runs optimization asynchronously
    ->
FastAPI stores result payload under app_data/runs/<run_id>
    ->
Streamlit polls job status and renders the completed run
    ->
User can reopen saved runs without recomputing
```

## Core Technologies

- Streamlit
- FastAPI
- Uvicorn
- Pandas
- NumPy
- GeoPandas
- Shapely
- NetworkX
- DEAP
- Plotly
- Pyogrio
- OpenPyXL

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements-ui.txt
```

## Running Locally

Velora needs both the backend and the Streamlit app running.

### Terminal 1: Start the backend

```bash
uvicorn backend_api:app --host 127.0.0.1 --port 8000
```

### Terminal 2: Start the Streamlit app

```bash
streamlit run streamlit_app.py --server.port 8501
```

### Open the app

- Frontend: `http://127.0.0.1:8501`
- Backend health check: `http://127.0.0.1:8000/health`

The default backend URL shown in the Streamlit sidebar is:

```text
http://127.0.0.1:8000
```

## Input Data Requirements

### Station file

Accepted formats:

- `.csv`
- `.xlsx`
- `.xls`

Required columns:

- `Station_Name`
- `Latitude`
- `Longitude`
- `Trips`

Optional columns:

- `estimated_docks`

Supported aliases are normalized by the optimizer, including:

- `station`, `name`, `station_name` -> `Station_Name`
- `lat`, `latitude` -> `Latitude`
- `lon`, `lng`, `longitude` -> `Longitude`
- `trips`, `total_trips`, `demand` -> `Trips`
- `docks`, `estimated_docks` -> `estimated_docks`

If `estimated_docks` is missing, Velora derives a fallback estimate from demand.

### Network file

Accepted vector formats:

- `.gpkg`
- `.geojson`
- `.json`
- `.shp`
- `.parquet`

Required columns:

- `geometry`
- `lts`

Optional columns:

- `length`

If `length` is missing, Velora computes it from geometry. The platform expects line geometries representing street or path segments.

## Optimization Workflow

The optimization logic lives in [ui/optimizer.py](/Users/natomanzolli/Documents/GitHub/BIXIdataset/ui/optimizer.py).

Main stages:

1. Validate and normalize uploaded station data
2. Read and validate the uploaded network
3. Build a station GeoDataFrame
4. Generate candidate station points around each station
5. Clip and prepare the network
6. Build a graph from the network geometry
7. Build candidate links between relevant nodes
8. Run NSGA-II
9. Score the final population
10. Extract representative solutions
11. Generate Plotly figures and saved payloads

## Representative Solutions

Each completed run stores four main representative scenarios:

- `Balanced`
- `Best Demand`
- `Best Stress`
- `Best Cost`

Each scenario includes:

- metrics
- map figure JSON
- selected station records
- selected link records
- available LTS levels

## User Interface Features

The current Streamlit interface includes:

- branded left sidebar with upload and model controls
- scenario selector
- comparison mode
- interactive decision map
- route-length and LTS filters
- demand-bubble overlay
- decision snapshot KPIs
- trade-off explorer
- scenario comparison table
- downloadable selected stations and links
- on-demand decision report generation
- saved-run reopening and deletion

## Saved Runs

Saved runs are stored under:

```text
app_data/runs/<run_id>/
```

A run directory typically contains:

- `summary.json`
- `pareto.json`
- `pareto_rows.csv`
- `map_*.json`
- `stations_*.csv`
- `links_*.csv`

Saved runs can be reopened directly from the Streamlit sidebar without rerunning optimization.

## Backend API

### `GET /health`

Simple health check.

Returns:

```json
{"status":"ok"}
```

### `GET /runs`

Returns saved run summaries.

### `GET /runs/{run_id}`

Returns a full saved run payload.

### `GET /jobs/{job_id}`

Returns live job status:

- `queued`
- `running`
- `completed`
- `failed`

### `POST /optimize`

Starts an optimization job.

Expected multipart fields:

- `station_file`
- `network_file`
- `candidate_points_per_station`
- `station_buffer_meters`
- `area_of_interest_buffer_meters`
- `station_minimum`
- `link_minimum`
- `population_size`
- `generations`
- `seed`
- `dock_unit_cost`
- `station_fixed_cost`
- `link_cost_lts1_per_km`
- `link_cost_lts2_per_km`
- `link_cost_lts3_per_km`
- `link_cost_lts4_per_km`

## Verification Dataset

This repo includes a tiny verification case for quick sanity checks:

- [verification_stations.csv](/Users/natomanzolli/Documents/GitHub/BIXIdataset/verification_stations.csv)
- [verification_network.geojson](/Users/natomanzolli/Documents/GitHub/BIXIdataset/verification_network.geojson)

These are useful for:

- confirming the backend boots correctly
- validating the job lifecycle
- testing saved-run serialization
- checking UI behavior quickly without a large dataset

## Deployment Options

Velora is easiest to deploy as two services:

### Option A

- Backend on Render or Railway
- Frontend on Streamlit Community Cloud

### Option B

- Backend on Render
- Frontend on Render

### Backend start command

```bash
uvicorn backend_api:app --host 0.0.0.0 --port $PORT
```

### Streamlit start command

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

If deploying online, the frontend must point to the public backend URL rather than `127.0.0.1`.

## Performance Notes

Runtime depends heavily on:

- number of uploaded stations
- network size after clipping
- candidate points per station
- population size
- number of generations
- number of candidate links created

The app is much faster on the small verification dataset than on a full city network.

## Troubleshooting

### The frontend loads but optimization fails

Check:

- backend is running
- backend URL in the sidebar is correct
- both files are uploaded
- the network file includes an `lts` field

### The map feels slow

Large networks and large Plotly payloads can make rerenders slower. Reduce:

- population size
- generations
- network extent
- candidate density

### Saved runs do not appear

Check that:

- optimization completed successfully
- `app_data/runs/` is writable
- the backend process has permission to write files

### Geo errors or missing geometry

Make sure the uploaded network:

- contains valid line geometries
- has a usable CRS or can be interpreted by GeoPandas
- includes the required `lts` field

## Development Notes

- The current app is intentionally still contained in a single Streamlit file for speed of iteration.
- The UI has been cleaned up with helper functions for repeated layout patterns.
- The backend serializes Plotly and NumPy-heavy payloads for saved runs and job responses.

## License

This repository includes a [LICENSE](/Users/natomanzolli/Documents/GitHub/BIXIdataset/LICENSE) file. Review it before redistribution or commercial deployment.
