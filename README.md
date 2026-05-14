# Velora

Velora is a Streamlit planning platform for bike-share and cycling-infrastructure decisions. It helps users upload station demand and street-network data, run a multi-objective optimizer, compare scenarios, inspect a decision map, and export stakeholder-ready outputs.

This repository is intentionally platform-only. Paper assets, exploratory notebooks, case-study figures, and local data exports have been removed from the production repo.

## Product Capabilities

- Upload station demand files and street-network files.
- Generate a Canadian population-based station demand CSV when station trips do not exist yet.
- Calculate an LTS-scored street network from a map-selected area, OSM streets, and optional municipal layers.
- Compare point-to-point cycling routes by shortest distance, lowest LTS, and a balanced distance/stress objective.
- Generate candidate station alternatives and corridor upgrade options.
- Run an NSGA-II multi-objective optimization.
- Review representative scenarios: `Balanced`, `Best Demand`, `Best Cost`, and `Best Stress`.
- Compare scenarios using demand, cost, LTS, mode-shift, and emissions indicators.
- Explore selected stations and corridor upgrades on an interactive map.
- Load a built-in demo scenario without uploading files.
- Export selected stations, selected links, and a generated decision report.

## Repository Layout

```text
.
├── streamlit_app.py              # Streamlit application entry point
├── pages/                        # Streamlit pages for LTS and population-demand preparation
├── ui/
│   ├── optimizer.py              # Data loading, graph construction, NSGA-II, map generation
│   ├── storage.py                # Local saved-run and scenario-snapshot persistence
│   ├── demand.py                 # Canadian population-to-demand generation
│   ├── lts/                      # LTS network building, feature extraction, scoring, and export
│   └── assets/
│       ├── velora_badge.png      # App icon and brand mark
│       └── velora_demo_run.json  # Pre-optimized demo scenario
├── .streamlit/config.toml        # Streamlit theme
├── requirements.txt              # Runtime dependencies
├── runtime.txt                   # Streamlit Cloud Python runtime
├── LICENSE
└── README.md
```

## Quick Start

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

To preview the product without uploading data, click **Load demo scenario** in the sidebar.

Use **Population Demand Builder** when there is no station CSV yet. Upload a Canadian census population polygon layer, draw/select the planning area, and export the generated station CSV for the optimizer.

Use **LTS Calculator** to create a street-network file with `geometry` and `lts` from a map-selected area before running the optimizer.

Use **Route Planner** to upload an LTS network, enter point A and point B as street/place names, and compare shortest, lowest-stress, and balanced routes.

## Input Data

### Station File

Accepted formats:

- `.csv`
- `.xlsx`
- `.xls`

Required fields:

- `Station_Name`
- `Latitude`
- `Longitude`
- `Trips`

Optional field:

- `estimated_docks`

Supported aliases include:

- `station`, `name`, `station_name` -> `Station_Name`
- `lat`, `latitude` -> `Latitude`
- `lon`, `lng`, `longitude` -> `Longitude`
- `trips`, `total_trips`, `demand` -> `Trips`
- `docks`, `estimated_docks` -> `estimated_docks`

If historical station trips are unavailable, generate this file from the **Population Demand Builder** page. It accepts Canadian census population polygons such as dissemination areas or census tracts and outputs an optimizer-ready CSV with the required fields.

### Network File

Accepted formats:

- `.gpkg`
- `.geojson`
- `.json`
- `.shp`
- `.parquet`

Required fields:

- `geometry`
- `lts`

Optional field:

- `length`

If `length` is missing, Velora computes segment length from geometry.

## Workflow

1. Upload station and network files, or load the demo scenario.
2. Configure study area, costs, impact assumptions, network rules, and optimizer settings.
3. Run the optimization.
4. Compare scenarios in the planning workspace.
5. Inspect map, KPI, Pareto, scenario-library, and comparison sections.
6. Export CSV assets or generate a decision report.

## Deployment

Velora is ready for Streamlit Community Cloud.

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Use `streamlit_app.py` as the main file.

Streamlit Cloud will use:

```text
requirements.txt
runtime.txt
.streamlit/config.toml
```

## Validation

Recommended checks before release:

```bash
python -m py_compile streamlit_app.py ui/optimizer.py ui/storage.py
```

## Release Status

Current release target: `v0.1.0-beta`.

This beta is suitable for demos, stakeholder walkthroughs, and early product feedback. For production use, add persistent cloud storage for saved runs, authentication, upload limits, and job timeout controls.
