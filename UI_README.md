# Optimization UI

This repo now includes a Streamlit front-end plus a FastAPI backend for uploading data, running the optimization, persisting runs, and reopening saved results.

## Files

- `streamlit_app.py`: Streamlit user interface
- `backend_api.py`: persistent backend service and API
- `ui/optimizer.py`: reusable optimization and visualization pipeline extracted from the notebook workflow
- `ui/storage.py`: run persistence for saved outputs and figures

## Expected station file

Upload a CSV or Excel file with at least these columns:

- `Station_Name`
- `Latitude`
- `Longitude`
- `Trips`

Optional:

- `estimated_docks`

## Expected network file

Upload a vector file such as `.gpkg`, `.geojson`, `.shp`, or `.parquet` with:

- `geometry`
- `lts`

Optional:

- `length`

If `length` is missing, the app computes it from the geometry.

## Run locally

1. Install the app dependencies.
2. Start the backend from the repo root:

```bash
pip install -r requirements-ui.txt
uvicorn backend_api:app --reload
```

3. In another terminal, launch the front-end:

```bash
streamlit run streamlit_app.py
```

4. Open the Streamlit URL shown in the terminal. The default backend URL in the UI is `http://127.0.0.1:8000`.

## Notes

- The app stores runs under `app_data/runs/`.
- The UI computes candidate station locations, clips the uploaded network to the study area, runs an NSGA-II search, and exposes representative solutions such as balanced, best-demand, best-stress, and best-cost.
- Saved runs can be reopened from the front-end without recomputing them.
- Runtime depends strongly on the number of uploaded stations, the clipped network size, and the population/generation settings.
