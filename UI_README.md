# Velora UI Notes

This file is a short companion to the main [README.md](README.md). Use it when working specifically on the Streamlit frontend.

## Frontend Entry Point

```text
streamlit_app.py
```

The frontend is responsible for:

- upload controls
- optimization configuration
- job submission and polling
- saved-run loading
- scenario selection and comparison
- decision map rendering
- report generation
- exports for selected stations and links

## Backend Contract

The UI expects the FastAPI backend to run at:

```text
http://127.0.0.1:8000
```

The sidebar allows this URL to be changed at runtime.

## Key UI Sections

- Sidebar setup rail: data, study area, costs, impact assumptions, network rules, optimizer settings, saved runs
- Planning workspace: active scenario, baseline scenario, map filters, impact assumptions
- Decision map: selected stations, selected links, route stress, demand bubbles, performance mode
- Decision snapshot: demand, mode shift, emissions, cost, average LTS
- Scenario Studio: save and manage stakeholder scenario snapshots
- Scenario Library: compare optimizer and saved stakeholder scenarios
- Decision Report: generated on demand

## Local Run Commands

Backend:

```bash
uvicorn backend_api:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
streamlit run streamlit_app.py --server.port 8501
```

## Validation

```bash
python -m py_compile streamlit_app.py backend_api.py ui/optimizer.py ui/storage.py
```

