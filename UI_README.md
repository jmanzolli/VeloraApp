# Velora UI Notes

This file is a short companion to the main [README.md](README.md). Use it when working specifically on the Streamlit frontend.

## Frontend Entry Point

```text
streamlit_app.py
```

The frontend is responsible for:

- upload controls
- optimization configuration
- in-process optimization with progress feedback
- saved-run loading
- scenario selection and comparison
- decision map rendering
- report generation
- exports for selected stations and links

## Runtime Contract

The hosted UI runs as a single Streamlit app. It calls `ui.optimizer.run_pipeline()` directly and stores saved-run payloads through `ui.storage`.

`backend_api.py` remains available when an API service is useful locally, but it is not required for the Streamlit Community Cloud deployment.

## Key UI Sections

- Sidebar setup rail: data, study area, costs, impact assumptions, network rules, optimizer settings, saved runs
- Planning workspace: active scenario, baseline scenario, map filters, impact assumptions
- Decision map: selected stations, selected links, route stress, demand bubbles, performance mode
- Decision snapshot: demand, mode shift, emissions, cost, average LTS
- Scenario Studio: save and manage stakeholder scenario snapshots
- Scenario Library: compare optimizer and saved stakeholder scenarios
- Decision Report: generated on demand

## Local Run Commands

```bash
streamlit run streamlit_app.py --server.port 8501
```

## Validation

```bash
python -m py_compile streamlit_app.py backend_api.py ui/optimizer.py ui/storage.py
```
