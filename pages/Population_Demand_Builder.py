from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import shape
from streamlit_folium import st_folium

from ui.demand import CanadianPopulationDemandBuilder, DemandConfig


ASSETS_DIR = Path(__file__).resolve().parents[1] / "ui" / "assets"
BADGE_PATH = ASSETS_DIR / "velora_badge.png"
DEFAULT_MONTREAL_BBOX = {"north": 45.58, "south": 45.46, "east": -73.50, "west": -73.70}


def load_asset_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


for key, value in DEFAULT_MONTREAL_BBOX.items():
    st.session_state.setdefault(f"demand_bbox_{key}", value)
st.session_state.setdefault("demand_map_center", [45.52, -73.60])
st.session_state.setdefault("demand_map_zoom", 11)


st.set_page_config(
    page_title="Velora | Population Demand Builder",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=str(BADGE_PATH),
)

BADGE_URI = load_asset_data_uri(BADGE_PATH)

st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(180deg, #eff4f7 0%, #e7eef3 100%);
      }
      .block-container {
        max-width: 1440px;
        padding-top: 3rem;
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2b44 0%, #092137 100%);
      }
      [data-testid="stSidebar"] * {
        color: #eef6fb;
      }
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] textarea,
      [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #173042 !important;
        -webkit-text-fill-color: #173042 !important;
      }
      .velora-hero {
        padding: 1.15rem 1.3rem;
        border-radius: 20px;
        background: linear-gradient(90deg, #0b3552 0%, #0d6c75 100%);
        color: white;
        box-shadow: 0 16px 36px rgba(7,39,61,0.16);
      }
      .velora-title {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin: 0;
        font-size: 1.55rem;
        font-weight: 800;
      }
      .velora-title img {
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 12px;
        padding: 0.22rem;
        background: rgba(255,255,255,0.12);
      }
      .velora-subtitle {
        margin: 0.75rem 0 0;
        color: rgba(255,255,255,0.86);
        max-width: 66rem;
      }
      .metric-card {
        padding: 1rem 1.05rem;
        border: 1px solid rgba(11,53,82,0.12);
        border-radius: 14px;
        background: rgba(255,255,255,0.94);
      }
      .metric-label {
        margin: 0;
        color: #667986;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .metric-value {
        margin: 0.35rem 0 0;
        color: #14212b;
        font-size: 1.55rem;
        font-weight: 800;
      }
      .note-panel {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(67,184,163,0.22);
        background: rgba(232,246,242,0.9);
        color: #173042;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def geocode_city_bbox(city_name: str) -> dict[str, Any]:
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("City lookup requires osmnx. Install project requirements and try again.") from exc

    city_gdf = ox.geocode_to_gdf(city_name)
    if city_gdf.empty:
        raise ValueError(f"Could not find a boundary for {city_name}.")
    west, south, east, north = [float(value) for value in city_gdf.to_crs("EPSG:4326").total_bounds]
    return {
        "north": north,
        "south": south,
        "east": east,
        "west": west,
        "center": [float((south + north) / 2), float((west + east) / 2)],
    }


def selected_bbox() -> dict[str, float]:
    return {
        "north": float(st.session_state["demand_bbox_north"]),
        "south": float(st.session_state["demand_bbox_south"]),
        "east": float(st.session_state["demand_bbox_east"]),
        "west": float(st.session_state["demand_bbox_west"]),
    }


def set_selected_bbox(bbox: dict[str, float]) -> None:
    for key in ["north", "south", "east", "west"]:
        st.session_state[f"demand_bbox_{key}"] = float(bbox[key])
    st.session_state["demand_map_center"] = [
        float((bbox["south"] + bbox["north"]) / 2),
        float((bbox["west"] + bbox["east"]) / 2),
    ]


def bbox_from_drawing(drawing: dict[str, Any] | None) -> dict[str, float] | None:
    if not drawing or not drawing.get("geometry"):
        return None
    drawn_shape = shape(drawing["geometry"])
    if drawn_shape.is_empty:
        return None
    west, south, east, north = drawn_shape.bounds
    if north <= south or east <= west:
        return None
    return {"north": float(north), "south": float(south), "east": float(east), "west": float(west)}


def format_bbox(bbox: dict[str, float]) -> str:
    return f"N {bbox['north']:.5f}, S {bbox['south']:.5f}, E {bbox['east']:.5f}, W {bbox['west']:.5f}"


def render_area_selector(city_name: str) -> dict[str, float] | None:
    bbox = selected_bbox()
    map_obj = folium.Map(
        location=st.session_state.get("demand_map_center", [45.52, -73.60]),
        zoom_start=int(st.session_state.get("demand_map_zoom", 11)),
        tiles="OpenStreetMap",
    )
    folium.Rectangle(
        bounds=[(bbox["south"], bbox["west"]), (bbox["north"], bbox["east"])],
        color="#0b6b74",
        fill=True,
        fill_opacity=0.08,
        weight=2,
        tooltip="Current demand generation area",
    ).add_to(map_obj)
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": True,
            "polygon": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(map_obj)
    st.caption(f"Draw the planning area for {city_name}, then apply it. Current area: `{format_bbox(bbox)}`.")
    map_data = st_folium(map_obj, height=500, use_container_width=True, returned_objects=["last_active_drawing"])
    return bbox_from_drawing(map_data.get("last_active_drawing") if map_data else None)


def read_population_upload(uploaded_file) -> gpd.GeoDataFrame:
    builder = CanadianPopulationDemandBuilder()
    path = builder.write_uploaded_file(uploaded_file)
    try:
        return builder.load_population_layer(path)
    finally:
        path.unlink(missing_ok=True)


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <p class="metric-label">{label}</p>
          <p class="metric-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def make_candidate_map(points: gpd.GeoDataFrame) -> go.Figure:
    fig = go.Figure()
    if points.empty:
        return fig
    display = points.to_crs("EPSG:4326")
    marker_size = (display["Trips"].astype(float) / max(1.0, display["Trips"].max()) * 18 + 6).clip(6, 24)
    fig.add_trace(
        go.Scattermapbox(
            lon=display.geometry.x,
            lat=display.geometry.y,
            mode="markers",
            marker=dict(size=marker_size, color=display["Trips"], colorscale="Viridis", showscale=True),
            text=[
                f"{row.Station_Name}<br>Population: {row.population:,.0f}<br>Trips: {row.Trips:,.0f}<br>Docks: {row.estimated_docks:,.0f}"
                for row in display.itertuples()
            ],
            hoverinfo="text",
            name="Population candidates",
        )
    )
    bounds = display.total_bounds
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={"lat": float((bounds[1] + bounds[3]) / 2), "lon": float((bounds[0] + bounds[2]) / 2)},
            zoom=11,
        ),
        height=540,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


st.markdown(
    f"""
    <div class="velora-hero">
      <h1 class="velora-title"><img src="{BADGE_URI}" alt="">Population Demand Builder</h1>
      <p class="velora-subtitle">Create a starter station-demand file from Canadian census population polygons when historical station trips do not exist yet.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Area")
    city_name = st.text_input("City name", value="Montreal")
    if st.button("Load City On Map", use_container_width=True):
        try:
            with st.spinner("Looking up city boundary..."):
                city_bbox = geocode_city_bbox(city_name)
            set_selected_bbox(city_bbox)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.caption("Use the map first. Coordinates are an advanced fallback.")
    left, right = st.columns(2)
    with left:
        st.number_input("North", format="%.6f", key="demand_bbox_north")
        st.number_input("West", format="%.6f", key="demand_bbox_west")
    with right:
        st.number_input("South", format="%.6f", key="demand_bbox_south")
        st.number_input("East", format="%.6f", key="demand_bbox_east")

    st.markdown("### Census Population")
    population_upload = st.file_uploader(
        "StatsCan DA/CT population layer",
        type=["gpkg", "geojson", "json", "shp", "parquet", "zip"],
        help="Upload a Canadian census polygon layer with a population field. Zip shapefiles are supported.",
    )

    st.markdown("### Demand Assumptions")
    adoption_rate = st.slider("Bike adoption rate (%)", 0.0, 40.0, 8.0, step=0.5) / 100.0
    daily_trip_rate = st.slider("Daily trips per adopting resident", 0.05, 2.0, 0.35, step=0.05)
    annualization_days = st.slider("Annualization days", 30, 365, 365, step=5)
    minimum_population = st.number_input("Minimum population per candidate", min_value=0.0, value=50.0, step=25.0)
    minimum_docks = st.number_input("Minimum docks", min_value=1.0, value=8.0, step=1.0)
    trips_per_dock = st.number_input("Trips per dock heuristic", min_value=100.0, value=4000.0, step=250.0)
    generate_button = st.button("Generate Demand CSV", type="primary", use_container_width=True)

drawn_bbox = render_area_selector(city_name)
if drawn_bbox is not None:
    st.info(f"Drawn area detected: {format_bbox(drawn_bbox)}")
    if st.button("Use drawn area", type="primary"):
        set_selected_bbox(drawn_bbox)
        st.rerun()

active_bbox = selected_bbox()
if active_bbox["north"] <= active_bbox["south"] or active_bbox["east"] <= active_bbox["west"]:
    st.error("The selected area is invalid. North must exceed south, and east must exceed west.")
    st.stop()

if generate_button:
    if population_upload is None:
        st.error("Upload a Canadian census population polygon layer first.")
    else:
        try:
            with st.spinner("Generating population-based demand candidates..."):
                builder = CanadianPopulationDemandBuilder()
                population_gdf = read_population_upload(population_upload)
                clipped = builder.clip_to_bbox(population_gdf, active_bbox)
                config = DemandConfig(
                    adoption_rate=adoption_rate,
                    daily_trip_rate=daily_trip_rate,
                    annualization_days=annualization_days,
                    minimum_population=minimum_population,
                    minimum_docks=minimum_docks,
                    trips_per_dock=trips_per_dock,
                )
                station_table, candidate_points = builder.generate_station_table(clipped, config)
            st.session_state["population_station_table"] = station_table
            st.session_state["population_candidate_points"] = candidate_points
            st.session_state["population_clipped_gdf"] = clipped
            st.success("Population demand file generated.")
        except Exception as exc:
            st.error(str(exc))

station_table = st.session_state.get("population_station_table")
candidate_points = st.session_state.get("population_candidate_points")
clipped_gdf = st.session_state.get("population_clipped_gdf")

if station_table is None or candidate_points is None:
    st.markdown(
        """
        <div class="note-panel">
          Upload a Statistics Canada dissemination-area or census-tract population layer, select the planning area, and generate a Velora-compatible station-demand file. The output can be uploaded directly into the Bike Allocation page.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric("Candidates", f"{len(station_table):,}")
with metric_cols[1]:
    render_metric("Population", f"{station_table['population'].sum():,.0f}")
with metric_cols[2]:
    render_metric("Estimated Trips", f"{station_table['Trips'].sum():,.0f}")
with metric_cols[3]:
    render_metric("Estimated Docks", f"{station_table['estimated_docks'].sum():,.0f}")

tab_map, tab_data, tab_export = st.tabs(["Map", "Data", "Export"])

with tab_map:
    st.plotly_chart(make_candidate_map(candidate_points), use_container_width=True)

with tab_data:
    st.dataframe(station_table.head(300), use_container_width=True, hide_index=True)
    if clipped_gdf is not None:
        st.caption(f"Intersecting census polygons used: {len(clipped_gdf):,}")

with tab_export:
    optimizer_columns = ["Station_Name", "Latitude", "Longitude", "Trips", "estimated_docks"]
    st.download_button(
        "Download optimizer-ready station CSV",
        data=csv_bytes(station_table[optimizer_columns]),
        file_name="velora_population_demand.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
    st.caption("Upload this CSV as the station file in Velora's Bike Allocation page.")
