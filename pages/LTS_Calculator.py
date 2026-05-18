from __future__ import annotations

import base64
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

from ui.lts import CityNetworkBuilder, LTSCalculator, LTSFeatureExtractor
from ui.lts.builder import BoundingBox
from ui.lts.export import export_network_bytes, prepare_optimizer_export
from ui.effort import EffortCalculator, fetch_online_endpoint_elevations, sample_dem_endpoints
from ui.storage import save_lts_artifact


ASSETS_DIR = Path(__file__).resolve().parents[1] / "ui" / "assets"
BADGE_PATH = ASSETS_DIR / "velora_badge.png"


def load_asset_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


st.set_page_config(
    page_title="Bike Network Planner | LTS Network Builder",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=str(BADGE_PATH),
)

BADGE_URI = load_asset_data_uri(BADGE_PATH)

DEFAULT_MONTREAL_BBOX = {
    "north": 45.58,
    "south": 45.46,
    "east": -73.50,
    "west": -73.70,
}

for key, value in DEFAULT_MONTREAL_BBOX.items():
    st.session_state.setdefault(f"lts_bbox_{key}", value)
st.session_state.setdefault("lts_map_center", [45.52, -73.60])
st.session_state.setdefault("lts_map_zoom", 11)

pending_bbox = st.session_state.pop("lts_pending_bbox", None)
if pending_bbox is not None:
    for key in ["north", "south", "east", "west"]:
        st.session_state[f"lts_bbox_{key}"] = float(pending_bbox[key])
    st.session_state["lts_map_center"] = [
        float((pending_bbox["south"] + pending_bbox["north"]) / 2),
        float((pending_bbox["west"] + pending_bbox["east"]) / 2),
    ]

st.markdown(
    """
    <style>
      :root {
        --navy: #0b3552;
        --teal: #43b8a3;
        --amber: #d9901a;
        --ink: #14212b;
        --muted: #667986;
        --surface: #f7fbfd;
        --panel-border: rgba(11,53,82,0.12);
      }
      .stApp {
        background: linear-gradient(180deg, #eff4f7 0%, #e7eef3 100%);
        color: var(--ink);
      }
      .block-container {
        max-width: 1440px;
        padding-top: 3rem;
        padding-bottom: 2rem;
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
      .lts-hero {
        padding: 1.15rem 1.3rem;
        border-radius: 20px;
        background: linear-gradient(90deg, #0b3552 0%, #0d6c75 100%);
        color: white;
        box-shadow: 0 16px 36px rgba(7,39,61,0.16);
      }
      .lts-title {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin: 0;
        font-size: 1.55rem;
        font-weight: 800;
      }
      .lts-title img {
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 12px;
        padding: 0.22rem;
        background: rgba(255,255,255,0.12);
      }
      .lts-subtitle {
        margin: 0.75rem 0 0;
        color: rgba(255,255,255,0.86);
        max-width: 62rem;
      }
      .metric-card {
        padding: 1rem 1.05rem;
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        background: rgba(255,255,255,0.94);
      }
      .metric-label {
        margin: 0;
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .metric-value {
        margin: 0.35rem 0 0;
        color: var(--ink);
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


def save_uploaded_vector(uploaded_file) -> Path:
    return CityNetworkBuilder.write_uploaded_file(uploaded_file)


def read_uploaded_vector(uploaded_file) -> gpd.GeoDataFrame:
    path = save_uploaded_vector(uploaded_file)
    try:
        return CityNetworkBuilder().load_vector_layer(path)
    finally:
        path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def geocode_city_bbox(city_name: str) -> dict[str, Any]:
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("City lookup requires osmnx. Install project requirements and try again.") from exc

    city_gdf = ox.geocode_to_gdf(city_name)
    if city_gdf.empty:
        raise ValueError(f"Could not find a boundary for {city_name}.")
    bounds = city_gdf.to_crs("EPSG:4326").total_bounds
    west, south, east, north = [float(value) for value in bounds]
    return {
        "north": north,
        "south": south,
        "east": east,
        "west": west,
        "center": [float((south + north) / 2), float((west + east) / 2)],
    }


def selected_bbox() -> dict[str, float]:
    return {
        "north": float(st.session_state["lts_bbox_north"]),
        "south": float(st.session_state["lts_bbox_south"]),
        "east": float(st.session_state["lts_bbox_east"]),
        "west": float(st.session_state["lts_bbox_west"]),
    }


def set_selected_bbox(bbox: dict[str, float]) -> None:
    for key in ["north", "south", "east", "west"]:
        st.session_state[f"lts_bbox_{key}"] = float(bbox[key])
    st.session_state["lts_map_center"] = [
        float((bbox["south"] + bbox["north"]) / 2),
        float((bbox["west"] + bbox["east"]) / 2),
    ]


def queue_selected_bbox(bbox: dict[str, float]) -> None:
    st.session_state["lts_pending_bbox"] = {key: float(bbox[key]) for key in ["north", "south", "east", "west"]}


def bbox_from_drawing(drawing: dict[str, Any] | None) -> dict[str, float] | None:
    if not drawing:
        return None
    geometry = drawing.get("geometry")
    if not geometry:
        return None
    drawn_shape = shape(geometry)
    if drawn_shape.is_empty:
        return None
    west, south, east, north = drawn_shape.bounds
    if north <= south or east <= west:
        return None
    return {
        "north": float(north),
        "south": float(south),
        "east": float(east),
        "west": float(west),
    }


def format_bbox(bbox: dict[str, float]) -> str:
    return (
        f"N {bbox['north']:.5f}, S {bbox['south']:.5f}, "
        f"E {bbox['east']:.5f}, W {bbox['west']:.5f}"
    )


def render_area_selector(city_name: str) -> dict[str, float] | None:
    bbox = selected_bbox()
    center = st.session_state.get("lts_map_center", [45.52, -73.60])
    map_obj = folium.Map(location=center, zoom_start=int(st.session_state.get("lts_map_zoom", 11)), tiles="OpenStreetMap")
    folium.Rectangle(
        bounds=[(bbox["south"], bbox["west"]), (bbox["north"], bbox["east"])],
        color="#0b6b74",
        fill=True,
        fill_opacity=0.08,
        weight=2,
        tooltip="Current LTS calculation area",
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
    folium.LayerControl().add_to(map_obj)

    st.markdown("#### Select the calculation area")
    st.caption(f"Draw a rectangle or polygon for {city_name}, then click **Use drawn area**. The current selected area is `{format_bbox(bbox)}`.")
    map_data = st_folium(map_obj, height=520, use_container_width=True, returned_objects=["last_active_drawing"])
    return bbox_from_drawing(map_data.get("last_active_drawing") if map_data else None)


def build_lts_network(
    bbox_values: dict[str, float],
    use_osm: bool,
    base_network_upload,
    municipal_uploads: list[Any],
    include_inaccessible: bool,
    elevation_source: str,
    dem_upload,
    steepness_threshold: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    builder = CityNetworkBuilder()
    extractor = LTSFeatureExtractor()
    calculator = LTSCalculator()

    municipal_layers = [read_uploaded_vector(upload) for upload in municipal_uploads if upload is not None]
    base_network = read_uploaded_vector(base_network_upload) if base_network_upload is not None else None

    if use_osm:
        bbox = BoundingBox(
            north=float(bbox_values["north"]),
            south=float(bbox_values["south"]),
            east=float(bbox_values["east"]),
            west=float(bbox_values["west"]),
        )
        raw_edges = builder.build_network(bbox=bbox, municipal_layers=municipal_layers)
    elif base_network is not None:
        raw_edges = builder.build_network(base_network=base_network, municipal_layers=municipal_layers)
    else:
        raise ValueError("Fetch from OSM or upload a base network before calculating LTS.")

    normalized = extractor.normalize(raw_edges)
    scored = calculator.score_dataframe(normalized)
    if elevation_source == "Upload DEM" and dem_upload is not None:
        dem_path = save_uploaded_vector(dem_upload)
        try:
            scored = sample_dem_endpoints(scored, str(dem_path))
        finally:
            dem_path.unlink(missing_ok=True)
    elif elevation_source == "Online elevation":
        scored = fetch_online_endpoint_elevations(scored)
    scored = EffortCalculator().score_dataframe(scored)
    exportable = prepare_optimizer_export(scored, include_inaccessible=include_inaccessible)
    exportable.attrs["steepness_threshold"] = steepness_threshold
    return scored, exportable


def make_lts_map(gdf: gpd.GeoDataFrame, mode: str = "Traffic Stress") -> go.Figure:
    colors = {
        1: "#2f9e44",
        2: "#74b816",
        3: "#f08c00",
        4: "#c92a2a",
    }
    fig = go.Figure()
    if gdf.empty:
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
        return fig

    display = gdf.to_crs("EPSG:4326")
    field = "lts"
    levels = [1, 2, 3, 4]
    labels = {level: f"LTS {level}" for level in levels}
    if mode == "Cyclist Effort":
        field = "steepness_level"
        levels = [3.5, 5.0, 6.5, 8.0, 9.5]
        colors = {3.5: "#2f9e44", 5.0: "#74b816", 6.5: "#f08c00", 8.0: "#e8590c", 9.5: "#c92a2a"}
        labels = {level: f"SL {level:.1f}" for level in levels}
    elif mode == "Combined Comfort":
        display = display.copy()
        sl_to_band = {3.5: 1, 5.0: 2, 6.5: 3, 8.0: 4, 9.5: 4}
        display["combined_comfort"] = pd.concat(
            [
                pd.to_numeric(display["lts"], errors="coerce"),
                pd.to_numeric(display["steepness_level"], errors="coerce").map(sl_to_band),
            ],
            axis=1,
        ).max(axis=1)
        field = "combined_comfort"
        labels = {level: f"Comfort {level}" for level in levels}
    for level in levels:
        subset = display[pd.to_numeric(display[field], errors="coerce").round(1) == level]
        if subset.empty:
            continue
        lon: list[float | None] = []
        lat: list[float | None] = []
        hover: list[str | None] = []
        for _, row in subset.iterrows():
            geom = row.geometry
            parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
            for part in parts:
                coords = list(part.coords)
                lon.extend([coord[0] for coord in coords] + [None])
                lat.extend([coord[1] for coord in coords] + [None])
                hover.extend(
                    [
                        (
                            f"Segment: {row.get('segment_id', '')}<br>"
                            f"LTS: {row.get('lts', '')}<br>"
                            f"Steepness Level: {row.get('steepness_level', 'n/a')}<br>"
                            f"Uphill grade: {float(row.get('uphill_grade_pct', 0.0)):.1f}%<br>"
                            f"Confidence: {float(row.get('confidence', 0.0)):.2f}<br>"
                            f"Missing: {row.get('missing_inputs', '') or 'none'}"
                        )
                    ]
                    * len(coords)
                    + [None]
                )
        fig.add_trace(
            go.Scattermapbox(
                lon=lon,
                lat=lat,
                mode="lines",
                line=dict(width=4, color=colors[level]),
                name=labels[level],
                text=hover,
                hoverinfo="text",
            )
        )

    bounds = display.total_bounds
    center = {"lat": float((bounds[1] + bounds[3]) / 2), "lon": float((bounds[0] + bounds[2]) / 2)}
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=center, zoom=11),
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


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


st.markdown(
    f"""
    <div class="lts-hero">
      <h1 class="lts-title"><img src="{BADGE_URI}" alt="">LTS Network Builder</h1>
      <p class="lts-subtitle">Generate a bicycle Level of Traffic Stress network from OSM streets and optional municipal overrides, then save it for optimization and route testing.</p>
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
            st.session_state["lts_map_zoom"] = 11
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.caption("Use the map as the main selector. Coordinates remain available as an advanced fallback.")
    col_a, col_b = st.columns(2)
    with col_a:
        north = st.number_input("North", format="%.6f", key="lts_bbox_north")
        west = st.number_input("West", format="%.6f", key="lts_bbox_west")
    with col_b:
        south = st.number_input("South", format="%.6f", key="lts_bbox_south")
        east = st.number_input("East", format="%.6f", key="lts_bbox_east")

    st.markdown("### Data")
    use_osm = st.checkbox("Fetch OSM within bounding box", value=True)
    base_network_upload = st.file_uploader(
        "Fallback base network",
        type=["gpkg", "geojson", "json", "shp", "parquet", "zip"],
        help="Used when OSM fetching is off or unavailable. Zip shapefiles are supported.",
    )
    municipal_uploads = st.file_uploader(
        "Municipal override layers",
        type=["gpkg", "geojson", "json", "shp", "parquet", "zip"],
        accept_multiple_files=True,
        help="Optional road/cycle layers. Nearest municipal values override OSM attributes when fields are recognized.",
    )
    st.markdown("### Elevation")
    elevation_source = st.radio("Elevation source", ["Online elevation", "Upload DEM", "None"], index=0)
    dem_upload = st.file_uploader("DEM raster", type=["tif", "tiff"], disabled=elevation_source != "Upload DEM")
    steepness_threshold = st.select_slider("Steepness threshold", options=[3.5, 5.0, 6.5, 8.0, 9.5], value=5.0)

    st.markdown("### Export")
    export_format = st.selectbox("Format", ["GeoPackage", "GeoJSON"])
    include_inaccessible = st.checkbox("Include bike-prohibited/freeway segments", value=False)
    calculate_button = st.button("Calculate LTS Network", type="primary", use_container_width=True)

drawn_bbox = render_area_selector(city_name)
if drawn_bbox is not None:
    st.info(f"Drawn area detected: {format_bbox(drawn_bbox)}")
    if st.button("Use drawn area", type="primary"):
        queue_selected_bbox(drawn_bbox)
        st.rerun()

active_bbox = selected_bbox()

if active_bbox["north"] <= active_bbox["south"] or active_bbox["east"] <= active_bbox["west"]:
    st.error("The bounding box is invalid. North must exceed south, and east must exceed west.")
    st.stop()

if calculate_button:
    with st.spinner("Building and scoring the LTS network..."):
        try:
            scored_gdf, export_gdf = build_lts_network(
                bbox_values=active_bbox,
                use_osm=use_osm,
                base_network_upload=base_network_upload,
                municipal_uploads=municipal_uploads or [],
                include_inaccessible=include_inaccessible,
                elevation_source=elevation_source,
                dem_upload=dem_upload,
                steepness_threshold=steepness_threshold,
            )
            st.session_state["lts_scored_gdf"] = scored_gdf
            st.session_state["lts_export_gdf"] = export_gdf
            st.session_state["lts_city_name"] = city_name
            st.success("LTS network calculated.")
        except Exception as exc:
            st.error(str(exc))

scored = st.session_state.get("lts_scored_gdf")
exportable = st.session_state.get("lts_export_gdf")

if scored is None or exportable is None:
    st.markdown(
        """
        <div class="note-panel">
          Start with the default Montreal bounding box, fetch OSM, and calculate a first network. Add municipal road or cycle layers when you want higher-confidence local attributes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

metric_cols = st.columns(5)
with metric_cols[0]:
    render_metric("Scored Segments", f"{len(scored):,}")
with metric_cols[1]:
    render_metric("Optimizer Export", f"{len(exportable):,}")
with metric_cols[2]:
    render_metric("Mean LTS", f"{exportable['lts'].mean():.2f}" if not exportable.empty else "n/a")
with metric_cols[3]:
    render_metric("Mean Confidence", f"{scored['confidence'].mean():.2f}" if not scored.empty else "n/a")
with metric_cols[4]:
    render_metric("SL Exceedance", f"{(exportable['steepness_level'] > steepness_threshold).mean() * 100:.1f}%" if exportable["steepness_level"].notna().any() else "n/a")

tab_map, tab_quality, tab_data, tab_export = st.tabs(["Map", "Quality", "Data", "Export"])

with tab_map:
    map_mode = st.radio("Map mode", ["Traffic Stress", "Cyclist Effort", "Combined Comfort"], horizontal=True)
    st.plotly_chart(make_lts_map(exportable, mode=map_mode), use_container_width=True)

with tab_quality:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        counts = exportable["lts"].round().astype(int).value_counts().sort_index()
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[f"LTS {idx}" for idx in counts.index],
                    y=counts.values,
                    marker_color=["#2f9e44", "#74b816", "#f08c00", "#c92a2a"][: len(counts)],
                )
            ]
        )
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Segments")
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        missing = (
            scored["missing_inputs"]
            .fillna("")
            .str.split(";")
            .explode()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .reset_index()
        )
        missing.columns = ["Missing Input", "Segments"] if not missing.empty else ["Missing Input", "Segments"]
        if missing.empty:
            st.success("No fallback inputs were recorded.")
        else:
            st.dataframe(missing, use_container_width=True, hide_index=True)
    if exportable["steepness_level"].notna().any():
        st.markdown("#### Cyclist effort distribution")
        effort_counts = exportable["steepness_level"].value_counts().sort_index().rename_axis("Steepness Level").reset_index(name="Segments")
        st.dataframe(effort_counts, use_container_width=True, hide_index=True)
        if effort_counts["Steepness Level"].nunique() == 1:
            st.warning("All scored segments share one Steepness Level. Check elevation quality, segment lengths, and the selected source before relying on effort outputs.")

with tab_data:
    preview_columns = [
        "segment_id",
        "road_class",
        "bike_facility_type",
        "lts",
        "lts_winter",
        "lts_contraflow",
        "confidence",
        "steepness_level",
        "uphill_grade_pct",
        "effort_exposure",
        "missing_inputs",
    ]
    st.dataframe(scored[[column for column in preview_columns if column in scored.columns]].head(200), use_container_width=True)

with tab_export:
    payload, filename, mime = export_network_bytes(exportable, export_format)
    artifact_label = st.text_input(
        "Saved input name",
        value=f"{city_name} LTS network",
        help="This name appears in the planner and Route Planner saved/generated LTS network lists.",
    )
    st.download_button(
        "Download optimizer-ready LTS network",
        data=payload,
        file_name=filename,
        mime=mime,
        type="primary",
        use_container_width=True,
    )
    save_cols = st.columns(2)
    if save_cols[0].button("Save and open planner", use_container_width=True):
        artifact = save_lts_artifact(
            artifact_label,
            exportable,
            {
                "city_name": city_name,
                "bbox": active_bbox,
                "format": "GeoJSON",
            },
        )
        st.session_state["selected_lts_artifact_id"] = artifact["artifact_id"]
        st.success(f"Saved {artifact['label']}.")
        st.switch_page("Bike_Network_Planner.py")
    if save_cols[1].button("Save and open Route Planner", use_container_width=True):
        artifact = save_lts_artifact(
            artifact_label,
            exportable,
            {
                "city_name": city_name,
                "bbox": active_bbox,
                "format": "GeoJSON",
            },
        )
        st.session_state["selected_lts_artifact_id"] = artifact["artifact_id"]
        st.success(f"Saved {artifact['label']}.")
        st.switch_page("pages/Route_Planner.py")
    st.caption("Saved LTS networks can be selected directly in the planner and Route Planner without downloading and re-uploading.")
