from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.routing import RouteEndpoint, RoutePlanner, RouteResult


ASSETS_DIR = Path(__file__).resolve().parents[1] / "ui" / "assets"
BADGE_PATH = ASSETS_DIR / "velora_badge.png"


def load_asset_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


st.set_page_config(
    page_title="Velora | Route Planner",
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


def load_uploaded_network(uploaded_file):
    planner = RoutePlanner()
    path = planner.write_uploaded_file(uploaded_file)
    try:
        return planner.load_lts_network(path)
    finally:
        path.unlink(missing_ok=True)


def calculate_route_set(
    uploaded_file,
    city_context: str,
    origin_text: str,
    destination_text: str,
    balanced_weight_pct: int,
) -> dict[str, object]:
    planner = RoutePlanner()
    network = load_uploaded_network(uploaded_file)
    graph = planner.build_graph(network)
    origin = planner.geocode_endpoint("Point A", origin_text, city_context, graph)
    destination = planner.geocode_endpoint("Point B", destination_text, city_context, graph)
    routes = planner.calculate_routes(
        graph,
        origin.node,
        destination.node,
        balanced_stress_weight=float(balanced_weight_pct) / 100.0,
    )
    return {
        "network": network,
        "graph": graph,
        "origin": origin,
        "destination": destination,
        "routes": routes,
        "geojson": planner.export_routes_geojson(routes, graph),
    }


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


def route_summary_dataframe(routes: list[RouteResult]) -> pd.DataFrame:
    rows = [route.summary_row() for route in routes]
    return pd.DataFrame(rows)


def make_route_map(routes: list[RouteResult], graph_crs: object, origin: RouteEndpoint, destination: RouteEndpoint) -> go.Figure:
    colors = {
        "Shortest Distance": "#2563eb",
        "Lowest LTS": "#16a34a",
        "Balanced": "#d97706",
    }
    fig = go.Figure()
    routes_gdf = RoutePlanner.routes_to_geodataframe(routes, graph_crs)

    for route, (_, row) in zip(routes, routes_gdf.iterrows()):
        label = " / ".join(route.labels)
        primary_label = route.labels[0]
        coords = list(row.geometry.coords)
        fig.add_trace(
            go.Scattermapbox(
                lon=[coord[0] for coord in coords],
                lat=[coord[1] for coord in coords],
                mode="lines",
                line=dict(width=6, color=colors.get(primary_label, "#7c3aed")),
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Distance: {route.distance_m / 1000.0:.2f} km<br>"
                    f"Mean LTS: {route.mean_lts:.2f}<br>"
                    f"Max LTS: {route.max_lts:.0f}<br>"
                    f"Time: {route.estimated_minutes:.0f} min"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lon=[origin.longitude, destination.longitude],
            lat=[origin.latitude, destination.latitude],
            mode="markers+text",
            marker=dict(size=14, color=["#0b3552", "#b54b32"]),
            text=["A", "B"],
            textposition="top center",
            hovertext=[origin.query, destination.query],
            hoverinfo="text",
            name="A / B",
        )
    )

    all_lats = [origin.latitude, destination.latitude]
    all_lons = [origin.longitude, destination.longitude]
    for _, row in routes_gdf.iterrows():
        coords = list(row.geometry.coords)
        all_lons.extend([coord[0] for coord in coords])
        all_lats.extend([coord[1] for coord in coords])

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={"lat": float(sum(all_lats) / len(all_lats)), "lon": float(sum(all_lons) / len(all_lons))},
            zoom=12,
        ),
        height=590,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


st.markdown(
    f"""
    <div class="velora-hero">
      <h1 class="velora-title"><img src="{BADGE_URI}" alt="">Route Planner</h1>
      <p class="velora-subtitle">Compare point-to-point cycling routes by shortest distance, lowest traffic stress, and a tunable balanced option.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Network")
    network_upload = st.file_uploader(
        "LTS network",
        type=["gpkg", "geojson", "json", "shp", "parquet"],
        help="Upload an LTS-scored network with geometry and lts. The LTS Calculator page can export this file.",
    )
    st.markdown("### Route")
    city_context = st.text_input("City context", value="Montreal, Canada")
    origin_text = st.text_input("Point A", value="McGill University")
    destination_text = st.text_input("Point B", value="Jean-Talon Market")
    balanced_weight_pct = st.slider(
        "Balanced stress weight",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="0 follows distance, 100 follows lowest LTS. 50 is an even distance/stress balance.",
    )
    calculate_button = st.button("Calculate Routes", type="primary", use_container_width=True)

if calculate_button:
    if network_upload is None:
        st.error("Upload an LTS network first.")
    elif not origin_text.strip() or not destination_text.strip():
        st.error("Enter both point A and point B.")
    else:
        try:
            with st.spinner("Calculating route alternatives..."):
                st.session_state["route_planner_result"] = calculate_route_set(
                    network_upload,
                    city_context=city_context,
                    origin_text=origin_text,
                    destination_text=destination_text,
                    balanced_weight_pct=balanced_weight_pct,
                )
            st.success("Routes calculated.")
        except Exception as exc:
            st.error(str(exc))

result = st.session_state.get("route_planner_result")
if result is None:
    st.markdown(
        """
        <div class="note-panel">
          Upload the LTS network created by the LTS Calculator, enter two street or place names, and calculate routes. Velora will geocode both locations, snap them to the network, and compare distance and stress alternatives.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

routes = result["routes"]
origin = result["origin"]
destination = result["destination"]
graph = result["graph"]
summary_df = route_summary_dataframe(routes)

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric("Routes", f"{len(routes)}")
with metric_cols[1]:
    render_metric("Best Distance", f"{summary_df['Distance (km)'].min():.2f} km")
with metric_cols[2]:
    render_metric("Best Mean LTS", f"{summary_df['Mean LTS'].min():.2f}")
with metric_cols[3]:
    render_metric("Fastest Time", f"{summary_df['Estimated Time (min)'].min():.0f} min")

tab_map, tab_summary, tab_export = st.tabs(["Map", "Summary", "Export"])

with tab_map:
    st.plotly_chart(make_route_map(routes, graph.graph["crs"], origin, destination), use_container_width=True)

with tab_summary:
    display_df = summary_df.copy()
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Distance (km)": st.column_config.NumberColumn(format="%.2f"),
            "Mean LTS": st.column_config.NumberColumn(format="%.2f"),
            "Max LTS": st.column_config.NumberColumn(format="%.0f"),
            "Stress Exposure": st.column_config.NumberColumn(format="%.0f"),
            "Estimated Time (min)": st.column_config.NumberColumn(format="%.0f"),
            "LTS 1 Share": st.column_config.NumberColumn(format="%.0%%"),
            "LTS 2 Share": st.column_config.NumberColumn(format="%.0%%"),
            "LTS 3 Share": st.column_config.NumberColumn(format="%.0%%"),
            "LTS 4 Share": st.column_config.NumberColumn(format="%.0%%"),
        },
    )
    st.caption(f"Point A geocoded as `{origin.query}`. Point B geocoded as `{destination.query}`.")

with tab_export:
    st.download_button(
        "Download route GeoJSON",
        data=result["geojson"],
        file_name="velora_routes.geojson",
        mime="application/geo+json",
        type="primary",
        use_container_width=True,
    )
