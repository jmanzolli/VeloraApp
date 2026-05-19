from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from ui.routing import RouteEndpoint, RoutePlanner, RouteResult
from ui.storage import list_lts_artifacts


ASSETS_DIR = Path(__file__).resolve().parents[1] / "ui" / "assets"
BADGE_PATH = ASSETS_DIR / "velora_badge.png"
ROUTE_COLORS = {
    "Shortest Distance": "#2563eb",
    "Lowest LTS": "#16a34a",
    "Lowest Effort": "#7c3aed",
    "Balanced Comfort": "#d97706",
}
MAP_STYLES = {
    "Detailed streets": "open-street-map",
    "Google-like light": "carto-positron",
    "Dark contrast": "carto-darkmatter",
}
MONTREAL_CENTER = (45.5089, -73.5617)
LANDMARK_ALIASES = {
    "mcgill": "McGill University",
    "jean talon": "Jean-Talon Market",
    "jean-talon": "Jean-Talon Market",
    "old port": "Old Port of Montreal",
    "vieux port": "Old Port of Montreal",
    "berri uqam": "Berri-UQAM station",
    "berri-uqam": "Berri-UQAM station",
    "concordia": "Concordia University Sir George Williams Campus",
    "udem": "Universite de Montreal",
    "plateau": "Le Plateau-Mont-Royal",
    "mile end": "Mile End Montreal",
    "downtown": "Downtown Montreal",
    "atwater": "Atwater Market",
    "olympic stadium": "Olympic Stadium Montreal",
    "parc lafontaine": "Parc La Fontaine",
    "mount royal": "Mount Royal Park",
    "mont royal": "Mount Royal Park",
}

st.session_state.setdefault("montreal_origin_coords", None)
st.session_state.setdefault("montreal_destination_coords", None)
st.session_state.setdefault("montreal_pick_target", "Origin")


def load_asset_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


st.set_page_config(
    page_title="Bike Network Planner | Montreal Route Finder",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=str(BADGE_PATH),
)

BADGE_URI = load_asset_data_uri(BADGE_PATH)

st.markdown(
    """
    <style>
      .stApp { background: #eef4f7; }
      .block-container { max-width: 1480px; padding-top: 2rem; }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a2b44 0%, #092137 100%); }
      [data-testid="stSidebar"] * { color: #eef6fb; }
      [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #173042 !important;
        -webkit-text-fill-color: #173042 !important;
      }
      .finder-hero {
        color: white;
        background: linear-gradient(90deg, #0b3552 0%, #0d6c75 100%);
        border-radius: 18px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
      }
      .finder-title {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 800;
      }
      .finder-title img {
        width: 2.1rem;
        height: 2.1rem;
        border-radius: 10px;
        padding: 0.18rem;
        background: rgba(255,255,255,0.14);
      }
      .finder-subtitle { margin: 0.55rem 0 0; color: rgba(255,255,255,0.86); }
      .route-card {
        min-height: 146px;
        border: 1px solid rgba(11,53,82,0.14);
        background: rgba(255,255,255,0.96);
        border-radius: 8px;
        padding: 0.9rem;
      }
      .route-card.selected { border: 2px solid #0d6c75; box-shadow: 0 8px 20px rgba(11,53,82,0.12); }
      .route-name { font-weight: 800; color: #14212b; margin-bottom: 0.45rem; }
      .route-meta { color: #526674; font-size: 0.9rem; line-height: 1.45; }
      .index-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.9rem 0;
      }
      .index-cell {
        border-radius: 8px;
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(11,53,82,0.12);
        padding: 0.8rem;
      }
      .index-label { color: #667986; font-size: 0.75rem; text-transform: uppercase; }
      .index-value { color: #14212b; font-size: 1.2rem; font-weight: 800; margin-top: 0.2rem; }
      .picker-panel {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(11,53,82,0.12);
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 1rem;
      }
      .picker-status {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        color: #173042;
        font-size: 0.9rem;
        margin: 0.4rem 0 0.75rem;
      }
      .picker-pill {
        background: #eef6fb;
        border: 1px solid rgba(11,53,82,0.12);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
      }
      @media (max-width: 900px) {
        .index-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def montreal_network_path() -> Path:
    artifacts = [
        artifact
        for artifact in list_lts_artifacts()
        if str(artifact.get("metadata", {}).get("city_name", "")).lower() == "montreal"
    ]
    if not artifacts:
        raise FileNotFoundError("No saved Montreal network is available yet.")
    preferred = next((artifact for artifact in artifacts if artifact.get("artifact_id") == "lts-20260515-122729"), artifacts[0])
    path = Path(str(preferred["file_path"]))
    if not path.exists():
        raise FileNotFoundError("The saved Montreal network file could not be found.")
    return path


@st.cache_resource(show_spinner=False)
def load_montreal_graph() -> tuple[RoutePlanner, object, object]:
    planner = RoutePlanner()
    network = planner.load_lts_network(montreal_network_path())
    graph = planner.build_graph(network)
    return planner, network, graph


def calculate_routes(
    origin_text: str,
    destination_text: str,
    stress_weight: int,
    effort_weight: int,
    origin_coords: tuple[float, float] | None = None,
    destination_coords: tuple[float, float] | None = None,
) -> dict[str, object]:
    planner, network, graph = load_montreal_graph()
    origin = (
        endpoint_from_coords(planner, "Origin", origin_coords, graph)
        if origin_coords
        else planner.geocode_endpoint("Origin", resolve_place_text(origin_text), "Montreal, Quebec, Canada", graph)
    )
    destination = (
        endpoint_from_coords(planner, "Destination", destination_coords, graph)
        if destination_coords
        else planner.geocode_endpoint("Destination", resolve_place_text(destination_text), "Montreal, Quebec, Canada", graph)
    )
    routes = planner.calculate_routes(
        graph,
        origin.node,
        destination.node,
        balanced_stress_weight=stress_weight / 100.0,
        balanced_effort_weight=effort_weight / 100.0,
        steepness_threshold=5.0,
    )
    return {"planner": planner, "network": network, "graph": graph, "origin": origin, "destination": destination, "routes": routes}


def resolve_place_text(text: str) -> str:
    normalized = " ".join(text.lower().replace(",", " ").split())
    return LANDMARK_ALIASES.get(normalized, text.strip())


def endpoint_from_coords(
    planner: RoutePlanner,
    label: str,
    coords: tuple[float, float],
    graph: object,
) -> RouteEndpoint:
    latitude, longitude = coords
    return RouteEndpoint(
        label=label,
        query=f"{latitude:.5f}, {longitude:.5f}",
        latitude=float(latitude),
        longitude=float(longitude),
        node=planner.snap_lonlat_to_graph(longitude=float(longitude), latitude=float(latitude), graph=graph),
    )


def make_picker_map(map_style: str) -> dict:
    tile = MAP_STYLES.get(map_style, "open-street-map")
    tiles = "OpenStreetMap" if tile == "open-street-map" else None
    map_obj = folium.Map(location=MONTREAL_CENTER, zoom_start=12, tiles=tiles)
    if tile == "carto-positron":
        folium.TileLayer("CartoDB positron", name="Google-like light").add_to(map_obj)
    elif tile == "carto-darkmatter":
        folium.TileLayer("CartoDB dark_matter", name="Dark contrast").add_to(map_obj)

    origin = st.session_state.get("montreal_origin_coords")
    destination = st.session_state.get("montreal_destination_coords")
    if origin:
        folium.Marker(
            location=origin,
            tooltip="Origin",
            icon=folium.Icon(color="darkblue", icon="play", prefix="fa"),
        ).add_to(map_obj)
    if destination:
        folium.Marker(
            location=destination,
            tooltip="Destination",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(map_obj)
    if origin and destination:
        folium.PolyLine([origin, destination], color="#0d6c75", weight=2, opacity=0.55, dash_array="6,8").add_to(map_obj)
    return st_folium(map_obj, height=430, use_container_width=True, returned_objects=["last_clicked"])


def route_by_label(routes: list[RouteResult], label: str) -> RouteResult | None:
    return next((route for route in routes if label in route.labels), None)


def selected_route_label(routes: list[RouteResult]) -> str:
    available = [label for label in ["Balanced Comfort", "Shortest Distance", "Lowest LTS", "Lowest Effort"] if route_by_label(routes, label)]
    return st.session_state.get("montreal_selected_route", available[0])


def make_map(
    routes: list[RouteResult],
    graph_crs: object,
    origin: RouteEndpoint,
    destination: RouteEndpoint,
    selected_label: str,
    map_style: str,
) -> go.Figure:
    fig = go.Figure()
    routes_gdf = RoutePlanner.routes_to_geodataframe(routes, graph_crs)
    for route, (_, row) in zip(routes, routes_gdf.iterrows()):
        label = " / ".join(route.labels)
        primary_label = route.labels[0]
        coords = list(row.geometry.coords)
        is_selected = selected_label in route.labels
        fig.add_trace(
            go.Scattermapbox(
                lon=[coord[0] for coord in coords],
                lat=[coord[1] for coord in coords],
                mode="lines",
                line=dict(width=8 if is_selected else 5, color=ROUTE_COLORS.get(primary_label, "#7c3aed")),
                opacity=1.0 if is_selected else 0.4,
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Distance: {route.distance_m / 1000.0:.2f} km<br>"
                    f"Time: {route.estimated_minutes:.0f} min<br>"
                    f"Mean LTS: {route.mean_lts:.2f}<br>"
                    f"Elevation gain: {route.elevation_gain_m:.0f} m<br>"
                    + (f"Mean effort: {route.mean_effort:.2f}<br>" if route.mean_effort is not None else "")
                    + "<extra></extra>"
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
    routes_4326 = RoutePlanner.routes_to_geodataframe(routes, graph_crs)
    bounds = routes_4326.total_bounds
    fig.update_layout(
        mapbox=dict(
            style=MAP_STYLES.get(map_style, "open-street-map"),
            center={"lat": float((bounds[1] + bounds[3]) / 2), "lon": float((bounds[0] + bounds[2]) / 2)},
            zoom=13,
        ),
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


st.markdown(
    f"""
    <div class="finder-hero">
      <h1 class="finder-title"><img src="{BADGE_URI}" alt="">Montreal Route Finder</h1>
      <p class="finder-subtitle">Compare cycling routes using distance, traffic stress, cyclist effort, and combined comfort.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Find a route")
    input_mode = st.radio("Choose points by", ["Typing", "Map click"], horizontal=True)
    origin_text = st.text_input("Origin", value="McGill University", disabled=input_mode == "Map click")
    destination_text = st.text_input("Destination", value="Jean-Talon Market", disabled=input_mode == "Map click")
    if input_mode == "Map click":
        st.session_state["montreal_pick_target"] = st.radio("Next map click sets", ["Origin", "Destination"], horizontal=True)
        clear_cols = st.columns(2)
        if clear_cols[0].button("Clear A", use_container_width=True):
            st.session_state["montreal_origin_coords"] = None
        if clear_cols[1].button("Clear B", use_container_width=True):
            st.session_state["montreal_destination_coords"] = None
    with st.expander("Customize route priorities", expanded=True):
        stress_weight = st.slider("Traffic stress weight", 0, 100, 35, 5)
        effort_weight = st.slider("Cyclist effort weight", 0, 100 - stress_weight, 35, 5)
        distance_weight = 100 - stress_weight - effort_weight
        st.caption(f"Distance {distance_weight}% | Traffic stress {stress_weight}% | Effort {effort_weight}%")
    map_style = st.selectbox(
        "Map style",
        list(MAP_STYLES.keys()),
        index=0,
        help="Detailed streets keeps a Google-Maps-like city context. Light and dark styles are optional viewing modes.",
    )
    find_button = st.button("Find routes", type="primary", use_container_width=True)

if input_mode == "Map click":
    origin = st.session_state.get("montreal_origin_coords")
    destination = st.session_state.get("montreal_destination_coords")
    st.markdown(
        f"""
        <div class="picker-panel">
          <strong>Pick locations on the Montreal map</strong>
          <div class="picker-status">
            <span class="picker-pill">A: {f"{origin[0]:.4f}, {origin[1]:.4f}" if origin else "not set"}</span>
            <span class="picker-pill">B: {f"{destination[0]:.4f}, {destination[1]:.4f}" if destination else "not set"}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    picker_data = make_picker_map(map_style)
    last_clicked = picker_data.get("last_clicked") if picker_data else None
    if last_clicked:
        clicked = (round(float(last_clicked["lat"]), 6), round(float(last_clicked["lng"]), 6))
        key = "montreal_origin_coords" if st.session_state["montreal_pick_target"] == "Origin" else "montreal_destination_coords"
        if st.session_state.get(key) != clicked:
            st.session_state[key] = clicked
            if st.session_state["montreal_pick_target"] == "Origin":
                st.session_state["montreal_pick_target"] = "Destination"
            st.rerun()

if find_button:
    origin_coords = st.session_state.get("montreal_origin_coords") if input_mode == "Map click" else None
    destination_coords = st.session_state.get("montreal_destination_coords") if input_mode == "Map click" else None
    if input_mode == "Map click" and (origin_coords is None or destination_coords is None):
        st.error("Click the map to set both A and B before finding routes.")
    elif input_mode == "Typing" and (not origin_text.strip() or not destination_text.strip()):
        st.error("Enter both an origin and a destination.")
    else:
        try:
            with st.spinner("Finding Montreal cycling routes..."):
                st.session_state["montreal_route_result"] = calculate_routes(
                    origin_text,
                    destination_text,
                    stress_weight,
                    effort_weight,
                    origin_coords=origin_coords,
                    destination_coords=destination_coords,
                )
                st.session_state["montreal_selected_route"] = "Balanced Comfort"
        except Exception as exc:
            st.error(str(exc))

result = st.session_state.get("montreal_route_result")
if result is None:
    st.info("Enter two Montreal locations to compare routes by distance, traffic stress, cyclist effort, and combined comfort.")
    st.stop()

routes = result["routes"]
selected_label = selected_route_label(routes)
selected_route = route_by_label(routes, selected_label) or routes[0]

route_labels = [label for label in ["Shortest Distance", "Lowest LTS", "Lowest Effort", "Balanced Comfort"] if route_by_label(routes, label)]
cols = st.columns(len(route_labels))
for col, label in zip(cols, route_labels):
    route = route_by_label(routes, label)
    assert route is not None
    if col.button(label, use_container_width=True):
        st.session_state["montreal_selected_route"] = label
        st.rerun()
    selected_class = " selected" if label == selected_label else ""
    col.markdown(
        f"""
        <div class="route-card{selected_class}">
          <div class="route-name">{label}</div>
          <div class="route-meta">
            {route.estimated_minutes:.0f} min · {route.distance_m / 1000.0:.2f} km<br>
            Mean LTS {route.mean_lts:.2f}<br>
            Elevation gain {route.elevation_gain_m:.0f} m<br>
            Mean effort {route.mean_effort:.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="index-strip">
      <div class="index-cell"><div class="index-label">Selected route</div><div class="index-value">{selected_label}</div></div>
      <div class="index-cell"><div class="index-label">Traffic stress</div><div class="index-value">LTS {selected_route.mean_lts:.2f}</div></div>
      <div class="index-cell"><div class="index-label">Cyclist effort</div><div class="index-value">{selected_route.mean_effort:.2f}</div></div>
      <div class="index-cell"><div class="index-label">Elevation gain</div><div class="index-value">{selected_route.elevation_gain_m:.0f} m</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

map_col, detail_col = st.columns([1.7, 0.9])
with map_col:
    st.plotly_chart(
        make_map(
            routes,
            result["graph"].graph["crs"],
            result["origin"],
            result["destination"],
            selected_label,
            map_style,
        ),
        use_container_width=True,
    )
with detail_col:
    st.markdown("### Route indices")
    summary = pd.DataFrame([route.summary_row() for route in routes])
    st.dataframe(
        summary[
            [
                "Route",
                "Distance (km)",
                "Estimated Time (min)",
                "Mean LTS",
                "Mean Effort",
                "Elevation Gain (m)",
                "Max Steepness Level",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("The route colors distinguish the different route objectives. Lower LTS means lower traffic stress; lower effort favors easier riding.")
