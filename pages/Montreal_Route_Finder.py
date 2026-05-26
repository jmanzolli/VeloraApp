from __future__ import annotations

import base64
import difflib
import hashlib
import mimetypes
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
from streamlit_searchbox import st_searchbox

from ui.effort import EffortCalculator, fetch_online_endpoint_elevations
from ui.lts import CityNetworkBuilder, LTSCalculator, LTSFeatureExtractor
from ui.lts.builder import BoundingBox
from ui.lts.export import prepare_optimizer_export
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
MONTREAL_SEARCH_BOUNDS = (-74.05, 45.35, -73.35, 45.75)
GENERATED_NETWORK_DIR = Path(__file__).resolve().parents[1] / "app_data" / "artifacts" / "lts" / "montreal_route_finder"
MONTREAL_CONTEXT_TERMS = (
    "montreal",
    "montréal",
    "westmount",
    "mount royal",
    "mont royal",
    "cote saint luc",
    "côte saint luc",
    "hampstead",
    "dorval",
)
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
    "old montreal": "Old Montreal",
    "vieux montreal": "Old Montreal",
}
SUGGESTED_PLACES = [
    "McGill University",
    "Jean-Talon Market",
    "Berri-UQAM station",
    "Old Port of Montreal",
    "Mount Royal Park",
    "Parc La Fontaine",
    "Atwater Market",
    "Concordia University Sir George Williams Campus",
    "Universite de Montreal",
    "Le Plateau-Mont-Royal",
    "Mile End Montreal",
    "Downtown Montreal",
    "Olympic Stadium Montreal",
    "Place des Arts",
    "Quartier des Spectacles",
    "Jarry Park",
    "La Fontaine Park",
    "Lachine Canal",
    "Griffintown Montreal",
    "Saint-Laurent Boulevard Montreal",
    "Westmount Quebec",
    "Outremont Montreal",
    "Verdun Montreal",
    "Cote-des-Neiges Montreal",
    "Rosemont Montreal",
]

st.session_state.setdefault("montreal_origin_coords", None)
st.session_state.setdefault("montreal_destination_coords", None)
st.session_state.setdefault("montreal_pick_target", "Origin")
st.session_state.setdefault("montreal_origin_text", "McGill University")
st.session_state.setdefault("montreal_destination_text", "Jean-Talon Market")


def load_asset_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


st.set_page_config(
    page_title="Bike Network Planner | Montreal Route Finder",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon=str(BADGE_PATH),
)

BADGE_URI = load_asset_data_uri(BADGE_PATH)

st.markdown(
    """
    <style>
      .stApp { background: #e8eef2; }
      .block-container { max-width: 100%; padding: 0.75rem 1rem 1rem; }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a2b44 0%, #092137 100%); }
      [data-testid="stSidebar"] * { color: #eef6fb; }
      [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #173042 !important;
        -webkit-text-fill-color: #173042 !important;
      }
      .map-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.65rem;
      }
      .finder-brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        color: #102a3a;
        font-size: 1.15rem;
        font-weight: 800;
      }
      .finder-brand img {
        width: 2.1rem;
        height: 2.1rem;
        border-radius: 10px;
        padding: 0.18rem;
        background: white;
        box-shadow: 0 2px 10px rgba(15,35,52,0.12);
      }
      .mode-chips {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        justify-content: flex-end;
      }
      .mode-chip {
        border: 1px solid rgba(15,35,52,0.14);
        background: white;
        color: #172b3a;
        border-radius: 999px;
        padding: 0.45rem 0.75rem;
        font-size: 0.88rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(15,35,52,0.10);
      }
      .search-panel {
        background: linear-gradient(180deg, #ffffff 0%, #f7fbfd 100%);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 24px rgba(15,35,52,0.10);
        border: 1px solid rgba(11,53,82,0.12);
      }
      .panel-kicker {
        margin: 0;
        color: #0d6c75;
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
      }
      .search-panel .panel-title {
        margin: 0.2rem 0 0.1rem;
        color: #14212b;
        font-size: 1.28rem !important;
        line-height: 1.2 !important;
        font-weight: 850;
      }
      div[data-testid="stHorizontalBlock"]:has(.search-panel) > div[data-testid="column"]:first-child {
        background: rgba(247,251,253,0.96);
        border-radius: 16px;
        padding: 0.95rem;
        box-shadow: 0 8px 28px rgba(15,35,52,0.10);
        border: 1px solid rgba(11,53,82,0.10);
      }
      div[data-testid="stHorizontalBlock"]:has(.search-panel) > div[data-testid="column"]:first-child label,
      div[data-testid="stHorizontalBlock"]:has(.search-panel) > div[data-testid="column"]:first-child p,
      div[data-testid="stHorizontalBlock"]:has(.search-panel) > div[data-testid="column"]:first-child .stMarkdown,
      div[data-testid="stHorizontalBlock"]:has(.search-panel) > div[data-testid="column"]:first-child [data-testid="stWidgetLabel"] {
        color: #14212b !important;
      }
      div[data-testid="stHorizontalBlock"]:has(.search-panel) > div[data-testid="column"]:first-child input,
      div[data-testid="stHorizontalBlock"]:has(.search-panel) > div[data-testid="column"]:first-child [data-baseweb="select"] * {
        color: #173042 !important;
        -webkit-text-fill-color: #173042 !important;
      }
      .route-card {
        min-height: 104px;
        border: 1px solid rgba(11,53,82,0.14);
        background: rgba(255,255,255,0.96);
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.65rem;
      }
      .route-card.selected { border: 2px solid #0d6c75; box-shadow: 0 8px 20px rgba(11,53,82,0.13); }
      .route-name { font-weight: 850; color: #14212b; margin-bottom: 0.35rem; }
      .route-meta { color: #526674; font-size: 0.9rem; line-height: 1.45; }
      .index-strip {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.55rem;
        margin: 0.7rem 0;
      }
      .index-cell {
        border-radius: 8px;
        background: #f4f8fa;
        border: 1px solid rgba(11,53,82,0.12);
        padding: 0.65rem;
      }
      .index-label { color: #667986; font-size: 0.75rem; text-transform: uppercase; }
      .index-value { color: #14212b; font-size: 1.05rem; font-weight: 800; margin-top: 0.2rem; }
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
      .map-frame {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 28px rgba(15,35,52,0.18);
        border: 1px solid rgba(15,35,52,0.08);
        background: white;
      }
      .results-title {
        margin: 1rem 0 0.6rem;
        color: #172b3a;
        font-weight: 850;
        font-size: 1.12rem;
      }
      .suggest-panel {
        background: rgba(255,255,255,0.98);
        border: 1px solid rgba(11,53,82,0.12);
        border-radius: 10px;
        padding: 0.6rem;
        margin: -0.2rem 0 0.65rem;
        box-shadow: 0 10px 22px rgba(15,35,52,0.10);
      }
      .suggest-title {
        color: #526674;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
      }
      .map-click-hint {
        border-radius: 10px;
        border: 1px solid rgba(11,53,82,0.12);
        background: #ffffff;
        padding: 0.65rem 0.75rem;
        color: #526674;
        font-size: 0.88rem;
        margin: 0.25rem 0 0.9rem;
      }
      .display-route-label {
        margin: 0.15rem 0 0.35rem;
        color: #526674;
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
      }
      @media (max-width: 900px) {
        .map-topbar { align-items: flex-start; flex-direction: column; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def generated_network_bounds(
    origin_coords: tuple[float, float],
    destination_coords: tuple[float, float],
) -> BoundingBox:
    margin = 0.022
    west, south, east, north = MONTREAL_SEARCH_BOUNDS
    return BoundingBox(
        north=min(north, max(origin_coords[0], destination_coords[0]) + margin),
        south=max(south, min(origin_coords[0], destination_coords[0]) - margin),
        east=min(east, max(origin_coords[1], destination_coords[1]) + margin),
        west=max(west, min(origin_coords[1], destination_coords[1]) - margin),
    )


def generated_network_path(bounds: BoundingBox) -> Path:
    signature = f"{bounds.north:.3f}_{bounds.south:.3f}_{bounds.east:.3f}_{bounds.west:.3f}"
    digest = hashlib.sha1(signature.encode("ascii")).hexdigest()[:12]
    return GENERATED_NETWORK_DIR / f"montreal_{digest}.geojson"


def build_generated_network(origin_coords: tuple[float, float], destination_coords: tuple[float, float]) -> Path:
    bounds = generated_network_bounds(origin_coords, destination_coords)
    path = generated_network_path(bounds)
    if path.exists():
        existing = gpd.read_file(path)
        if "effort_exposure" in existing and existing["effort_exposure"].notna().any():
            return path
    else:
        raw_edges = CityNetworkBuilder().build_network(bbox=bounds)
        normalized = LTSFeatureExtractor().normalize(raw_edges)
        scored = LTSCalculator().score_dataframe(normalized)
        existing = prepare_optimizer_export(scored)
    if existing.empty:
        raise ValueError("No routable cycling streets were returned for these Montreal locations.")
    elevated = fetch_online_endpoint_elevations(existing)
    enriched = EffortCalculator().score_dataframe(elevated)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_file(path, driver="GeoJSON")
    return path


def montreal_network_path(
    origin_coords: tuple[float, float] | None = None,
    destination_coords: tuple[float, float] | None = None,
) -> Path:
    artifacts = [
        artifact
        for artifact in list_lts_artifacts()
        if str(artifact.get("metadata", {}).get("city_name", "")).lower() == "montreal"
    ]
    if artifacts:
        preferred = next((artifact for artifact in artifacts if artifact.get("artifact_id") == "lts-20260515-122729"), artifacts[0])
        path = Path(str(preferred["file_path"]))
        if path.exists():
            network = gpd.read_file(path)
            if "effort_exposure" in network and network["effort_exposure"].notna().any():
                return path
            if origin_coords is None or destination_coords is None:
                return path
    if origin_coords is not None and destination_coords is not None:
        return build_generated_network(origin_coords, destination_coords)
    raise FileNotFoundError("Select two suggested Montreal locations to prepare a route network.")


@st.cache_resource(show_spinner=False)
def load_montreal_graph(
    origin_coords: tuple[float, float] | None = None,
    destination_coords: tuple[float, float] | None = None,
) -> tuple[RoutePlanner, object, object]:
    planner = RoutePlanner()
    network = planner.load_lts_network(montreal_network_path(origin_coords, destination_coords))
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
    planner, network, graph = load_montreal_graph(origin_coords, destination_coords)
    connected_nodes = None
    if origin_coords is not None and destination_coords is not None:
        connected_nodes = planner.snap_connected_endpoint_nodes(
            origin_coords[1],
            origin_coords[0],
            destination_coords[1],
            destination_coords[0],
            graph,
        )
    origin = (
        endpoint_from_coords(planner, "Origin", origin_coords, graph, query=origin_text, node=connected_nodes[0] if connected_nodes else None)
        if origin_coords
        else planner.geocode_endpoint("Origin", resolve_place_text(origin_text), "Montreal, Quebec, Canada", graph)
    )
    destination = (
        endpoint_from_coords(planner, "Destination", destination_coords, graph, query=destination_text, node=connected_nodes[1] if connected_nodes else None)
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


def normalize_place_text(text: str) -> str:
    return " ".join(text.lower().replace(",", " ").replace("-", " ").split())


def matching_places(text: str) -> list[str]:
    normalized = normalize_place_text(text)
    searchable = {normalize_place_text(place): place for place in SUGGESTED_PLACES}
    searchable.update({normalize_place_text(alias): place for alias, place in LANDMARK_ALIASES.items()})
    if not normalized:
        return []
    suggestions: list[str] = []
    exact = searchable.get(normalized)
    if exact:
        suggestions.append(exact)
    for key, place in searchable.items():
        if normalized in key or any(token.startswith(normalized) for token in key.split()):
            suggestions.append(place)
    if suggestions:
        return list(dict.fromkeys(suggestions))
    fuzzy_keys = difflib.get_close_matches(normalized, searchable.keys(), n=5, cutoff=0.48)
    suggestions.extend(searchable[key] for key in fuzzy_keys)
    return list(dict.fromkeys(suggestions))


def inside_montreal_area(longitude: float, latitude: float, properties: dict[str, object]) -> bool:
    west, south, east, north = MONTREAL_SEARCH_BOUNDS
    if not (west <= longitude <= east and south <= latitude <= north):
        return False
    context = normalize_place_text(
        " ".join(
            str(properties.get(key) or "")
            for key in ("city", "county", "district", "state")
        )
    )
    return any(normalize_place_text(term) in context for term in MONTREAL_CONTEXT_TERMS)


def place_feature_label(properties: dict[str, object]) -> tuple[str, str] | None:
    name = str(properties.get("name") or "").strip()
    if not name:
        return None
    city = str(properties.get("city") or properties.get("county") or "").strip()
    district = str(properties.get("district") or "").strip()
    context = city if city else district
    display = f"{name}  |  {context}, QC" if context else f"{name}  |  Quebec"
    query_parts = [name]
    if context and normalize_place_text(context) != normalize_place_text(name):
        query_parts.append(context)
    query_parts.extend(["Quebec", "Canada"])
    return display, ", ".join(query_parts)


@st.cache_data(ttl=300, show_spinner=False)
def search_montreal_places(text: str) -> list[tuple[str, str, tuple[float, float]]]:
    cleaned = text.strip()
    if len(cleaned) < 2:
        return []
    queries = [cleaned]
    first_token = cleaned.split()[0]
    if len(first_token) >= 3 and normalize_place_text(first_token) != normalize_place_text(cleaned):
        queries.append(first_token)
    results: list[tuple[str, str, tuple[float, float]]] = []
    for query in queries:
        try:
            response = requests.get(
                "https://photon.komoot.io/api/",
                params={
                    "q": query,
                    "lat": MONTREAL_CENTER[0],
                    "lon": MONTREAL_CENTER[1],
                    "zoom": 11,
                    "location_bias_scale": 0.15,
                    "limit": 8,
                    "lang": "en",
                },
                headers={"User-Agent": "BikeNetworkPlanner/1.0"},
                timeout=8,
            )
            response.raise_for_status()
        except requests.RequestException:
            break
        for feature in response.json().get("features", []):
            coordinates = feature.get("geometry", {}).get("coordinates", [])
            properties = feature.get("properties", {})
            if len(coordinates) < 2 or not inside_montreal_area(float(coordinates[0]), float(coordinates[1]), properties):
                continue
            label_and_query = place_feature_label(properties)
            if label_and_query:
                suggestion = (
                    label_and_query[0],
                    label_and_query[1],
                    (float(coordinates[1]), float(coordinates[0])),
                )
                if suggestion not in results:
                    results.append(suggestion)
        if results:
            break
    typed = normalize_place_text(cleaned)
    return sorted(
        results,
        key=lambda item: difflib.SequenceMatcher(None, typed, normalize_place_text(item[0].split("  |  ")[0])).ratio(),
        reverse=True,
    )[:5]


def resolve_place_text(text: str) -> str:
    normalized = normalize_place_text(text)
    aliases = {normalize_place_text(alias): place for alias, place in LANDMARK_ALIASES.items()}
    places = {normalize_place_text(place): place for place in SUGGESTED_PLACES}
    return aliases.get(normalized, places.get(normalized, text.strip()))


def place_suggestions(text: str) -> list[tuple[str, str, tuple[float, float] | None]]:
    live_results = search_montreal_places(text)
    if live_results:
        return live_results
    return [(place, place, None) for place in matching_places(text)[:6]]


def needs_suggestions(text: str) -> bool:
    normalized = normalize_place_text(text)
    if not normalized:
        return False
    exact_places = {normalize_place_text(place) for place in SUGGESTED_PLACES}
    aliases = {normalize_place_text(alias) for alias in LANDMARK_ALIASES}
    return normalized not in exact_places and normalized not in aliases


def autocomplete_place_options(text: str) -> list[tuple[str, dict[str, object]]]:
    return [
        (label, {"query": query, "coords": coords})
        for label, query, coords in place_suggestions(text)
    ]


def format_selected_coords(coords: tuple[float, float] | None, placeholder: str) -> str:
    if not coords:
        return placeholder
    return f"{coords[0]:.5f}, {coords[1]:.5f}"


def endpoint_from_coords(
    planner: RoutePlanner,
    label: str,
    coords: tuple[float, float],
    graph: object,
    query: str | None = None,
    node: tuple[float, float] | None = None,
) -> RouteEndpoint:
    latitude, longitude = coords
    return RouteEndpoint(
        label=label,
        query=query or f"{latitude:.5f}, {longitude:.5f}",
        latitude=float(latitude),
        longitude=float(longitude),
        node=node or planner.snap_lonlat_to_graph(longitude=float(longitude), latitude=float(latitude), graph=graph),
    )


def make_picker_map(map_style: str, height: int = 520) -> dict:
    tile = MAP_STYLES.get(map_style, "open-street-map")
    tiles = "OpenStreetMap" if tile == "open-street-map" else None
    map_obj = folium.Map(location=MONTREAL_CENTER, zoom_start=12, tiles=tiles, zoom_control=True, control_scale=True)
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
    return st_folium(map_obj, height=height, use_container_width=True, returned_objects=["last_clicked"])


def route_by_label(routes: list[RouteResult], label: str) -> RouteResult | None:
    return next((route for route in routes if label in route.labels), None)


def selected_route_label(routes: list[RouteResult]) -> str:
    available = [label for label in ["Balanced Comfort", "Shortest Distance", "Lowest LTS", "Lowest Effort"] if route_by_label(routes, label)]
    selected = st.session_state.get("montreal_selected_route", available[0])
    return selected if selected in available else available[0]


def make_route_map(
    routes: list[RouteResult],
    graph_crs: object,
    origin: RouteEndpoint,
    destination: RouteEndpoint,
    selected_label: str,
    map_style: str,
) -> folium.Map:
    tile = MAP_STYLES.get(map_style, "open-street-map")
    tiles = "OpenStreetMap" if tile == "open-street-map" else None
    map_obj = folium.Map(location=MONTREAL_CENTER, zoom_start=13, tiles=tiles, zoom_control=True, control_scale=True)
    if tile == "carto-positron":
        folium.TileLayer("CartoDB positron", name="Google-like light").add_to(map_obj)
    elif tile == "carto-darkmatter":
        folium.TileLayer("CartoDB dark_matter", name="Dark contrast").add_to(map_obj)

    routes_gdf = RoutePlanner.routes_to_geodataframe(routes, graph_crs)
    selected_geometry = None
    for route, (_, row) in zip(routes, routes_gdf.iterrows()):
        if selected_label not in route.labels:
            continue
        selected_geometry = row.geometry
        label = selected_label
        coords = list(row.geometry.coords)
        tooltip = (
            f"<b>{label}</b><br>"
            f"{route.distance_m / 1000.0:.2f} km | {route.estimated_minutes:.0f} min<br>"
            f"LTS {route.mean_lts:.2f} | Effort {route.mean_effort:.2f}<br>"
            f"Elevation gain {route.elevation_gain_m:.0f} m"
        )
        folium.PolyLine(
            [(coord[1], coord[0]) for coord in coords],
            color=ROUTE_COLORS.get(selected_label, "#0d6c75"),
            weight=7,
            opacity=0.95,
            tooltip=tooltip,
        ).add_to(map_obj)
    folium.Marker(
        location=(origin.latitude, origin.longitude),
        tooltip=f"A: {origin.query}",
        icon=folium.Icon(color="darkblue", icon="play", prefix="fa"),
    ).add_to(map_obj)
    folium.Marker(
        location=(destination.latitude, destination.longitude),
        tooltip=f"B: {destination.query}",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(map_obj)
    bounds = selected_geometry.bounds if selected_geometry is not None else routes_gdf.total_bounds
    map_obj.fit_bounds([[float(bounds[1]), float(bounds[0])], [float(bounds[3]), float(bounds[2])]], padding=(28, 28))
    return map_obj


st.markdown(
    f"""
    <div class="map-topbar">
      <div class="finder-brand"><img src="{BADGE_URI}" alt="">Montreal Route Finder</div>
      <div class="mode-chips">
        <span class="mode-chip">Routes</span>
        <span class="mode-chip">Traffic stress</span>
        <span class="mode-chip">Cyclist effort</span>
        <span class="mode-chip">Combined comfort</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

panel_col, map_col = st.columns([0.32, 0.68], gap="medium")

with panel_col:
    st.markdown(
        """
        <div class="search-panel">
          <p class="panel-kicker">Montreal cycling directions</p>
          <h1 class="panel-title">Choose a start and destination</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    input_mode = st.radio("Choose points by", ["Typing", "Map click"], horizontal=True)
    if input_mode == "Map click":
        st.session_state["montreal_origin_display"] = format_selected_coords(
            st.session_state.get("montreal_origin_coords"),
            "Click map to set origin",
        )
        st.session_state["montreal_destination_display"] = format_selected_coords(
            st.session_state.get("montreal_destination_coords"),
            "Click map to set destination",
        )
        origin_text = st.text_input("Origin", key="montreal_origin_display", disabled=True)
        destination_text = st.text_input("Destination", key="montreal_destination_display", disabled=True)
        origin_choice = origin_text
        destination_choice = destination_text
        st.markdown(
            '<div class="map-click-hint">Choose A or B, then click directly on the map. Selected coordinates will appear here.</div>',
            unsafe_allow_html=True,
        )
    else:
        origin_selection = st_searchbox(
            autocomplete_place_options,
            placeholder="Search for an origin in Montreal",
            label="Origin",
            key="montreal_origin_search",
            clear_on_submit=False,
            edit_after_submit="option",
            style_absolute=True,
        )
        destination_selection = st_searchbox(
            autocomplete_place_options,
            placeholder="Search for a destination in Montreal",
            label="Destination",
            key="montreal_destination_search",
            clear_on_submit=False,
            edit_after_submit="option",
            style_absolute=True,
        )
        origin_choice = str(origin_selection["query"]) if origin_selection else ""
        destination_choice = str(destination_selection["query"]) if destination_selection else ""
    if input_mode == "Map click":
        st.session_state["montreal_pick_target"] = st.radio("Next map click sets", ["Origin", "Destination"], horizontal=True)
        clear_cols = st.columns(2)
        if clear_cols[0].button("Clear A", use_container_width=True):
            st.session_state["montreal_origin_coords"] = None
            st.session_state.pop("montreal_route_result", None)
            st.rerun()
        if clear_cols[1].button("Clear B", use_container_width=True):
            st.session_state["montreal_destination_coords"] = None
            st.session_state.pop("montreal_route_result", None)
            st.rerun()
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

if find_button:
    origin_coords = (
        st.session_state.get("montreal_origin_coords")
        if input_mode == "Map click"
        else origin_selection.get("coords") if origin_selection else None
    )
    destination_coords = (
        st.session_state.get("montreal_destination_coords")
        if input_mode == "Map click"
        else destination_selection.get("coords") if destination_selection else None
    )
    if input_mode == "Map click" and (origin_coords is None or destination_coords is None):
        st.error("Click the map to set both A and B before finding routes.")
    elif input_mode == "Typing" and not origin_selection:
        st.error("Search for and select an origin from the location suggestions.")
    elif input_mode == "Typing" and not destination_selection:
        st.error("Search for and select a destination from the location suggestions.")
    else:
        try:
            with st.spinner("Preparing LTS and cyclist-effort routes for the selected locations..."):
                st.session_state["montreal_route_result"] = calculate_routes(
                    origin_choice,
                    destination_choice,
                    stress_weight,
                    effort_weight,
                    origin_coords=origin_coords,
                    destination_coords=destination_coords,
                )
                st.session_state["montreal_selected_route"] = "Balanced Comfort"
        except Exception as exc:
            st.error(f"Could not calculate routes for the selected locations: {exc}")

result = st.session_state.get("montreal_route_result")

with panel_col:
    if result is None:
        st.info("Enter two Montreal locations or switch to map click to set A/B points.")
    else:
        routes = result["routes"]
        selected_label = selected_route_label(routes)
        selected_route = route_by_label(routes, selected_label) or routes[0]
        st.success("Routes ready. Select the route to display above the map.")
        st.markdown(
            f"""
            <div class="index-strip">
              <div class="index-cell"><div class="index-label">Selected</div><div class="index-value">{selected_label}</div></div>
              <div class="index-cell"><div class="index-label">Traffic stress</div><div class="index-value">LTS {selected_route.mean_lts:.2f}</div></div>
              <div class="index-cell"><div class="index-label">Effort</div><div class="index-value">{selected_route.mean_effort:.2f}</div></div>
              <div class="index-cell"><div class="index-label">Elevation</div><div class="index-value">{selected_route.elevation_gain_m:.0f} m</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "elevation_source" in result["network"]:
            source = str(result["network"]["elevation_source"].dropna().iloc[0])
            st.caption(f"Elevation source: {source}.")

with map_col:
    st.markdown('<div class="map-frame">', unsafe_allow_html=True)
    if result is None:
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
        picker_data = make_picker_map(map_style, height=520)
        if input_mode == "Map click":
            last_clicked = picker_data.get("last_clicked") if picker_data else None
            if last_clicked:
                clicked = (round(float(last_clicked["lat"]), 6), round(float(last_clicked["lng"]), 6))
                key = "montreal_origin_coords" if st.session_state["montreal_pick_target"] == "Origin" else "montreal_destination_coords"
                if st.session_state.get(key) != clicked:
                    st.session_state[key] = clicked
                    st.session_state.pop("montreal_route_result", None)
                    if st.session_state["montreal_pick_target"] == "Origin":
                        st.session_state["montreal_pick_target"] = "Destination"
                    st.rerun()
    else:
        routes = result["routes"]
        route_labels = [label for label in ["Shortest Distance", "Lowest LTS", "Lowest Effort", "Balanced Comfort"] if route_by_label(routes, label)]
        selected_label = selected_route_label(routes)
        if st.session_state.get("montreal_selected_route") not in route_labels:
            st.session_state["montreal_selected_route"] = selected_label
        st.markdown('<div class="display-route-label">Route shown on map</div>', unsafe_allow_html=True)
        st.radio(
            "Route shown on map",
            route_labels,
            key="montreal_selected_route",
            horizontal=True,
            label_visibility="collapsed",
        )
        selected_label = selected_route_label(routes)
        st_folium(
            make_route_map(
                routes,
                result["graph"].graph["crs"],
                result["origin"],
                result["destination"],
                selected_label,
                map_style,
            ),
            height=520,
            use_container_width=True,
            returned_objects=[],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if result is not None:
        routes = result["routes"]
        selected_label = selected_route_label(routes)
        route_labels = [label for label in ["Shortest Distance", "Lowest LTS", "Lowest Effort", "Balanced Comfort"] if route_by_label(routes, label)]
        st.markdown('<div class="results-title">Route options</div>', unsafe_allow_html=True)
        card_cols = st.columns(len(route_labels))
        for col, label in zip(card_cols, route_labels):
            route = route_by_label(routes, label)
            assert route is not None
            selected_class = " selected" if label == selected_label else ""
            col.markdown(
                f"""
                <div class="route-card{selected_class}">
                  <div class="route-name">{label}</div>
                  <div class="route-meta">
                    <strong>{route.estimated_minutes:.0f} min</strong> · {route.distance_m / 1000.0:.2f} km<br>
                    Mean LTS {route.mean_lts:.2f} · Effort {route.mean_effort:.2f}<br>
                    Elevation gain {route.elevation_gain_m:.0f} m
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with st.expander("Route index table", expanded=False):
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
