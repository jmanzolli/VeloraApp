from __future__ import annotations

import base64
import html
import json
import time
import mimetypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.storage import (
    delete_all_runs,
    delete_run,
    delete_scenario_snapshot,
    list_scenario_snapshots,
    save_scenario_snapshot,
)

ASSETS_DIR = Path(__file__).parent / "ui" / "assets"


def load_asset_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


LOGO_PATH = ASSETS_DIR / "velora_wordmark.png"
BADGE_PATH = ASSETS_DIR / "velora_badge.png"
PAGE_ICON_PATH = BADGE_PATH
LOGO_URI = load_asset_data_uri(LOGO_PATH)
BADGE_URI = load_asset_data_uri(BADGE_PATH)

DEFAULT_MODE_SHIFT_RATE = 0.15
DEFAULT_AVERAGE_TRIP_DISTANCE_KM = 2.5
DEFAULT_CAR_EMISSION_FACTOR_G_PER_KM = 190.0

st.set_page_config(
    page_title="Velora",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=str(PAGE_ICON_PATH),
)

st.markdown(
    """
    <style>
      :root {
        --navy: #0b3552;
        --navy-deep: #07273d;
        --teal: #43b8a3;
        --teal-soft: #e8f6f2;
        --amber: #d9901a;
        --red: #b54b32;
        --panel: rgba(255,255,255,0.94);
        --panel-strong: rgba(255,255,255,0.985);
        --panel-border: rgba(11,53,82,0.12);
        --ink: #14212b;
        --muted: #6a7c89;
        --surface: #f7fbfd;
        --surface-strong: #ffffff;
        --shadow-soft: 0 18px 42px rgba(12,44,66,0.08);
      }
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(67,184,163,0.18), transparent 24%),
          radial-gradient(circle at top right, rgba(217,144,26,0.16), transparent 20%),
          linear-gradient(180deg, #eff4f7 0%, #e7eef3 100%);
        color: var(--ink);
      }
      .block-container {
        max-width: 1480px;
        padding-top: 3.4rem;
        padding-bottom: 2.2rem;
      }
      [data-testid="stHeader"] {
        background: rgba(255,255,255,0.9);
      }
      [data-testid="stSidebar"] {
        background:
          radial-gradient(circle at top left, rgba(88,205,180,0.18), transparent 24%),
          radial-gradient(circle at 80% 8%, rgba(213,232,112,0.16), transparent 18%),
          linear-gradient(180deg, #0a2b44 0%, #092137 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
      }
      [data-testid="stSidebar"] > div:first-child {
        background:
          linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0)),
          linear-gradient(180deg, rgba(10,43,68,0.94), rgba(9,33,55,0.98));
      }
      [data-testid="stSidebar"] .block-container {
        padding-top: 1.35rem;
      }
      [data-testid="stSidebar"] * {
        color: #eef6fb;
      }
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] .stMarkdown,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] small,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #eef6fb !important;
      }
      [data-testid="stSidebar"] .stExpander {
        background: rgba(255,255,255,0.065);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        box-shadow: 0 16px 36px rgba(3,17,29,0.18);
        overflow: hidden;
        backdrop-filter: blur(12px);
      }
      [data-testid="stSidebar"] .stExpander summary:hover,
      [data-testid="stSidebar"] .stExpander summary:focus,
      [data-testid="stSidebar"] .stExpander summary:active {
        background: rgba(255,255,255,0.12) !important;
      }
      [data-testid="stSidebar"] .stExpander details {
        border-radius: 14px;
      }
      [data-testid="stSidebar"] .stExpander summary {
        background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05)) !important;
        border-radius: 18px;
        color: #f5fbff !important;
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
      }
      [data-testid="stSidebar"] .stExpander details[open] > summary,
      [data-testid="stSidebar"] .stExpander details[open] > summary:hover,
      [data-testid="stSidebar"] .stExpander details[open] > summary:focus,
      [data-testid="stSidebar"] .stExpander details[open] > summary:active {
        background: rgba(255,255,255,0.06) !important;
        color: #f5fbff !important;
        border-bottom-left-radius: 12px !important;
        border-bottom-right-radius: 12px !important;
      }
      [data-testid="stSidebar"] .stExpander summary p,
      [data-testid="stSidebar"] .stExpander summary span {
        color: #f5fbff !important;
        font-weight: 600;
      }
      [data-testid="stSidebar"] .stExpander details[open] > summary p,
      [data-testid="stSidebar"] .stExpander details[open] > summary span,
      [data-testid="stSidebar"] .stExpander details[open] > summary svg {
        color: #f5fbff !important;
        fill: #f5fbff !important;
      }
      [data-testid="stSidebar"] .stNumberInput input,
      [data-testid="stSidebar"] .stTextInput input,
      [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
      [data-testid="stSidebar"] .stFileUploader {
        background: rgba(255,255,255,0.95);
        border-radius: 14px;
      }
      button, a, [role="button"], summary, input, textarea, [data-baseweb="select"] {
        transform: none !important;
      }
      .stApp button,
      .stApp [role="button"],
      .stApp summary,
      .stApp [data-baseweb="select"] > div,
      .stApp [data-baseweb="base-input"] > div,
      .stApp [data-testid="stFileUploaderDropzone"] {
        transition: background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
      }
      button:hover, button:focus, button:active,
      [role="button"]:hover, [role="button"]:focus, [role="button"]:active,
      summary:hover, summary:focus, summary:active,
      .stButton button:hover, .stButton button:focus, .stButton button:active,
      .stDownloadButton button:hover, .stDownloadButton button:focus, .stDownloadButton button:active {
        transform: none !important;
        box-shadow: none !important;
      }
      .stButton button,
      .stDownloadButton button {
        background: linear-gradient(180deg, #ffffff, #eef4f7) !important;
        color: #173042 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(11,53,82,0.12) !important;
        min-height: 2.8rem !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 0 rgba(255,255,255,0.9), 0 8px 18px rgba(12,44,66,0.06) !important;
      }
      .stButton button:hover,
      .stDownloadButton button:hover {
        background: linear-gradient(180deg, #ffffff, #e8f1f6) !important;
        border-color: rgba(11,53,82,0.2) !important;
      }
      .stButton button:focus-visible,
      .stDownloadButton button:focus-visible,
      .stApp [data-baseweb="select"] > div:focus-within,
      .stApp [data-baseweb="base-input"] > div:focus-within,
      .stApp textarea:focus,
      .stApp input:focus {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(67,184,163,0.22) !important;
        border-color: rgba(67,184,163,0.88) !important;
      }
      .stButton button:disabled,
      .stDownloadButton button:disabled {
        background: linear-gradient(180deg, #eef3f6, #e6edf2) !important;
        color: #7b8e9d !important;
        border-color: rgba(11,53,82,0.08) !important;
        box-shadow: none !important;
        opacity: 1 !important;
      }
      .stButton button:disabled *,
      .stDownloadButton button:disabled * {
        color: #7b8e9d !important;
        fill: #7b8e9d !important;
      }
      .stButton button *,
      .stDownloadButton button * {
        color: #173042 !important;
        fill: #173042 !important;
      }
      [data-testid="stSidebar"] .stButton button,
      [data-testid="stSidebar"] .stDownloadButton button {
        background: rgba(255,255,255,0.10) !important;
        color: #f5fbff !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        min-height: 2.7rem !important;
        border-radius: 13px !important;
        font-weight: 700 !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 22px rgba(2,14,24,0.10) !important;
      }
      [data-testid="stSidebar"] .stButton button *,
      [data-testid="stSidebar"] .stDownloadButton button * {
        color: #f5fbff !important;
        fill: #f5fbff !important;
      }
      [data-testid="stSidebar"] .stButton button:hover,
      [data-testid="stSidebar"] .stButton button:focus,
      [data-testid="stSidebar"] .stDownloadButton button:hover,
      [data-testid="stSidebar"] .stDownloadButton button:focus {
        background: rgba(255,255,255,0.16) !important;
        color: #ffffff !important;
        border-color: rgba(67,184,163,0.75) !important;
        box-shadow: 0 0 0 1px rgba(67,184,163,0.25) !important;
      }
      [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #134b73 0%, #16736a 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.13) !important;
      }
      [data-testid="stSidebar"] .stButton button[kind="primary"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
      }
      .stButton button[kind="primary"] {
        background: linear-gradient(90deg, var(--navy) 0%, #0d486f 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 14px 28px rgba(7,39,61,0.18) !important;
      }
      .stButton button[kind="primary"] *,
      .stDownloadButton button[kind="primary"] * {
        color: white !important;
        fill: white !important;
      }
      .stButton button[kind="primary"]:hover,
      .stButton button[kind="primary"]:focus,
      .stButton button[kind="primary"]:active {
        background: linear-gradient(90deg, #0c3c5d 0%, #12608f 100%) !important;
        color: white !important;
      }
      .stApp [data-baseweb="select"] > div,
      .stApp [data-baseweb="base-input"] > div,
      .stApp textarea,
      .stApp input {
        background: rgba(255,255,255,0.96) !important;
        color: #173042 !important;
        border-color: rgba(11,53,82,0.12) !important;
      }
      [data-baseweb="select"] > div,
      [data-baseweb="base-input"] > div,
      [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px !important;
      }
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] textarea,
      [data-testid="stSidebar"] [data-baseweb="input"] input,
      [data-testid="stSidebar"] [data-baseweb="base-input"] input {
        color: #173042 !important;
        -webkit-text-fill-color: #173042 !important;
      }
      [data-testid="stSidebar"] input::placeholder,
      [data-testid="stSidebar"] textarea::placeholder {
        color: #7a8d9a !important;
        -webkit-text-fill-color: #7a8d9a !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #173042 !important;
      }
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.96) !important;
        border: 1px dashed rgba(11,53,82,0.18) !important;
        padding: 1rem 0.85rem !important;
      }
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #173042 !important;
      }
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: #ffffff !important;
        color: #173042 !important;
        border: 1px solid rgba(11,53,82,0.14) !important;
        min-height: 2.25rem !important;
        box-shadow: none !important;
      }
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
        color: #173042 !important;
        fill: #173042 !important;
      }
      [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
        display: none !important;
      }
      [data-baseweb="select"] > div:hover,
      [data-baseweb="base-input"] > div:hover,
      [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(11,53,82,0.18) !important;
        box-shadow: none !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
      [data-testid="stSidebar"] [data-baseweb="base-input"] > div:hover,
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(67,184,163,0.9) !important;
        box-shadow: 0 0 0 1px rgba(67,184,163,0.22) !important;
      }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
      [data-testid="stSidebar"] .stCaption * {
        color: rgba(238,246,251,0.78) !important;
      }
      [data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] {
        color: #eef6fb !important;
      }
      [data-testid="stSidebar"] .stSlider label,
      [data-testid="stSidebar"] .stSlider small,
      [data-testid="stSidebar"] .stSlider p,
      [data-testid="stSidebar"] .stSlider span {
        color: #eef6fb !important;
        -webkit-text-fill-color: #eef6fb !important;
      }
      [data-testid="stSidebar"] .stSlider * {
        color: #eef6fb !important;
        fill: #eef6fb !important;
        stroke: #eef6fb !important;
      }
      [data-testid="stSidebar"] .stSlider [aria-hidden="true"],
      [data-testid="stSidebar"] .stSlider [data-testid*="TickBar"],
      [data-testid="stSidebar"] .stSlider [data-testid*="tickBar"],
      [data-testid="stSidebar"] .stSlider [class*="tick"],
      [data-testid="stSidebar"] .stSlider [class*="Tick"],
      [data-testid="stSidebar"] .stSlider [class*="mark"],
      [data-testid="stSidebar"] .stSlider [class*="Mark"] {
        opacity: 1 !important;
        visibility: visible !important;
      }
      [data-testid="stSidebar"] .stSlider div[role="slider"] {
        background: #43b8a3 !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.18) !important;
      }
      [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
        background: rgba(255,255,255,0.22) !important;
      }
      [data-testid="stSidebar"] .stSlider input,
      [data-testid="stSidebar"] .stSlider input:hover,
      [data-testid="stSidebar"] .stSlider input:focus,
      [data-testid="stSidebar"] .stSlider [data-baseweb="input"] input,
      [data-testid="stSidebar"] .stSlider [data-baseweb="base-input"] input {
        color: #eef6fb !important;
        -webkit-text-fill-color: #eef6fb !important;
        background: #49677d !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        border-radius: 6px !important;
      }
      [data-testid="stSidebar"] .stSlider [data-baseweb="input"],
      [data-testid="stSidebar"] .stSlider [data-baseweb="base-input"] {
        background: #49677d !important;
        border-radius: 6px !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        box-shadow: none !important;
      }
      [data-testid="stSidebar"] .stSlider [data-baseweb="input"] *,
      [data-testid="stSidebar"] .stSlider [data-baseweb="base-input"] * {
        background-color: #49677d !important;
      }
      [data-testid="stSidebar"] .stSlider [data-baseweb="input"] > div,
      [data-testid="stSidebar"] .stSlider [data-baseweb="base-input"] > div,
      [data-testid="stSidebar"] .stSlider [data-baseweb="input"] > div:hover,
      [data-testid="stSidebar"] .stSlider [data-baseweb="base-input"] > div:hover,
      [data-testid="stSidebar"] .stSlider [data-baseweb="input"] > div:focus-within,
      [data-testid="stSidebar"] .stSlider [data-baseweb="base-input"] > div:focus-within {
        background: #49677d !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        box-shadow: none !important;
      }
      [data-testid="stSidebar"] .stSlider [data-baseweb="input"] input[disabled],
      [data-testid="stSidebar"] .stSlider [data-baseweb="base-input"] input[disabled] {
        opacity: 1 !important;
        color: #eef6fb !important;
        -webkit-text-fill-color: #eef6fb !important;
        background: #49677d !important;
      }
      [data-testid="stSidebar"] .stSlider [data-baseweb="input"] {
        display: none !important;
      }
      [data-testid="stSidebar"] .stSlider [data-testid*="TickBar"] *,
      [data-testid="stSidebar"] .stSlider [data-testid*="tickBar"] *,
      [data-testid="stSidebar"] .stSlider [class*="tick"] *,
      [data-testid="stSidebar"] .stSlider [class*="Tick"] *,
      [data-testid="stSidebar"] .stSlider [class*="mark"] *,
      [data-testid="stSidebar"] .stSlider [class*="Mark"] *,
      [data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] + div *,
      [data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] ~ div * {
        color: rgba(238,246,251,0.92) !important;
        fill: rgba(238,246,251,0.92) !important;
        stroke: rgba(238,246,251,0.92) !important;
        opacity: 1 !important;
        visibility: visible !important;
      }
      .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 0.95rem 1.3rem;
        margin-top: 0.15rem;
        border-radius: 22px 22px 0 0;
        background: linear-gradient(90deg, var(--navy) 0%, #0d486f 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(7,39,61,0.15);
      }
      .topbar-title {
        display: flex;
        align-items: center;
        gap: 1rem;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.01em;
      }
      .brand-lockup {
        display: flex;
        align-items: center;
        gap: 1rem;
      }
      .brand-logo {
        height: 3rem;
        width: auto;
        max-width: 15rem;
        object-fit: contain;
        border-radius: 16px;
        padding: 0.32rem 0.72rem;
        background: rgba(255,255,255,0.94);
        box-shadow: inset 0 0 0 1px rgba(16,47,67,0.08), 0 10px 22px rgba(2,14,24,0.12);
      }
      .sidebar-brand-card {
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
        padding: 1.1rem 1rem 1rem;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.12);
        background:
          radial-gradient(circle at top right, rgba(215,243,106,0.18), transparent 30%),
          linear-gradient(145deg, rgba(255,255,255,0.14), rgba(255,255,255,0.05));
        box-shadow: 0 20px 42px rgba(2,14,24,0.22);
        backdrop-filter: blur(14px);
      }
      .sidebar-brand-card::after {
        content: "";
        position: absolute;
        inset: auto -12% -34% auto;
        width: 9rem;
        height: 9rem;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(67,200,176,0.28), rgba(67,200,176,0));
      }
      .sidebar-brand-top {
        display: flex;
        align-items: center;
        gap: 0.85rem;
      }
      .sidebar-brand-logo {
        width: 3.4rem;
        height: 3.4rem;
        border-radius: 18px;
        object-fit: contain;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 0.36rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
      }
      .sidebar-kicker {
        margin: 0;
        color: rgba(238,246,251,0.72);
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
      }
      .sidebar-wordmark {
        margin: 0.18rem 0 0;
        font-size: 1.42rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        color: #ffffff;
      }
      .sidebar-tagline {
        margin: 0.85rem 0 0;
        font-size: 0.9rem;
        line-height: 1.45;
        color: rgba(238,246,251,0.82);
      }
      .sidebar-stat-row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
        margin-top: 0.95rem;
      }
      .sidebar-stat {
        padding: 0.7rem 0.75rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.08);
      }
      .sidebar-stat-label {
        display: block;
        color: rgba(238,246,251,0.68);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .sidebar-stat-value {
        display: block;
        margin-top: 0.3rem;
        color: #ffffff;
        font-size: 0.96rem;
        font-weight: 700;
      }
      .sidebar-section-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin: 1rem 0 0.45rem;
      }
      .sidebar-section-title {
        margin: 0;
        color: #ffffff;
        font-size: 0.98rem;
        font-weight: 700;
        letter-spacing: 0.01em;
      }
      .sidebar-section-meta {
        margin: 0.12rem 0 0;
        color: rgba(238,246,251,0.7);
        font-size: 0.77rem;
      }
      .sidebar-help {
        position: relative;
        flex: 0 0 auto;
      }
      .sidebar-help-badge {
        width: 1.55rem;
        height: 1.55rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        color: #dff8ff;
        font-size: 0.95rem;
        font-weight: 700;
        cursor: help;
        line-height: 1;
      }
      .sidebar-help-panel {
        position: absolute;
        right: 0;
        top: calc(100% + 0.45rem);
        width: 15rem;
        padding: 0.7rem 0.8rem;
        border-radius: 14px;
        background: rgba(4,19,31,0.96);
        border: 1px solid rgba(138,247,214,0.18);
        color: #ecfbff;
        font-size: 0.77rem;
        line-height: 1.45;
        box-shadow: 0 18px 32px rgba(0,0,0,0.28);
        opacity: 0;
        visibility: hidden;
        transform: translateY(6px);
        transition: opacity 0.18s ease, transform 0.18s ease, visibility 0.18s ease;
        z-index: 20;
      }
      .sidebar-help:hover .sidebar-help-panel,
      .sidebar-help:focus-within .sidebar-help-panel {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
      }
      .sidebar-note {
        margin: 0.8rem 0 1rem;
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: rgba(238,246,251,0.8);
        font-size: 0.84rem;
        line-height: 1.45;
      }
      .sidebar-upload-status {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0.35rem 0 0.15rem;
        padding: 0.6rem 0.7rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
      }
      .sidebar-upload-icon {
        width: 1.9rem;
        height: 1.9rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: rgba(67,184,163,0.18);
        color: #d7fbf4;
        font-size: 1rem;
        flex: 0 0 auto;
      }
      .sidebar-upload-text {
        min-width: 0;
      }
      .sidebar-upload-name {
        color: #f5fbff;
        font-size: 0.92rem;
        font-weight: 600;
        line-height: 1.1;
        word-break: break-word;
      }
      .sidebar-upload-meta {
        color: rgba(238,246,251,0.78);
        font-size: 0.78rem;
        margin-top: 0.18rem;
      }
      .topbar-nav {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
      }
      .topbar-pill {
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.14);
        color: rgba(248,252,254,0.96);
        font-size: 0.88rem;
        font-weight: 600;
      }
      .topbar-pill:hover,
      .topbar-pill:focus,
      .topbar-pill:active {
        background: rgba(255,255,255,0.18);
        border-color: rgba(255,255,255,0.24);
      }
      .hero {
        padding: 1.15rem 1.35rem 1.25rem;
        border-radius: 0 0 22px 22px;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(245,250,252,1));
        border: 1px solid var(--panel-border);
        border-top: none;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-soft);
      }
      .hero h1 {
        margin: 0;
        font-size: 1.95rem;
        line-height: 1.1;
        color: var(--ink);
      }
      .hero p {
        margin: 0.6rem 0 0;
        font-size: 1rem;
        max-width: 60rem;
        color: var(--muted);
      }
      .subhero {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1rem;
      }
      .subhero-card {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: white;
        border: 1px solid var(--panel-border);
      }
      .subhero-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }
      .subhero-value {
        margin-top: 0.3rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--navy);
      }
      .dashboard-section {
        background: var(--panel-strong);
        border: 1px solid var(--panel-border);
        border-radius: 20px;
        padding: 1rem 1rem 0.75rem;
        box-shadow: var(--shadow-soft);
        margin-bottom: 1rem;
      }
      .control-shell {
        padding-bottom: 1rem;
        background:
          linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,250,252,0.98));
      }
      .panel-lead {
        margin: 0 0 0.85rem;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.5;
      }
      .insight-shell {
        background:
          linear-gradient(180deg, rgba(255,255,255,0.98), rgba(241,247,250,0.98));
      }
      .comparison-shell,
      .action-shell {
        padding-bottom: 1rem;
      }
      .workspace-rail {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0 0 0.95rem;
      }
      .workspace-step {
        padding: 0.78rem 0.9rem;
        border-radius: 14px;
        border: 1px solid rgba(11,53,82,0.09);
        background: rgba(255,255,255,0.82);
      }
      .workspace-step strong {
        display: block;
        color: var(--navy);
        font-size: 0.92rem;
      }
      .workspace-step span {
        display: block;
        margin-top: 0.2rem;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.35;
      }
      .workspace-step--active {
        background: linear-gradient(180deg, rgba(232,247,241,0.98), rgba(255,255,255,0.98));
        border-color: rgba(67,184,163,0.34);
      }
      .scenario-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 0.95rem;
      }
      .scenario-card {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(11,53,82,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(242,248,251,0.98));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.88);
      }
      .scenario-card--accent {
        background: linear-gradient(180deg, rgba(232,247,241,0.92), rgba(255,255,255,0.98));
        border-color: rgba(67,184,163,0.26);
      }
      .scenario-card--comparison {
        background: linear-gradient(180deg, rgba(242,238,252,0.96), rgba(255,255,255,0.98));
        border-color: rgba(109,40,217,0.18);
      }
      .scenario-kicker {
        display: block;
        color: var(--muted);
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .scenario-name {
        display: block;
        margin-top: 0.28rem;
        color: var(--navy);
        font-size: 1.05rem;
        font-weight: 700;
      }
      .scenario-meta {
        margin-top: 0.35rem;
        color: #4e6372;
        font-size: 0.84rem;
        line-height: 1.45;
      }
      .planning-metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 0.95rem;
      }
      .planning-metric {
        padding: 0.85rem 0.9rem;
        border-radius: 14px;
        border: 1px solid rgba(11,53,82,0.09);
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(246,250,252,0.98));
        min-height: 6.2rem;
      }
      .planning-metric--wide {
        grid-column: 1 / -1;
      }
      .planning-metric-label {
        display: block;
        color: var(--muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .planning-metric-value {
        display: block;
        margin-top: 0.34rem;
        color: var(--navy);
        font-size: 1.55rem;
        line-height: 1.05;
        font-weight: 800;
      }
      .planning-metric-note {
        display: block;
        margin-top: 0.38rem;
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.35;
      }
      .delta-pill {
        display: inline-flex;
        align-items: center;
        margin-top: 0.45rem;
        padding: 0.18rem 0.48rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 800;
      }
      .delta-pill--positive {
        background: rgba(44,162,95,0.12);
        color: #0f6d42;
      }
      .delta-pill--negative {
        background: rgba(209,73,91,0.12);
        color: #a43849;
      }
      .delta-pill--neutral {
        background: rgba(106,124,137,0.12);
        color: #4e6372;
      }
      .asset-summary {
        padding: 0.75rem 0.85rem;
        border-radius: 14px;
        background: rgba(11,53,82,0.05);
        color: #4e6372;
        font-size: 0.86rem;
        line-height: 1.45;
      }
      .scenario-library-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.35rem;
      }
      .library-card {
        padding: 0.9rem;
        border-radius: 16px;
        border: 1px solid rgba(11,53,82,0.09);
        background: linear-gradient(180deg, rgba(255,255,255,0.99), rgba(246,250,252,0.98));
      }
      .library-card--active {
        border-color: rgba(67,184,163,0.5);
        box-shadow: inset 0 0 0 1px rgba(67,184,163,0.14);
      }
      .library-card-title {
        display: block;
        color: var(--navy);
        font-size: 1rem;
        font-weight: 800;
      }
      .library-card-meta,
      .library-card-note {
        display: block;
        margin-top: 0.28rem;
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.35;
      }
      .library-card-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.45rem;
        margin-top: 0.7rem;
      }
      .library-card-metrics span {
        display: block;
        padding: 0.42rem 0.5rem;
        border-radius: 10px;
        background: rgba(11,53,82,0.045);
        color: #173042;
        font-size: 0.78rem;
        font-weight: 700;
      }
      .comparison-delta-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.75rem 0 0.95rem;
      }
      .comparison-delta-card {
        padding: 0.8rem;
        border-radius: 14px;
        border: 1px solid rgba(11,53,82,0.08);
        background: rgba(255,255,255,0.86);
      }
      .comparison-delta-card strong {
        display: block;
        color: var(--navy);
        font-size: 0.9rem;
      }
      .comparison-delta-card span {
        display: block;
        margin-top: 0.22rem;
        color: var(--muted);
        font-size: 0.8rem;
      }
      .empty-state {
        padding: 1rem 1.05rem;
        border-radius: 16px;
        border: 1px dashed rgba(11,53,82,0.18);
        background: linear-gradient(180deg, rgba(245,249,252,0.96), rgba(255,255,255,0.98));
        color: var(--muted);
      }
      .section-divider {
        height: 1px;
        margin: 0.9rem 0 0.95rem;
        background: linear-gradient(90deg, rgba(11,53,82,0.04), rgba(11,53,82,0.16), rgba(11,53,82,0.04));
      }
      .section-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.8rem;
      }
      .section-title h3 {
        margin: 0;
        font-size: 1.1rem;
        color: var(--ink);
      }
      .section-chip {
        padding: 0.34rem 0.7rem;
        border-radius: 999px;
        background: var(--teal-soft);
        color: #0f6d5d;
        font-size: 0.78rem;
        font-weight: 600;
      }
      .section-chip:hover,
      .section-chip:focus,
      .section-chip:active {
        background: var(--teal-soft);
        color: #0f6d5d;
      }
      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.9rem;
        margin-bottom: 1rem;
      }
      .kpi-card {
        padding: 1rem 1rem 0.9rem;
        border-radius: 18px;
        background: white;
        border: 1px solid var(--panel-border);
        min-height: 120px;
      }
      .kpi-label {
        font-size: 0.84rem;
        color: var(--muted);
        margin-bottom: 0.45rem;
      }
      .kpi-value {
        font-size: 2.25rem;
        line-height: 1;
        font-weight: 700;
        color: var(--navy);
      }
      .kpi-note {
        margin-top: 0.45rem;
        font-size: 0.82rem;
        color: var(--muted);
      }
      .report-shell {
        background: white;
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 1rem 1.15rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
      }
      .map-control-panel {
        margin-bottom: 0.95rem;
        padding: 1rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(232,247,241,0.78), rgba(255,255,255,0.98));
        border: 1px solid rgba(11,53,82,0.08);
      }
      .map-control-title {
        margin: 0 0 0.3rem;
        font-size: 1rem;
        font-weight: 700;
        color: var(--navy);
      }
      .map-control-copy {
        margin: 0;
        color: var(--muted);
        font-size: 0.9rem;
      }
      .map-mini-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.9rem 0 0.25rem;
      }
      .map-mini-stat {
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(11,53,82,0.08);
      }
      .map-mini-label {
        display: block;
        color: var(--muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .map-mini-value {
        display: block;
        margin-top: 0.25rem;
        color: var(--navy);
        font-size: 1.15rem;
        font-weight: 700;
      }
      .upload-section-copy {
        margin: 0 0 0.75rem;
        color: rgba(238,246,251,0.76);
        font-size: 0.84rem;
        line-height: 1.45;
      }
      .stDataFrame, .stPlotlyChart {
        background: white;
        border-radius: 18px;
        border: 1px solid rgba(11,53,82,0.08);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
      }
      div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,249,252,0.98));
        border: 1px solid rgba(11,53,82,0.08);
        border-radius: 16px;
        padding: 0.85rem 0.9rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.86);
      }
      div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
      }
      div[data-testid="stMetricValue"] {
        color: var(--navy) !important;
      }
      div[data-testid="stMetricDelta"] {
        color: #0f6d5d !important;
      }
      div[data-testid="stAlert"] {
        border-radius: 16px !important;
        border: 1px solid rgba(11,53,82,0.08) !important;
        background: linear-gradient(180deg, rgba(235,244,252,0.9), rgba(255,255,255,0.96)) !important;
      }
      div[data-testid="stInfo"] {
        background: linear-gradient(180deg, rgba(227,240,252,0.92), rgba(246,250,253,0.96)) !important;
      }
      div[data-testid="stSuccess"] {
        background: linear-gradient(180deg, rgba(231,247,241,0.96), rgba(248,252,250,0.98)) !important;
      }
      div[data-testid="stWarning"] {
        background: linear-gradient(180deg, rgba(252,243,226,0.96), rgba(255,250,245,0.98)) !important;
      }
      .stMultiSelect [data-baseweb="tag"] {
        background: rgba(67,184,163,0.14) !important;
        border: 1px solid rgba(67,184,163,0.22) !important;
      }
      .stMultiSelect [data-baseweb="tag"] * {
        color: var(--navy) !important;
      }
      @media (max-width: 1200px) {
        .kpi-grid, .subhero, .map-mini-stats, .scenario-strip, .workspace-rail, .scenario-library-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
      }
      @media (max-width: 820px) {
        .kpi-grid, .subhero, .map-mini-stats, .scenario-strip, .workspace-rail, .planning-metric-grid, .scenario-library-grid, .comparison-delta-grid {
          grid-template-columns: 1fr;
        }
        .planning-metric--wide {
          grid-column: auto;
        }
        .brand-logo {
          height: 2.4rem;
        }
        .sidebar-stat-row {
          grid-template-columns: 1fr;
        }
      }

      /* Velora Planning Workspace visual refresh: cleaner, sharper, and more technical. */
      :root {
        --navy: #102f43;
        --navy-deep: #071b29;
        --teal: #16a389;
        --teal-soft: #e6f7f2;
        --amber: #c98d22;
        --red: #c24152;
        --panel: rgba(255,255,255,0.88);
        --panel-strong: rgba(255,255,255,0.96);
        --panel-border: rgba(16,47,67,0.10);
        --ink: #101820;
        --muted: #657685;
        --surface: #edf3f6;
        --surface-strong: #ffffff;
        --shadow-soft: 0 18px 46px rgba(16,47,67,0.08);
      }
      html, body, .stApp {
        font-family: "Aptos", "SF Pro Display", "Segoe UI", sans-serif;
      }
      .stApp {
        background:
          radial-gradient(circle at 88% 0%, rgba(22,163,137,0.09), transparent 24%),
          linear-gradient(180deg, #f8fbfc 0%, #eef4f7 100%);
        background-size: auto;
      }
      .block-container {
        max-width: 1540px;
        padding-top: 3.25rem;
      }
      [data-testid="stHeader"] {
        background: rgba(255,255,255,0.82);
        border-bottom: 1px solid rgba(16,47,67,0.08);
        backdrop-filter: blur(14px);
      }
      [data-testid="stSidebar"] {
        background:
          linear-gradient(180deg, #071b29 0%, #0b2638 54%, #071a27 100%) !important;
        background-size: auto !important;
        border-right: 1px solid rgba(159,232,215,0.12);
      }
      [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
      }
      [data-testid="stSidebar"] .block-container {
        padding-top: 1.05rem;
        padding-left: 1.05rem;
        padding-right: 1.05rem;
      }
      [data-testid="stSidebar"] .stExpander {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        box-shadow: none;
        backdrop-filter: blur(18px);
      }
      [data-testid="stSidebar"] .stExpander summary {
        border-radius: 10px !important;
        background: rgba(255,255,255,0.075) !important;
        padding: 0.15rem 0.2rem !important;
      }
      [data-testid="stSidebar"] .stExpander summary p,
      [data-testid="stSidebar"] .stExpander summary span {
        font-size: 0.9rem;
        font-weight: 750;
      }
      [data-testid="stSidebar"] .stSlider div[role="slider"] {
        background: #54d5bd !important;
        box-shadow: 0 0 0 3px rgba(84,213,189,0.18) !important;
      }
      [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
        background: rgba(255,255,255,0.20) !important;
      }
      [data-testid="stSidebar"] .stNumberInput input,
      [data-testid="stSidebar"] .stTextInput input,
      [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
      [data-testid="stSidebar"] .stFileUploader {
        border-radius: 10px !important;
      }
      [data-testid="stSidebar"] .stButton button,
      [data-testid="stSidebar"] .stDownloadButton button {
        border-radius: 10px !important;
        min-height: 2.35rem !important;
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.14) !important;
        box-shadow: none !important;
        padding: 0.45rem 0.5rem !important;
        font-size: 0.8rem !important;
        line-height: 1.15 !important;
        white-space: normal !important;
      }
      [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #145c7a 0%, #16a389 100%) !important;
        box-shadow: 0 12px 26px rgba(22,163,137,0.18) !important;
      }
      .sidebar-brand-card {
        border-radius: 16px;
        padding: 1rem;
        background:
          linear-gradient(145deg, rgba(255,255,255,0.13), rgba(255,255,255,0.04)),
          linear-gradient(180deg, rgba(22,163,137,0.10), rgba(16,47,67,0));
        border-color: rgba(255,255,255,0.12);
        box-shadow: none;
      }
      .sidebar-brand-logo {
        border-radius: 12px;
        width: 3rem;
        height: 3rem;
      }
      .sidebar-wordmark {
        font-size: 1.2rem;
        letter-spacing: 0;
      }
      .sidebar-tagline,
      .sidebar-note {
        font-size: 0.82rem;
      }
      .sidebar-stat {
        border-radius: 10px;
        background: rgba(255,255,255,0.065);
      }
      .sidebar-section-head {
        margin-top: 0.95rem;
      }
      .sidebar-section-title {
        font-size: 0.92rem;
      }
      .sidebar-section-meta {
        font-size: 0.74rem;
      }
      .sidebar-help-badge {
        width: 1.35rem;
        height: 1.35rem;
        font-size: 0.8rem;
        background: rgba(84,213,189,0.12);
        border-color: rgba(84,213,189,0.24);
      }
      .topbar {
        border-radius: 12px 12px 0 0;
        margin-top: 0.25rem;
        padding: 0.82rem 1.05rem;
        background:
          linear-gradient(90deg, rgba(84,213,189,0.13), transparent 42%),
          linear-gradient(90deg, #0d2a3d 0%, #113d56 100%);
        box-shadow: 0 16px 38px rgba(16,47,67,0.12);
      }
      .topbar-title {
        font-size: 1.08rem;
        letter-spacing: 0;
      }
      .brand-logo {
        height: 2.55rem;
        max-width: 13.5rem;
        padding: 0.26rem 0.62rem;
      }
      .topbar-pill {
        border-radius: 10px;
        padding: 0.42rem 0.72rem;
        background: rgba(255,255,255,0.08);
        border-color: rgba(255,255,255,0.16);
        font-size: 0.78rem;
      }
      .hero {
        border-radius: 0 0 12px 12px;
        padding: 1.05rem 1.15rem 1.1rem;
        background:
          linear-gradient(180deg, rgba(255,255,255,0.96), rgba(251,253,254,0.92));
        box-shadow: 0 18px 44px rgba(16,47,67,0.07);
      }
      .hero h1 {
        font-size: clamp(1.55rem, 2.3vw, 2.15rem);
        letter-spacing: 0;
      }
      .hero p {
        font-size: 0.94rem;
      }
      .subhero {
        gap: 0.7rem;
      }
      .subhero-card,
      .dashboard-section,
      .scenario-card,
      .planning-metric,
      .library-card,
      .comparison-delta-card,
      .map-mini-stat,
      .asset-summary,
      .empty-state,
      .report-shell {
        border-radius: 10px;
      }
      .subhero-card,
      .scenario-card,
      .planning-metric,
      .library-card,
      .comparison-delta-card,
      .map-mini-stat {
        background: rgba(255,255,255,0.82);
        border-color: rgba(16,47,67,0.09);
        box-shadow: none;
      }
      .dashboard-section {
        background: rgba(255,255,255,0.84);
        border-color: rgba(16,47,67,0.09);
        box-shadow: 0 16px 42px rgba(16,47,67,0.06);
        backdrop-filter: blur(10px);
      }
      .control-shell,
      .insight-shell {
        background: rgba(255,255,255,0.86);
      }
      .workspace-step {
        border-radius: 10px;
        background: rgba(255,255,255,0.72);
        box-shadow: none;
      }
      .workspace-step--active {
        background: linear-gradient(180deg, rgba(230,247,242,0.95), rgba(255,255,255,0.78));
        border-color: rgba(22,163,137,0.34);
      }
      .scenario-card--accent {
        background: linear-gradient(180deg, rgba(230,247,242,0.92), rgba(255,255,255,0.82));
        border-color: rgba(22,163,137,0.32);
      }
      .scenario-card--comparison {
        background: linear-gradient(180deg, rgba(237,243,246,0.94), rgba(255,255,255,0.82));
        border-color: rgba(16,47,67,0.12);
      }
      .scenario-kicker,
      .planning-metric-label,
      .map-mini-label,
      .subhero-label {
        letter-spacing: 0.12em;
        color: #718290;
      }
      .scenario-name,
      .planning-metric-value,
      .map-mini-value,
      .library-card-title,
      .subhero-value {
        color: #102f43;
      }
      .planning-metric {
        min-height: 5.75rem;
      }
      .planning-metric-value {
        font-size: clamp(1.25rem, 1.8vw, 1.7rem);
      }
      .library-card--active {
        border-color: rgba(22,163,137,0.48);
        box-shadow: inset 0 0 0 1px rgba(22,163,137,0.15);
      }
      .library-card-metrics span {
        border-radius: 8px;
        background: rgba(16,47,67,0.05);
      }
      div[data-testid="stMetric"] {
        border-radius: 10px;
        background: rgba(255,255,255,0.82);
        box-shadow: none;
      }
      .stDataFrame,
      .stPlotlyChart {
        border-radius: 10px;
        box-shadow: none;
        border-color: rgba(16,47,67,0.08);
      }
      .stButton button,
      .stDownloadButton button {
        border-radius: 10px !important;
        min-height: 2.55rem !important;
        box-shadow: none !important;
      }
      .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #102f43 0%, #16a389 100%) !important;
        box-shadow: 0 14px 28px rgba(22,163,137,0.16) !important;
      }
      .stApp input[type="checkbox"] {
        accent-color: #16a389 !important;
      }
      .stApp .stCheckbox [data-testid="stWidgetLabel"] p,
      .stApp .stCheckbox label p {
        color: #2d3a43 !important;
        font-weight: 650 !important;
      }
      .stApp .stCheckbox svg {
        color: #16a389 !important;
        fill: #16a389 !important;
      }
      .stApp .stSlider div[role="slider"] {
        background: #16a389 !important;
        box-shadow: 0 0 0 4px rgba(22,163,137,0.13) !important;
      }
      .stApp .stSlider [data-baseweb="slider"] > div > div {
        background: rgba(16,47,67,0.16) !important;
      }
      .stApp .stSlider [data-baseweb="slider"] div[style*="background"] {
        background-color: #16a389 !important;
      }
      .stApp .stSlider [data-testid="stWidgetLabel"] p,
      .stApp .stSlider label p {
        color: #2d3a43 !important;
        font-weight: 650 !important;
      }
      .stApp .stSlider [data-baseweb="input"] {
        display: none !important;
      }
      [data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] p,
      [data-testid="stSidebar"] .stSlider label p,
      [data-testid="stSidebar"] .stCheckbox [data-testid="stWidgetLabel"] p,
      [data-testid="stSidebar"] .stCheckbox label p {
        color: #eef6fb !important;
      }
      [data-testid="stSidebar"] .stSlider div[role="slider"] {
        background: #54d5bd !important;
      }
      [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[style*="background"] {
        background-color: #54d5bd !important;
      }
    </style>
    <div class="topbar">
      <div class="topbar-title">
        <div class="brand-lockup">
          <img class="brand-logo" src="__VELORA_WORDMARK__" alt="Velora logo" />
          <span>Control Center</span>
        </div>
      </div>
      <div class="topbar-nav">
        <span class="topbar-pill">Data</span>
        <span class="topbar-pill">Scenarios</span>
        <span class="topbar-pill">City Report</span>
      </div>
    </div>
    <div class="hero">
      <h1>Plan better cycling networks with Velora</h1>
      <p>Upload your station data and street network, run the optimizer, and review the recommended plan, map, and city-ready report.</p>
      <div class="subhero">
        <div class="subhero-card">
          <div class="subhero-label">Workflow</div>
          <div class="subhero-value">Upload, run, decide</div>
        </div>
        <div class="subhero-card">
          <div class="subhero-label">What it does</div>
          <div class="subhero-value">Tests network upgrade options</div>
        </div>
        <div class="subhero-card">
          <div class="subhero-label">What you get</div>
          <div class="subhero-value">Map, choices, and report</div>
        </div>
      </div>
    </div>
    """.replace("__VELORA_WORDMARK__", LOGO_URI),
    unsafe_allow_html=True,
)

def format_cost(value: float) -> str:
    return f"{value:,.0f}"


def format_compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,.0f}"


def format_emissions(value_kg: float) -> str:
    if abs(value_kg) >= 1000:
        return f"{value_kg / 1000:,.1f} t CO2e"
    return f"{value_kg:,.0f} kg CO2e"


def format_uploaded_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def open_panel(class_name: str = "") -> None:
    panel_class = f"dashboard-section {class_name}".strip()
    st.markdown(f'<div class="{panel_class}">', unsafe_allow_html=True)


def close_panel() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_panel_header(title: str, description: str | None = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.markdown(f'<p class="panel-lead">{description}</p>', unsafe_allow_html=True)


def render_workspace_rail(active_step: str = "Compare") -> None:
    steps = [
        ("Prepare", "Data and assumptions"),
        ("Optimize", "Run and save results"),
        ("Compare", "Choose plans and baselines"),
        ("Package", "Export decisions"),
    ]
    cards = []
    for label, description in steps:
        state_class = " workspace-step--active" if label == active_step else ""
        cards.append(
            f'<div class="workspace-step{state_class}">'
            f"<strong>{html.escape(label)}</strong>"
            f"<span>{html.escape(description)}</span>"
            "</div>"
        )
    st.markdown(f'<div class="workspace-rail">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_station_preview(stations_upload) -> None:
    stations_preview = pd.read_csv(stations_upload) if stations_upload.name.endswith(".csv") else pd.read_excel(stations_upload)
    preview_open = st.session_state.get("station_preview_open", True)
    with st.expander("Station preview", expanded=preview_open):
        st.caption("Preview of the first uploaded station rows.")
        st.dataframe(stations_preview.head(10), use_container_width=True)
    stations_upload.seek(0)


def render_run_details(run_payload: dict[str, Any]) -> None:
    with st.expander("Run details"):
        st.write("Run ID:", run_payload["run_id"])
        st.write("Created at:", run_payload["created_at"])
        st.write("Uploaded stations:", run_payload["station_count"])
        st.write("Generated candidate locations:", run_payload["candidate_count"])
        st.write("Candidate link pairs:", run_payload["link_pair_count"])
        st.write("Optimization population rows:", run_payload["population_rows"])
        st.dataframe(pd.DataFrame(run_payload["pareto_rows"]).head(20), use_container_width=True)


def render_sidebar_upload_status(uploaded_file, kind_label: str) -> None:
    if uploaded_file is None:
        return
    st.sidebar.markdown(
        f"""
        <div class="sidebar-upload-status">
          <div class="sidebar-upload-icon">✓</div>
          <div class="sidebar-upload-text">
            <div class="sidebar-upload-name">{uploaded_file.name}</div>
            <div class="sidebar-upload-meta">{kind_label} ready · {format_uploaded_size(len(uploaded_file.getvalue()))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand-card">
          <div class="sidebar-brand-top">
            <img class="sidebar-brand-logo" src="{BADGE_URI}" alt="Velora badge" />
            <div>
              <p class="sidebar-kicker">Urban Cycling Studio</p>
              <p class="sidebar-wordmark">Velora</p>
            </div>
          </div>
          <p class="sidebar-tagline">Design safer, smarter bike networks with a control panel built for exploration, trade-offs, and decisions.</p>
          <div class="sidebar-stat-row">
            <div class="sidebar-stat">
              <span class="sidebar-stat-label">Mode</span>
              <span class="sidebar-stat-value">Planner view</span>
            </div>
            <div class="sidebar-stat">
              <span class="sidebar-stat-label">Outputs</span>
              <span class="sidebar-stat-value">Maps + report</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_section_heading(title: str, subtitle: str, help_text: str) -> None:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-section-head">
          <div>
            <p class="sidebar-section-title">{title}</p>
            <p class="sidebar-section-meta">{subtitle}</p>
          </div>
          <div class="sidebar-help" tabindex="0" aria-label="More information about {title}">
            <span class="sidebar-help-badge">?</span>
            <div class="sidebar-help-panel">{help_text}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_report_markdown(run_payload: dict[str, Any]) -> str:
    solutions = run_payload["solutions"]
    recommended_name = "Balanced" if "Balanced" in solutions else next(iter(solutions))
    recommended_solution = solutions[recommended_name]
    recommended = recommended_solution["metrics"]
    best_demand = solutions.get("Best Demand", {}).get("metrics", {})
    best_stress = solutions.get("Best Stress", {}).get("metrics", {})
    best_cost = solutions.get("Best Cost", {}).get("metrics", {})
    config = run_payload.get("config", {})
    recommended_impact = estimate_impact_metrics(float(recommended["demand"]), config)

    selected_station_rows = pd.DataFrame(recommended_solution.get("selected_stations", []))
    if not selected_station_rows.empty:
        if "trips" not in selected_station_rows.columns:
            selected_station_rows["trips"] = pd.to_numeric(
                selected_station_rows.get("Trips", 0.0),
                errors="coerce",
            ).fillna(0.0)
        if "station_name" not in selected_station_rows.columns:
            selected_station_rows["station_name"] = selected_station_rows.get("Station_Name", "Unknown station")
        if "estimated_docks" not in selected_station_rows.columns:
            selected_station_rows["estimated_docks"] = pd.to_numeric(
                selected_station_rows.get("estimated_docks", selected_station_rows.get("Estimated_Docks", 0.0)),
                errors="coerce",
            ).fillna(0.0)
        selected_station_rows = selected_station_rows.sort_values("trips", ascending=False).head(10)

    selected_links_df = pd.DataFrame(recommended_solution.get("selected_links", []))
    if not selected_links_df.empty:
        if "total_length" not in selected_links_df.columns:
            selected_links_df["total_length"] = pd.to_numeric(
                selected_links_df.get("length", 0.0),
                errors="coerce",
            ).fillna(0.0)
        if "from_station" not in selected_links_df.columns:
            selected_links_df["from_station"] = selected_links_df.get("from", "Unknown")
        if "to_station" not in selected_links_df.columns:
            selected_links_df["to_station"] = selected_links_df.get("to", "Unknown")
        if "mean_lts" not in selected_links_df.columns:
            selected_links_df["mean_lts"] = pd.to_numeric(
                selected_links_df.get("lts", 0.0),
                errors="coerce",
            ).fillna(0.0)
        selected_links_df = selected_links_df.sort_values("total_length", ascending=False).head(10)

    demand_gap = stress_reduction = cost_reduction = None
    if best_demand:
        demand_gap = float(best_demand["demand"]) - float(recommended["demand"])
        stress_reduction = float(best_demand["stress"]) - float(recommended["stress"])
        cost_reduction = float(best_demand["cost"]) - float(recommended["cost"])

    lines = [
        "# City Decision Report",
        "",
        "## Executive Summary",
        (
            f"The recommended implementation strategy is the **{recommended_name} solution**. "
            f"It covers approximately **{recommended['demand']:,.0f} trips**, using "
            f"**{int(recommended['selected_stations'])} stations** and **{int(recommended['selected_links'])} upgraded links**, "
            f"with total modeled stress of **{recommended['stress']:,.1f}** and total modeled cost of **{recommended['cost']:,.0f}**. "
            f"Using the impact assumptions below, this scenario has an estimated **{recommended_impact['Mode Shift Potential']:,.0f} shifted trips** "
            f"and **{format_emissions(recommended_impact['Emissions Reduction'])}** of emissions reduction potential for the modeled demand period."
        ),
    ]
    if demand_gap is not None and stress_reduction is not None and cost_reduction is not None:
        lines.append(
            f"Compared with the demand-maximizing alternative, this recommendation gives up only **{demand_gap:,.0f} trips** "
            f"while reducing modeled stress by **{stress_reduction:,.1f}** and reducing modeled cost by **{cost_reduction:,.0f}**."
        )
    lines.extend(
        [
            "",
            "## Recommended Actions for the City",
            "1. Approve the balanced package as the preferred near-term investment program.",
            "2. Prioritize station implementation in the highest-demand selected locations listed below.",
            "3. Prioritize corridor upgrades along the selected links identified below to secure low-stress network continuity.",
            "4. Use the high-demand and low-stress solutions as boundary cases for budget and policy negotiations, but not as the default implementation plan.",
            "",
            "## Priority Station Actions",
        ]
    )
    if not selected_station_rows.empty:
        for _, row in selected_station_rows.iterrows():
            lines.append(
                f"- **{row['station_name']}**: demand {float(row['trips']):,.0f} trips, estimated dock need {float(row['estimated_docks']):,.0f}."
            )
    else:
        lines.append("- No station summary could be derived from the current run outputs.")

    lines.extend(["", "## Priority Link Upgrade Actions"])
    if not selected_links_df.empty:
        for _, row in selected_links_df.iterrows():
            lines.append(
                f"- **{row['from_station']} to {row['to_station']}**: approximately {float(row['total_length'])/1000.0:.2f} km, "
                f"mean current stress {float(row['mean_lts']):.2f}."
            )
    else:
        lines.append("- No link summary could be derived from the current run outputs.")

    lines.extend(
        [
            "",
            "## Policy Interpretation",
            "The balanced solution is the most suitable package for decision-makers because it preserves most of the attainable demand while avoiding the extreme cost and stress associated with the demand-maximizing alternative.",
            "This means the city should emphasize targeted station deployment in high-demand areas and selective low-stress corridor upgrades, rather than pursuing network densification everywhere at once.",
            "",
            "## Climate and Mode Shift Assumptions",
            f"- Mode shift capture rate: {float(config.get('mode_shift_rate', DEFAULT_MODE_SHIFT_RATE)) * 100:.1f}% of covered demand",
            f"- Average replaced trip distance: {float(config.get('average_trip_distance_km', DEFAULT_AVERAGE_TRIP_DISTANCE_KM)):.1f} km",
            f"- Car emissions factor: {float(config.get('car_emission_factor_g_per_km', DEFAULT_CAR_EMISSION_FACTOR_G_PER_KM)):.0f} g CO2e per km",
            "",
            "## Implementation Guidance",
            "1. Deliver the selected stations and corridor upgrades as one coordinated package rather than separate projects.",
            "2. Sequence early implementation around the highest-demand stations first, then add the remaining selected corridors to complete continuity.",
            "3. Re-run the optimization when cost assumptions, budget ceilings, or policy constraints change.",
            "",
            "## Cost Assumptions Used in This Run",
            f"- Station fixed cost: {float(config.get('station_fixed_cost', 0.0)):,.0f}",
            f"- Dock or capacity unit cost: {float(config.get('dock_unit_cost', 0.0)):,.0f}",
            f"- Link upgrade cost for LTS 1: {float(config.get('link_cost_lts1_per_km', 0.0)):,.0f} per km",
            f"- Link upgrade cost for LTS 2: {float(config.get('link_cost_lts2_per_km', 0.0)):,.0f} per km",
            f"- Link upgrade cost for LTS 3: {float(config.get('link_cost_lts3_per_km', 0.0)):,.0f} per km",
            f"- Link upgrade cost for LTS 4: {float(config.get('link_cost_lts4_per_km', 0.0)):,.0f} per km",
        ]
    )
    if best_stress or best_cost or best_demand:
        lines.extend(["", "## Comparison to Other Representative Alternatives"])
        if best_stress:
            impact = estimate_impact_metrics(float(best_stress["demand"]), config)
            lines.append(
                f"- **Best Stress**: {best_stress['demand']:,.0f} trips, stress {best_stress['stress']:,.1f}, cost {best_stress['cost']:,.0f}, emissions reduction {format_emissions(impact['Emissions Reduction'])}."
            )
        if best_cost:
            impact = estimate_impact_metrics(float(best_cost["demand"]), config)
            lines.append(
                f"- **Best Cost**: {best_cost['demand']:,.0f} trips, stress {best_cost['stress']:,.1f}, cost {best_cost['cost']:,.0f}, emissions reduction {format_emissions(impact['Emissions Reduction'])}."
            )
        if best_demand:
            impact = estimate_impact_metrics(float(best_demand["demand"]), config)
            lines.append(
                f"- **Best Demand**: {best_demand['demand']:,.0f} trips, stress {best_demand['stress']:,.1f}, cost {best_demand['cost']:,.0f}, emissions reduction {format_emissions(impact['Emissions Reduction'])}."
            )
    return "\n".join(lines)


render_sidebar_brand()
backend_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")

render_sidebar_section_heading(
    "1. Data Files",
    "Bring in the inputs for this scenario.",
    "Upload the demand table for existing or candidate stations and the street network file with geometry plus LTS values. Velora uses these two files to build the study area and test upgrades.",
)
with st.sidebar.expander("Open data setup", expanded=True):
    st.markdown(
        '<p class="upload-section-copy">Upload one station file and one street network file to start a scenario.</p>',
        unsafe_allow_html=True,
    )
    stations_upload = st.file_uploader(
        "Station file",
        type=["csv", "xlsx", "xls"],
        help="Needs station name, coordinates, and trip totals. The app also accepts names like station, lat, lon, and total_trips.",
    )
    render_sidebar_upload_status(stations_upload, "Station file")
    network_upload = st.file_uploader(
        "Street network",
        type=["gpkg", "geojson", "json", "shp", "parquet"],
        help="Needs geometry and lts. Length is optional.",
    )
    render_sidebar_upload_status(network_upload, "Network file")

render_sidebar_section_heading(
    "2. Study Area Settings",
    "Control how far Velora explores nearby options.",
    "These settings decide how many alternative station locations are generated and how much surrounding network is included. Increase them to explore more of the city, or keep them tighter for faster runs.",
)
with st.sidebar.expander("Open study area controls", expanded=True):
    candidate_points_per_station = st.slider(
        "Extra station options",
        0,
        5,
        2,
        help="How many nearby station alternatives to test for each uploaded station.",
    )
    station_buffer_meters = st.slider(
        "Station search radius (m)",
        25,
        500,
        150,
        step=25,
        help="How far the model can move a station when testing alternatives.",
    )
    area_of_interest_buffer_meters = st.slider(
        "Network clip buffer (m)",
        250,
        4000,
        2000,
        step=250,
        help="How much surrounding network to include around the study area.",
    )

render_sidebar_section_heading(
    "3. Cost Assumptions",
    "Shape the investment logic behind every plan.",
    "Use this section to define station, capacity, and link-upgrade costs. The optimizer balances these values against demand coverage and stress reduction, so even small changes here can shift the recommended plan.",
)
with st.sidebar.expander("Open cost model", expanded=True):
    station_fixed_cost = st.number_input(
        "Station fixed cost",
        min_value=0.0,
        value=5000.0,
        step=500.0,
        help="Base cost for each selected station.",
    )
    dock_unit_cost = st.number_input(
        "Capacity unit cost",
        min_value=0.0,
        value=1000.0,
        step=50.0,
        help="Cost per unit of station capacity.",
    )
    st.caption("Link upgrade costs are entered per kilometer.")
    link_cost_lts1_per_km = st.number_input("Link cost for LTS 1 (per km)", min_value=0.0, value=0.0, step=10000.0)
    link_cost_lts2_per_km = st.number_input("Link cost for LTS 2 (per km)", min_value=0.0, value=5000.0, step=1000.0)
    link_cost_lts3_per_km = st.number_input("Link cost for LTS 3 (per km)", min_value=0.0, value=8000.0, step=1000.0)
    link_cost_lts4_per_km = st.number_input("Link cost for LTS 4 (per km)", min_value=0.0, value=10000.0, step=1000.0)

render_sidebar_section_heading(
    "4. Impact Assumptions",
    "Estimate climate and travel behavior effects.",
    "These assumptions translate covered bike-share demand into approximate shifted trips and avoided car emissions. They do not rerun the optimizer; they make scenario comparisons easier for stakeholders.",
)
with st.sidebar.expander("Open impact model", expanded=False):
    mode_shift_rate_pct = st.slider(
        "Mode shift capture (%)",
        min_value=0.0,
        max_value=100.0,
        value=DEFAULT_MODE_SHIFT_RATE * 100,
        step=1.0,
        help="Share of covered demand assumed to represent trips shifted from other modes.",
    )
    average_trip_distance_km = st.number_input(
        "Average shifted trip distance (km)",
        min_value=0.1,
        value=DEFAULT_AVERAGE_TRIP_DISTANCE_KM,
        step=0.1,
        help="Average length of each shifted trip used for emissions estimates.",
    )
    car_emission_factor_g_per_km = st.number_input(
        "Car emissions factor (g CO2e/km)",
        min_value=0.0,
        value=DEFAULT_CAR_EMISSION_FACTOR_G_PER_KM,
        step=5.0,
        help="Tailpipe or lifecycle factor used to estimate avoided emissions.",
    )

render_sidebar_section_heading(
    "5. Network Requirements",
    "Set the minimum scale of the final network.",
    "These thresholds act as guardrails. They tell Velora the smallest number of stations and upgraded links that still count as an acceptable solution.",
)
with st.sidebar.expander("Open network rules", expanded=True):
    station_minimum = st.number_input(
        "Minimum stations",
        min_value=1,
        value=15,
        step=1,
        help="Smallest number of stations the model must keep.",
    )
    link_minimum = st.number_input(
        "Minimum links",
        min_value=1,
        value=50,
        step=1,
        help="Smallest number of links the model must select.",
    )

render_sidebar_section_heading(
    "6. Optimization Engine",
    "Tune the search depth and repeatability.",
    "Population size and generations control how aggressively the evolutionary search explores alternatives. Use the seed to reproduce a run when you want a consistent comparison.",
)
with st.sidebar.expander("Open engine controls", expanded=True):
    population_size = st.slider(
        "Population size",
        40,
        300,
        120,
        step=20,
        help="Higher values test more options but take longer.",
    )
    generations = st.slider(
        "Generations",
        4,
        40,
        16,
        step=2,
        help="More generations usually improve results but take longer.",
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1,
        help="Keep this fixed if you want the same run again.",
    )

st.sidebar.markdown(
    '<div class="sidebar-note">Start with the default setup, run one scenario, then refine assumptions after you see the first trade-off chart and network map.</div>',
    unsafe_allow_html=True,
)
run_button = st.sidebar.button("Run Optimization", type="primary", use_container_width=True)
if run_button:
    st.session_state["station_preview_open"] = False


def get_json(url: str) -> Any:
    with urlopen(url) as response:
        return json_loads_bytes(response.read())


def json_loads_bytes(payload: bytes) -> Any:
    import json

    return json.loads(payload.decode("utf-8"))


@st.cache_data(ttl=15, show_spinner=False)
def fetch_runs(base_url: str) -> list[dict[str, Any]]:
    # Saved runs change infrequently, so a short cache keeps the sidebar responsive.
    try:
        return get_json(f"{base_url}/runs")
    except Exception:
        return []


@st.cache_data(ttl=15, show_spinner=False)
def fetch_run(base_url: str, run_id: str) -> dict[str, Any]:
    # Cache reopened runs briefly to avoid repeated backend fetches while users compare scenarios.
    return get_json(f"{base_url}/runs/{run_id}")


def fetch_job(base_url: str, job_id: str) -> dict[str, Any]:
    return get_json(f"{base_url}/jobs/{job_id}")


def post_optimize(base_url: str, station_file, network_file, form_data: dict[str, Any]) -> dict[str, Any]:
    import uuid

    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()
    for key, value in form_data.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())

    uploads = [
        ("station_file", station_file),
        ("network_file", network_file),
    ]
    for field_name, uploaded in uploads:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{uploaded.name}"\r\n'
            ).encode()
        )
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(uploaded.getvalue())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = Request(
        f"{base_url}/optimize",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=3600) as response:
            return json_loads_bytes(response.read())
    except HTTPError as exc:
        detail = exc.reason
        try:
            payload = json_loads_bytes(exc.read())
            if isinstance(payload, dict) and "detail" in payload:
                detail = payload["detail"]
        except Exception:
            pass
        raise RuntimeError(f"Backend optimization error: {detail}") from exc


def wait_for_job(base_url: str, job_id: str) -> dict[str, Any]:
    # Poll the backend and surface progress while optimization runs asynchronously.
    progress_bar = st.progress(0, text="Job queued")
    status_box = st.empty()
    while True:
        job = fetch_job(base_url, job_id)
        message = job.get("message", "Working...")
        progress = float(job.get("progress", 0.0))
        progress_bar.progress(min(max(progress, 0.0), 1.0), text=message)
        status_box.info(
            f"Status: {job.get('status', 'unknown')} | "
            f"Step: {job.get('stage', 'n/a')} | "
            f"Progress: {progress * 100:.0f}%"
        )
        if job.get("status") == "completed":
            progress_bar.progress(1.0, text="Optimization complete")
            status_box.success("Optimization finished successfully.")
            return job["result"]
        if job.get("status") == "failed":
            progress_bar.progress(1.0, text="Optimization failed")
            raise RuntimeError(f"Backend optimization error: {job.get('error', 'Unknown error')}")
        time.sleep(1.5)


def build_solution_summary_rows(run_payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, payload in run_payload["solutions"].items():
        metrics = get_solution_display_metrics(payload, run_payload.get("config", {}))
        rows.append(
            {
                "Solution": name,
                "Demand Coverage": metrics["Demand Coverage"],
                "Average LTS": metrics["Average LTS"],
                "Total Cost": metrics["Total Cost"],
                "Mode Shift Potential": metrics["Mode Shift Potential"],
                "Emissions Reduction": metrics["Emissions Reduction"],
                "Stations": metrics["Stations"],
                "Links": metrics["Links"],
            }
        )
    return pd.DataFrame(rows)


def build_scenario_summary_rows(
    scenario_entries: dict[str, dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_id, entry in scenario_entries.items():
        metrics = get_solution_display_metrics(entry["solution"], config)
        rows.append(
            {
                "Scenario ID": scenario_id,
                "Scenario": entry["label"],
                "Type": entry["kind"],
                "Source": entry["source_label"],
                "Demand Coverage": metrics["Demand Coverage"],
                "Average LTS": metrics["Average LTS"],
                "Total Cost": metrics["Total Cost"],
                "Mode Shift Potential": metrics["Mode Shift Potential"],
                "Emissions Reduction": metrics["Emissions Reduction"],
                "Stations": metrics["Stations"],
                "Links": metrics["Links"],
                "Notes": entry.get("notes", ""),
                "Created": entry.get("created_at", ""),
            }
        )
    return pd.DataFrame(rows)


def get_impact_assumptions(config: dict[str, Any] | None = None) -> dict[str, float]:
    config = config or {}
    return {
        "mode_shift_rate": float(config.get("mode_shift_rate", DEFAULT_MODE_SHIFT_RATE)),
        "average_trip_distance_km": float(config.get("average_trip_distance_km", DEFAULT_AVERAGE_TRIP_DISTANCE_KM)),
        "car_emission_factor_g_per_km": float(
            config.get("car_emission_factor_g_per_km", DEFAULT_CAR_EMISSION_FACTOR_G_PER_KM)
        ),
    }


def estimate_impact_metrics(demand_coverage: float, config: dict[str, Any] | None = None) -> dict[str, float]:
    assumptions = get_impact_assumptions(config)
    mode_shift_trips = max(0.0, demand_coverage * assumptions["mode_shift_rate"])
    emissions_kg = (
        mode_shift_trips
        * assumptions["average_trip_distance_km"]
        * assumptions["car_emission_factor_g_per_km"]
        / 1000.0
    )
    return {
        "Mode Shift Potential": mode_shift_trips,
        "Emissions Reduction": emissions_kg,
    }


def get_solution_display_metrics(solution: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, float]:
    metrics = solution["metrics"]
    avg_lts = metrics.get("avg_lts")
    if avg_lts is None:
        avg_lts = float(metrics.get("stress", 0.0)) / max(1, int(metrics.get("selected_links", 0)))
    demand_coverage = float(metrics["demand"])
    impact_metrics = estimate_impact_metrics(demand_coverage, config)
    return {
        "Demand Coverage": demand_coverage,
        "Average LTS": float(avg_lts),
        "Total Cost": float(metrics["cost"]),
        **impact_metrics,
        "Stations": int(metrics["selected_stations"]),
        "Links": int(metrics["selected_links"]),
    }


def build_scenario_catalog(
    run_payload: dict[str, Any],
    snapshots: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for solution_name, solution in run_payload["solutions"].items():
        scenario_id = f"core::{solution_name}"
        catalog[scenario_id] = {
            "id": scenario_id,
            "label": solution_name,
            "kind": "Optimizer scenario",
            "source_label": solution_name,
            "created_at": run_payload.get("created_at", ""),
            "notes": "",
            "solution": solution,
        }
    for snapshot in snapshots:
        snapshot_id = str(snapshot.get("snapshot_id", "")).strip()
        if not snapshot_id:
            continue
        scenario_id = f"snapshot::{snapshot_id}"
        catalog[scenario_id] = {
            "id": scenario_id,
            "label": str(snapshot.get("label", "Untitled scenario")),
            "kind": "Saved stakeholder scenario",
            "source_label": str(snapshot.get("source_label", "Saved scenario")),
            "created_at": str(snapshot.get("created_at", "")),
            "notes": str(snapshot.get("notes", "")),
            "solution": snapshot.get("solution", {}),
            "snapshot_id": snapshot_id,
        }
    return catalog


def build_difference_summary(candidate_entry: dict[str, Any], baseline_entry: dict[str, Any]) -> dict[str, Any]:
    candidate_stations = pd.DataFrame(candidate_entry["solution"].get("selected_stations", []))
    baseline_stations = pd.DataFrame(baseline_entry["solution"].get("selected_stations", []))
    candidate_links = pd.DataFrame(candidate_entry["solution"].get("selected_links", []))
    baseline_links = pd.DataFrame(baseline_entry["solution"].get("selected_links", []))

    candidate_station_names = set(candidate_stations.get("station_name", candidate_stations.get("Station_Name", pd.Series(dtype=str))).astype(str))
    baseline_station_names = set(baseline_stations.get("station_name", baseline_stations.get("Station_Name", pd.Series(dtype=str))).astype(str))

    def link_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
        if df.empty:
            return set()
        from_series = df.get("from_station", df.get("from_node", pd.Series(dtype=str))).astype(str)
        to_series = df.get("to_station", df.get("to_node", pd.Series(dtype=str))).astype(str)
        return {tuple(sorted((from_name, to_name))) for from_name, to_name in zip(from_series, to_series)}

    candidate_link_keys = link_keys(candidate_links)
    baseline_link_keys = link_keys(baseline_links)

    return {
        "added_stations": sorted(candidate_station_names - baseline_station_names),
        "removed_stations": sorted(baseline_station_names - candidate_station_names),
        "added_links": sorted(candidate_link_keys - baseline_link_keys),
        "removed_links": sorted(baseline_link_keys - candidate_link_keys),
    }


def format_percent_delta(current: float, baseline: float) -> str:
    if abs(baseline) < 1e-9:
        return "n/a"
    return f"{((current - baseline) / baseline) * 100:+.1f}%"


def delta_class_from_values(current: float, baseline: float, lower_is_better: bool = False) -> str:
    diff = current - baseline
    if abs(diff) < 1e-9:
        return "neutral"
    improved = diff < 0 if lower_is_better else diff > 0
    return "positive" if improved else "negative"


def render_planning_metric_grid(metrics: dict[str, Any], baseline: dict[str, Any] | None = None) -> None:
    metric_specs = [
        ("Demand coverage", f"{metrics['Demand Coverage']:,.0f}", "covered trips", "Demand Coverage", False, False),
        ("Mode shift potential", f"{metrics['Mode Shift Potential']:,.0f}", "shifted trips", "Mode Shift Potential", False, False),
        ("Emissions reduction", format_emissions(metrics["Emissions Reduction"]), "modeled demand period", "Emissions Reduction", False, False),
        ("Total cost", format_cost(metrics["Total Cost"]), "investment estimate", "Total Cost", True, False),
        ("Average LTS", f"{metrics['Average LTS']:.2f}", "lower stress is better", "Average LTS", True, True),
    ]
    cards = []
    for label, value, note, key, lower_is_better, wide in metric_specs:
        delta_markup = ""
        if baseline:
            delta_text = (
                f"{metrics[key] - baseline[key]:+.2f}"
                if key == "Average LTS"
                else format_percent_delta(metrics[key], baseline[key])
            )
            delta_class = delta_class_from_values(metrics[key], baseline[key], lower_is_better=lower_is_better)
            delta_markup = f'<span class="delta-pill delta-pill--{delta_class}">{html.escape(delta_text)}</span>'
        wide_class = " planning-metric--wide" if wide else ""
        cards.append(
            f'<div class="planning-metric{wide_class}">'
            f'<span class="planning-metric-label">{html.escape(label)}</span>'
            f'<span class="planning-metric-value">{html.escape(value)}</span>'
            f"{delta_markup}"
            f'<span class="planning-metric-note">{html.escape(note)}</span>'
            "</div>"
        )
    st.markdown(f'<div class="planning-metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def summarize_selected_assets(solution: dict[str, Any]) -> dict[str, float]:
    links = solution.get("selected_links", [])
    total_length_km = sum(float(item.get("total_length", 0.0)) for item in links) / 1000.0
    return {
        "stations": float(len(solution.get("selected_stations", []))),
        "links": float(len(links)),
        "length_km": total_length_km,
    }


def render_asset_summary(solution: dict[str, Any], baseline_solution: dict[str, Any] | None = None) -> None:
    assets = summarize_selected_assets(solution)
    summary = (
        f"{int(assets['stations'])} selected stations, {int(assets['links'])} upgraded links, "
        f"and {assets['length_km']:.1f} km of selected corridors."
    )
    if baseline_solution:
        baseline_assets = summarize_selected_assets(baseline_solution)
        summary += (
            f" Compared with the baseline: {int(assets['stations'] - baseline_assets['stations']):+d} stations, "
            f"{int(assets['links'] - baseline_assets['links']):+d} links, "
            f"{assets['length_km'] - baseline_assets['length_km']:+.1f} km."
        )
    st.markdown(f'<div class="asset-summary">{html.escape(summary)}</div>', unsafe_allow_html=True)


def render_scenario_cards(
    scenario_entries: dict[str, dict[str, Any]],
    selected_scenario_id: str,
    config: dict[str, Any] | None = None,
) -> None:
    cards = []
    for scenario_id, entry in scenario_entries.items():
        metrics = get_solution_display_metrics(entry["solution"], config)
        active_class = " library-card--active" if scenario_id == selected_scenario_id else ""
        note = entry.get("notes") or f"{entry['kind']} from {entry['source_label']}"
        cards.append(
            f'<div class="library-card{active_class}">'
            f'<span class="library-card-title">{html.escape(entry["label"])}</span>'
            f'<span class="library-card-meta">{html.escape(entry["kind"])} · {html.escape(entry["source_label"])}</span>'
            f'<span class="library-card-note">{html.escape(str(note))}</span>'
            '<div class="library-card-metrics">'
            f"<span>{metrics['Demand Coverage']:,.0f} demand</span>"
            f"<span>{format_emissions(metrics['Emissions Reduction'])}</span>"
            f"<span>{format_cost(metrics['Total Cost'])} cost</span>"
            "</div>"
            "</div>"
        )
    st.markdown(f'<div class="scenario-library-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_comparison_delta_cards(candidate_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> None:
    specs = [
        ("Demand", candidate_metrics["Demand Coverage"], baseline_metrics["Demand Coverage"], False, "covered trips"),
        ("Mode shift", candidate_metrics["Mode Shift Potential"], baseline_metrics["Mode Shift Potential"], False, "shifted trips"),
        ("Emissions", candidate_metrics["Emissions Reduction"], baseline_metrics["Emissions Reduction"], False, "kg CO2e"),
        ("Cost", candidate_metrics["Total Cost"], baseline_metrics["Total Cost"], True, "investment"),
        ("Average LTS", candidate_metrics["Average LTS"], baseline_metrics["Average LTS"], True, "network stress"),
    ]
    cards = []
    for label, current, baseline, lower_is_better, note in specs:
        if label == "Emissions":
            value = format_emissions(current - baseline)
        elif label == "Average LTS":
            value = f"{current - baseline:+.2f}"
        else:
            value = f"{current - baseline:+,.0f}"
        direction = delta_class_from_values(current, baseline, lower_is_better=lower_is_better)
        cards.append(
            '<div class="comparison-delta-card">'
            f"<strong>{html.escape(label)}</strong>"
            f'<span class="delta-pill delta-pill--{direction}">{html.escape(value)}</span>'
            f"<span>{html.escape(note)}</span>"
            "</div>"
        )
    st.markdown(f'<div class="comparison-delta-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def build_comparison_interpretation(candidate_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> str:
    demand_delta = format_percent_delta(candidate_metrics["Demand Coverage"], baseline_metrics["Demand Coverage"])
    emissions_delta = format_percent_delta(candidate_metrics["Emissions Reduction"], baseline_metrics["Emissions Reduction"])
    cost_delta = format_percent_delta(candidate_metrics["Total Cost"], baseline_metrics["Total Cost"])
    lts_delta = candidate_metrics["Average LTS"] - baseline_metrics["Average LTS"]
    stress_phrase = "lower" if lts_delta < 0 else "higher" if lts_delta > 0 else "unchanged"
    return (
        f"This candidate changes demand by {demand_delta}, emissions reduction potential by {emissions_delta}, "
        f"and cost by {cost_delta}. Average LTS is {stress_phrase} by {abs(lts_delta):.2f}."
    )


def build_decision_insight(adjusted_solution: dict[str, Any], baseline_name: str | None = None, baseline_metrics: dict[str, Any] | None = None) -> str:
    metrics = adjusted_solution["metrics"]
    insight = (
        f"This plan activates {metrics['Stations']} stations and {metrics['Links']} upgraded links, "
        f"with an estimated mode shift potential of {metrics['Mode Shift Potential']:,.0f} trips and "
        f"{format_emissions(metrics['Emissions Reduction'])} avoided for the modeled demand period."
    )
    if baseline_name and baseline_metrics:
        demand_delta = metrics["Demand Coverage"] - baseline_metrics["Demand Coverage"]
        cost_delta = metrics["Total Cost"] - baseline_metrics["Total Cost"]
        stress_delta = metrics["Average LTS"] - baseline_metrics["Average LTS"]
        emissions_delta = metrics["Emissions Reduction"] - baseline_metrics["Emissions Reduction"]
        insight += (
            f" Compared with {baseline_name}, demand changes by {demand_delta:,.0f} "
            f"({format_percent_delta(metrics['Demand Coverage'], baseline_metrics['Demand Coverage'])}), "
            f"cost changes by {cost_delta:,.0f} "
            f"({format_percent_delta(metrics['Total Cost'], baseline_metrics['Total Cost'])}), "
            f"average LTS changes by {stress_delta:+.2f}, and emissions reduction changes by "
            f"{format_emissions(emissions_delta)}."
        )
    return insight


def classify_map_trace(trace: go.BaseTraceType) -> str:
    name = str(getattr(trace, "name", "") or "").lower()
    if "current stations" in name:
        return "current_stations"
    if "candidate options" in name:
        return "candidate_stations"
    if "selected stations" in name:
        return "selected_stations"
    if "routes with lts" in name:
        return "links"
    return "other"


def build_map_figure(
    solution: dict[str, Any],
    compare_solution: dict[str, Any] | None,
    show_stations: bool,
    show_links: bool,
    show_demand_overlay: bool,
    performance_mode: bool,
    max_lts_filter: int,
    route_length_range_km: tuple[float, float],
) -> go.Figure:
    # Start from the backend-generated scenario map and apply lightweight UI-only filters here.
    fig = go.Figure(solution["map_figure"])
    min_length_km, max_length_km = route_length_range_km

    for trace in fig.data:
        trace_kind = classify_map_trace(trace)
        if trace_kind in {"current_stations", "candidate_stations"}:
            trace.visible = show_stations and not performance_mode
        elif trace_kind == "selected_stations":
            trace.visible = show_stations
        elif trace_kind == "links":
            meta = getattr(trace, "meta", None)
            if isinstance(meta, dict):
                lts_level = int(meta.get("lts_level", 1))
                distance_km = float(meta.get("distance_km", 0.0))
            else:
                trace_name = str(getattr(trace, "name", "") or "")
                lts_level = int(trace_name.rsplit(" ", 1)[-1]) if trace_name.rsplit(" ", 1)[-1].isdigit() else 1
                distance_km = 0.0
            trace.visible = (
                show_links and lts_level <= max_lts_filter and min_length_km <= distance_km <= max_length_km
            )

    selected_station_trace = next((trace for trace in fig.data if classify_map_trace(trace) == "selected_stations"), None)
    demand_bubble_trace = None
    if show_demand_overlay and selected_station_trace is not None:
        customdata = getattr(selected_station_trace, "customdata", None)
        station_lat_values = getattr(selected_station_trace, "lat", None)
        station_lon_values = getattr(selected_station_trace, "lon", None)
        station_lats = list(station_lat_values) if station_lat_values is not None else []
        station_lons = list(station_lon_values) if station_lon_values is not None else []
        point_count = min(len(station_lats), len(station_lons))
        station_lats = station_lats[:point_count]
        station_lons = station_lons[:point_count]
        custom_rows = list(customdata) if customdata is not None else []
        if station_lats and station_lons:
            weights = []
            for index, _lat in enumerate(station_lats):
                try:
                    row = custom_rows[index]
                    weights.append(float(row[1]) if len(row) > 1 else 1.0)
                except (IndexError, TypeError, ValueError):
                    weights.append(1.0)
            max_weight = max(weights) if weights else 1.0
            min_weight = min(weights) if weights else 0.0
            weight_span = max(max_weight - min_weight, 1.0)
            bubble_sizes = [18 + ((weight - min_weight) / weight_span) * 34 for weight in weights]
            demand_bubble_trace = go.Scattermapbox(
                lat=station_lats,
                lon=station_lons,
                mode="markers",
                marker=dict(
                    size=bubble_sizes,
                    color="rgba(20, 163, 137, 0.28)",
                    opacity=0.82,
                ),
                customdata=custom_rows if len(custom_rows) == len(station_lats) else None,
                hovertemplate=(
                    "<b>Demand bubble</b><br>"
                    "Station: %{customdata[0]}<br>"
                    "Trips: %{customdata[1]:,.0f}<extra></extra>"
                    if len(custom_rows) == len(station_lats)
                    else "<b>Demand bubble</b><extra></extra>"
                ),
                showlegend=True,
                name="Demand bubbles",
            )

    if compare_solution is not None:
        compare_fig = go.Figure(compare_solution["map_figure"])
        for trace in compare_fig.data:
            trace_kind = classify_map_trace(trace)
            if trace_kind == "selected_stations":
                trace.name = "Comparison stations"
                trace.marker = dict(size=11, color="#6d28d9", opacity=0.6)
                trace.textfont = dict(color="#6d28d9")
                trace.visible = show_stations
                fig.add_trace(trace)
            elif trace_kind == "links":
                meta = getattr(trace, "meta", None)
                if isinstance(meta, dict):
                    lts_level = int(meta.get("lts_level", 1))
                    distance_km = float(meta.get("distance_km", 0.0))
                else:
                    lts_level = 1
                    distance_km = 0.0
                trace.name = "Comparison links"
                trace.line = dict(width=4, color="#6d28d9")
                trace.opacity = 0.38
                trace.visible = (
                    show_links and lts_level <= max_lts_filter and min_length_km <= distance_km <= max_length_km
                )
                fig.add_trace(trace)

    if demand_bubble_trace is not None:
        # Draw after the network traces so the checkbox produces an obvious visual change.
        fig.add_trace(demand_bubble_trace)

    fig.update_layout(
        title=None,
        height=660 if performance_mode else 720,
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(style="carto-positron", zoom=11.8),
        legend=dict(
            orientation="h",
            y=0.01,
            x=0.01,
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="rgba(17,24,39,0.10)",
            borderwidth=1,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="rgba(11,53,82,0.94)",
            bordercolor="rgba(67,184,163,0.9)",
            font=dict(color="#f8fbfd", size=13),
        ),
    )
    return fig


def render_kpis(adjusted_solution: dict[str, Any], compare_metrics: dict[str, Any] | None = None) -> None:
    metrics = adjusted_solution["metrics"]
    baseline = adjusted_solution.get("baseline")
    baseline_label = adjusted_solution.get("baseline_label")
    st.markdown("### Decision snapshot")
    if baseline and baseline_label:
        st.caption(f"Delta values are shown against {baseline_label}.")
    else:
        st.caption("This snapshot summarizes the currently selected scenario.")
    render_planning_metric_grid(metrics, baseline)
    if compare_metrics is not None:
        st.caption(
            f"Against the comparison scenario: demand {metrics['Demand Coverage'] - compare_metrics['Demand Coverage']:,.0f}, "
            f"cost {metrics['Total Cost'] - compare_metrics['Total Cost']:,.0f}, "
            f"emissions {format_emissions(metrics['Emissions Reduction'] - compare_metrics['Emissions Reduction'])}, "
            f"LTS {metrics['Average LTS'] - compare_metrics['Average LTS']:+.2f}."
        )


def render_pareto(run_payload: dict[str, Any], selected_solution_name: str) -> str:
    summary_df = build_solution_summary_rows(run_payload)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=summary_df["Total Cost"],
                y=summary_df["Demand Coverage"],
                mode="markers+text",
                text=summary_df["Solution"],
                textposition="top center",
                marker=dict(
                    size=18,
                    color=summary_df["Average LTS"],
                    colorscale=[
                        [0.0, "#2ca25f"],
                        [0.35, "#f2c14e"],
                        [0.7, "#f28e2b"],
                        [1.0, "#d1495b"],
                    ],
                    showscale=True,
                    colorbar=dict(title="Avg LTS"),
                    line=dict(
                        width=[3 if name == selected_solution_name else 1.4 for name in summary_df["Solution"]],
                        color=["#0b3552" if name == selected_solution_name else "#ffffff" for name in summary_df["Solution"]],
                    ),
                ),
                customdata=np.stack(
                    [
                        summary_df["Solution"],
                        summary_df["Mode Shift Potential"],
                        summary_df["Emissions Reduction"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Cost: %{x:,.0f}<br>"
                    "Demand: %{y:,.0f}<br>"
                    "Mode shift: %{customdata[1]:,.0f} trips<br>"
                    "Emissions reduction: %{customdata[2]:,.0f} kg CO2e<br>"
                    "Average LTS: %{marker.color:.2f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=None,
        clickmode="event+select",
        xaxis_title="Total cost",
        yaxis_title="Demand coverage",
        margin=dict(l=8, r=8, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.98)",
        font=dict(color="#173042"),
        xaxis=dict(
            gridcolor="rgba(11,53,82,0.10)",
            zerolinecolor="rgba(11,53,82,0.10)",
        ),
        yaxis=dict(
            gridcolor="rgba(11,53,82,0.10)",
            zerolinecolor="rgba(11,53,82,0.10)",
        ),
    )
    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"pareto_summary_{run_payload['run_id']}",
        on_select="rerun",
    )
    if selection and getattr(selection, "selection", None):
        points = selection.selection.get("points", [])
        if points:
            point_index = int(points[0]["point_index"])
            return str(summary_df.iloc[point_index]["Solution"])
    return selected_solution_name


def render_map(
    run_id: str,
    selected_entry: dict[str, Any],
    compare_entry: dict[str, Any] | None,
    show_stations: bool,
    show_links: bool,
    show_demand_overlay: bool,
    performance_mode: bool,
    max_lts_filter: int,
    route_length_range_km: tuple[float, float],
) -> None:
    solution = selected_entry["solution"]
    compare_solution = compare_entry["solution"] if compare_entry else None
    assets = summarize_selected_assets(solution)
    st.markdown(
        '<div class="map-mini-stats">'
        '<div class="map-mini-stat"><span class="map-mini-label">Selected stations</span>'
        f'<span class="map-mini-value">{int(assets["stations"])}</span></div>'
        '<div class="map-mini-stat"><span class="map-mini-label">Upgraded links</span>'
        f'<span class="map-mini-value">{int(assets["links"])}</span></div>'
        '<div class="map-mini-stat"><span class="map-mini-label">Corridor length</span>'
        f'<span class="map-mini-value">{assets["length_km"]:.1f} km</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    map_fig = build_map_figure(
        solution,
        compare_solution,
        show_stations=show_stations,
        show_links=show_links,
        show_demand_overlay=show_demand_overlay,
        performance_mode=performance_mode,
        max_lts_filter=max_lts_filter,
        route_length_range_km=route_length_range_km,
    )
    st.plotly_chart(
        map_fig,
        use_container_width=True,
        key=f"decision_map_{run_id}_{selected_entry['id']}_{compare_entry['id'] if compare_entry else 'none'}",
    )
    if performance_mode:
        st.caption("Performance mode is showing selected scenario assets first. Turn it off when you need wider station context.")
    else:
        st.caption("Hover the map to inspect station demand and docks, or route distance and previous LTS for each upgraded corridor.")
    station_layer_notes: list[str] = []
    for trace in map_fig.data:
        meta = getattr(trace, "meta", None)
        if isinstance(meta, dict) and meta.get("trace_type") == "station_layer" and meta.get("sampled"):
            layer_name = str(meta.get("layer_name", "stations")).replace("_", " ")
            rendered_points = int(meta.get("rendered_points", 0))
            station_layer_notes.append(f"{layer_name}: showing {rendered_points} highest-demand points")
    if station_layer_notes:
        st.caption("For performance, large background station layers are sampled on the map: " + " | ".join(station_layer_notes) + ".")


def render_scenario_studio(
    run_id: str,
    selected_entry: dict[str, Any],
    scenario_entries: dict[str, dict[str, Any]],
    built_in_solution_names: list[str],
) -> None:
    open_panel("action-shell")
    render_panel_header(
        "Scenario studio",
        "Save, annotate, and manage stakeholder-ready planning alternatives.",
    )
    with st.expander("Save or manage stakeholder scenarios", expanded=False):
        studio_cols = st.columns([1.1, 1.6])
        with studio_cols[0]:
            scenario_label = st.text_input(
                "Scenario name",
                value=st.session_state.get(f"scenario_label_{run_id}", selected_entry["label"]),
                key=f"scenario_label_input_{run_id}",
                help="Use a clear planning name like Safety First, East-West Focus, or Low Budget Alternative.",
            )
            scenario_notes = st.text_area(
                "Scenario notes",
                value=st.session_state.get(f"scenario_notes_{run_id}", ""),
                key=f"scenario_notes_input_{run_id}",
                height=110,
                help="Capture why this scenario matters, what assumptions it reflects, or who requested it.",
            )
            if st.button("Save Active Scenario", type="primary", use_container_width=True, key=f"save_scenario_{run_id}"):
                clean_label = scenario_label.strip()
                if not clean_label:
                    st.warning("Provide a scenario name before saving.")
                else:
                    save_scenario_snapshot(
                        run_id,
                        {
                            "snapshot_id": uuid.uuid4().hex[:10],
                            "label": clean_label,
                            "notes": scenario_notes.strip(),
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "source_label": selected_entry["label"],
                            "solution": selected_entry["solution"],
                        },
                    )
                    st.session_state[f"scenario_label_{run_id}"] = ""
                    st.session_state[f"scenario_notes_{run_id}"] = ""
                    st.rerun()
        with studio_cols[1]:
            saved_snapshot_rows = [
                {
                    "Scenario": entry["label"],
                    "Source": entry["source_label"],
                    "Created": entry["created_at"],
                    "Notes": entry.get("notes", ""),
                }
                for entry in scenario_entries.values()
                if entry["kind"] == "Saved stakeholder scenario"
            ]
            if saved_snapshot_rows:
                st.dataframe(pd.DataFrame(saved_snapshot_rows), use_container_width=True, hide_index=True)
                snapshot_options = [
                    entry["id"]
                    for entry in scenario_entries.values()
                    if entry["kind"] == "Saved stakeholder scenario"
                ]
                snapshot_to_delete = st.selectbox(
                    "Delete saved scenario",
                    snapshot_options,
                    format_func=lambda item: scenario_entries[item]["label"],
                    key=f"snapshot_delete_select_{run_id}",
                )
                if st.button("Delete Saved Scenario", use_container_width=True, key=f"delete_snapshot_{run_id}"):
                    delete_scenario_snapshot(run_id, scenario_entries[snapshot_to_delete]["snapshot_id"])
                    if st.session_state.get(f"selected_scenario_id_{run_id}") == snapshot_to_delete:
                        st.session_state[f"selected_scenario_id_{run_id}"] = f"core::{built_in_solution_names[0]}"
                    st.rerun()
            else:
                st.markdown(
                    '<div class="empty-state">No stakeholder scenarios saved yet. Save one from the active scenario to build a reusable planning library.</div>',
                    unsafe_allow_html=True,
                )
    close_panel()


def render_run(run_payload: dict[str, Any]) -> None:
    run_id = run_payload["run_id"]
    built_in_solution_names = list(run_payload["solutions"].keys())
    snapshots = list_scenario_snapshots(run_id)
    scenario_entries = build_scenario_catalog(run_payload, snapshots)
    scenario_ids = list(scenario_entries.keys())
    default_scenario_id = st.session_state.get(
        f"selected_scenario_id_{run_id}",
        f"core::{'Balanced' if 'Balanced' in built_in_solution_names else built_in_solution_names[0]}",
    )
    if default_scenario_id not in scenario_entries:
        default_scenario_id = scenario_ids[0]

    open_panel("control-shell")
    render_workspace_rail("Compare")
    render_panel_header(
        "Planning workspace",
        "Choose the active plan, set a baseline, and keep map filters close to the decision surface.",
    )
    top_controls = st.columns([1.45, 0.85, 1.15, 0.85])
    with top_controls[0]:
        selected_scenario_id = st.selectbox(
            "Active scenario",
            scenario_ids,
            index=scenario_ids.index(default_scenario_id),
            format_func=lambda item: scenario_entries[item]["label"],
            key=f"active_scenario_select_{run_id}",
        )
    with top_controls[1]:
        compare_mode = st.checkbox("Enable comparison", key=f"compare_mode_{run_id}")
    compare_scenario_id = None
    if compare_mode:
        with top_controls[2]:
            compare_options = [scenario_id for scenario_id in scenario_ids if scenario_id != selected_scenario_id]
            compare_scenario_id = st.selectbox(
                "Baseline scenario",
                compare_options,
                index=0,
                format_func=lambda item: scenario_entries[item]["label"],
                key=f"compare_scenario_select_{run_id}",
            )
    with top_controls[3]:
        max_lts_filter = st.slider(
            "Max LTS",
            min_value=1,
            max_value=4,
            value=4,
            step=1,
            key=f"max_lts_filter_{run_id}",
        )

    selected_entry = scenario_entries[selected_scenario_id]
    compare_entry = scenario_entries.get(compare_scenario_id) if compare_scenario_id else None
    run_config = run_payload.get("config", {})
    compare_metrics = get_solution_display_metrics(compare_entry["solution"], run_config) if compare_entry else None
    baseline_metrics = None
    baseline_label = None
    if compare_entry:
        baseline_metrics = compare_metrics
        baseline_label = compare_entry["label"]
    elif "Balanced" in run_payload["solutions"] and selected_entry["label"] != "Balanced":
        baseline_metrics = get_solution_display_metrics(run_payload["solutions"]["Balanced"], run_config)
        baseline_label = "Balanced"

    filter_row = st.columns([0.9, 0.9, 1.1, 1.1, 1.35])
    with filter_row[0]:
        show_stations = st.checkbox("Show stations", value=True, key=f"show_stations_{run_id}")
    with filter_row[1]:
        show_links = st.checkbox("Show links", value=True, key=f"show_links_{run_id}")
    with filter_row[2]:
        performance_mode = st.checkbox(
            "Performance mode",
            value=True,
            key=f"performance_mode_{run_id}",
            help="Keeps the map focused on selected assets and hides large context layers.",
        )
    with filter_row[3]:
        show_demand_overlay = st.checkbox(
            "Show demand bubbles",
            value=True,
            key=f"show_demand_overlay_{run_id}",
            help="Displays a lightweight demand-size bubble around each selected station.",
        )
    selected_links_rows = selected_entry["solution"].get("selected_links", [])
    max_lane_length_km = max(
        [float(item.get("total_length", 0.0)) / 1000.0 for item in selected_links_rows],
        default=0.0,
    )
    with filter_row[4]:
        if max_lane_length_km > 0:
            route_length_range_km = st.slider(
                "Route length (km)",
                min_value=0.0,
                max_value=max_lane_length_km,
                value=(0.0, max_lane_length_km),
                step=max(0.1, max_lane_length_km / 20),
                key=f"route_length_range_{run_id}",
            )
        else:
            route_length_range_km = (0.0, 0.0)
            st.caption("Route length filter becomes available once route geometry is present.")
    baseline_caption = compare_entry["label"] if compare_entry else (baseline_label or "No explicit baseline")
    impact_assumptions = get_impact_assumptions(run_config)
    selected_label_html = html.escape(str(selected_entry["label"]))
    selected_kind_html = html.escape(str(selected_entry["kind"]))
    selected_source_html = html.escape(str(selected_entry["source_label"]))
    baseline_caption_html = html.escape(str(baseline_caption))
    st.markdown(
        f"""
        <div class="scenario-strip">
          <div class="scenario-card scenario-card--accent">
            <span class="scenario-kicker">Active Scenario</span>
            <span class="scenario-name">{selected_label_html}</span>
            <div class="scenario-meta">{selected_kind_html} · Source: {selected_source_html}</div>
          </div>
          <div class="scenario-card scenario-card--comparison">
            <span class="scenario-kicker">Baseline</span>
            <span class="scenario-name">{baseline_caption_html}</span>
            <div class="scenario-meta">Used for KPI deltas and comparison workspace summaries.</div>
          </div>
          <div class="scenario-card">
            <span class="scenario-kicker">Map Filters</span>
            <span class="scenario-name">LTS ≤ {max_lts_filter}</span>
            <div class="scenario-meta">Route range: {route_length_range_km[0]:.1f} to {route_length_range_km[1]:.1f} km · {'Performance mode' if performance_mode else 'Full context'}</div>
          </div>
          <div class="scenario-card">
            <span class="scenario-kicker">Impact Model</span>
            <span class="scenario-name">{impact_assumptions['mode_shift_rate'] * 100:.0f}% shift capture</span>
            <div class="scenario-meta">{impact_assumptions['average_trip_distance_km']:.1f} km/trip · {impact_assumptions['car_emission_factor_g_per_km']:.0f} g CO2e/km</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    close_panel()

    selected_metrics = get_solution_display_metrics(selected_entry["solution"], run_config)
    snapshot_payload = {
        "metrics": selected_metrics,
        "baseline": baseline_metrics,
        "baseline_label": baseline_label,
    }

    st.session_state[f"selected_scenario_id_{run_id}"] = selected_scenario_id

    main_cols = st.columns([1.9, 1], gap="large")
    with main_cols[0]:
        open_panel()
        render_panel_header(
            "Interactive decision map",
            "Use the map as the main decision surface. Hover stations and corridor upgrades to inspect demand, docks, distance, and stress.",
        )
        render_map(
            run_id,
            selected_entry,
            compare_entry,
            show_stations=show_stations,
            show_links=show_links,
            show_demand_overlay=show_demand_overlay,
            performance_mode=performance_mode,
            max_lts_filter=max_lts_filter,
            route_length_range_km=route_length_range_km,
        )
        close_panel()
    with main_cols[1]:
        open_panel("insight-shell")
        render_kpis(snapshot_payload, compare_metrics=compare_metrics)
        baseline_solution = compare_entry["solution"] if compare_entry else (
            run_payload["solutions"].get("Balanced") if baseline_label == "Balanced" else None
        )
        render_asset_summary(selected_entry["solution"], baseline_solution)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Why this solution?")
        st.info(build_decision_insight(snapshot_payload, baseline_label, baseline_metrics))
        if selected_entry.get("notes"):
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Scenario notes")
            st.caption(selected_entry["notes"])
        close_panel()

    render_scenario_studio(run_id, selected_entry, scenario_entries, built_in_solution_names)

    open_panel()
    render_panel_header(
        "Trade-off explorer",
        "Click a point to switch the active optimizer scenario. Saved stakeholder scenarios still remain available in Scenario Studio.",
    )
    highlighted_solution_name = (
        selected_entry["source_label"]
        if selected_entry["kind"] == "Saved stakeholder scenario" and selected_entry["source_label"] in run_payload["solutions"]
        else (selected_entry["label"] if selected_entry["label"] in run_payload["solutions"] else built_in_solution_names[0])
    )
    pareto_selected_solution_name = render_pareto(run_payload, highlighted_solution_name)
    close_panel()
    if pareto_selected_solution_name != highlighted_solution_name:
        st.session_state[f"selected_scenario_id_{run_id}"] = f"core::{pareto_selected_solution_name}"
        st.rerun()

    scenario_summary_df = build_scenario_summary_rows(scenario_entries, run_config)
    bottom_cols = st.columns([1.25, 1.05], gap="large")
    with bottom_cols[0]:
        open_panel("comparison-shell")
        render_panel_header(
            "Scenario library",
            "Review optimizer scenarios and saved stakeholder alternatives as reusable planning options.",
        )
        render_scenario_cards(scenario_entries, selected_scenario_id, run_config)
        comparison_df = scenario_summary_df.copy()
        comparison_df["Cost / Demand"] = comparison_df["Total Cost"] / comparison_df["Demand Coverage"].replace(0, np.nan)
        with st.expander("Open detailed scenario table", expanded=False):
            st.dataframe(
                comparison_df[
                    [
                        "Scenario",
                        "Type",
                        "Source",
                        "Demand Coverage",
                        "Mode Shift Potential",
                        "Emissions Reduction",
                        "Average LTS",
                        "Total Cost",
                        "Stations",
                        "Links",
                        "Notes",
                    ]
                ].style.format(
                    {
                        "Demand Coverage": "{:,.0f}",
                        "Mode Shift Potential": "{:,.0f}",
                        "Emissions Reduction": "{:,.0f}",
                        "Average LTS": "{:.2f}",
                        "Total Cost": "{:,.0f}",
                        "Cost / Demand": "{:,.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        close_panel()
    with bottom_cols[1]:
        open_panel("comparison-shell")
        render_panel_header(
            "Comparison workspace",
            "Use the active scenario as the candidate and compare it against the chosen baseline.",
        )
        if compare_entry:
            render_comparison_delta_cards(selected_metrics, compare_metrics)
            st.info(build_comparison_interpretation(selected_metrics, compare_metrics))
            compare_rows = scenario_summary_df[scenario_summary_df["Scenario ID"].isin([selected_scenario_id, compare_scenario_id])].copy()
            with st.expander("Open side-by-side values", expanded=False):
                st.dataframe(
                    compare_rows[
                        [
                            "Scenario",
                            "Demand Coverage",
                            "Mode Shift Potential",
                            "Emissions Reduction",
                            "Average LTS",
                            "Total Cost",
                            "Stations",
                            "Links",
                        ]
                    ].style.format(
                        {
                            "Demand Coverage": "{:,.0f}",
                            "Mode Shift Potential": "{:,.0f}",
                            "Emissions Reduction": "{:,.0f}",
                            "Average LTS": "{:.2f}",
                            "Total Cost": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            diff_summary = build_difference_summary(selected_entry, compare_entry)
            st.markdown("### What changed")
            diff_cols = st.columns(2)
            diff_cols[0].metric("Added stations", len(diff_summary["added_stations"]))
            diff_cols[1].metric("Removed stations", len(diff_summary["removed_stations"]))
            diff_cols = st.columns(2)
            diff_cols[0].metric("Added links", len(diff_summary["added_links"]))
            diff_cols[1].metric("Removed links", len(diff_summary["removed_links"]))
            if diff_summary["added_stations"]:
                st.caption("Added stations: " + ", ".join(diff_summary["added_stations"][:6]))
            if diff_summary["removed_stations"]:
                st.caption("Removed stations: " + ", ".join(diff_summary["removed_stations"][:6]))
            if diff_summary["added_links"]:
                st.caption(
                    "Added links: "
                    + "; ".join([f"{edge[0]} - {edge[1]}" for edge in diff_summary["added_links"][:4]])
                )
            if diff_summary["removed_links"]:
                st.caption(
                    "Removed links: "
                    + "; ".join([f"{edge[0]} - {edge[1]}" for edge in diff_summary["removed_links"][:4]])
                )
        else:
            st.markdown(
                '<div class="empty-state">Enable comparison and choose a baseline scenario to activate the comparison workspace.</div>',
                unsafe_allow_html=True,
            )
        close_panel()

    render_run_details(run_payload)

    selected_label_slug = selected_entry["label"].lower().replace(" ", "_")
    stations_csv = pd.DataFrame(selected_entry["solution"].get("selected_stations", [])).to_csv(index=False).encode("utf-8")
    links_csv = pd.DataFrame(selected_entry["solution"].get("selected_links", [])).to_csv(index=False).encode("utf-8")
    open_panel("action-shell")
    render_panel_header(
        "Export selected scenario",
        "Download the current scenario assets for external review, analysis, or reporting.",
    )
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download selected stations",
        data=stations_csv,
        file_name=f"{selected_label_slug}_stations.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"stations_download_{run_id}_{selected_label_slug}",
    )
    download_cols[1].download_button(
        "Download selected links",
        data=links_csv,
        file_name=f"{selected_label_slug}_links.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"links_download_{run_id}_{selected_label_slug}",
    )
    close_panel()

    report_key = f"report_markdown_{run_id}"
    trigger_key = f"report_generated_{run_id}"
    open_panel("action-shell")
    render_panel_header(
        "Decision report",
        "Generate the narrative report only when you are ready to package the current run into a decision summary.",
    )
    if st.button(
        "Generate Decision Report",
        type="primary",
        use_container_width=True,
        key=f"generate_report_{run_id}",
    ):
        st.session_state[report_key] = generate_report_markdown(run_payload)
        st.session_state[trigger_key] = True
    if st.session_state.get(trigger_key):
        report_markdown = st.session_state.get(report_key, "")
        st.markdown('<div class="report-shell">', unsafe_allow_html=True)
        st.markdown(report_markdown or "_Report generation did not produce any content._")
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "Download decision report",
            data=report_markdown.encode("utf-8"),
            file_name=f"decision_report_{run_id}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"report_download_{run_id}",
        )
    close_panel()


runs = fetch_runs(backend_url.rstrip("/"))
selected_saved_run = None
if runs:
    render_sidebar_section_heading(
        "7. Saved Runs",
        "Jump back into previous scenarios.",
        "Saved runs let you reopen past optimization results without rerunning the model. Use them to compare scenarios, revisit assumptions, or export a previously generated plan.",
    )
    labels = [f"{run['run_id']} | {run['created_at']}" for run in runs]
    selected_label = st.sidebar.selectbox("Saved runs", ["None"] + labels)
    if selected_label != "None":
        selected_saved_run = selected_label.split(" | ")[0]
    delete_cols = st.sidebar.columns(2)
    if delete_cols[0].button("Delete run", use_container_width=True):
        if selected_saved_run is not None:
            delete_run(selected_saved_run)
            fetch_runs.clear()
            fetch_run.clear()
            st.session_state.pop(f"selected_solution_name_{selected_saved_run}", None)
            st.session_state.pop(f"selected_scenario_id_{selected_saved_run}", None)
            st.rerun()
    confirm_delete_all = st.sidebar.checkbox(
        "Confirm delete all runs",
        value=False,
        help="Turn this on before removing the entire saved run history.",
    )
    if delete_cols[1].button("Clear all", use_container_width=True):
        if not confirm_delete_all:
            st.sidebar.warning("Enable confirmation first to delete every saved run.")
        else:
            delete_all_runs()
            fetch_runs.clear()
            fetch_run.clear()
            st.session_state.pop("selected_saved_run", None)
            st.rerun()

if stations_upload is None or network_upload is None:
    st.info(
        "Upload a station file and a street network to begin."
    )
else:
    render_station_preview(stations_upload)

if run_button:
    if stations_upload is None or network_upload is None:
        st.error("Please upload both files first.")
        st.stop()

    with st.spinner("Running optimization and generating visualizations..."):
        try:
            job_response = post_optimize(
                backend_url.rstrip("/"),
                stations_upload,
                network_upload,
                {
                    "candidate_points_per_station": candidate_points_per_station,
                    "station_buffer_meters": float(station_buffer_meters),
                    "area_of_interest_buffer_meters": float(area_of_interest_buffer_meters),
                    "station_minimum": int(station_minimum),
                    "link_minimum": int(link_minimum),
                    "population_size": int(population_size),
                    "generations": int(generations),
                    "seed": int(seed),
                    "dock_unit_cost": float(dock_unit_cost),
                    "station_fixed_cost": float(station_fixed_cost),
                    "link_cost_lts1_per_km": float(link_cost_lts1_per_km),
                    "link_cost_lts2_per_km": float(link_cost_lts2_per_km),
                    "link_cost_lts3_per_km": float(link_cost_lts3_per_km),
                    "link_cost_lts4_per_km": float(link_cost_lts4_per_km),
                    "mode_shift_rate": float(mode_shift_rate_pct) / 100.0,
                    "average_trip_distance_km": float(average_trip_distance_km),
                    "car_emission_factor_g_per_km": float(car_emission_factor_g_per_km),
                },
            )
            run_payload = wait_for_job(backend_url.rstrip("/"), job_response["job_id"])
        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.success("Optimization complete.")
    render_run(run_payload)
elif selected_saved_run:
    try:
        render_run(fetch_run(backend_url.rstrip("/"), selected_saved_run))
    except Exception as exc:
        st.exception(exc)
