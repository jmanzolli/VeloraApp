from __future__ import annotations

from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).parent


navigation = st.navigation(
    {
        "Start": [
            st.Page(
                ROOT_DIR / "Bike_Network_Planner.py",
                title="Bike Network Planner",
                default=True,
            ),
        ],
        "Prepare Inputs": [
            st.Page(
                ROOT_DIR / "pages" / "Population_Demand_Builder.py",
                title="Demand",
            ),
            st.Page(
                ROOT_DIR / "pages" / "LTS_Calculator.py",
                title="LTS Network Builder",
            ),
        ],
        "Test Routes": [
            st.Page(
                ROOT_DIR / "pages" / "Route_Planner.py",
                title="Route Planner",
            ),
        ],
    },
    position="sidebar",
)

navigation.run()
