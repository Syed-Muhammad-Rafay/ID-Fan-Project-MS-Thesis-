from pathlib import Path

import streamlit as st

from app.utils.style import footer

PROJ_ROOT = Path(__file__).resolve().parents[1]


def run() -> None:
    st.set_page_config(
        page_title="ID Fan Failure Prediction System",
        page_icon="🔧",
        initial_sidebar_state="expanded",
        layout="wide",
    )

    home_page = st.Page(
        page="app/views/home.py",
        title="Home",
        icon="🏠",
        default=True,
    )

    predictions_page = st.Page(
        page="app/views/predictions.py",
        title="Predictions",
        icon=None,
    )

    about_page = st.Page(
        page="app/views/about.py",
        title="About",
        icon=None,
    )

    pg = st.navigation([home_page, predictions_page, about_page])
    pg.run()

    footer()


if __name__ == "__main__":
    run()
