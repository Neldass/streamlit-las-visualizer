import streamlit as st
import numpy as np

from src.las_utils import list_numeric_logs, subset_depth
from src.plot_utils import (
    plot_logs_custom_tracks,
)

st.set_page_config(page_title="Visualiser les logs", page_icon="📈", layout="wide")

st.title("📈 Visualiser les logs")

if "datasets" not in st.session_state or not st.session_state["datasets"]:
    st.warning("Veuillez d'abord importer des fichiers LAS dans la page précédente.")
    st.stop()

datasets = st.session_state["datasets"]
names = list(datasets.keys())
name = st.selectbox(
    "Choisir un puits",
    names,
    index=0,
    format_func=lambda n: datasets[n].get("display_name", n),
)

selected = st.session_state["datasets"][name]
df = selected["df"].copy()

# Determine units map from curves info
units = {c.name: (c.unit or None) for c in selected["curves"]}

numeric_logs = list_numeric_logs(df, exclude=["DEPTH"])

with st.sidebar:
    st.header("Paramètres")
    depth_min = float(df["DEPTH"].min()) if not df.empty else 0.0
    depth_max = float(df["DEPTH"].max()) if not df.empty else 1.0
    m, M = st.slider(
        "Intervalle de profondeur",
        min_value=float(np.floor(depth_min)),
        max_value=float(np.ceil(depth_max)),
        value=(float(depth_min), float(depth_max)),
        step=0.1,
    )

# Filter by depth
view = subset_depth(df, m, M)

display_name = selected.get("display_name", name)

st.subheader("Configurer manuellement les tracks")
n_tracks = st.number_input("Nombre de tracks", min_value=1, max_value=8, value=min(3, len(numeric_logs)))
track_groups = []
for i in range(int(n_tracks)):
    default_i = [numeric_logs[i]] if i < len(numeric_logs) else []
    sel = st.multiselect(f"Logs du track {i+1}", options=numeric_logs, default=default_i, key=f"track_{i}")
    track_groups.append(sel)

# Vérifier qu'au moins un log est sélectionné sur l'ensemble des tracks
chosen = sorted({c for grp in track_groups for c in grp})
if not chosen:
    st.info("Sélectionnez au moins un log dans les tracks ci-dessus.")
else:
    fig_custom = plot_logs_custom_tracks(view, track_groups, depth_col="DEPTH", units=units, title=f"Tracks - {display_name}")
    st.plotly_chart(fig_custom, config={"displaylogo": False, "responsive": True})
