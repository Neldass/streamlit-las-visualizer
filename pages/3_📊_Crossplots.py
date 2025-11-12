import streamlit as st
import numpy as np
import pandas as pd

from src.las_utils import list_numeric_logs, subset_depth
from src.plot_utils import plot_crossplot

st.set_page_config(page_title="Crossplots", page_icon="📊", layout="wide")

st.title("📊 Crossplots")

if "datasets" not in st.session_state or not st.session_state["datasets"]:
    st.warning("Veuillez d'abord importer des fichiers LAS dans la page précédente.")
    st.stop()

datasets_all = st.session_state["datasets"]
names = list(datasets_all.keys())
# Allow selecting multiple wells/datasets (display well name if available)
selected_names = st.multiselect(
    "Choisir un ou plusieurs puits",
    names,
    default=names[:1],
    format_func=lambda n: datasets_all[n].get("display_name", n),
)
if not selected_names:
    st.stop()

datasets = {n: datasets_all[n] for n in selected_names}

# Compute the intersection of numeric logs present in ALL selected wells
per_well_numeric = {n: set(list_numeric_logs(d["df"], exclude=["DEPTH"])) for n, d in datasets.items()}
common_numeric = set.intersection(*per_well_numeric.values()) if per_well_numeric else set()
numeric_logs = sorted(common_numeric)
if len(numeric_logs) < 2:
    st.info("Les puits sélectionnés n'ont pas au moins deux logs numériques en commun.")
    st.stop()

with st.sidebar:
    st.header("Paramètres crossplot")
    c1, c2 = st.columns(2)
    with c1:
        x = st.selectbox("X", options=numeric_logs, index=0)
    with c2:
        y = st.selectbox("Y", options=numeric_logs, index=1 if len(numeric_logs) > 1 else 0)

    color_choice = st.selectbox(
        "Couleur (optionnel)", options=["(aucune)", "(puits)"] + numeric_logs, index=0,
        help="Colorer par puits ou par un log commun."
    )
    color = None if color_choice == "(aucune)" else ("WELL" if color_choice == "(puits)" else color_choice)

    use_density = st.checkbox("Mode densité (heatmap)", value=False, help="Utile pour de très gros nuages de points")

    # Global depth range across selected wells
    depth_min = float(min(d["df"]["DEPTH"].min() for d in datasets.values()))
    depth_max = float(max(d["df"]["DEPTH"].max() for d in datasets.values()))
    m, M = st.slider(
        "Intervalle de profondeur",
        min_value=float(np.floor(depth_min)),
        max_value=float(np.ceil(depth_max)),
        value=(float(depth_min), float(depth_max)),
        step=0.1,
    )

# Build concatenated view with WELL column
frames = []
needed_cols = {x, y} | ({color} if color and color != "WELL" else set())
for n, d in datasets.items():
    tmp = subset_depth(d["df"].copy(), m, M)
    cols = [c for c in needed_cols if c in tmp.columns]
    chunk = tmp[cols].copy()
    chunk["WELL"] = n
    frames.append(chunk)

combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[x, y])

title_suffix_raw = [datasets_all[n].get("display_name", n) for n in selected_names]
title_suffix = ", ".join(title_suffix_raw) if len(title_suffix_raw) <= 3 else f"{len(title_suffix_raw)} puits"
fig = plot_crossplot(combined, x=str(x), y=str(y), color=color, use_density=use_density, title=f"{y} vs {x} - {title_suffix}")
st.plotly_chart(fig, config={"displaylogo": False, "responsive": True})
