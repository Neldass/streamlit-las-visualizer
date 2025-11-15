import streamlit as st
import numpy as np

from src.las_utils import list_numeric_logs, subset_depth
from src.plot_utils import plot_logs_custom_tracks

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
    st.subheader("Mise en page")
    track_width = st.slider("Largeur par track (px)", min_value=120, max_value=420, value=270, step=10)
    fig_height = st.slider("Hauteur du graphique (px)", min_value=400, max_value=1200, value=750, step=50)
    trim_depth = st.checkbox("Raccourcir la profondeur aux valeurs présentes", value=True)

view = subset_depth(df, m, M)

display_name = selected.get("display_name", name)

st.subheader("Configurer les tracks")
n_tracks = st.number_input("Nombre de tracks", min_value=1, max_value=8, value=min(1, len(numeric_logs)))
track_groups = []
for i in range(int(n_tracks)):
    default_i = [numeric_logs[i]] if i < len(numeric_logs) else []
    sel = st.multiselect(f"Logs du track {i+1}", options=numeric_logs, default=default_i, key=f"track_{i}")
    track_groups.append(sel)

chosen = sorted({c for grp in track_groups for c in grp})

color_map = {}
if chosen:
    with st.expander("Couleurs personnalisées"):
        st.caption("Définissez une couleur par log (facultatif).")
        if "logs_colors" not in st.session_state:
            st.session_state["logs_colors"] = {}
        colors_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for idx, log in enumerate(chosen):
            default_color = st.session_state["logs_colors"].get(log, colors_cycle[idx % len(colors_cycle)])
            picked = st.color_picker(f"{log}", default_color, key=f"color_{log}")
            st.session_state["logs_colors"][log] = picked
            color_map[log] = picked
if not chosen:
    st.info("Sélectionnez au moins un log dans les tracks ci-dessus.")
else:
    fig_custom = plot_logs_custom_tracks(
        view,
        track_groups,
        depth_col="DEPTH",
        units=units,
        title=f"Tracks — {display_name}",
        color_map=color_map if color_map else None,
        track_width=int(track_width),
        height=int(fig_height),
        trim_depth_gaps=trim_depth,
    )
    non_empty_tracks = [tg for tg in track_groups if tg]
    single = len(non_empty_tracks) == 1
    # Toujours respecter la largeur calculée (track_width * n). Quand plusieurs tracks,
    # use_container_width=True écrasait width -> on le désactive.
    st.plotly_chart(
        fig_custom,
        use_container_width=False,
        config={"displaylogo": False}
    )
    if not single:
        st.caption("Astuce: réduisez ou augmentez la largeur par track dans la sidebar pour ajuster l'espace horizontal.")
