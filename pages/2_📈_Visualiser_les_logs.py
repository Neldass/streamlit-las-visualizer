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
    st.subheader("Mise en page")
    track_width = st.slider("Largeur par track (px)", min_value=120, max_value=420, value=270, step=10)
    fig_height = st.slider("Hauteur du graphique (px)", min_value=400, max_value=1200, value=750, step=50)
    trim_depth = st.checkbox(
        "Raccourcir la profondeur aux valeurs présentes", value=True,
        help="Commencer/terminer chaque track là où il y a des valeurs (ignorer les zones entièrement vides)."
    )

# Filter by depth
view = subset_depth(df, m, M)

display_name = selected.get("display_name", name)

st.subheader("Configurer manuellement les tracks")
n_tracks = st.number_input("Nombre de tracks", min_value=1, max_value=8, value=min(1, len(numeric_logs)))
track_groups = []
for i in range(int(n_tracks)):
    default_i = [numeric_logs[i]] if i < len(numeric_logs) else []
    sel = st.multiselect(f"Logs du track {i+1}", options=numeric_logs, default=default_i, key=f"track_{i}")
    track_groups.append(sel)

# Vérifier qu'au moins un log est sélectionné sur l'ensemble des tracks
chosen = sorted({c for grp in track_groups for c in grp})

# Couleurs personnalisées (expander) si au moins un log choisi
color_map = {}
if chosen:
    with st.expander("Couleurs personnalisées"):
        st.caption("Définissez une couleur par log (facultatif).")
        if "logs_colors" not in st.session_state:
            st.session_state["logs_colors"] = {}
        for log in chosen:
            default_color = st.session_state["logs_colors"].get(log, None)
            # Default palette fallback cycle
            if default_color is None:
                palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                idx = chosen.index(log) % len(palette)
                default_color = palette[idx]
            picked = st.color_picker(f"{log}", default_color, key=f"color_{log}")
            st.session_state["logs_colors"][log] = picked
            color_map[log] = picked

# Echelles X par track (manuelles) 
x_ranges: list | None = None
if any(track_groups):
    with st.expander("Échelle X par track (optionnel)"):
        st.caption("Définissez des min/max d'axe X par track. Laissez vide pour 'Auto'.")
        x_ranges = []
        for i, logs in enumerate(track_groups, start=1):
            if not logs:
                x_ranges.append(None)
                continue
            # Calcule des bornes suggérées sur la vue filtrée (sans dépendre de pandas ici)
            vals = []
            for l in logs:
                if l in view.columns:
                    arr = np.asarray(view[l].to_numpy(dtype=float))
                    if arr.size:
                        vals.append(arr)
            if vals:
                joined = np.concatenate(vals)
                finite = joined[np.isfinite(joined)]
                if finite.size:
                    sugg_min = float(np.min(finite))
                    sugg_max = float(np.max(finite))
                else:
                    sugg_min = sugg_max = None
            else:
                sugg_min = sugg_max = None
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                xmin = st.text_input(f"Track {i} • X min", value=(f"{sugg_min:.3f}" if sugg_min is not None else ""), key=f"xmin_{i}")
            with c2:
                xmax = st.text_input(f"Track {i} • X max", value=(f"{sugg_max:.3f}" if sugg_max is not None else ""), key=f"xmax_{i}")
            with c3:
                st.write("\u00A0")
                st.write("Auto si vide")
            try:
                xr = (float(xmin), float(xmax)) if xmin.strip() != "" and xmax.strip() != "" else None
            except Exception:
                xr = None
            x_ranges.append(xr)
if not chosen:
    st.info("Sélectionnez au moins un log dans les tracks ci-dessus.")
else:
    fig_custom = plot_logs_custom_tracks(
        view,
        track_groups,
        depth_col="DEPTH",
        units=units,
        title=f"Tracks - {display_name}",
        color_map=color_map if color_map else None,
        x_ranges=x_ranges,
        track_width=int(track_width),
        height=int(fig_height),
        trim_depth_gaps=trim_depth,
    )
    # If only one non-empty track, disable container width so custom width applies
    non_empty_tracks = [tg for tg in track_groups if tg]
    single = len(non_empty_tracks) == 1
    st.plotly_chart(
        fig_custom,
        use_container_width=not single,
        config={"displaylogo": False}
    )
