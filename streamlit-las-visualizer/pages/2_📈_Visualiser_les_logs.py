import streamlit as st
import numpy as np

from src.las_utils import list_numeric_logs, subset_depth
from src.plot_utils import plot_logs_custom_tracks, plot_logs_multi_wells

st.set_page_config(page_title="Visualiser les logs", page_icon="📈", layout="wide")

st.title("📈 Visualiser les logs")

if "datasets" not in st.session_state or not st.session_state["datasets"]:
    st.warning("Veuillez d'abord importer des fichiers LAS dans la page précédente.")
    st.stop()

datasets = st.session_state["datasets"]
well_names = list(datasets.keys())
selected_wells = st.multiselect(
    "Choisir un ou plusieurs puits",
    well_names,
    default=well_names[:1],
    format_func=lambda n: datasets[n].get("display_name", n),
)
if not selected_wells:
    st.info("Sélectionnez au moins un puits.")
    st.stop()

# Déterminer les logs disponibles (communs si multi-puits)
if len(selected_wells) == 1:
    ds0 = datasets[selected_wells[0]]
    numeric_logs_available = list_numeric_logs(ds0["df"], exclude=["DEPTH"])
    units_map = {c.name: (c.unit or None) for c in ds0["curves"]}
else:
    per_well_numeric = {w: set(list_numeric_logs(datasets[w]["df"], exclude=["DEPTH"])) for w in selected_wells}
    common_logs = sorted(set.intersection(*per_well_numeric.values())) if per_well_numeric else []
    if not common_logs:
        st.info("Aucun log numérique commun entre les puits sélectionnés.")
        st.stop()
    numeric_logs_available = common_logs
    first_ds = datasets[selected_wells[0]]
    units_map = {c.name: (c.unit or None) for c in first_ds["curves"]}

with st.sidebar:
    st.header("Paramètres")
    if len(selected_wells) == 1:
        df_depth = datasets[selected_wells[0]]["df"]
        depth_min = float(df_depth["DEPTH"].min()) if not df_depth.empty else 0.0
        depth_max = float(df_depth["DEPTH"].max()) if not df_depth.empty else 1.0
    else:
        depth_min = float(min(datasets[w]["df"]["DEPTH"].min() for w in selected_wells))
        depth_max = float(max(datasets[w]["df"]["DEPTH"].max() for w in selected_wells))
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
    st.subheader("Style")
    line_width = st.slider("Épaisseur des courbes", min_value=0.5, max_value=5.0, value=1.6, step=0.1)
    opacity = st.slider("Opacité", min_value=0.2, max_value=1.0, value=1.0, step=0.05)
    st.caption("La légende est toujours affichée en bas de chaque figure.")

st.subheader("Configurer les tracks")
n_tracks = st.number_input("Nombre de tracks", min_value=1, max_value=8, value=min(1, len(numeric_logs_available)))
track_groups = []
for i in range(int(n_tracks)):
    default_i = [numeric_logs_available[i]] if i < len(numeric_logs_available) else []
    sel = st.multiselect(f"Logs du track {i+1}", options=numeric_logs_available, default=default_i, key=f"track_{i}")
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

x_ranges: list | None = None
if any(track_groups):
    with st.expander("Échelle X par track (optionnel)"):
        st.caption("Min/max X par track (vide = auto)")
        x_ranges = []
        for i, logs in enumerate(track_groups, start=1):
            if not logs:
                x_ranges.append(None)
                continue
            vals = []
            if len(selected_wells) == 1:
                df_range = subset_depth(datasets[selected_wells[0]]["df"].copy(), m, M)
                for l in logs:
                    if l in df_range.columns:
                        arr = np.asarray(df_range[l].to_numpy(dtype=float))
                        if arr.size:
                            vals.append(arr)
            else:
                for w in selected_wells:
                    df_range = subset_depth(datasets[w]["df"].copy(), m, M)
                    if not df_range.empty:
                        for l in logs:
                            if l in df_range.columns:
                                arr = np.asarray(df_range[l].to_numpy(dtype=float))
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
    if len(selected_wells) == 1:
        w = selected_wells[0]
        ds = datasets[w]
        df_w = subset_depth(ds["df"].copy(), m, M)
        fig_single = plot_logs_custom_tracks(
            df_w,
            track_groups,
            depth_col="DEPTH",
            units=units_map,
            title=f"Tracks — {ds.get('display_name', w)}",
            color_map=color_map if color_map else None,
            x_ranges=x_ranges,
            track_width=int(track_width),
            height=int(fig_height),
            trim_depth_gaps=trim_depth,
            line_width=float(line_width),
            opacity=float(opacity),
            show_legend=True,
            legend_position="bottom",
        )
        st.plotly_chart(fig_single, use_container_width=False, config={"displaylogo": False})
    else:
        # Figure unique avec tous les puits en colonnes pour respecter la largeur par track
        df_by_well = {w: subset_depth(datasets[w]["df"].copy(), m, M) for w in selected_wells}
        filtered_groups = [
            [l for l in grp if any(l in df_by_well[w].columns for w in df_by_well)] for grp in track_groups
        ]
        fig_multi = plot_logs_multi_wells(
            df_by_well=df_by_well,
            track_groups=filtered_groups,
            depth_col="DEPTH",
            units=units_map,
            title="Tracks — " + ", ".join([datasets[w].get("display_name", w) for w in selected_wells]),
            color_map=color_map if color_map else None,
            x_ranges=x_ranges,
            track_width=int(track_width),
            height=int(fig_height),
            trim_depth_gaps=trim_depth,
            line_width=float(line_width),
            opacity=float(opacity),
            show_legend=True,
        )
        st.plotly_chart(fig_multi, use_container_width=False, config={"displaylogo": False})
