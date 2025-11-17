import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from src.las_utils import list_numeric_logs, subset_depth
from src.plot_utils import plot_crossplot


def build_well_color_map(wells: list[str], custom_enabled: bool) -> dict[str, str]:
    base_cycle = px.colors.qualitative.Plotly
    color_map: dict[str, str] = {w: base_cycle[i % len(base_cycle)] for i, w in enumerate(wells)}
    if custom_enabled:
        st.sidebar.markdown("— Couleurs par puits —")
        if "cross_colors" not in st.session_state:
            st.session_state["cross_colors"] = {}
        for w in wells:
            label = st.session_state["datasets"].get(w, {}).get("display_name", w) if "datasets" in st.session_state else w
            default_hex = st.session_state["cross_colors"].get(w, color_map[w])
            picked = st.sidebar.color_picker(label, default_hex, key=f"cross_color_{w}")
            st.session_state["cross_colors"][w] = picked
            color_map[w] = picked
    return color_map

st.set_page_config(page_title="Crossplots", page_icon="📊", layout="wide")

st.title("📊 Crossplots")

if "datasets" not in st.session_state or not st.session_state["datasets"]:
    st.warning("Veuillez d'abord importer des fichiers LAS dans la page précédente.")
    st.stop()

datasets_all = st.session_state["datasets"]
names = list(datasets_all.keys())
select_all = st.checkbox("Sélectionner tous les puits", value=False)
selected_names = st.multiselect(
    "Choisir un ou plusieurs puits",
    names,
    default=names if select_all else names[:1],
    format_func=lambda n: datasets_all[n].get("display_name", n),
)
if select_all:
    selected_names = names
if not selected_names:
    st.stop()

datasets = {n: datasets_all[n] for n in selected_names}

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

    color_choice = st.selectbox("Couleur (optionnel)", options=["(aucune)", "(puits)"] + numeric_logs, index=0)
    color = None if color_choice == "(aucune)" else ("WELL" if color_choice == "(puits)" else color_choice)

    use_density = st.checkbox("Mode densité (heatmap)", value=False)

    st.divider()
    st.subheader("Normalisation des points")
    norm_method = st.selectbox("Méthode", options=["(aucune)", "Min-Max [0,1]", "Z-score (moy=0, std=1)", "Robuste (IQR)"], index=0)

    # Couleurs personnalisées (seulement quand on colorie par puits)
    color_map = None
    if color == "WELL":
        st.subheader("Couleurs")
        st.caption("Couleurs par défaut (Plotly). Activez pour personnaliser.")
        use_custom_colors = st.checkbox("Couleurs personnalisées par puits", value=False)
        color_map = build_well_color_map(selected_names, use_custom_colors)

    depth_min = float(min(d["df"]["DEPTH"].min() for d in datasets.values()))
    depth_max = float(max(d["df"]["DEPTH"].max() for d in datasets.values()))
    m, M = st.slider(
        "Intervalle de profondeur",
        min_value=float(np.floor(depth_min)),
        max_value=float(np.ceil(depth_max)),
        value=(float(depth_min), float(depth_max)),
        step=0.1,
    )

frames = []
needed_cols = {x, y} | ({color} if color and color != "WELL" else set())
for n, d in datasets.items():
    tmp = subset_depth(d["df"].copy(), m, M)
    cols = [c for c in needed_cols if c in tmp.columns]
    chunk = tmp[cols].copy()
    chunk["WELL"] = n
    frames.append(chunk)

combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[x, y])

def _normalize_inplace(df: pd.DataFrame, cols: list[str], method: str, by: str | None):
    if method == "(aucune)":
        return
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if by is None:
            s = df[col]
            if method == "Min-Max [0,1]":
                vmin, vmax = s.min(skipna=True), s.max(skipna=True)
                denom = (vmax - vmin)
                if pd.isna(denom) or denom == 0:
                    df[col] = 0.0
                else:
                    df[col] = (s - vmin) / denom
            elif method == "Z-score (moy=0, std=1)":
                mean, std = s.mean(skipna=True), s.std(skipna=True)
                df[col] = 0.0 if (pd.isna(std) or std == 0) else (s - mean) / std
            elif method == "Robuste (IQR)":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                df[col] = 0.0 if (pd.isna(iqr) or iqr == 0) else (s - (q1 + q3) / 2) / iqr
        else:
            gb = df.groupby(by)[col]
            if method == "Min-Max [0,1]":
                vmin = gb.transform("min")
                vmax = gb.transform("max")
                denom = (vmax - vmin)
                denom_safe = denom.mask(denom == 0, other=np.nan)
                df[col] = ((df[col] - vmin) / denom_safe).fillna(0.0)
            elif method == "Z-score (moy=0, std=1)":
                mean = gb.transform("mean")
                std = gb.transform("std")
                std_safe = std.mask((std == 0) | (std.isna()), other=np.nan)
                df[col] = ((df[col] - mean) / std_safe).fillna(0.0)
            elif method == "Robuste (IQR)":
                q1 = gb.transform(lambda s: s.quantile(0.25))
                q3 = gb.transform(lambda s: s.quantile(0.75))
                iqr = (q3 - q1)
                iqr_safe = iqr.mask((iqr == 0) | (iqr.isna()), other=np.nan)
                df[col] = ((df[col] - (q1 + q3) / 2) / iqr_safe).fillna(0.0)

norm_by = None
cols_to_norm = [str(x), str(y)]
_normalize_inplace(combined, cols_to_norm, norm_method, by=norm_by)

# Note: pas de normalisation de couleur; on retire cette option (aligné avec la demande)

title_suffix_raw = [datasets_all[n].get("display_name", n) for n in selected_names]
title_suffix = ", ".join(title_suffix_raw) if len(title_suffix_raw) <= 3 else f"{len(title_suffix_raw)} puits"
norm_suffix = "" if norm_method == "(aucune)" else (" • norm=Globale/" + norm_method)
fig = plot_crossplot(
    combined,
    x=str(x),
    y=str(y),
    color=color,
    use_density=use_density,
    title=f"{y} vs {x} - {title_suffix}{norm_suffix}",
    color_discrete_map=color_map,
)
st.plotly_chart(fig, config={"displaylogo": False, "responsive": True})
