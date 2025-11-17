import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from src.las_utils import list_numeric_logs, subset_depth


def build_color_map(selected: list[str], custom_enabled: bool) -> dict[str, str]:
    """Return a color map keyed by the WELL label used in charts.

    The label is LAS well name when available, otherwise display name/filename.
    """
    base_cycle = px.colors.qualitative.Plotly
    def label_for(key: str) -> str:
        ds = st.session_state.get("datasets", {}).get(key, {})
        return ds.get("well_name") or ds.get("display_name", key)

    labels = [label_for(w) for w in selected]
    color_map: dict[str, str] = {lbl: base_cycle[i % len(base_cycle)] for i, lbl in enumerate(labels)}
    if custom_enabled:
        st.sidebar.markdown("— Couleurs par puits —")
        if "stats_colors" not in st.session_state:
            st.session_state["stats_colors"] = {}
        for key, lbl in zip(selected, labels):
            default_hex = st.session_state["stats_colors"].get(lbl, color_map[lbl])
            picked = st.sidebar.color_picker(f"{lbl}", default_hex, key=f"stats_color_{key}")
            st.session_state["stats_colors"][lbl] = picked
            color_map[lbl] = picked
    return color_map


st.set_page_config(page_title="Statistiques", page_icon="📊", layout="wide")

st.title("📊 Statistiques")

if "datasets" not in st.session_state or not st.session_state["datasets"]:
    st.warning("Veuillez d'abord importer des fichiers LAS dans la page d'import.")
    st.stop()

datasets_all = st.session_state["datasets"]
well_names = list(datasets_all.keys())

select_all_wells = st.checkbox("Sélectionner tous les puits", value=False, help="Activez pour inclure tous les puits.")
selected_wells = st.multiselect(
    "Choisir un ou plusieurs puits",
    well_names,
    default=well_names[:1],
    format_func=lambda n: datasets_all[n].get("display_name", n),
    key="stats_selected_wells",
)
if select_all_wells:
    selected_wells = well_names
if not selected_wells:
    st.info("Sélectionnez au moins un puits pour continuer.")
    st.stop()

datasets = {n: datasets_all[n] for n in selected_wells}

per_well_numeric = {n: set(list_numeric_logs(d["df"], exclude=["DEPTH"])) for n, d in datasets.items()}
common_numeric = sorted(set.intersection(*per_well_numeric.values())) if per_well_numeric else []
if not common_numeric:
    st.info("Aucun log numérique commun entre les puits sélectionnés.")
    st.stop()

with st.sidebar:
    st.header("Paramètres globaux")
    depth_min = float(min(d["df"]["DEPTH"].min() for d in datasets.values()))
    depth_max = float(max(d["df"]["DEPTH"].max() for d in datasets.values()))
    m, M = st.slider(
        "Intervalle de profondeur",
        min_value=float(np.floor(depth_min)),
        max_value=float(np.ceil(depth_max)),
        value=(float(depth_min), float(depth_max)),
        step=0.1,
    )
    st.caption("Les statistiques utilisent uniquement les données dans cet intervalle.")

    st.subheader("Normalisation des valeurs")
    norm_method = st.selectbox(
        "Méthode",
        options=["(aucune)", "Min-Max [0,1]", "Z-score (moy=0, std=1)", "Robuste (IQR)"],
        index=0,
        help="Normalise globalement sans modifier les données sources.",
    )

    st.subheader("Couleurs")
    use_custom_colors = st.checkbox("Couleurs personnalisées par puits", value=False)

    # Build color map ONCE to avoid duplicate widget keys
    color_map = build_color_map(selected_wells, use_custom_colors)


def build_combined_df(cols: list[str]) -> pd.DataFrame:
    frames = []
    needed = list(set(cols))  # de-dup
    for n, d in datasets.items():
        df = subset_depth(d["df"].copy(), m, M)
        present = [c for c in needed if c in df.columns]
        if not present:
            continue
        chunk = df[present].copy()
        chunk["WELL"] = d.get("well_name") or d.get("display_name", n)
        frames.append(chunk)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        for c in cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan).dropna(how="all", subset=cols)
    return pd.DataFrame(columns=list(cols) + ["WELL"])  # empty


def _normalize_inplace(df: pd.DataFrame, cols: list[str], method: str) -> None:
    if method == "(aucune)":
        return
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if method == "Min-Max [0,1]":
            vmin, vmax = s.min(skipna=True), s.max(skipna=True)
            denom = (vmax - vmin)
            df[col] = 0.0 if (pd.isna(denom) or denom == 0) else (s - vmin) / denom
        elif method == "Z-score (moy=0, std=1)":
            mean, std = s.mean(skipna=True), s.std(skipna=True)
            df[col] = 0.0 if (pd.isna(std) or std == 0) else (s - mean) / std
        elif method == "Robuste (IQR)":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            df[col] = 0.0 if (pd.isna(iqr) or iqr == 0) else (s - (q1 + q3) / 2) / iqr


tabs = st.tabs(["Histogrammes", "Boxplots", "Pairplot", "Corrélation"])

with tabs[0]:
    st.subheader("Histogrammes")
    sel_logs_hist = st.multiselect(
        "Sélectionner un ou plusieurs logs",
        options=common_numeric,
        default=common_numeric[:1],
        key="stats_hist_logs",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        nbins = st.slider("Nombre de classes (bins)", 5, 200, 50)
    with c2:
        histnorm = st.selectbox("Normalisation", ["count", "probability", "percent", "density"], index=0)
    with c3:
        barmode = st.selectbox("Mode", ["overlay", "group"], index=0)

    if sel_logs_hist:
        data = build_combined_df(sel_logs_hist)
        _normalize_inplace(data, sel_logs_hist, norm_method)
        if data.empty:
            st.info("Pas de données à l'intervalle sélectionné.")
        else:
            if len(sel_logs_hist) == 1:
                v = sel_logs_hist[0]
                fig = px.histogram(
                    data,
                    x=v,
                    color="WELL",
                    color_discrete_map=color_map,
                    nbins=nbins,
                    histnorm=None if histnorm == "count" else histnorm,
                    opacity=0.7,
                )
                suffix = "" if norm_method == "(aucune)" else (" • norm=Globale/" + norm_method)
                fig.update_layout(barmode=barmode, height=650, title=f"Histogramme • {v}{suffix}")
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            else:
                long = data.melt(id_vars=["WELL"], value_vars=sel_logs_hist, var_name="log", value_name="value")
                fig = px.histogram(
                    long.dropna(subset=["value"]),
                    x="value",
                    color="WELL",
                    color_discrete_map=color_map,
                    nbins=nbins,
                    histnorm=None if histnorm == "count" else histnorm,
                    facet_col="log",
                    facet_col_wrap=3,
                    opacity=0.7,
                )
                suffix = "" if norm_method == "(aucune)" else (" • norm=Globale/" + norm_method)
                fig.update_layout(barmode=barmode, height=700, title=f"Histogrammes par log{suffix}")
                fig.for_each_xaxis(lambda a: a.update(title=""))
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


with tabs[1]:
    st.subheader("Boxplots")
    sel_logs_box = st.multiselect(
        "Sélectionner un ou plusieurs logs",
        options=common_numeric,
        default=common_numeric[:min(4, len(common_numeric))],
        key="stats_box_logs",
    )
    if sel_logs_box:
        data = build_combined_df(sel_logs_box)
        _normalize_inplace(data, sel_logs_box, norm_method)
        if data.empty:
            st.info("Pas de données à l'intervalle sélectionné.")
        else:
            long = data.melt(id_vars=["WELL"], value_vars=sel_logs_box, var_name="log", value_name="value")
            fig = px.box(
                long.dropna(subset=["value"]),
                x="log",
                y="value",
                color="WELL",
                points=False,
                color_discrete_map=color_map,
            )
            suffix = "" if norm_method == "(aucune)" else (" • norm=Globale/" + norm_method)
            fig.update_layout(height=650, title=f"Boxplots par log{suffix}", boxmode="group")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


with tabs[2]:
    st.subheader("Pairplot (scatter matrix)")
    sel_logs_pair = st.multiselect(
        "Sélectionner les logs (2-6)",
        options=common_numeric,
        default=common_numeric[: min(4, len(common_numeric))],
        help="<= 6 recommandé.",
        key="stats_pair_logs",
    )
    if len(sel_logs_pair) >= 2:
        if len(sel_logs_pair) > 8:
            st.warning("Limitation d'affichage: maximum 8 logs pour le pairplot.")
            sel_logs_pair = sel_logs_pair[:8]
        data = build_combined_df(sel_logs_pair)
        _normalize_inplace(data, sel_logs_pair, norm_method)
        if data.empty:
            st.info("Pas de données à l'intervalle sélectionné.")
        else:
            fig = px.scatter_matrix(
                data.dropna(subset=sel_logs_pair),
                dimensions=sel_logs_pair,
                color="WELL",
                color_discrete_map=color_map,
                opacity=0.6,
            )
            n = len(sel_logs_pair)
            size = 180 + n * 120
            suffix = "" if norm_method == "(aucune)" else (" • norm=Globale/" + norm_method)
            fig.update_layout(height=max(600, size), width=None, title=f"Pairplot{suffix}")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Sélectionnez au moins deux logs.")


with tabs[3]:
    st.subheader("Matrice de corrélation (Pearson)")
    sel_logs_corr = st.multiselect(
        "Sélectionner les logs",
        options=common_numeric,
        default=common_numeric[: min(6, len(common_numeric))],
        key="stats_corr_logs",
    )
    if len(sel_logs_corr) >= 2:
        data = build_combined_df(sel_logs_corr)
        _normalize_inplace(data, sel_logs_corr, norm_method)
        if data.empty:
            st.info("Pas de données à l'intervalle sélectionné.")
        else:
            corr = data[sel_logs_corr].corr(method="pearson", numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            suffix = "" if norm_method == "(aucune)" else (" • norm=Globale/" + norm_method)
            fig.update_layout(height=650, title=f"Corrélation (Pearson){suffix}")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Sélectionnez au moins deux logs.")
