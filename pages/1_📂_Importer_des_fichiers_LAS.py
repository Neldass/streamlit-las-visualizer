import streamlit as st
import pandas as pd

from src.las_utils import read_las_file

st.set_page_config(page_title="Importer des fichiers LAS", page_icon="📂", layout="wide")

st.title("📂 Importer des fichiers LAS")

uploaded = st.file_uploader(
    "Déposez un ou plusieurs fichiers .LAS",
    type=["las", "LAS"],
    accept_multiple_files=True,
)

if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}

def _curves_table(curves) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Courbe": [c.name for c in curves],
            "Unité": [c.unit or "" for c in curves],
            "Description": [c.descr for c in curves],
        }
    )

if uploaded:
    for up in uploaded:
        bytes_data = up.getvalue()
        dataset = read_las_file(bytes_data, up.name)
        st.session_state["datasets"][up.name] = dataset
    st.success(f"{len(uploaded)} fichier(s) importé(s). Allez aux autres pages pour explorer.")

if st.session_state["datasets"]:
    st.subheader("Jeux de données disponibles")
    for name, ds in st.session_state["datasets"].items():
        df = ds["df"]
        depth_series = pd.to_numeric(df["DEPTH"], errors="coerce")
        depth_min, depth_max = depth_series.min(skipna=True), depth_series.max(skipna=True)
        depth_min_f = float(depth_min) if pd.notna(depth_min) else None
        depth_max_f = float(depth_max) if pd.notna(depth_max) else None
        display = ds.get("display_name", name)
        st.markdown(f"### {display}")
        if display != name:
            st.caption(f"Fichier: {name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nbre de logs", len(ds["curves"]))
        c2.metric("Nbre d'échantillons", len(df))
        c3.metric("Profondeur min", f"{depth_min_f:.2f}" if depth_min_f is not None else "-")
        c4.metric("Profondeur max", f"{depth_max_f:.2f}" if depth_max_f is not None else "-")
        with st.expander("Logs disponibles"):
            st.dataframe(_curves_table(ds["curves"]))
else:
    st.info("Aucun fichier importé pour le moment.")
