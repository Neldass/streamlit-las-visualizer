import streamlit as st

st.set_page_config(
    page_title="LAS Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("LAS Visualizer")

st.markdown(
    """
Bienvenue dans l'outil de visualisation LAS.

- Allez dans la page « Importer des fichiers LAS » pour charger vos données.
- Utilisez ensuite « Visualiser les logs » ou « Crossplots » pour explorer.

Astuce: vous pouvez charger plusieurs fichiers pour comparer.
    """
)

st.info(
    "Utilisez le menu de gauche pour naviguer entre les pages."
)
