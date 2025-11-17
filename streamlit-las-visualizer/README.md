# Streamlit LAS Visualizer

Application Streamlit minimaliste pour importer des fichiers LAS, visualiser des logs en tracks personnalisés et générer des crossplots. Gestion des dépendances via `uv`.

## Prérequis
Python 3.9+ et `uv` installé (voir documentation officielle si absent).

## Installation

Depuis le dossier du projet :

```bash
# Crée un environnement virtuel et installe les dépendances
uv sync
```

## Lancer l'application

```bash
# Démarre Streamlit avec l'environnement configuré par uv
uv run streamlit run app.py
```

Ensuite :
- Importer des fichiers `.LAS`
- Construire des tracks (plusieurs logs côte à côte)
- Faire des crossplots (X vs Y, couleur optionnelle)
- Explorer des statistiques (histogrammes, boxplots, pairplot, corrélation)

## Structure

- `app.py` – point d'entrée Streamlit
- `pages/` – pages multipages (import, visualisation, crossplots, statistiques)
- `src/las_utils.py` – lecture et normalisation des LAS
- `src/plot_utils.py` – fonctions de tracé Plotly
(- `.streamlit/config.toml` si présent – thème Streamlit)

## Astuces

- Conversion automatique des valeurs NULL en NaN.
- Profondeur standardisée en `DEPTH` (axe Y inversé).
- Unités présentes affichées dans la légende.

### Optionnel
Pour rechargements plus rapides : installer `watchdog`.

## Licence

Usage interne / démonstration.
