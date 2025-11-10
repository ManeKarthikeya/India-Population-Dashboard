
# 🇮🇳 India Population Dashboard (Starter)

A professional-grade Streamlit dashboard inspired by Data Professor's India Population. This starter pack ships with:

- Heatmap (Altair), Donut charts (Altair), Choropleth (Plotly)
- Caching for performance
- A small **demo dataset** (36 states/UTs, 2011–2021)
- A **tiny demo GeoJSON** (toy squares) so the app boots instantly
- Dark theme and minimal CSS tweaks
S
## 📦 Project structure

```
india_population_dashboard/
├── assets/
│   └── geo/
│       └── india_states.geojson
├── data/
│   └── raw/
│       └── population_state_year.csv
├── config.toml
├── requirements.txt
└── streamlit_app.py
```

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> If Streamlit can't find `config.toml`, run from the project root (`india_population_dashboard/`).

## 🔁 Swap in real data (Production)

- Replace `data/raw/population_state_year.csv` with your official dataset.
  Required columns: `state_code, state_name, year, population`.
- Replace `assets/geo/india_states.geojson` with a real India states GeoJSON.
  It **must** have `properties.st_code` that matches the CSV `state_code`.

## 🗺️ Where to get real data
- Census of India (2011) and intercensal estimates (official portals)
- Ministry of Health & Family Welfare / Sample Registration System (estimates)
- India state boundaries GeoJSON from open-data repositories (e.g., Datameet)

## 🛠️ Ideas & Extensions
- Add pages for age/sex and urban/rural breakdowns.
- Add state-level trend lines and forecasting.
- Add CSV/XLSX download buttons.
- Deploy via Streamlit Community Cloud or Docker.

