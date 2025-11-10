# ================================
# 🌏 India Population Dashboard – Developed by Karthikeya
# “Interactive visualization of India’s population growth and forecast trends (2011–2026).”
# ================================

from __future__ import annotations
import os, json, math
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ------------------------------
# Page / Theme
# ------------------------------
st.set_page_config(
    page_title="India Population Dashboard – Developed by Karthikeya",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme for Altair
alt.themes.enable("dark")

# Subtle title + italic tagline
st.markdown(
    """
    <h1 style="margin-bottom:0">🌏 India Population Dashboard – Developed by Karthikeya</h1>
    <p style="opacity:0.85; font-style:italic; margin-top:4px;">
    “Interactive visualization of India’s population growth and forecast trends (2011–2026).”
    </p>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Constants / Paths
# ------------------------------
DATA_CSV = "data/raw/population_state_year.csv"
GEOJSON_PATH = "assets/geo/india_states.geojson"

# Color themes available
COLOR_THEMES = ["blues","cividis","greens","inferno","magma","plasma","reds","rainbow","turbo","viridis"]

# Canonical state-code map (2-letter) for convenience
STATE_NAME_TO_CODE = {
    "Andaman & Nicobar Islands": "AN", "Andaman and Nicobar Islands": "AN",
    "Andhra Pradesh": "AP", "Arunachal Pradesh": "AR", "Assam": "AS",
    "Bihar": "BR", "Chandigarh": "CH", "Chhattisgarh": "CT",
    "Dadra and Nagar Haveli and Daman and Diu": "DN",
    "Delhi": "DL", "NCT of Delhi": "DL",
    "Goa": "GA", "Gujarat": "GJ", "Haryana": "HR", "Himachal Pradesh": "HP",
    "Jammu & Kashmir": "JK", "Jammu and Kashmir": "JK", "Jharkhand": "JH",
    "Karnataka": "KA", "Kerala": "KL", "Ladakh": "LA", "Lakshadweep": "LD",
    "Madhya Pradesh": "MP", "Maharashtra": "MH", "Manipur": "MN",
    "Meghalaya": "ML", "Mizoram": "MZ", "Nagaland": "NL",
    "Odisha": "OD", "Orissa": "OD", "Puducherry": "PY", "Pondicherry": "PY",
    "Punjab": "PB", "Rajasthan": "RJ", "Sikkim": "SK",
    "Tamil Nadu": "TN", "Telangana": "TS", "Tripura": "TR",
    "Uttar Pradesh": "UP", "Uttarakhand": "UK", "Uttaranchal": "UK",
    "West Bengal": "WB",
}
STATE_CODE_TO_NAME = {v:k for k,v in STATE_NAME_TO_CODE.items() if v not in ["DL","JK","OD","PY","UK"]}

# ------------------------------
# Helpers
# ------------------------------
def _format_number(num: int) -> str:
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f} M" if num % 1_000_000 else f"{num//1_000_000} M"
    if num >= 1_000:
        return f"{num//1_000} K"
    return str(num)

def _coalesce_state_code(state_name: str) -> str | None:
    return STATE_NAME_TO_CODE.get(state_name)

# ------------------------------
# Data loaders (cached)
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"❌ Data file not found at `{path}`. Make sure your CSV exists.")
        st.stop()
    df = pd.read_csv(path)
    required = {"state_code","state_name","year","population"}
    if missing := (required - set(df.columns)):
        st.error(f"❌ CSV missing columns: {missing}")
        st.stop()
    df["year"] = df["year"].astype(int)
    df["population"] = df["population"].astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_geojson(path: str = GEOJSON_PATH) -> Tuple[dict, str]:
    """
    Returns (geojson, join_mode)
    join_mode:
        - "name"  -> use properties.ST_NM with df.state_name
        - "code"  -> use properties.st_code with df.state_code
    """
    if not os.path.exists(path):
        st.error(f"❌ GeoJSON not found at `{path}`. Place a real India states GeoJSON there.")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # Choose join mode
    sample_props = gj["features"][0]["properties"]
    if "ST_NM" in sample_props:
        # ensure st_code is present (nice to have)
        for feat in gj["features"]:
            props = feat["properties"]
            if "st_code" not in props:
                code = _coalesce_state_code(props.get("ST_NM",""))
                if code: props["st_code"] = code
        return gj, "name"
    elif "st_code" in sample_props:
        return gj, "code"
    else:
        # Try to synthesize from NAME_1 or NAME
        for feat in gj["features"]:
            props = feat.get("properties", {})
            name = props.get("NAME_1") or props.get("NAME") or props.get("state_name")
            if name and "st_code" not in props:
                code = _coalesce_state_code(name)
                if code: props["st_code"] = code
            feat["properties"] = props
        return gj, "code"

# ------------------------------
# Sidebar (global controls)
# ------------------------------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Overview", "Forecast"], index=0, horizontal=False)

    st.markdown("### Global")
    theme = st.selectbox("Color theme", COLOR_THEMES, index=0)
    st.caption("Theme affects map and heatmap color scales.")

# Load all data
df = load_data()
gj, join_mode = load_geojson()

# ------------------------------
# Components
# ------------------------------
def make_choropleth(df_year: pd.DataFrame, geojson: dict, scheme: str,
                    legend: bool = True, height: int = 600) -> "plotly.graph_objects.Figure":
    """
    Optimized, clear, zoomable India map with smooth transitions.
    """

    if join_mode == "name":
        fig = px.choropleth(
            df_year,
            geojson=geojson,
            featureidkey="properties.ST_NM",
            locations="state_name",
            color="population",
            color_continuous_scale=scheme,
            labels={"population": "Population"},
        )
    else:
        fig = px.choropleth(
            df_year,
            geojson=geojson,
            featureidkey="properties.st_code",
            locations="state_code",
            color="population",
            color_continuous_scale=scheme,
            labels={"population": "Population"},
        )

    # --- Improved visual settings ---
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showsubunits=False,
        projection_type="mercator",       # correct projection for India
        lataxis_range=[6, 38],            # focus on India's lat range
        lonaxis_range=[68, 98],           # focus on India's long range
    )

    # --- Better layout for Streamlit ---
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        dragmode="zoom",
        hovermode="closest",
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_scale=1.2,          # zoom in slightly for clarity
            center=dict(lat=22.5, lon=80), # center on India
        ),
    )

    # Enable scroll zoom and hover label improvements
    fig.update_layout(
        coloraxis_showscale=legend,
        hoverlabel=dict(bgcolor="black", font_size=12, font_color="white"),
    )
    fig.update_traces(marker_line_width=0.4, marker_line_color="black")

    return fig

    if join_mode == "name":
        fig = px.choropleth(
            df_year,
            geojson=geojson,
            featureidkey="properties.ST_NM",
            locations="state_name",
            color="population",
            color_continuous_scale=scheme,
            labels={"population":"Population"},
        )
    else:
        fig = px.choropleth(
            df_year,
            geojson=geojson,
            featureidkey="properties.st_code",
            locations="state_code",
            color="population",
            color_continuous_scale=scheme,
            labels={"population":"Population"},
        )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), height=height)
    if not legend:
        fig.update_layout(coloraxis_showscale=False)
    return fig

def make_heatmap(df_all: pd.DataFrame, scheme: str) -> alt.Chart:
    return (
        alt.Chart(df_all)
        .mark_rect()
        .encode(
            y=alt.Y("year:O", axis=alt.Axis(title="Year", labelAngle=0)),
            x=alt.X("state_name:O", axis=alt.Axis(title="")),
            color=alt.Color("max(population):Q", legend=None, scale=alt.Scale(scheme=scheme)),
            stroke=alt.value("black"),
            strokeWidth=alt.value(0.25),
        )
        .properties(height=220)
    )

def make_donut(value: int, label: str, palette="green") -> alt.Chart:
    if palette == "green":
        ring = ["#27AE60", "#12783D"]
    elif palette == "red":
        ring = ["#E74C3C", "#781F16"]
    else:
        ring = ["#29b5e8", "#155F7A"]
    src = pd.DataFrame({"k":["x",label], "v":[100-value, value]})
    src_bg = pd.DataFrame({"k":["x",label], "v":[100,0]})
    plot = alt.Chart(src).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="v", color=alt.Color("k:N", scale=alt.Scale(domain=[label,"x"], range=ring), legend=None)
    ).properties(width=130, height=130)
    text = plot.mark_text(align="center", fontSize=28, fontWeight=700).encode(text=alt.value(f"{value} %"))
    bg = alt.Chart(src_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="v", color=alt.Color("k:N", scale=alt.Scale(domain=[label,"x"], range=ring), legend=None)
    ).properties(width=130, height=130)
    return bg + plot + text

def yoy_change(df_all: pd.DataFrame, y: int) -> pd.DataFrame:
    cur = df_all[df_all.year == y][["state_code","state_name","population"]].copy()
    prev = df_all[df_all.year == (y-1)][["state_code","population"]].rename(columns={"population":"prev"})
    x = cur.merge(prev, on="state_code", how="left")
    x["prev"] = x["prev"].fillna(x["population"])
    x["diff"] = x["population"] - x["prev"]
    return x.sort_values("diff", ascending=False)

# ------------------------------
# OVERVIEW PAGE
# ------------------------------
if page == "Overview":
    with st.sidebar:
        st.markdown("### Overview Controls")
        years = sorted(df["year"].unique().tolist(), reverse=True)
        year = st.selectbox("Select year", years, index=0)
        state_options = ["All states"] + sorted(df["state_name"].unique().tolist())
        state_filter = st.selectbox("Focus state", state_options, index=0)

    # Filter data for year
    df_y = df[df.year == year].copy().sort_values("population", ascending=False)
    diff = yoy_change(df, year)

    # Left column: metrics + donuts
    c1, c2, c3 = st.columns((1.2, 4.6, 2.2), gap="medium")

    with c1:
        st.markdown("#### Gains/Losses")
        g = diff.iloc[0]
        l = diff.iloc[-1]
        st.metric(label=g["state_name"], value=_format_number(int(g["population"])), delta=_format_number(int(g["diff"])))
        st.metric(label=l["state_name"], value=_format_number(int(l["population"])), delta=_format_number(int(l["diff"])))

        st.markdown("#### States Migration")
        inbound_pct = int(round((diff["diff"] > 300_000).mean() * 100))
        outbound_pct = int(round((diff["diff"] < -300_000).mean() * 100))
        donut_in = make_donut(inbound_pct, "Inbound Migration", "green")
        donut_out = make_donut(outbound_pct, "Outbound Migration", "red")
        pad, mid, pad2 = st.columns((0.15, 0.7, 0.15))
        with mid:
            st.write("Inbound")
            st.altair_chart(donut_in, use_container_width=False)
            st.write("Outbound")
            st.altair_chart(donut_out, use_container_width=False)

    with c2:
        st.markdown("#### Total Population")

        with st.spinner("🧭 Building Map…"):
            fig_overview = make_choropleth(df_y, gj, scheme=theme, legend=True, height=600)
            st.plotly_chart(fig_overview, use_container_width=True, config={"scrollZoom": True})



        # Heatmap (optionally filtered for a single state)
        with st.spinner("📊 Updating Charts…"):
            if state_filter != "All states":
                df_heat = df[df["state_name"] == state_filter].copy()
            else:
                df_heat = df.copy()
            st.altair_chart(make_heatmap(df_heat, theme), use_container_width=True)

    with c3:
        st.markdown("#### Top States")
        d_display = df_y.rename(columns={"state_name": "State", "population": "Population"})
        st.dataframe(
            d_display,
            hide_index=True,
            column_order=("state_code","State","Population"),
            column_config={
                "state_code": st.column_config.TextColumn("Code"),
                "State": st.column_config.TextColumn("State"),
                "Population": st.column_config.ProgressColumn(
                    "Population", format="%d", min_value=0, max_value=int(d_display["Population"].max())
                ),
            },
            use_container_width=True,
        )

        # Download current-year table
        csv_bytes = d_display.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download this table (CSV)", data=csv_bytes, file_name=f"top_states_{year}.csv", mime="text/csv")

        with st.expander("About", expanded=True):
            st.write(
                "- Data: 36 States + UTs, 2011–2021 (synthetic demo; replace with official data when available)\n"
                "- Map: Works with GeoJSON having `properties.ST_NM` (Datameet) or `properties.st_code`.\n"
                "- Gains/Losses: Year-over-year population change.\n"
                "- Migration donuts: Share of states beyond ±300k change."
            )

# ------------------------------
# FORECAST UTILITIES (cached)
# ------------------------------
@st.cache_data(show_spinner=False)
def fit_models_return_forecasts(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Returns wide table of forecasts for each state, years 2022-2026,
    plus returns historical rows for 2011-2021 to allow combined charts.
    """
    out_rows = []
    for code, grp in df_all.groupby("state_code"):
        grp = grp.sort_values("year")
        X = grp[["year"]].values
        y = grp["population"].values
        model = LinearRegression().fit(X, y)

        # Historical
        for _, r in grp.iterrows():
            out_rows.append({
                "state_code": code,
                "state_name": r["state_name"],
                "year": int(r["year"]),
                "population": int(r["population"]),
                "source": "actual"
            })

        # Forecast 2022–2026
        for yr in range(2022, 2027):
            pred = float(model.predict(np.array([[yr]])))
            pred = max(pred, 0.0)
            # add ±5% CI
            lo = pred * 0.95
            hi = pred * 1.05
            out_rows.append({
                "state_code": code,
                "state_name": grp["state_name"].iloc[0],
                "year": yr,
                "population": int(round(pred)),
                "ci_low": int(round(lo)),
                "ci_high": int(round(hi)),
                "source": "pred"
            })
    return pd.DataFrame(out_rows)

def forecast_line_chart(df_state: pd.DataFrame, theme: str, show_combined: bool) -> alt.Chart:
    """
    show_combined=True  -> actual(2011–2021) + predicted(2022–2026)
    show_combined=False -> predicted only (2022–2026)
    """
    if show_combined:
        src = df_state.copy()
    else:
        src = df_state[df_state["source"] == "pred"].copy()

    # confidence band (only for predicted rows)
    band = (
        alt.Chart(src[src["source"]=="pred"])
        .mark_area(opacity=0.25)
        .encode(
            x="year:O",
            y="ci_low:Q",
            y2="ci_high:Q",
        )
    )

    # line style: dashed for predicted, solid for actual
    line = (
        alt.Chart(src)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", axis=alt.Axis(title="Year")),
            y=alt.Y("population:Q", axis=alt.Axis(title="Population")),
            strokeDash=alt.condition("datum.source == 'pred'", alt.value([6,4]), alt.value([1,0])),
            color=alt.Color("source:N", legend=alt.Legend(title="Type"), scale=alt.Scale(range=["#1f77b4","#ff7f0e"])),
            tooltip=["year", "population","source"]
        )
    )
    return (band + line).properties(height=320)

# ------------------------------
# FORECAST PAGE
# ------------------------------
if page == "Forecast":
    with st.sidebar:
        st.markdown("### Forecast Controls")
        state_list = sorted(df["state_name"].unique().tolist())
        state_sel = st.selectbox("Select state (chart)", state_list, index=0)
        forecast_year = st.slider("Forecast year for map", min_value=2022, max_value=2026, value=2026, step=1)
        show_legend = st.toggle("Show forecast map legend", value=True)

    with st.spinner("📈 Forecasting Population Trends…"):
        df_fc = fit_models_return_forecasts(df)

    # Charts for the selected state
    df_state = df_fc[df_fc["state_name"] == state_sel].copy().sort_values("year")

    colA, colB = st.columns(2, gap="large")
    with colA:
        st.markdown("#### Combined: Actual (2011–2021) + Predicted (2022–2026)")
        st.altair_chart(forecast_line_chart(df_state, theme, show_combined=True), use_container_width=True)
    with colB:
        st.markdown("#### Predicted Only: 2022–2026")
        st.altair_chart(forecast_line_chart(df_state, theme, show_combined=False), use_container_width=True)

    # Forecast map (full-width, centered)
    st.markdown("#### Forecast Choropleth (selectable year)")
    df_map_year = df_fc[(df_fc["year"] == forecast_year) & (df_fc["source"] == "pred")][["state_code","state_name","population"]]
    with st.spinner("🕰️ Rendering Future Map…"):
        fig_forecast = make_choropleth(df_map_year, gj, scheme=theme, legend=show_legend, height=600)
        st.plotly_chart(fig_forecast, use_container_width=True, config={"scrollZoom": True})



    # Download forecast rows
    csv_bytes = df_fc.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download forecast data (CSV)",
        data=csv_bytes,
        file_name="forecast_2011_2026_by_state.csv",
        mime="text/csv",
    )

# ------------------------------
# Footer
# ------------------------------
st.markdown(
    """
    <div style="opacity:0.6; font-size:0.9rem; padding-top:0.5rem;">
      Built with Streamlit, Plotly, Altair & scikit-learn.
    </div>
    """,
    unsafe_allow_html=True,
)
