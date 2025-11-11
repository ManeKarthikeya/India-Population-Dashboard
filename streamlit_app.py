# ================================
# 🌏 India Population Dashboard – Developed by Karthikeya
# “Interactive visualization of India’s population growth and forecast trends.”
# ================================

from __future__ import annotations
import os, json
from typing import Tuple
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
alt.themes.enable("dark")

st.markdown(
    """
    <h1 style="margin-bottom:0">🌏 India Population Dashboard – Developed by Karthikeya</h1>
    <p style="opacity:0.85; font-style:italic; margin-top:4px;">
    “Interactive visualization of India’s population growth and forecast trends.”
    </p>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Constants / Paths
# ------------------------------
DEFAULT_DATA_CSV = "data/raw/population_state_year.csv"
GEOJSON_PATH = "assets/geo/india_states.geojson"
COLOR_THEMES = ["blues","cividis","greens","inferno","magma","plasma","reds","rainbow","turbo","viridis"]

# ------------------------------
# Utilities
# ------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}
    need = ["state_code","state_name","year","population"]
    if not all(n in lower for n in need):
        st.error("❌ CSV must contain columns: state_code, state_name, year, population")
        st.stop()
    df = df.rename(columns={
        lower["state_code"]: "state_code",
        lower["state_name"]: "state_name",
        lower["year"]: "year",
        lower["population"]: "population",
    })
    df["year"] = df["year"].astype(int)
    df["population"] = df["population"].astype(int)
    return df

def fmt_num(n: int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f} M"
    if n >= 1_000:     return f"{n//1_000} K"
    return str(n)

@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    if not os.path.exists(DEFAULT_DATA_CSV):
        st.error(f"❌ Default data not found at `{DEFAULT_DATA_CSV}`.")
        st.stop()
    return normalize_columns(pd.read_csv(DEFAULT_DATA_CSV))

@st.cache_data(show_spinner=False)
def load_geojson(path: str = GEOJSON_PATH) -> Tuple[dict, str]:
    if not os.path.exists(path):
        st.error(f"❌ GeoJSON not found at `{path}`.")
        st.stop()
    gj = json.load(open(path, "r", encoding="utf-8"))
    props = gj["features"][0]["properties"]
    if "ST_NM" in props:
        return gj, "name"
    elif "st_code" in props:
        return gj, "code"
    for f in gj["features"]:
        p = f["properties"]
        if "ST_NM" not in p and ("NAME_1" in p or "NAME" in p):
            p["ST_NM"] = p.get("NAME_1", p.get("NAME"))
    return gj, "name"

# ------------------------------
# Session-backed data (upload temp)
# ------------------------------
def set_uploaded_csv(file):
    try:
        if file is None:
            st.warning("No file selected.")
            return
        df = pd.read_csv(file)
        if df.empty:
            st.error("❌ Uploaded CSV is empty.")
            return
        df = normalize_columns(df)
        st.session_state["uploaded_df"] = df
        st.success(f"✅ Loaded {len(df):,} rows from uploaded CSV.")
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")

def get_active_df() -> pd.DataFrame:
    """Return uploaded DataFrame if present (and not None); otherwise default."""
    df_up = st.session_state.get("uploaded_df", None)
    if df_up is not None:
        return df_up
    return load_default_data()

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Overview", "Forecast", "📈 Growth Insights"], index=0)

    st.markdown("### Global")
    theme = st.selectbox("Color theme", COLOR_THEMES, index=0)
    st.caption("Affects numeric choropleths and heatmaps.")
    st.markdown("---")

    st.markdown("### Use your own CSV")
    up = st.file_uploader("Upload CSV (temporary, in-memory)", type=["csv"])
    c1, c2 = st.columns(2)
    with c1:
        st.button("Use uploaded", disabled=(up is None), on_click=lambda: set_uploaded_csv(up))
    with c2:
        def _reset():
            st.session_state["uploaded_df"] = None
            st.success("🔄 Using default dataset.")
        st.button("Reset to default", on_click=_reset)

df = get_active_df()
gj, join_mode = load_geojson()

# ------------------------------
# Core computations
# ------------------------------
def yoy_change(df_all: pd.DataFrame, y: int) -> pd.DataFrame:
    cur = df_all[df_all.year == y][["state_code","state_name","population"]].copy()
    prev = df_all[df_all.year == y-1][["state_code","population"]].rename(columns={"population":"prev"})
    x = cur.merge(prev, on="state_code", how="left")
    x["prev"] = x["prev"].fillna(x["population"])
    x["diff"] = x["population"] - x["prev"]
    return x.sort_values("diff", ascending=False)

@st.cache_data(show_spinner=False)
def fit_forecast(df_all: pd.DataFrame, start_year: int, end_year: int, ci_pct: float = 5.0) -> pd.DataFrame:
    out = []
    for code, grp in df_all.groupby("state_code"):
        grp = grp.sort_values("year")
        model = LinearRegression().fit(grp[["year"]].values, grp["population"].values)
        for _, r in grp.iterrows():
            out.append({"state_code":code,"state_name":r["state_name"],"year":int(r["year"]),
                        "population":int(r["population"]), "source":"actual"})
        for yr in range(start_year, end_year+1):
            pred = float(model.predict(np.array([[yr]])))
            pred = max(0.0, pred)
            lo = pred * (1-ci_pct/100); hi = pred * (1+ci_pct/100)
            out.append({"state_code":code,"state_name":grp["state_name"].iloc[0],"year":yr,
                        "population":int(round(pred)),"ci_low":int(round(lo)),"ci_high":int(round(hi)),
                        "source":"pred"})
    return pd.DataFrame(out)

def classify_growth(df_fc: pd.DataFrame, base_year: int, target_year: int,
                    low_upper: float=4.0, med_upper: float=10.0) -> pd.DataFrame:
    base = df_fc[df_fc["year"]==base_year][["state_code","population"]].rename(columns={"population":"pop_base"})
    tgt  = df_fc[df_fc["year"]==target_year][["state_code","population"]].rename(columns={"population":"pop_target"})
    merged = base.merge(tgt, on="state_code", how="inner")
    merged["growth_rate"] = ((merged["pop_target"]-merged["pop_base"])/merged["pop_base"])*100
    def bucket(r):
        if r <= low_upper: return "Low Growth"
        if r <= med_upper: return "Medium Growth"
        return "High Growth"
    merged["growth_class"] = merged["growth_rate"].apply(bucket)
    return df_fc.merge(merged[["state_code","growth_rate","growth_class"]], on="state_code", how="left")

# ------------------------------
# Visualization helpers
# ------------------------------
def choropleth_numeric(df_year: pd.DataFrame, legend=True, height=600):
    feature = "properties.ST_NM" if join_mode=="name" else "properties.st_code"
    loccol  = "state_name" if join_mode=="name" else "state_code"
    fig = px.choropleth(df_year, geojson=gj, featureidkey=feature,
                        locations=loccol, color="population",
                        color_continuous_scale=theme, labels={"population":"Population"})
    fig.update_geos(fitbounds="locations", visible=False,
                    projection_type="mercator", lataxis_range=[6,38], lonaxis_range=[68,98])
    fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0),
                      height=height, dragmode="zoom", hovermode="closest",
                      geo=dict(showframe=False, showcoastlines=False,
                               projection_scale=1.2, center=dict(lat=22.5, lon=80)),
                      coloraxis_showscale=legend)
    fig.update_traces(marker_line_width=0.4, marker_line_color="black")
    return fig

def choropleth_class(df_cat: pd.DataFrame, legend=True, height=620):
    color_map = {"High Growth":"#00FF7F","Medium Growth":"#FFD700","Low Growth":"#FF6347"}
    feature = "properties.ST_NM" if join_mode=="name" else "properties.st_code"
    loccol  = "state_name" if join_mode=="name" else "state_code"
    fig = px.choropleth(df_cat, geojson=gj, featureidkey=feature, locations=loccol,
                        color="growth_class", color_discrete_map=color_map,
                        category_orders={"growth_class":["High Growth","Medium Growth","Low Growth"]})
    fig.update_geos(fitbounds="locations", visible=False,
                    projection_type="mercator", lataxis_range=[6,38], lonaxis_range=[68,98])
    fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0),
                      height=height, dragmode="zoom", hovermode="closest",
                      geo=dict(showframe=False, showcoastlines=False,
                               projection_scale=1.2, center=dict(lat=22.5, lon=80)))
    if not legend:
        fig.update_layout(showlegend=False)
    fig.update_traces(marker_line_width=0.4, marker_line_color="black")
    return fig

def heatmap(df_all: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df_all)
        .mark_rect()
        .encode(
            y=alt.Y("year:O", axis=alt.Axis(title="Year", labelAngle=0)),
            x=alt.X("state_name:O", sort=alt.Sort(field="state_name"),
                    axis=alt.Axis(title="State", labelAngle=300, labelOverlap=False)),
            color=alt.Color("max(population):Q", legend=None, scale=alt.Scale(scheme=theme)),
            tooltip=[alt.Tooltip("state_name:N"), alt.Tooltip("year:O"),
                     alt.Tooltip("max(population):Q", title="Population")]
        )
        .properties(height=380)
        .configure_view(strokeOpacity=0)
        .configure(padding={"left":10,"right":10,"top":10,"bottom":10})
    )

def donut(value: int, label: str, color="#27AE60",
          text_color = ("black" if st.get_option("theme.base") == "light" else "white")) -> alt.Chart:
    ring = [color, "#1A1A1A"]
    src = pd.DataFrame({"k": ["fill", "empty"], "v": [value, 100 - value]})
    chart = (
        alt.Chart(src)
        .mark_arc(innerRadius=45, cornerRadius=25)
        .encode(
            theta="v",
            color=alt.Color("k:N", scale=alt.Scale(domain=["fill","empty"], range=ring), legend=None),
        )
        .properties(width=140, height=140)
    )
    text = (
        alt.Chart(pd.DataFrame({"text": [f"{value}%"]}))
        .mark_text(fontSize=28, fontWeight=700, color=text_color)
        .encode(text="text")
    )
    return chart + text

def forecast_line_chart(df_state: pd.DataFrame, show_combined: bool) -> alt.Chart:
    src = df_state.copy() if show_combined else df_state[df_state["source"]=="pred"].copy()
    band = alt.Chart(src[src["source"]=="pred"]).mark_area(opacity=0.25).encode(
        x="year:O", y="ci_low:Q", y2="ci_high:Q"
    )
    line = (
        alt.Chart(src)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O"), y=alt.Y("population:Q"),
            strokeDash=alt.condition("datum.source == 'pred'", alt.value([6,4]), alt.value([1,0])),
            color=alt.Color("source:N", legend=alt.Legend(title="Type"),
                            scale=alt.Scale(range=["#1f77b4","#ff7f0e"]))
        )
    )
    return (band + line).properties(height=340)

# ------------------------------
# OVERVIEW
# ------------------------------
if page == "Overview":
    with st.sidebar:
        st.markdown("### Overview Controls")
        all_years = sorted(df["year"].unique().tolist())
        year = st.selectbox("Select year", all_years[::-1], index=0)
        state_options = ["All states"] + sorted(df["state_name"].unique().tolist())
        focus_state = st.selectbox("Focus state", state_options, index=0)

    df_y = df[df.year == year].copy().sort_values("population", ascending=False)
    diff = yoy_change(df, year)

    c1, c2, c3 = st.columns((1.2, 4.8, 2.0), gap="large")

    with c1:
        st.markdown("#### Gains/Losses")
        g, l = diff.iloc[0], diff.iloc[-1]
        st.metric(g["state_name"], fmt_num(int(g["population"])), delta=fmt_num(int(g["diff"])))
        st.metric(l["state_name"], fmt_num(int(l["population"])), delta=fmt_num(int(l["diff"])))
        st.markdown("#### Migration Indicators")
        inbound_pct = int(round((diff["diff"] > 300_000).mean() * 100))
        outbound_pct = int(round((diff["diff"] < -300_000).mean() * 100))
        st.write("Inbound");  st.altair_chart(donut(inbound_pct, "Inbound", "#27AE60"), use_container_width=False)
        st.write("Outbound"); st.altair_chart(donut(outbound_pct, "Outbound", "#E74C3C"), use_container_width=False)

    with c2:
        st.markdown("#### Total Population")
        with st.spinner("🧭 Building Map…"):
            fig_overview = choropleth_numeric(df_y, legend=True, height=600)
            st.plotly_chart(fig_overview, use_container_width=True, config={"scrollZoom": True})
        if st.button("🔄 Reset Map Zoom", key="reset_overview"):
            fig_overview.update_geos(fitbounds="locations", visible=False)
            st.rerun()
        st.markdown("#### States Heatmap")
        with st.spinner("📊 Drawing heatmap…"):
            df_heat = df if focus_state=="All states" else df[df["state_name"]==focus_state]
            st.altair_chart(heatmap(df_heat), use_container_width=True)

    with c3:
        st.markdown("#### Top States")
        d_display = df_y.rename(columns={"state_name":"State","population":"Population"})
        st.dataframe(
            d_display,
            hide_index=True,
            column_order=("state_code","State","Population"),
            column_config={
                "state_code": st.column_config.TextColumn("Code"),
                "Population": st.column_config.ProgressColumn(
                    "Population", format="%d", min_value=0, max_value=int(d_display["Population"].max())
                ),
            },
            use_container_width=True,
        )
        st.download_button("⬇️ Download table (CSV)",
                           data=d_display.to_csv(index=False).encode("utf-8"),
                           file_name=f"top_states_{year}.csv", mime="text/csv")

# ------------------------------
# FORECAST
# ------------------------------
if page == "Forecast":
    with st.sidebar:
        st.markdown("### Forecast Controls")
        last_actual_year = int(df["year"].max())
        default_start = last_actual_year + 1
        forecast_start = st.number_input("Forecast start year", value=default_start, step=1)
        forecast_end   = st.number_input("Forecast end year", value=max(default_start+5, default_start),
                                         step=1, min_value=forecast_start)
        state_sel = st.selectbox("Select state (chart)",
                                 sorted(df["state_name"].unique().tolist()), index=0)
        show_legend = st.toggle("Show forecast map legend", value=True)

    with st.spinner("📈 Forecasting Population Trends…"):
        df_fc = fit_forecast(df, start_year=int(forecast_start), end_year=int(forecast_end))
    df_state = df_fc[df_fc["state_name"]==state_sel].sort_values("year")

    colA, colB = st.columns(2, gap="large")
    with colA:
        st.markdown("#### Combined: Actual + Predicted")
        st.altair_chart(forecast_line_chart(df_state, True), use_container_width=True)
    with colB:
        st.markdown("#### Predicted Only")
        st.altair_chart(forecast_line_chart(df_state, False), use_container_width=True)

    st.markdown("#### Forecast Choropleth")
    forecast_year_for_map = st.slider("Year for map", min_value=int(forecast_start),
                                      max_value=int(forecast_end), value=int(forecast_end), step=1)
    df_map_year = df_fc[(df_fc["year"]==forecast_year_for_map)&(df_fc["source"]=="pred")][
        ["state_code","state_name","population"]
    ]
    with st.spinner("🕰️ Rendering Future Map…"):
        fig_forecast = choropleth_numeric(df_map_year, legend=show_legend, height=600)
        st.plotly_chart(fig_forecast, use_container_width=True, config={"scrollZoom": True})
    if st.button("🔄 Reset Map Zoom", key="reset_forecast"):
        fig_forecast.update_geos(fitbounds="locations", visible=False)
        st.rerun()

    st.download_button("⬇️ Download forecast data (CSV)",
                       data=df_fc.to_csv(index=False).encode("utf-8"),
                       file_name=f"forecast_{forecast_start}_{forecast_end}.csv",
                       mime="text/csv")

# ------------------------------
# 📈 GROWTH INSIGHTS
# ------------------------------
if page == "📈 Growth Insights":
    with st.sidebar:
        st.markdown("### Growth Insights Controls")
        last_actual_year = int(df["year"].max())
        default_start = last_actual_year + 1
        gi_start = st.number_input("Forecast start year", value=default_start, step=1)
        gi_end   = st.number_input("Forecast end year", value=max(default_start+5, default_start),
                                   step=1, min_value=gi_start)
        base_year   = st.number_input("Baseline year (actual)", value=last_actual_year,
                                      min_value=int(df["year"].min()), max_value=last_actual_year)
        target_year = st.number_input("Target year (within forecast)", value=int(gi_end),
                                      min_value=int(gi_start), max_value=int(gi_end))
        low_thr = st.slider("Low vs Medium threshold (%)", 0.0, 10.0, 4.0, 0.5)
        med_thr = st.slider("Medium vs High threshold (%)", low_thr+0.5, 25.0, 10.0, 0.5)
        show_class_legend = st.toggle("Show legend on map", True)
        class_filter = st.multiselect("Filter classes",
                                      ["High Growth","Medium Growth","Low Growth"],
                                      default=["High Growth","Medium Growth","Low Growth"])

    with st.spinner("🔎 Computing forecasts and classes…"):
        df_fc_all = fit_forecast(df, start_year=int(gi_start), end_year=int(gi_end))
        df_fc_cls = classify_growth(df_fc_all, base_year=int(base_year),
                                    target_year=int(target_year),
                                    low_upper=float(low_thr), med_upper=float(med_thr))

    df_growth = (
        df_fc_cls[df_fc_cls["year"]==int(target_year)][["state_code","state_name","growth_rate","growth_class"]]
        .drop_duplicates(subset=["state_code"]).sort_values("growth_rate", ascending=False)
    )
    if class_filter:
        df_growth = df_growth[df_growth["growth_class"].isin(class_filter)]
    df_growth["growth_rate"] = df_growth["growth_rate"].round(2)

    st.markdown(f"#### Growth Classification Table ({base_year} → {target_year})")
    st.dataframe(
        df_growth.rename(columns={"state_code":"Code","state_name":"State",
                                  "growth_rate":"Growth %","growth_class":"Class"}),
        hide_index=True, use_container_width=True
    )
    st.download_button("⬇️ Download classification (CSV)",
                       data=df_growth.to_csv(index=False).encode("utf-8"),
                       file_name=f"growth_class_{base_year}_{target_year}.csv", mime="text/csv")

    st.markdown("#### Growth Class Choropleth")
    with st.spinner("🧭 Building Classification Map…"):
        fig_class = choropleth_class(df_growth, legend=show_class_legend, height=620)
        st.plotly_chart(fig_class, use_container_width=True, config={"scrollZoom": True})
    if st.button("🔄 Reset Map Zoom", key="reset_class"):
        fig_class.update_geos(fitbounds="locations", visible=False)
        st.rerun()

    with st.expander("Class definitions", expanded=False):
        st.write(
            f"- **Low Growth:** ≤ {low_thr:.1f}%\n"
            f"- **Medium Growth:** > {low_thr:.1f}% and ≤ {med_thr:.1f}%\n"
            f"- **High Growth:** > {med_thr:.1f}%"
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
