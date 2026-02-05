import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, Input, Output

# =========================================================
# 1. LOAD & CLEAN DATA
# =========================================================
RAW_PATH = "Crash_Reporting_-_Drivers_Data.csv"
df = pd.read_csv(RAW_PATH, low_memory=False)

# ---- Parse datetime ----
df["Crash_Datetime"] = pd.to_datetime(df["Crash Date/Time"], errors="coerce")
df = df.dropna(subset=["Crash_Datetime"]).copy()

# ---- Time features ----
df["Hour"] = df["Crash_Datetime"].dt.hour + 1      # 1–24 instead of 0–23
df["DayOfWeek"] = df["Crash_Datetime"].dt.day_name()
df["MonthNum"] = df["Crash_Datetime"].dt.month
df["MonthName"] = df["Crash_Datetime"].dt.month_name()
df["Year"] = df["Crash_Datetime"].dt.year

# ---------------- LIGHT ----------------


def normalize_light(val):
    if pd.isna(val):
        return "UNKNOWN"
    s = str(val).strip().upper()

    if s in ["", "NAN"]:
        return "UNKNOWN"

    if "DAWN" in s:
        return "DAWN"
    if "DUSK" in s:
        return "DUSK"
    if "DAYLIGHT" in s:
        return "DAYLIGHT"
    if "DARK" in s:
        # merge DARK_NO_LIGHTS + DARK_UNKNOWN into DARK_UNLIT
        if "NO LIGHT" in s or "NOT LIGHTED" in s or "UNKNOWN" in s:
            return "DARK_UNLIT"
        return "DARK_LIGHTED"

    if "OTHER" in s or "UNKNOWN" in s:
        return "UNKNOWN"

    return "UNKNOWN"


df["Light_norm"] = df["Light"].map(normalize_light)

# ---------------- WEATHER ----------------


def normalize_weather(val):
    if pd.isna(val):
        return "UNKNOWN"
    s = str(val).strip().upper()

    if s in ["", "NAN"]:
        return "UNKNOWN"

    if "CROSSWIND" in s or "WIND" in s or "WINDS" in s:
        return "WIND"

    if "CLEAR" in s:
        return "CLEAR"
    if "RAIN" in s or "DRIZZLE" in s or "SHOWER" in s:
        return "RAIN"
    if "SNOW" in s or "SLEET" in s or "WINTRY" in s or "FREEZING" in s:
        return "SNOW/ICE"
    if "FOG" in s or "SMOG" in s or "SMOKE" in s:
        return "FOG"
    if "CLOUD" in s or "OVERCAST" in s:
        return "CLOUDY"
    if "OTHER" in s:
        return "OTHER"
    if "UNKNOWN" in s:
        return "UNKNOWN"

    return "OTHER"


df["Weather_norm"] = df["Weather"].map(normalize_weather)

# Remove WIND and OTHER from weather analysis entirely
df = df[~df["Weather_norm"].isin(["WIND", "OTHER"])].copy()

# ---------------- SURFACE ----------------


def normalize_surface(val):
    if pd.isna(val):
        return "UNKNOWN"
    s = str(val).strip().upper()

    if "DRY" in s:
        return "DRY"
    if "WET" in s or "WATER" in s:
        return "WET"
    if "SNOW" in s or "ICE" in s or "SLUSH" in s or "FROST" in s:
        return "SNOW/ICE"
    if "GRAVEL" in s or "MUD" in s or "DIRT" in s:
        return "LOOSE MATERIAL"
    return "OTHER"


df["Surface_norm"] = df["Surface Condition"].map(normalize_surface)

# ---------------- INJURY ----------------


def normalize_injury(val):
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().upper()

    if "NO APPARENT" in s:
        return "No Injury"
    if "POSSIBLE" in s:
        return "Possible Injury"
    if "MINOR" in s:
        return "Minor Injury"
    if "SERIOUS" in s:
        return "Serious Injury"
    if "FATAL" in s:
        return "Fatal Injury"
    return "Unknown"


df["Injury_norm"] = df["Injury Severity"].map(normalize_injury)

# ---------------- VEHICLE MAKE ----------------


def normalize_make(val):
    if pd.isna(val):
        return "UNKNOWN"
    mapping = {
        "TOYT": "TOYOTA",
        "HOND": "HONDA",
        "CHEV": "CHEVROLET",
        "MERZ": "MERCEDES-BENZ",
        "VOLK": "VOLKSWAGEN",
    }
    s = str(val).strip().upper()
    if s in mapping:
        return mapping[s]
    if s in ["UNK", "UNKNOWN"]:
        return "UNKNOWN"
    return s


df["Vehicle_Make_norm"] = df["Vehicle Make"].map(normalize_make)

# ---------------- VEHICLE YEAR ----------------
df["Vehicle_Year_clean"] = df["Vehicle Year"].where(
    (df["Vehicle Year"] >= 1980) & (df["Vehicle Year"] <= 2025)
)

# ORDER DAYS
day_order = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
df["DayOfWeek"] = pd.Categorical(
    df["DayOfWeek"], categories=day_order, ordered=True)

# =========================================================
# 2. SMALL KPI NUMBERS FOR HEADER
# =========================================================
total_crashes = len(df)
date_min = df["Crash_Datetime"].dt.date.min()
date_max = df["Crash_Datetime"].dt.date.max()

# busiest hour
by_hour_tmp = df.groupby("Hour").size().reset_index(name="Crashes")
peak_hour_row = by_hour_tmp.loc[by_hour_tmp["Crashes"].idxmax()]
peak_hour = int(peak_hour_row["Hour"])
peak_hour_crashes = int(peak_hour_row["Crashes"])

# riskiest weather by % serious+fatal (excluding UNKNOWN now)
df["is_severe"] = df["Injury_norm"].isin(
    ["Serious Injury", "Fatal Injury"]).astype(int)
df_weather_injury = df[df["Weather_norm"] != "UNKNOWN"].copy()

weather_severity_tmp = (
    df_weather_injury.groupby("Weather_norm")
    .agg(total=("Injury_norm", "size"), severe=("is_severe", "sum"))
    .reset_index()
)
weather_severity_tmp["pct_severe"] = np.where(
    weather_severity_tmp["total"] > 0,
    weather_severity_tmp["severe"] / weather_severity_tmp["total"] * 100,
    0.0,
)
riskiest_weather_row = weather_severity_tmp.loc[weather_severity_tmp["pct_severe"].idxmax(
)]
riskiest_weather = riskiest_weather_row["Weather_norm"]
riskiest_weather_pct = riskiest_weather_row["pct_severe"]

# =========================================================
# 3. COMMON FIGURE STYLING
# =========================================================
COLORS = {
    "bg": "#f9f4e8",
    "card": "#ffffff",
    "primary": "#d35400",
    "accent": "#f1c40f",
    "text": "#333333",
    "muted": "#777777",
}


def style_figure(fig, legend_bottom=False):
    fig.update_layout(
        template="simple_white",
        font=dict(family="Arial", size=12, color=COLORS["text"]),
        title_font=dict(size=18, color=COLORS["primary"], family="Arial"),
        plot_bgcolor=COLORS["card"],
        paper_bgcolor=COLORS["card"],
        margin=dict(t=60, l=40, r=20, b=40),
    )
    if legend_bottom:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
            )
        )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    return fig

# =========================================================
# 4. BUILD FIGURES
# =========================================================

# ---------- STORY 1: When Do Crashes Happen the Most? ----------


# Line: crashes by hour
by_hour = df.groupby("Hour").size().reset_index(name="Crashes")
fig_hour_line = px.line(
    by_hour,
    x="Hour",
    y="Crashes",
    markers=True,
    title="Crashes by Hour of Day",
)
fig_hour_line.update_traces(line=dict(width=3), marker=dict(size=6))
fig_hour_line.update_layout(xaxis=dict(dtick=1))
fig_hour_line = style_figure(fig_hour_line)
fig_hour_line.update_layout(height=360)

# Heatmap: DayOfWeek x Hour
pivot_dow_hour = df.groupby(
    ["DayOfWeek", "Hour"]).size().reset_index(name="Crashes")
fig_dow_hour = px.density_heatmap(
    pivot_dow_hour,
    x="Hour",
    y="DayOfWeek",
    z="Crashes",
    color_continuous_scale="YlOrRd",
    title="Crash Density: Day of Week vs Hour",
)
fig_dow_hour.update_layout(xaxis=dict(dtick=1))
fig_dow_hour = style_figure(fig_dow_hour)
fig_dow_hour.update_layout(height=360)

# ---------- Monthly trend data prep (for dropdown-driven chart) ----------
MONTH_LABEL_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

df_month_year = (
    df.groupby(["Year", "MonthNum"])
    .size()
    .reset_index(name="Crashes")
)
df_month_year["MonthLabel"] = df_month_year["MonthNum"].map(MONTH_LABEL_MAP)

unique_years = sorted(df_month_year["Year"].unique())
year_options = (
    [{"label": "All Years", "value": "ALL"}] +
    [{"label": str(int(y)), "value": str(int(y))} for y in unique_years]
)

# Bar chart – total crashes by light condition (UNKNOWN removed)
light_label_map = {
    "DAYLIGHT": "Daylight",
    "DARK_LIGHTED": "Dark<br>(Lit)",
    "DARK_UNLIT": "Dark<br>(No Light)",
    "DAWN": "Dawn",
    "DUSK": "Dusk",
}
light_order = ["DAYLIGHT", "DARK_LIGHTED", "DARK_UNLIT", "DAWN", "DUSK"]

light_color_map = {
    "DAYLIGHT": "#e67e22",
    "DARK_LIGHTED": "#2980b9",
    "DARK_UNLIT": "#c0392b",
    "DAWN": "#27ae60",
    "DUSK": "#f39c12",
}

light_counts = (
    df[df["Light_norm"] != "UNKNOWN"]["Light_norm"]
    .value_counts()
    .reindex(light_order, fill_value=0)
    .reset_index()
)
light_counts.columns = ["Light_norm", "Crashes"]
light_counts["LightLabel"] = light_counts["Light_norm"].map(light_label_map)

fig_light_bar = px.bar(
    light_counts,
    x="LightLabel",
    y="Crashes",
    color="Light_norm",
    title="Total Crashes by Light Condition",
    category_orders={"LightLabel": [light_label_map[k] for k in light_order]},
    color_discrete_map=light_color_map,
)

fig_light_bar = style_figure(fig_light_bar)

# spacing so legend and x-axis title do not collide
fig_light_bar.update_layout(
    height=360,
    margin=dict(t=60, l=40, r=20, b=180),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.55,
        xanchor="center",
        x=0.5,
        title_text="Light Condition",
    ),
    xaxis=dict(
        title=dict(text="Light Condition", standoff=40),
        tickangle=0,
        tickfont=dict(size=12),
    ),
    yaxis_title="Number of Crashes",
)

# ---------- STORY 2: Weather and Danger ----------

df["Weather_norm"] = df["Weather_norm"].astype(str)
df["Injury_norm"] = df["Injury_norm"].astype(str)

df_weather_injury = df[df["Weather_norm"] != "UNKNOWN"].copy()
weather_order_injury = ["CLEAR", "RAIN", "FOG", "CLOUDY", "SNOW/ICE"]

# Bar: crashes by weather
weather_order_bar = ["CLEAR", "RAIN", "FOG", "CLOUDY", "SNOW/ICE"]
weather_counts = (
    df[df["Weather_norm"] != "UNKNOWN"]["Weather_norm"]
    .value_counts()
    .reindex(weather_order_bar, fill_value=0)
    .reset_index()
)
weather_counts.columns = ["Weather_norm", "Crashes"]

fig_weather_bar = px.bar(
    weather_counts,
    x="Weather_norm",
    y="Crashes",
    title="Crashes by Weather Condition",
    color="Weather_norm",
    color_discrete_sequence=["#f1c40f", "#3498db",
                             "#9b59b6", "#e67e22", "#c0392b"],
)
fig_weather_bar = style_figure(fig_weather_bar)
fig_weather_bar.update_layout(
    showlegend=False,
    xaxis_title="Weather",
    yaxis_title="Number of Crashes",
    height=360,
)

# Stacked bar: injury severity by weather
severity_order = ["No Injury", "Possible Injury",
                  "Minor Injury", "Serious Injury", "Fatal Injury"]

ct_weather_injury = pd.crosstab(
    df_weather_injury["Weather_norm"],
    df_weather_injury["Injury_norm"]
)

for sev in severity_order:
    if sev not in ct_weather_injury.columns:
        ct_weather_injury[sev] = 0

ct_weather_injury = (
    ct_weather_injury
    .reindex(weather_order_injury, fill_value=0)[severity_order]
    .reset_index()
)

fig_weather_injury = px.bar(
    ct_weather_injury.melt(id_vars="Weather_norm",
                           var_name="Injury", value_name="Count"),
    x="Weather_norm",
    y="Count",
    color="Injury",
    title="Injury Severity by Weather",
)
fig_weather_injury = style_figure(fig_weather_injury, legend_bottom=True)
fig_weather_injury.update_layout(
    xaxis_title="Weather",
    yaxis_title="Number of Crashes",
    height=360,
    margin=dict(t=60, l=40, r=20, b=130),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.4,
        xanchor="center",
        x=0.5,
        title_text="",
    ),
)

# % serious+fatal per weather
weather_severity = (
    df_weather_injury.groupby("Weather_norm")
    .agg(total_crashes=("Injury_norm", "size"),
         severe_crashes=("is_severe", "sum"))
)

weather_severity = (
    weather_severity
    .reindex(weather_order_injury, fill_value=0)
    .reset_index()
)

weather_severity["severe_pct"] = np.where(
    weather_severity["total_crashes"] > 0,
    weather_severity["severe_crashes"] /
    weather_severity["total_crashes"] * 100,
    0.0,
)

fig_weather_severe_pct = px.bar(
    weather_severity,
    x="Weather_norm",
    y="severe_pct",
    title="Share of Serious/Fatal Crashes by Weather",
    color="Weather_norm",
    color_discrete_sequence=["#f1c40f", "#3498db",
                             "#9b59b6", "#e67e22", "#c0392b"],
)
fig_weather_severe_pct = style_figure(fig_weather_severe_pct)
fig_weather_severe_pct.update_layout(
    showlegend=False,
    xaxis_title="Weather",
    yaxis_title="Serious/Fatal (%)",
    height=360,
)

# Weather vs Surface Condition
combo = df_weather_injury.groupby(
    ["Weather_norm", "Surface_norm"]).size().reset_index(name="Count")
fig_weather_surface = px.scatter(
    combo,
    x="Weather_norm",
    y="Surface_norm",
    size="Count",
    size_max=40,
    title="Weather vs Surface Condition (Bubble Size = Crashes)",
)
fig_weather_surface = style_figure(fig_weather_surface)
fig_weather_surface.update_layout(
    xaxis_title="Weather",
    yaxis_title="Surface Condition",
    height=360,
)

# ---------- STORY 3: Vehicle Story ----------

# FIX: exclude Unknown severity and missing years from the box plot
df_box = df[
    df["Vehicle_Year_clean"].notna() &
    df["Injury_norm"].isin(severity_order)
].copy()

fig_vehicle_box = px.box(
    df_box,
    x="Injury_norm",
    y="Vehicle_Year_clean",
    category_orders={"Injury_norm": severity_order},
    title="Vehicle Year vs Injury Severity",
)
fig_vehicle_box = style_figure(fig_vehicle_box)
fig_vehicle_box.update_layout(
    xaxis_title="Injury Severity",
    yaxis_title="Vehicle Year",
    height=360,
)

top_n = 15
make_counts = df["Vehicle_Make_norm"].value_counts().head(top_n).reset_index()
make_counts.columns = ["Vehicle_Make_norm", "Crashes"]
fig_make_bar = px.bar(
    make_counts,
    x="Vehicle_Make_norm",
    y="Crashes",
    title=f"Top {top_n} Crash-Involved Vehicle Makes",
)
fig_make_bar = style_figure(fig_make_bar)
fig_make_bar.update_layout(
    xaxis_title="Vehicle Make",
    yaxis_title="Number of Crashes",
    height=360,
)

top_body = df["Vehicle Body Type"].value_counts().head(8).index
top_collision = df["Collision Type"].value_counts().head(6).index
df_bc = df[df["Vehicle Body Type"].isin(
    top_body) & df["Collision Type"].isin(top_collision)].copy()
pivot_body_collision = (
    df_bc.groupby(["Vehicle Body Type", "Collision Type"])
    .size()
    .reset_index(name="Crashes")
)

fig_body_collision = px.density_heatmap(
    pivot_body_collision,
    x="Collision Type",
    y="Vehicle Body Type",
    z="Crashes",
    color_continuous_scale="YlOrRd",
    title="Vehicle Body Type vs Collision Type",
)
fig_body_collision = style_figure(fig_body_collision)
fig_body_collision.update_layout(height=360)

# =========================================================
# 5. DASH APP LAYOUT – INFOGRAPHIC STYLE
# =========================================================
app = Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": COLORS["bg"],
        "minHeight": "100vh",
        "padding": "30px",
    },
    children=[
        html.Div(
            style={
                "maxWidth": "1400px",
                "margin": "0 auto",
                "backgroundColor": COLORS["card"],
                "borderRadius": "16px",
                "padding": "24px 28px 32px 28px",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.12)",
            },
            children=[
                # HEADER
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between",
                           "alignItems": "baseline", "gap": "12px"},
                    children=[
                        html.Div(
                            children=[
                                html.H1(
                                    "Crash Insights Dashboard",
                                    style={
                                        "margin": 0,
                                        "fontFamily": "Arial",
                                        "fontSize": "28px",
                                        "color": COLORS["primary"],
                                    },
                                ),
                                html.P(
                                    "A data-driven exploration of how time, weather, road conditions, and vehicle factors shape crash outcomes.",
                                    style={
                                        "marginTop": "6px", "color": COLORS["muted"], "fontSize": "14px"},
                                ),
                            ]
                        ),
                        html.Div(
                            style={
                                "fontSize": "12px",
                                "color": COLORS["muted"],
                                "textAlign": "right",
                            },
                            children=[
                                html.Div(
                                    f"Data period: {date_min} to {date_max}"),
                                html.Div(
                                    f"Total records: {total_crashes:,} crashes"),
                            ],
                        ),
                    ],
                ),

                html.Hr(style={"margin": "18px 0 16px 0",
                        "borderColor": "#f0e1c5"}),

                # KPI CARDS
                html.Div(
                    style={"display": "flex",
                           "flexWrap": "wrap", "gap": "12px"},
                    children=[
                        html.Div(
                            style={
                                "flex": "1",
                                "minWidth": "180px",
                                "backgroundColor": "#fff7e0",
                                "borderRadius": "12px",
                                "padding": "12px 16px",
                            },
                            children=[
                                html.Div("Total Crashes", style={
                                         "fontSize": "11px", "color": COLORS["muted"]}),
                                html.Div(f"{total_crashes:,}", style={
                                         "fontSize": "22px", "fontWeight": "bold", "color": COLORS["primary"]}),
                            ],
                        ),
                        html.Div(
                            style={
                                "flex": "1",
                                "minWidth": "180px",
                                "backgroundColor": "#e8f6ff",
                                "borderRadius": "12px",
                                "padding": "12px 16px",
                            },
                            children=[
                                html.Div("Peak Crash Hour", style={
                                         "fontSize": "11px", "color": COLORS["muted"]}),
                                html.Div(
                                    f"{peak_hour}:00  (≈ {peak_hour_crashes:,} crashes)",
                                    style={
                                        "fontSize": "18px", "fontWeight": "bold", "color": "#2980b9"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "flex": "1",
                                "minWidth": "180px",
                                "backgroundColor": "#ffeef0",
                                "borderRadius": "12px",
                                "padding": "12px 16px",
                            },
                            children=[
                                html.Div("Riskiest Weather (by severity share)", style={
                                         "fontSize": "11px", "color": COLORS["muted"]}),
                                html.Div(
                                    f"{riskiest_weather}  –  {riskiest_weather_pct:.1f}% severe",
                                    style={
                                        "fontSize": "18px", "fontWeight": "bold", "color": "#c0392b"},
                                ),
                            ],
                        ),
                    ],
                ),

                html.Br(),

                # TABS
                dcc.Tabs(
                    children=[
                        # ================= TAB 1 =================
                        dcc.Tab(
                            label="1 • When Do Crashes Happen?",
                            children=[
                                html.Br(),
                                # Row 1: Hourly line + heatmap
                                html.Div(
                                    style={"display": "flex",
                                           "flexWrap": "wrap", "gap": "16px"},
                                    children=[
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[dcc.Graph(figure=fig_hour_line, style={
                                                                "height": "360px"})],
                                        ),
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[dcc.Graph(figure=fig_dow_hour, style={
                                                                "height": "360px"})],
                                        ),
                                    ],
                                ),
                                # Row 2: Monthly trend (with dropdown) + light condition bar
                                html.Div(
                                    style={"display": "flex", "flexWrap": "wrap",
                                           "gap": "16px", "marginTop": "10px"},
                                    children=[
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[
                                                html.Div(
                                                    style={
                                                        "marginBottom": "8px"},
                                                    children=[
                                                        html.Label(
                                                            "Select Year",
                                                            style={
                                                                "fontSize": "13px",
                                                                "color": COLORS["primary"],
                                                                "fontWeight": "bold",
                                                            },
                                                        ),
                                                        dcc.Dropdown(
                                                            id="year-dropdown",
                                                            options=year_options,
                                                            value="ALL",
                                                            clearable=False,
                                                            style={
                                                                "fontSize": "13px"},
                                                        ),
                                                    ],
                                                ),
                                                dcc.Graph(
                                                    id="monthly-trend-graph",
                                                    style={"height": "360px"},
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[dcc.Graph(figure=fig_light_bar, style={
                                                                "height": "360px"})],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # ================= TAB 2 =================
                        dcc.Tab(
                            label="2 • Weather and Danger",
                            children=[
                                html.Br(),
                                html.Div(
                                    style={"display": "flex",
                                           "flexWrap": "wrap", "gap": "16px"},
                                    children=[
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[dcc.Graph(figure=fig_weather_bar, style={
                                                                "height": "360px"})],
                                        ),
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[dcc.Graph(figure=fig_weather_injury, style={
                                                                "height": "360px"})],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"display": "flex", "flexWrap": "wrap",
                                           "gap": "16px", "marginTop": "10px"},
                                    children=[
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[dcc.Graph(figure=fig_weather_severe_pct, style={
                                                                "height": "360px"})],
                                        ),
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[dcc.Graph(figure=fig_weather_surface, style={
                                                                "height": "360px"})],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # ================= TAB 3 =================
                        dcc.Tab(
                            label="3 • Vehicle Story",
                            children=[
                                html.Br(),
                                html.Div(
                                    style={"display": "flex",
                                           "flexWrap": "wrap", "gap": "16px"},
                                    children=[
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[dcc.Graph(figure=fig_vehicle_box, style={
                                                                "height": "360px"})],
                                        ),
                                        html.Div(
                                            style={"flex": "1 1 380px"},
                                            children=[dcc.Graph(figure=fig_make_bar, style={
                                                                "height": "360px"})],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"marginTop": "12px"},
                                    children=[dcc.Graph(figure=fig_body_collision, style={
                                                        "height": "360px"})],
                                ),

                                # ===== STORYTELLING SUMMARY CARD =====
                                html.Div(
                                    style={
                                        "marginTop": "18px",
                                        "backgroundColor": "#fff7e0",
                                        "borderRadius": "12px",
                                        "padding": "16px 18px",
                                        "border": "1px solid #f0d8a8",
                                    },
                                    children=[
                                        html.H3(
                                            "Story Summary",
                                            style={
                                                "marginTop": 0,
                                                "marginBottom": "6px",
                                                "color": COLORS["primary"],
                                                "fontSize": "18px",
                                            },
                                        ),
                                        html.P(
                                            "Crashes peak during evening rush hours and on Fridays, with daylight showing the "
                                            "highest counts simply because that is when most driving occurs.",
                                            style={"marginBottom": "4px",
                                                   "fontSize": "13px"},
                                        ),
                                        html.P(
                                            "Rain, fog, and especially snow or ice do not produce the most crashes, but they "
                                            "create a much higher share of serious and fatal injuries—particularly on wet or icy roads.",
                                            style={"marginBottom": "4px",
                                                   "fontSize": "13px"},
                                        ),
                                        html.P(
                                            "Unlit dark conditions and older vehicles are associated with more severe outcomes, "
                                            "while sedans dominate crash counts mainly because they are the most common vehicles.",
                                            style={"marginBottom": "4px",
                                                   "fontSize": "13px"},
                                        ),
                                        html.P(
                                            "Overall, the dashboard shows how time of day, weather, surface condition, and vehicle "
                                            "characteristics combine to shape both the likelihood and the severity of crashes.",
                                            style={"marginBottom": "0px",
                                                   "fontSize": "13px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ]
                ),
            ],
        ),
    ],
)

# =========================================================
# 6. CALLBACKS
# =========================================================


@app.callback(
    Output("monthly-trend-graph", "figure"),
    Input("year-dropdown", "value"),
)
def update_monthly_trend(selected_year):
    if selected_year is not None and selected_year != "ALL":
        year_int = int(selected_year)
        df_filtered = df_month_year[df_month_year["Year"] == year_int].copy()
        title = f"Monthly Crash Trend – {year_int}"
    else:
        df_filtered = (
            df_month_year.groupby("MonthNum", as_index=False)["Crashes"].sum()
        )
        df_filtered["MonthLabel"] = df_filtered["MonthNum"].map(
            MONTH_LABEL_MAP)
        title = "Monthly Crash Trend – All Years"

    fig = px.line(
        df_filtered,
        x="MonthLabel",
        y="Crashes",
        markers=True,
        title=title,
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    fig = style_figure(fig)
    fig.update_layout(height=360)
    return fig


# =========================================================
# 7. RUN APP
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
