"""
SmartETA — ML-Driven Real-Time Bus Arrival Prediction & Transit Monitoring
---------------------------------------------------------------------------
A Streamlit application for a smart public-transport platform covering
Bangalore transit routes, with live tracking and ML-based prediction.

Features implemented:
  * Real-time occupancy tracking (simulated live buses)
  * AI-based crowd prediction (RandomForest trained on synthetic history)
  * Live GPS tracking (interpolated positions on a map)
  * Passenger mobile app view
  * Driver dashboard
  * Admin dashboard with analytics
  * Automated alerts for overcrowded buses
  * Peak-hour demand forecasting
  * Dynamic fleet allocation recommendations

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from auth import verify_login, create_user

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartETA",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUS_CAPACITY = 50
BUSES_PER_ROUTE = 3

# ----------------------------------------------------------------------------
# CUSTOM THEME (CSS injection)
# ----------------------------------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600 !important;
        }

        /* App background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #0f172a 0%, #131c31 100%);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #0b1120;
            border-right: 1px solid #1e293b;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: #161f36;
            border: 1px solid #26314d;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
        }
        [data-testid="stMetricValue"] {
            color: #38bdf8 !important;
            font-family: 'Poppins', sans-serif !important;
        }

        /* Body text */
        p, span, label, .stMarkdown {
            color: #cbd5e1;
        }
        h1, h2, h3, h4 {
            color: #f1f5f9 !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #38bdf8);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            transition: transform 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(56,189,248,0.35);
        }

        /* Dataframes / tables */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #26314d;
        }

        /* Radio nav in sidebar */
        [data-testid="stSidebar"] .stRadio label {
            font-size: 0.95rem;
        }

        /* Alerts */
        .stAlert {
            border-radius: 10px;
        }

        /* Divider spacing */
        hr {
            border-color: #26314d !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# ROUTE NETWORK (Bangalore, BMTC-corridor-inspired coordinates for demo)
# ----------------------------------------------------------------------------
ROUTES = {
    "Route 1 - Majestic to Koramangala": [
        ("Majestic (KBS)", 12.9767, 77.5713),
        ("Shivajinagar", 12.9855, 77.6057),
        ("MG Road", 12.9757, 77.6098),
        ("Koramangala", 12.9352, 77.6245),
    ],
    "Route 2 - Whitefield to MG Road": [
        ("Whitefield", 12.9698, 77.7500),
        ("Marathahalli", 12.9569, 77.7011),
        ("Indiranagar", 12.9719, 77.6412),
        ("MG Road", 12.9757, 77.6098),
    ],
    "Route 3 - Electronic City to Silk Board": [
        ("Electronic City", 12.8452, 77.6602),
        ("Bommanahalli", 12.8988, 77.6146),
        ("BTM Layout", 12.9166, 77.6101),
        ("Silk Board", 12.9172, 77.6228),
    ],
    "Route 4 - Jayanagar to Indiranagar": [
        ("Jayanagar", 12.9308, 77.5838),
        ("Lalbagh", 12.9507, 77.5848),
        ("Trinity Circle", 12.9757, 77.6098),
        ("Indiranagar", 12.9719, 77.6412),
    ],
    "Route 5 - Hebbal to Marathahalli": [
        ("Hebbal", 13.0355, 77.5970),
        ("Nagawara", 13.0359, 77.6206),
        ("KR Puram", 13.0027, 77.6975),
        ("Marathahalli", 12.9569, 77.7011),
    ],
}
ROUTE_NAMES = list(ROUTES.keys())

# ----------------------------------------------------------------------------
# SYNTHETIC MODEL TRAINING (AI-based crowd prediction)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_crowd_model():
    rng = np.random.default_rng(42)
    route_popularity = {r: rng.uniform(0.7, 1.35) for r in ROUTE_NAMES}

    rows = []
    for _ in range(6000):
        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))
        route = rng.choice(ROUTE_NAMES)
        weekend_factor = 0.55 if dow >= 5 else 1.0
        morning = np.exp(-((hour - 8) ** 2) / 8)
        evening = np.exp(-((hour - 18) ** 2) / 8)
        base = (0.22 + 0.68 * (morning + evening)) * route_popularity[route] * weekend_factor
        noise = rng.normal(0, 0.05)
        occ = float(np.clip(base + noise, 0.03, 1.0) * 100)
        rows.append([hour, dow, route, occ])

    df = pd.DataFrame(rows, columns=["hour", "dow", "route", "occupancy"])
    X = pd.get_dummies(df[["hour", "dow", "route"]], columns=["route"])
    y = df["occupancy"]

    model = RandomForestRegressor(n_estimators=120, max_depth=9, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist(), route_popularity


def predict_occupancy(model, feature_cols, hour, dow, route):
    row = {c: 0 for c in feature_cols}
    row["hour"] = hour % 24
    row["dow"] = dow % 7
    col = f"route_{route}"
    if col in row:
        row[col] = 1
    X = pd.DataFrame([row])[feature_cols]
    return float(model.predict(X)[0])


@st.cache_data(show_spinner=False)
def compare_models():
    """
    Train Linear Regression, Random Forest, and XGBoost on the same
    occupancy-prediction dataset with a proper train/test split, and
    return a metrics comparison table (MAE, RMSE, R2).

    This is the model-comparison study referenced in the paper's
    methodology/results section.
    """
    rng = np.random.default_rng(42)
    route_popularity = {r: rng.uniform(0.7, 1.35) for r in ROUTE_NAMES}

    rows = []
    for _ in range(6000):
        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))
        route = rng.choice(ROUTE_NAMES)
        weekend_factor = 0.55 if dow >= 5 else 1.0
        morning = np.exp(-((hour - 8) ** 2) / 8)
        evening = np.exp(-((hour - 18) ** 2) / 8)
        base = (0.22 + 0.68 * (morning + evening)) * route_popularity[route] * weekend_factor
        noise = rng.normal(0, 0.05)
        occ = float(np.clip(base + noise, 0.03, 1.0) * 100)
        rows.append([hour, dow, route, occ])

    df = pd.DataFrame(rows, columns=["hour", "dow", "route", "occupancy"])
    X = pd.get_dummies(df[["hour", "dow", "route"]], columns=["route"])
    y = df["occupancy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=9, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    }

    results = []
    for name, mdl in candidates.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2 Score": round(r2, 3)})

    return pd.DataFrame(results)


# ----------------------------------------------------------------------------
# SESSION STATE / SIMULATION HELPERS
# ----------------------------------------------------------------------------
def init_state():
    if "buses" not in st.session_state:
        rng = np.random.default_rng()
        buses = []
        bid = 1
        for route in ROUTE_NAMES:
            for _ in range(BUSES_PER_ROUTE):
                buses.append(
                    {
                        "id": f"BUS-{bid:03d}",
                        "route": route,
                        "progress": float(rng.uniform(0, 1)),
                        "speed": float(rng.uniform(35, 70)),  # progress units per tick (x1000)
                        "occupancy": float(rng.uniform(25, 55)),
                        "driver": f"Driver {bid}",
                    }
                )
                bid += 1
        st.session_state.buses = pd.DataFrame(buses)

    if "sim_time" not in st.session_state:
        st.session_state.sim_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    if "tick" not in st.session_state:
        st.session_state.tick = 0

    if "alerts_log" not in st.session_state:
        st.session_state.alerts_log = []


def seed_history(model, feature_cols):
    if "history" in st.session_state:
        return
    records = []
    start = st.session_state.sim_time - timedelta(hours=24)
    t = start
    rng = np.random.default_rng(7)
    while t <= st.session_state.sim_time:
        for _, row in st.session_state.buses.iterrows():
            target = predict_occupancy(model, feature_cols, t.hour, t.weekday(), row["route"])
            occ = float(np.clip(target + rng.normal(0, 5), 2, 100))
            records.append({"time": t, "bus": row["id"], "route": row["route"], "occupancy": occ})
        t += timedelta(minutes=30)
    st.session_state.history = pd.DataFrame(records)


def interpolate_position(stops, progress):
    n = len(stops) - 1
    progress = progress % 1.0
    seg = progress * n
    idx = min(int(seg), n - 1)
    frac = seg - idx
    _, lat1, lon1 = stops[idx]
    _, lat2, lon2 = stops[idx + 1]
    lat = lat1 + (lat2 - lat1) * frac
    lon = lon1 + (lon2 - lon1) * frac
    return lat, lon, stops[idx + 1][0]


def advance_simulation(model, feature_cols):
    st.session_state.tick += 1
    st.session_state.sim_time += timedelta(minutes=15)
    hour, dow = st.session_state.sim_time.hour, st.session_state.sim_time.weekday()

    df = st.session_state.buses
    new_occ, new_progress = [], []
    for _, row in df.iterrows():
        target = predict_occupancy(model, feature_cols, hour, dow, row["route"])
        new_val = row["occupancy"] + (target - row["occupancy"]) * 0.35 + np.random.normal(0, 4)
        new_occ.append(float(np.clip(new_val, 0, 100)))
        new_progress.append((row["progress"] + row["speed"] / 1000.0) % 1.0)
    df["occupancy"] = new_occ
    df["progress"] = new_progress
    st.session_state.buses = df

    new_hist = pd.DataFrame(
        [
            {"time": st.session_state.sim_time, "bus": r["id"], "route": r["route"], "occupancy": r["occupancy"]}
            for _, r in df.iterrows()
        ]
    )
    st.session_state.history = pd.concat([st.session_state.history, new_hist], ignore_index=True)
    if len(st.session_state.history) > 4000:
        st.session_state.history = st.session_state.history.tail(4000).reset_index(drop=True)

    overcrowded = df[df["occupancy"] >= 85]
    for _, r in overcrowded.iterrows():
        st.session_state.alerts_log.append(
            {
                "time": st.session_state.sim_time.strftime("%Y-%m-%d %H:%M"),
                "bus": r["id"],
                "route": r["route"],
                "occupancy": round(r["occupancy"], 1),
            }
        )
    st.session_state.alerts_log = st.session_state.alerts_log[-100:]


def get_live_bus_df():
    df = st.session_state.buses.copy()
    lats, lons, next_stops, statuses = [], [], [], []
    for _, row in df.iterrows():
        stops = ROUTES[row["route"]]
        lat, lon, nxt = interpolate_position(stops, row["progress"])
        lats.append(lat)
        lons.append(lon)
        next_stops.append(nxt)
        occ = row["occupancy"]
        if occ >= 85:
            statuses.append("Overcrowded")
        elif occ >= 60:
            statuses.append("Moderate")
        else:
            statuses.append("Comfortable")
    df["lat"] = lats
    df["lon"] = lons
    df["next_stop"] = next_stops
    df["status"] = statuses
    df["passengers"] = (df["occupancy"] / 100 * BUS_CAPACITY).round().astype(int)
    return df


STATUS_COLORS = {
    "Comfortable": [34, 197, 94],
    "Moderate": [234, 179, 8],
    "Overcrowded": [239, 68, 68],
}


def color_for_status(status):
    return STATUS_COLORS.get(status, [100, 100, 100])


# ----------------------------------------------------------------------------
# PAGE RENDERERS
# ----------------------------------------------------------------------------
def render_home(live_df):
    st.title("🚌 SmartETA")
    st.caption("Real-time bus tracking and ML-based arrival prediction for Bangalore transit routes.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Buses", len(live_df))
    c2.metric("Avg Occupancy", f"{live_df['occupancy'].mean():.0f}%")
    c3.metric("Overcrowded Buses", int((live_df["status"] == "Overcrowded").sum()))
    c4.metric("Routes Live", live_df["route"].nunique())

    st.markdown("---")
    left, right = st.columns([2, 1])

    with left:
        st.subheader("📍 Live Fleet Map")
        map_df = live_df.copy()
        map_df["color"] = map_df["status"].apply(color_for_status)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=110,
            pickable=True,
        )
        view = pdk.ViewState(latitude=12.9716, longitude=77.5946, zoom=10.3, pitch=0)
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                map_style=None,
                tooltip={"text": "{id}\nRoute: {route}\nOccupancy: {occupancy}%\nStatus: {status}"},
            )
        )
        legend_cols = st.columns(3)
        legend_cols[0].markdown("🟢 Comfortable (<60%)")
        legend_cols[1].markdown("🟡 Moderate (60-85%)")
        legend_cols[2].markdown("🔴 Overcrowded (85%+)")

    with right:
        st.subheader("Route-wise Avg Occupancy")
        route_avg = live_df.groupby("route")["occupancy"].mean().reset_index()
        fig = px.bar(
            route_avg,
            x="occupancy",
            y="route",
            orientation="h",
            color="occupancy",
            color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
            range_color=[0, 100],
        )
        fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_passenger(live_df):
    st.title("📱 Passenger Mobile App")
    st.caption("Check live occupancy before you board — pick the least crowded bus.")

    route = st.selectbox("Select your route", ROUTE_NAMES)
    route_buses = live_df[live_df["route"] == route].copy().sort_values("progress")

    if route_buses.empty:
        st.info("No buses currently active on this route.")
        return

    best_bus = route_buses.loc[route_buses["occupancy"].idxmin()]
    st.success(
        f"✅ Recommended: **{best_bus['id']}** is the least crowded "
        f"({best_bus['occupancy']:.0f}% full) heading to **{best_bus['next_stop']}**."
    )

    for _, bus in route_buses.iterrows():
        with st.container(border=True):
            cols = st.columns([1.2, 2, 1, 1])
            cols[0].markdown(f"**{bus['id']}**")
            cols[1].progress(min(int(bus["occupancy"]), 100), text=f"{bus['occupancy']:.0f}% full · {bus['passengers']}/{BUS_CAPACITY} seats")
            eta_min = int((1 - (bus["progress"] * len(ROUTES[route]) % 1)) * 8) + 1
            cols[2].markdown(f"Next: **{bus['next_stop']}**\nETA ~{eta_min} min")
            badge = {"Comfortable": "🟢", "Moderate": "🟡", "Overcrowded": "🔴"}[bus["status"]]
            cols[3].markdown(f"{badge} {bus['status']}")

    st.markdown("---")
    st.subheader("Route Map")
    map_df = route_buses.copy()
    map_df["color"] = map_df["status"].apply(color_for_status)
    stops = ROUTES[route]
    stops_df = pd.DataFrame(stops, columns=["name", "lat", "lon"])
    bus_layer = pdk.Layer("ScatterplotLayer", data=map_df, get_position="[lon, lat]", get_fill_color="color", get_radius=120, pickable=True)
    stop_layer = pdk.Layer("ScatterplotLayer", data=stops_df, get_position="[lon, lat]", get_fill_color=[59, 130, 246], get_radius=60)
    path_df = pd.DataFrame({"path": [[[lon, lat] for _, lat, lon in stops]]})
    path_layer = pdk.Layer("PathLayer", data=path_df, get_path="path", get_width=25, get_color=[59, 130, 246])
    view = pdk.ViewState(latitude=stops_df["lat"].mean(), longitude=stops_df["lon"].mean(), zoom=11.5)
    st.pydeck_chart(pdk.Deck(layers=[path_layer, stop_layer, bus_layer], initial_view_state=view, map_style=None,
                             tooltip={"text": "{id}\nOccupancy: {occupancy}%"}))


def render_driver(live_df, model, feature_cols):
    st.title("🚍 Driver Dashboard")

    bus_id = st.selectbox("Select your Bus ID", live_df["id"].tolist())
    bus = live_df[live_df["id"] == bus_id].iloc[0]

    if bus["status"] == "Overcrowded":
        st.error(f"⚠️ ALERT: {bus_id} is overcrowded ({bus['occupancy']:.0f}%). Consider requesting backup or skipping non-essential stops.")
    elif bus["status"] == "Moderate":
        st.warning(f"Occupancy is moderate ({bus['occupancy']:.0f}%). Keep monitoring.")
    else:
        st.success(f"Occupancy is comfortable ({bus['occupancy']:.0f}%).")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=bus["occupancy"],
                number={"suffix": "%"},
                title={"text": f"{bus_id} Occupancy"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f2937"},
                    "steps": [
                        {"range": [0, 60], "color": "#bbf7d0"},
                        {"range": [60, 85], "color": "#fde68a"},
                        {"range": [85, 100], "color": "#fecaca"},
                    ],
                },
            )
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Passengers Onboard", f"{bus['passengers']} / {BUS_CAPACITY}")
        st.metric("Next Stop", bus["next_stop"])
        st.metric("Route", bus["route"])

    with c2:
        st.subheader("Occupancy Trend (last 24h simulated)")
        hist = st.session_state.history
        bus_hist = hist[hist["bus"] == bus_id].sort_values("time")
        if not bus_hist.empty:
            fig2 = px.line(bus_hist, x="time", y="occupancy")
            fig2.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Overcrowding threshold")
            fig2.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)

        next_hour = (st.session_state.sim_time + timedelta(hours=1))
        predicted = predict_occupancy(model, feature_cols, next_hour.hour, next_hour.weekday(), bus["route"])
        st.info(f"🤖 AI Prediction: expected occupancy on this route in ~1 hour is **{predicted:.0f}%**.")


def render_model_comparison():
    st.title("📊 Model Comparison")
    st.caption("Evaluating candidate models for occupancy/ETA prediction on a held-out test split (80/20).")

    with st.spinner("Training and evaluating models..."):
        results_df = compare_models()

    st.dataframe(results_df, use_container_width=True, hide_index=True)

    best_model = results_df.loc[results_df["RMSE"].idxmin(), "Model"]
    st.success(f"✅ Lowest RMSE: **{best_model}**")

    c1, c2 = st.columns(2)
    with c1:
        fig_mae = px.bar(results_df, x="Model", y="MAE", title="Mean Absolute Error (lower is better)")
        fig_mae.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_mae, use_container_width=True)
    with c2:
        fig_rmse = px.bar(results_df, x="Model", y="RMSE", title="Root Mean Squared Error (lower is better)")
        fig_rmse.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_rmse, use_container_width=True)

    fig_r2 = px.bar(results_df, x="Model", y="R2 Score", title="R² Score (closer to 1 is better)")
    fig_r2.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown(
        """
        **Methodology:** All three models were trained on the same feature set
        (hour of day, day of week, one-hot encoded route) and the same
        80/20 train/test split, so the comparison isolates model choice as
        the only variable. MAE and RMSE are in percentage-occupancy units;
        R² measures the proportion of variance explained.
        """
    )


def render_admin(live_df):
    st.title("🛠️ Admin Dashboard & Analytics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Fleet", len(live_df))
    c2.metric("Avg Occupancy", f"{live_df['occupancy'].mean():.0f}%")
    c3.metric("Overcrowded Now", int((live_df["status"] == "Overcrowded").sum()))
    c4.metric("Total Alerts Logged", len(st.session_state.alerts_log))

    st.markdown("---")
    st.subheader("Fleet Status Table")
    display_df = live_df[["id", "route", "occupancy", "passengers", "next_stop", "status"]].sort_values("occupancy", ascending=False)
    st.dataframe(
        display_df.style.format({"occupancy": "{:.0f}%"}),
        use_container_width=True,
        height=320,
    )
    st.download_button("⬇️ Download live fleet data (CSV)", display_df.to_csv(index=False), file_name="fleet_status.csv")

    st.markdown("---")
    hist = st.session_state.history.copy()
    hist["hour"] = hist["time"].dt.hour

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Occupancy Heatmap (Hour vs Route)")
        pivot = hist.pivot_table(index="route", columns="hour", values="occupancy", aggfunc="mean")
        fig = px.imshow(pivot, color_continuous_scale=["#22c55e", "#eab308", "#ef4444"], aspect="auto",
                         labels=dict(color="Avg Occupancy %"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Historical Occupancy Trend")
        route_filter = st.multiselect("Filter routes", ROUTE_NAMES, default=ROUTE_NAMES)
        trend = hist[hist["route"].isin(route_filter)].groupby(["time", "route"])["occupancy"].mean().reset_index()
        fig2 = px.line(trend, x="time", y="occupancy", color="route")
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Route-wise Ridership (avg passengers)")
    ridership = live_df.groupby("route")["passengers"].mean().reset_index()
    fig3 = px.bar(ridership, x="route", y="passengers", color="passengers", color_continuous_scale="Blues")
    fig3.update_layout(height=350, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)


def render_alerts(live_df):
    st.title("🚨 Automated Overcrowding Alerts")

    current = live_df[live_df["status"] == "Overcrowded"]
    if current.empty:
        st.success("✅ No buses are currently overcrowded.")
    else:
        st.error(f"⚠️ {len(current)} bus(es) currently overcrowded!")
        for _, r in current.iterrows():
            st.warning(f"**{r['id']}** on *{r['route']}* is at **{r['occupancy']:.0f}%** capacity — approaching {r['next_stop']}.")

    st.markdown("---")
    st.subheader("Alert History Log")
    if st.session_state.alerts_log:
        log_df = pd.DataFrame(st.session_state.alerts_log).iloc[::-1]
        route_filter = st.selectbox("Filter by route", ["All routes"] + ROUTE_NAMES)
        if route_filter != "All routes":
            log_df = log_df[log_df["route"] == route_filter]
        st.dataframe(log_df, use_container_width=True, height=350)
    else:
        st.info("No alerts logged yet. Click 'Advance 15 min' in the sidebar a few times to simulate activity.")


def render_forecast(model, feature_cols):
    st.title("📈 Peak-Hour Demand Forecasting")
    st.caption("AI-based forecast using a model trained on synthetic historical ridership patterns.")

    route = st.selectbox("Select route to forecast", ROUTE_NAMES)
    horizon = st.slider("Forecast horizon (hours ahead)", 6, 48, 24)

    base_time = st.session_state.sim_time
    times = [base_time + timedelta(hours=h) for h in range(horizon)]
    preds = [predict_occupancy(model, feature_cols, t.hour, t.weekday(), route) for t in times]
    fdf = pd.DataFrame({"time": times, "predicted_occupancy": preds})

    fig = px.line(fdf, x="time", y="predicted_occupancy", markers=True)
    fig.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.1, line_width=0, annotation_text="High demand zone")
    fig.update_layout(height=420, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    peaks = fdf[fdf["predicted_occupancy"] >= 80]
    if not peaks.empty:
        st.warning("⏰ Predicted peak-demand windows: " + ", ".join(peaks["time"].dt.strftime("%a %H:%M").tolist()))
    else:
        st.success("No high-demand peaks predicted in this window.")

    st.markdown("---")
    st.subheader("All-Routes Forecast Heatmap")
    all_rows = []
    for r in ROUTE_NAMES:
        for t in times:
            all_rows.append({"route": r, "time": t.strftime("%a %H:%M"), "occupancy": predict_occupancy(model, feature_cols, t.hour, t.weekday(), r)})
    all_df = pd.DataFrame(all_rows)
    pivot = all_df.pivot_table(index="route", columns="time", values="occupancy", sort=False)
    pivot = pivot[[t.strftime("%a %H:%M") for t in times]]
    fig2 = px.imshow(pivot, color_continuous_scale=["#22c55e", "#eab308", "#ef4444"], aspect="auto", labels=dict(color="Occupancy %"))
    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)


def render_allocation(model, feature_cols, live_df):
    st.title("🔄 Dynamic Fleet Allocation Recommendations")
    st.caption("Simulated optimizer that reallocates buses across routes based on predicted demand.")

    next_time = st.session_state.sim_time + timedelta(hours=1)
    total_buses = len(live_df)

    demand_rows = []
    for r in ROUTE_NAMES:
        pred = predict_occupancy(model, feature_cols, next_time.hour, next_time.weekday(), r)
        current_buses = int((live_df["route"] == r).sum())
        demand_rows.append({"route": r, "predicted_demand": pred, "current_buses": current_buses})
    ddf = pd.DataFrame(demand_rows)

    ddf["demand_share"] = ddf["predicted_demand"] / ddf["predicted_demand"].sum()
    ddf["recommended_buses"] = np.maximum(1, np.round(ddf["demand_share"] * total_buses)).astype(int)

    diff = total_buses - ddf["recommended_buses"].sum()
    if diff != 0:
        idx = ddf["predicted_demand"].idxmax() if diff > 0 else ddf["predicted_demand"].idxmin()
        ddf.loc[idx, "recommended_buses"] += diff

    ddf["change"] = ddf["recommended_buses"] - ddf["current_buses"]

    st.subheader(f"Recommendation for {next_time.strftime('%a %H:%M')} (next hour)")
    st.dataframe(
        ddf[["route", "predicted_demand", "current_buses", "recommended_buses", "change"]]
        .rename(columns={
            "predicted_demand": "Predicted Occupancy (%)",
            "current_buses": "Current Buses",
            "recommended_buses": "Recommended Buses",
            "change": "Change",
            "route": "Route",
        })
        .style.format({"Predicted Occupancy (%)": "{:.0f}"}),
        use_container_width=True,
    )

    fig = go.Figure()
    fig.add_bar(name="Current", x=ddf["route"], y=ddf["current_buses"])
    fig.add_bar(name="Recommended", x=ddf["route"], y=ddf["recommended_buses"])
    fig.update_layout(barmode="group", height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Suggested Actions")
    moves_out = ddf[ddf["change"] < 0].sort_values("change")
    moves_in = ddf[ddf["change"] > 0].sort_values("change", ascending=False)
    if moves_out.empty and moves_in.empty:
        st.info("Current fleet allocation already matches predicted demand. No changes needed.")
    else:
        for _, row in moves_in.iterrows():
            st.success(f"➕ Add **{int(row['change'])} bus(es)** to *{row['route']}* — predicted occupancy {row['predicted_demand']:.0f}%.")
        for _, row in moves_out.iterrows():
            st.warning(f"➖ Reduce **{abs(int(row['change']))} bus(es)** from *{row['route']}* — predicted occupancy {row['predicted_demand']:.0f}%.")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# AUTH: LOGIN SCREEN + ROLE-BASED PAGE ACCESS
# ----------------------------------------------------------------------------
ROLE_PAGES = {
    "passenger": ["🏠 Home", "📱 Passenger App", "🚨 Alerts"],
    "driver": ["🏠 Home", "🚍 Driver Dashboard"],
    "admin": [
        "🏠 Home",
        "📱 Passenger App",
        "🚍 Driver Dashboard",
        "🛠️ Admin Dashboard",
        "🚨 Alerts",
        "📈 Demand Forecasting",
        "🔄 Fleet Allocation",
        "📊 Model Comparison",
    ],
}


def render_login():
    st.markdown(
        """
        <style>
        [data-testid="stVerticalBlockBorderWrapper"] {
            max-width: 380px;
            margin: 6vh auto 0 auto;
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: #161f36;
            border: 1px solid #26314d;
            border-radius: 18px;
            padding: 30px 30px 10px 30px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }
        .login-logo {
            font-size: 2.6rem;
            text-align: center;
            margin-bottom: 4px;
        }
        .login-title {
            text-align: center;
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.5rem;
            color: #f1f5f9;
            margin-bottom: 2px;
        }
        .login-subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 0.88rem;
            margin-bottom: 22px;
        }
        .login-demo {
            text-align: center;
            color: #64748b;
            font-size: 0.78rem;
            margin-top: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="login-logo">🚌</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-title">SmartETA</div>', unsafe_allow_html=True)

        if st.session_state.get("show_signup"):
            _render_signup_form()
        else:
            _render_login_form()


def _render_login_form():
    st.markdown('<div class="login-subtitle">Sign in to your account</div>', unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed")
    password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")

    role_hint = st.selectbox(
        "I am a",
        ["Passenger", "Driver", "Admin"],
        label_visibility="collapsed",
    )

    if st.button("Sign In", use_container_width=True):
        user = verify_login(username, password)
        if user:
            if user.role != role_hint.lower():
                st.error(f"This account is registered as '{user.role}', not '{role_hint}'. Select the correct role.")
            else:
                st.session_state.user = {"username": user.username, "role": user.role}
                st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown(
        '<div class="login-demo">Demo: passenger1/pass123 · driver1/drive123 · admin1/admin123</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Don't have an account? Sign up", use_container_width=True, type="secondary"):
        st.session_state.show_signup = True
        st.rerun()


def _render_signup_form():
    st.markdown('<div class="login-subtitle">Create a new account</div>', unsafe_allow_html=True)

    new_username = st.text_input("New Username", placeholder="Choose a username", label_visibility="collapsed")
    new_password = st.text_input("New Password", type="password", placeholder="Choose a password", label_visibility="collapsed")
    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password", label_visibility="collapsed")
    role_choice = st.selectbox(
        "Role",
        ["Passenger", "Driver", "Admin"],
        label_visibility="collapsed",
    )

    if st.button("Create Account", use_container_width=True):
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            success, error = create_user(new_username, new_password, role_choice.lower())
            if success:
                st.success("Account created! You can now sign in.")
                st.session_state.show_signup = False
                st.rerun()
            else:
                st.error(error)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Already have an account? Sign in", use_container_width=True, type="secondary"):
        st.session_state.show_signup = False
        st.rerun()


def main():
    inject_custom_css()

    if "user" not in st.session_state:
        render_login()
        return

    init_state()
    model, feature_cols, _ = train_crowd_model()
    seed_history(model, feature_cols)

    role = st.session_state.user["role"]
    allowed_pages = ROLE_PAGES.get(role, [])

    st.sidebar.title("🚌 SmartETA Control")
    st.sidebar.caption(f"Logged in as **{st.session_state.user['username']}** ({role})")
    if st.sidebar.button("Logout"):
        del st.session_state["user"]
        st.rerun()

    page = st.sidebar.radio("Navigate", allowed_pages)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"🕒 Simulated time: **{st.session_state.sim_time.strftime('%a, %I:%M %p')}**")
    col_a, col_b = st.sidebar.columns(2)
    manual_advance = col_a.button("⏭️ Advance 15 min")
    auto = col_b.checkbox("Auto-play")

    if manual_advance:
        advance_simulation(model, feature_cols)

    if auto:
        advance_simulation(model, feature_cols)
        time.sleep(1.5)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("SmartETA — real-time bus tracking and ML-based arrival/occupancy prediction for Bangalore transit routes.")

    live_df = get_live_bus_df()

    if page == "🏠 Home":
        render_home(live_df)
    elif page == "📱 Passenger App":
        render_passenger(live_df)
    elif page == "🚍 Driver Dashboard":
        render_driver(live_df, model, feature_cols)
    elif page == "🛠️ Admin Dashboard":
        render_admin(live_df)
    elif page == "🚨 Alerts":
        render_alerts(live_df)
    elif page == "📈 Demand Forecasting":
        render_forecast(model, feature_cols)
    elif page == "🔄 Fleet Allocation":
        render_allocation(model, feature_cols, live_df)
    elif page == "📊 Model Comparison":
        render_model_comparison()


if __name__ == "__main__":
    main()