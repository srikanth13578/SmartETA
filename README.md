# Smart Bus Transit System — Design Thinking Prototype

A single-file Streamlit web app that simulates a smart public-transport platform.
All data is synthetic/dummy, generated on the fly — no external datasets or APIs required.

## Features
- **Real-time occupancy tracking** — simulated live buses with occupancy % that updates each tick
- **AI-based crowd prediction** — a RandomForest model trained on synthetic historical ridership patterns
- **Live GPS tracking** — buses shown moving along their routes on an interactive map (pydeck)
- **Passenger mobile app** — pick a route, see live buses, get routed to the least-crowded one
- **Driver dashboard** — occupancy gauge, next stop, trend chart, AI prediction for the next hour
- **Admin dashboard with analytics** — fleet table, occupancy heatmap, historical trends, ridership charts, CSV export
- **Automated alerts** — buses at ≥85% occupancy are flagged live and logged
- **Peak-hour demand forecasting** — forecast next 6–48 hours per route, plus an all-routes heatmap
- **Dynamic fleet allocation** — recommends how many buses each route should have based on predicted demand

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## How to use it
- Use the sidebar **"⏭️ Advance 15 min"** button (or check **"Auto-play"**) to move the simulation forward —
  bus positions, occupancy, and alerts all update.
- Explore each page from the sidebar: Home, Passenger App, Driver Dashboard, Admin Dashboard,
  Alerts, Demand Forecasting, and Fleet Allocation.
- Everything resets if you restart the app — state lives only in the Streamlit session.

## Notes for your design-thinking write-up
- The route network (5 routes, 15 buses) and all coordinates are fictional, inspired by real
  Chennai localities, purely for a realistic-looking demo map.
- The "AI" crowd-prediction and forecasting model is a `RandomForestRegressor` trained on
  synthetic data that encodes a plausible morning/evening peak pattern — swap in real
  historical ridership data later without changing the rest of the app.
- Fleet allocation uses a simple proportional-demand heuristic; this is easy to explain in a
  design-thinking presentation and can be replaced with a more advanced optimizer later.
