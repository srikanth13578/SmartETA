# SmartETA

**ML-Driven Real-Time Bus Arrival Prediction & Transit Monitoring System**

SmartETA is a smart public-transport platform for Bangalore bus routes, combining live bus tracking with machine learning-based occupancy and demand prediction. Built as an internship research project.

## Features

- Login and signup with role-based access (Passenger / Driver / Admin)
- Passenger trip planner — pick a route and travel time, get ML-predicted occupancy for that slot
- Real-time occupancy tracking across simulated live buses
- ML-based occupancy prediction, with a model comparison study (Linear Regression, Random Forest, XGBoost)
- Live GPS position simulation on a map
- Driver dashboard
- Admin dashboard with analytics and model comparison metrics
- Automated alerts for overcrowded buses
- Peak-hour demand forecasting
- Dynamic fleet allocation recommendations

## Coverage

10 Bangalore transit corridors, including:
- Majestic ↔ Koramangala
- Whitefield ↔ MG Road
- Electronic City ↔ Silk Board
- Jayanagar ↔ Indiranagar
- Hebbal ↔ Marathahalli
- Banashankari ↔ Silk Board
- Yeshwantpur ↔ Majestic
- HSR Layout ↔ Domlur
- Vijayanagar ↔ Malleshwaram
- RT Nagar ↔ KR Puram

## Tech Stack

- **Frontend:** Streamlit, custom CSS theme
- **Backend logic:** Python (app.py, auth.py)
- **Database:** PostgreSQL, via SQLAlchemy ORM
- **Auth:** bcrypt password hashing, role-based access
- **ML:** scikit-learn (Linear Regression, Random Forest), XGBoost — with a comparison study (MAE, RMSE, R²)
- **Data visualization:** Plotly, PyDeck
- **Planned:** Docker, GitHub Actions (CI/CD), Azure (hosting)

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# set up Postgres and point the app at it
export DATABASE_URL="postgresql+psycopg2://smarteta_user:yourpassword@localhost:5432/smarteta"

# create tables and load routes/demo users
python seed_db.py

streamlit run app.py
```

Demo logins: `passenger1`/`pass123` · `driver1`/`drive123` · `admin1`/`admin123`

## Status

Actively under development as part of an internship research paper project. Core system (DB, auth, ML comparison, trip planning) is functional. Docker/CI/CD/Azure deployment planned once features are finalized. Position/tracking data is currently simulated; occupancy predictions are genuine ML inference trained on temporally realistic (peak-hour, weekday/weekend) synthetic patterns.