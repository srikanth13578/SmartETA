# SmartETA

**ML-Driven Real-Time Bus Arrival Prediction & Transit Monitoring System**

SmartETA is a smart public-transport platform for Bangalore bus routes, combining live bus tracking with machine learning-based occupancy and demand prediction. Built as an internship research project.

## Features

- Real-time occupancy tracking across simulated live buses
- ML-based crowd/demand prediction (RandomForest, trained on route/time patterns)
- Live GPS tracking with interpolated positions on a map
- Passenger-facing mobile-style dashboard
- Driver dashboard
- Admin dashboard with analytics
- Automated alerts for overcrowded buses
- Peak-hour demand forecasting
- Dynamic fleet allocation recommendations

## Coverage

Currently modeled around five Bangalore transit corridors:
- Majestic ↔ Koramangala
- Whitefield ↔ MG Road
- Electronic City ↔ Silk Board
- Jayanagar ↔ Indiranagar
- Hebbal ↔ Marathahalli

## Tech Stack

- **Frontend/Dashboard:** Streamlit
- **ML:** scikit-learn (RandomForest, with additional models planned)
- **Data visualization:** Plotly, PyDeck
- **Planned:** PostgreSQL for persistence, Docker for containerization, GitHub Actions for CI/CD, Azure for hosting

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Status

Actively under development as part of an internship research paper project. Currently migrating from synthetic simulation data toward real/semi-real transit data, with additional model comparison and cloud deployment in progress.