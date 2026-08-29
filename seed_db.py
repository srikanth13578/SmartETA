"""
One-time (re-runnable) seed script for SmartETA's database.

Run with:
    python seed_db.py

Creates all tables, loads the 5 Bangalore routes + stops, creates 3 buses
per route, and creates one demo login per role (passenger/driver/admin).
Safe to re-run — it skips anything that already exists.
"""

from passlib.hash import bcrypt

from db.database import init_db, get_session
from db.models import Route, Stop, Bus, User

# Same route data as app.py's ROUTES dict — kept in sync manually for now.
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
    "Route 6 - Banashankari to Silk Board": [
        ("Banashankari", 12.9255, 77.5468),
        ("JP Nagar", 12.9081, 77.5833),
        ("BTM Layout", 12.9166, 77.6101),
        ("Silk Board", 12.9172, 77.6228),
    ],
    "Route 7 - Yeshwantpur to Majestic": [
        ("Yeshwantpur", 13.0284, 77.5540),
        ("Malleshwaram", 13.0067, 77.5709),
        ("Rajajinagar", 12.9908, 77.5525),
        ("Majestic (KBS)", 12.9767, 77.5713),
    ],
    "Route 8 - HSR Layout to Domlur": [
        ("HSR Layout", 12.9121, 77.6446),
        ("Koramangala", 12.9352, 77.6245),
        ("Ejipura", 12.9413, 77.6335),
        ("Domlur", 12.9612, 77.6386),
    ],
    "Route 9 - Vijayanagar to Malleshwaram": [
        ("Vijayanagar", 12.9719, 77.5364),
        ("Rajajinagar", 12.9908, 77.5525),
        ("Sadashivanagar", 13.0068, 77.5807),
        ("Malleshwaram", 13.0067, 77.5709),
    ],
    "Route 10 - RT Nagar to KR Puram": [
        ("RT Nagar", 13.0198, 77.5959),
        ("Hennur", 13.0353, 77.6412),
        ("Horamavu", 13.0243, 77.6551),
        ("KR Puram", 13.0027, 77.6975),
    ],
}

BUSES_PER_ROUTE = 3
DEMO_USERS = [
    ("passenger1", "pass123", "passenger"),
    ("driver1", "drive123", "driver"),
    ("admin1", "admin123", "admin"),
]


def seed():
    init_db()
    session = get_session()

    bus_counter = 1
    for route_name, stops in ROUTES.items():
        route = session.query(Route).filter_by(name=route_name).first()
        if not route:
            route = Route(name=route_name)
            session.add(route)
            session.flush()  # get route.id before adding stops

            for seq, (stop_name, lat, lon) in enumerate(stops):
                session.add(Stop(route_id=route.id, name=stop_name, sequence=seq,
                                  latitude=lat, longitude=lon))

            for _ in range(BUSES_PER_ROUTE):
                bus_code = f"BUS-{bus_counter:03d}"
                if not session.query(Bus).filter_by(bus_code=bus_code).first():
                    session.add(Bus(bus_code=bus_code, route_id=route.id, capacity=50))
                bus_counter += 1

            print(f"Seeded route: {route_name}")
        else:
            print(f"Route already exists, skipping: {route_name}")

    for username, password, role in DEMO_USERS:
        if not session.query(User).filter_by(username=username).first():
            session.add(User(username=username, password_hash=bcrypt.hash(password), role=role))
            print(f"Created demo user: {username} ({role})")

    session.commit()
    session.close()
    print("Seed complete.")


if __name__ == "__main__":
    seed()