"""
Driver-reported crowd levels, stored in the database.

A driver submits how many passengers are onboard; the report is saved to
PostgreSQL (driver_reports table). Reports of 85% or more also create a row
in the existing alerts table. Passengers and admins read these reports back.

The DriverReport table is created automatically by ensure_tables(), so no
change to seed_db.py or db/models.py is needed.
"""

from datetime import datetime, timedelta

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func

from db.database import Base, engine, get_session
from db.models import Bus, Route, Alert  # importing models also registers the base tables

OVERCROWDED_PCT = 85
FRESH_WINDOW_HOURS = 12  # reports older than this are not shown to passengers


class DriverReport(Base):
    __tablename__ = "driver_reports"
    __table_args__ = {"extend_existing": True}  # safe if Streamlit re-imports this module

    id = Column(Integer, primary_key=True)
    bus_id = Column(Integer, ForeignKey("buses.id"), nullable=False)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False)
    reported_by = Column(String(50), nullable=False)
    occupancy_pct = Column(Float, nullable=False)
    note = Column(String(200))
    reported_at = Column(DateTime, nullable=False, default=datetime.now)


def ensure_tables():
    """Create any missing tables (including driver_reports). Safe to call repeatedly."""
    Base.metadata.create_all(bind=engine)


def save_driver_report(username, bus_code, occupancy_pct, note=""):
    """
    Save one driver report. Returns (True, None) on success or (False, message).
    Also writes an alerts row when the report is at or above the overcrowding level.
    """
    session = get_session()
    try:
        bus = session.query(Bus).filter_by(bus_code=bus_code).first()
        if bus is None:
            return False, f"{bus_code} was not found in the database. Run seed_db.py first."

        pct = float(max(0, min(100, occupancy_pct)))
        session.add(DriverReport(
            bus_id=bus.id,
            route_id=bus.route_id,
            reported_by=username,
            occupancy_pct=pct,
            note=(note or "").strip()[:200] or None,
            reported_at=datetime.now(),
        ))
        if pct >= OVERCROWDED_PCT:
            session.add(Alert(bus_id=bus.id, route_id=bus.route_id, occupancy_pct=pct,
                              triggered_at=datetime.now()))
        session.commit()
        return True, None
    except Exception as exc:  # surface DB problems to the UI instead of crashing the page
        session.rollback()
        return False, f"Could not save the report: {exc}"
    finally:
        session.close()


def get_recent_reports(bus_code=None, limit=10):
    """Most recent driver reports (newest first), optionally for one bus."""
    session = get_session()
    try:
        q = (session.query(DriverReport, Bus, Route)
             .join(Bus, DriverReport.bus_id == Bus.id)
             .join(Route, DriverReport.route_id == Route.id))
        if bus_code:
            q = q.filter(Bus.bus_code == bus_code)
        rows = q.order_by(DriverReport.reported_at.desc(), DriverReport.id.desc()).limit(limit).all()
        return [
            {
                "time": r.reported_at,
                "bus": b.bus_code,
                "route": rt.name,
                "driver": r.reported_by,
                "occupancy": r.occupancy_pct,
                "note": r.note or "",
            }
            for r, b, rt in rows
        ]
    except Exception:
        return []
    finally:
        session.close()


def get_latest_reports_for_route(route_name):
    """Latest fresh report per bus on a route, as {bus_code: {...}}."""
    session = get_session()
    try:
        cutoff = datetime.now() - timedelta(hours=FRESH_WINDOW_HOURS)
        rows = (session.query(DriverReport, Bus)
                .join(Bus, DriverReport.bus_id == Bus.id)
                .join(Route, Bus.route_id == Route.id)
                .filter(Route.name == route_name, DriverReport.reported_at >= cutoff)
                .order_by(DriverReport.reported_at.desc(), DriverReport.id.desc())
                .all())
        latest = {}
        for r, b in rows:
            if b.bus_code not in latest:  # rows are newest-first, so the first one wins
                latest[b.bus_code] = {"time": r.reported_at, "occupancy": r.occupancy_pct, "driver": r.reported_by}
        return latest
    except Exception:
        return {}
    finally:
        session.close()


def get_report_counts():
    """(total driver reports, total stored overcrowding alerts)."""
    session = get_session()
    try:
        reports = session.query(func.count(DriverReport.id)).scalar() or 0
        alerts = session.query(func.count(Alert.id)).scalar() or 0
        return int(reports), int(alerts)
    except Exception:
        return 0, 0
    finally:
        session.close()