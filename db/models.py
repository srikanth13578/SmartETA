"""
SQLAlchemy ORM models for SmartETA.
"""

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import relationship

from db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)  # "passenger" | "driver" | "admin"
    created_at = Column(DateTime, default=datetime.utcnow)

    # Only relevant for role == "driver": which bus they're assigned to
    assigned_bus_id = Column(Integer, ForeignKey("buses.id"), nullable=True)
    assigned_bus = relationship("Bus", back_populates="driver_user")


class Route(Base):
    __tablename__ = "routes"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)  # e.g. "Route 1 - Majestic to Koramangala"

    stops = relationship("Stop", back_populates="route", order_by="Stop.sequence")
    buses = relationship("Bus", back_populates="route")


class Stop(Base):
    __tablename__ = "stops"

    id = Column(Integer, primary_key=True)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False)
    name = Column(String(100), nullable=False)
    sequence = Column(Integer, nullable=False)  # order along the route, 0-indexed
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    route = relationship("Route", back_populates="stops")


class Bus(Base):
    __tablename__ = "buses"

    id = Column(Integer, primary_key=True)
    bus_code = Column(String(20), unique=True, nullable=False)  # e.g. "BUS-001"
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False)
    capacity = Column(Integer, default=50)

    route = relationship("Route", back_populates="buses")
    driver_user = relationship("User", back_populates="assigned_bus", uselist=False)


class LivePosition(Base):
    """Latest known position + occupancy for each bus. One row per bus, upserted."""
    __tablename__ = "live_positions"

    id = Column(Integer, primary_key=True)
    bus_id = Column(Integer, ForeignKey("buses.id"), unique=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    next_stop = Column(String(100))
    occupancy_pct = Column(Float, nullable=False)
    status = Column(String(20))  # "Comfortable" | "Moderate" | "Overcrowded"
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class OccupancyHistory(Base):
    """Time-series log of occupancy readings, used for training/eval and trend charts."""
    __tablename__ = "occupancy_history"

    id = Column(Integer, primary_key=True)
    bus_id = Column(Integer, ForeignKey("buses.id"), nullable=False)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False)
    recorded_at = Column(DateTime, nullable=False)
    occupancy_pct = Column(Float, nullable=False)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    bus_id = Column(Integer, ForeignKey("buses.id"), nullable=False)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False)
    occupancy_pct = Column(Float, nullable=False)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
