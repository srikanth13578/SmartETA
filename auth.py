"""
Simple authentication helper for SmartETA.
"""

from passlib.hash import bcrypt

from db.database import get_session
from db.models import User


def verify_login(username: str, password: str):
    """
    Returns the User object if username/password match, else None.
    """
    session = get_session()
    try:
        user = session.query(User).filter_by(username=username).first()
        if user and bcrypt.verify(password, user.password_hash):
            return user
        return None
    finally:
        session.close()


def create_user(username: str, password: str, role: str):
    """
    Creates a new user. Returns (True, None) on success,
    or (False, error_message) if the username is taken or input is invalid.
    """
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    if role not in ("passenger", "driver", "admin"):
        return False, "Invalid role."

    session = get_session()
    try:
        existing = session.query(User).filter_by(username=username).first()
        if existing:
            return False, "That username is already taken."
        user = User(username=username, password_hash=bcrypt.hash(password), role=role)
        session.add(user)
        session.commit()
        return True, None
    finally:
        session.close()