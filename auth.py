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
