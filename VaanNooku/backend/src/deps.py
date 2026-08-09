"""
FastAPI dependencies for authenticating store requests.
"""
from fastapi import Depends, HTTPException, Header, status
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.models import models
from src.security import decode_access_token


def get_current_store(
    authorization: str = Header(default=None),
    db: Session = Depends(get_db),
) -> models.Store:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or malformed Authorization header.")

    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_access_token(token)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    store_id = payload.get("sub")
    store = db.query(models.Store).filter(models.Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Store for this token no longer exists.")
    return store


def require_matching_store(store_id: str, current_store: models.Store = Depends(get_current_store)) -> models.Store:
    """Use as a route dependency to ensure the authenticated store can only act on its own data."""
    if current_store.id != store_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to act on this store's data.")
    return current_store


def get_current_admin(
    authorization: str = Header(default=None),
    db: Session = Depends(get_db),
) -> models.AdminUser:
    """
    Separate from get_current_store: an admin token carries {"type": "admin"}
    in its JWT payload, which a store login token never sets, so a merchant's
    token can never pass this check even if IDs happened to collide.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or malformed Authorization header.")

    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_access_token(token)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    if payload.get("type") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin credentials required.")

    admin = db.query(models.AdminUser).filter(models.AdminUser.id == payload.get("sub")).first()
    if not admin:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin account for this token no longer exists.")
    return admin
