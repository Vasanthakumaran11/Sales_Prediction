from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from src.models import models
from src.security import verify_password, create_access_token


def login_admin(db: Session, email: str, password: str):
    admin = db.query(models.AdminUser).filter(models.AdminUser.email.ilike(email)).first()
    if not admin or not verify_password(password or "", admin.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")

    return {
        "token": create_access_token({"sub": admin.id, "type": "admin"}),
        "admin": {"id": admin.id, "email": admin.email, "role": admin.role},
    }
