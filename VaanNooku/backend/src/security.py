"""
Password hashing + JWT issuance/verification for store accounts.

Uses the `bcrypt` library directly rather than passlib's CryptContext:
passlib (last released 2020) does its own bcrypt-backend version detection
that is incompatible with bcrypt>=4.0's API changes, and raises at hash time
on this environment's bcrypt 5.x. Calling bcrypt.hashpw/checkpw directly
sidesteps that broken detection entirely.
"""
import os
import bcrypt
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError

JWT_SECRET = os.getenv("JWT_SECRET", "retail_ai_secret_key_9988")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# bcrypt only uses the first 72 bytes of the input — truncate rather than
# let long passwords raise ValueError.
_MAX_PASSWORD_BYTES = 72


def hash_password(plain: str) -> str:
    truncated = plain.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    return bcrypt.hashpw(truncated, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        truncated = (plain or "").encode("utf-8")[:_MAX_PASSWORD_BYTES]
        return bcrypt.checkpw(truncated, hashed.encode("utf-8"))
    except ValueError:
        # Not a valid bcrypt hash (e.g. legacy plaintext row not yet migrated)
        return False


def is_hashed(value: str) -> bool:
    return bool(value) and value.startswith(("$2a$", "$2b$", "$2y$"))


def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as e:
        raise ValueError(f"Invalid or expired token: {e}")
