from datetime import datetime, timedelta
from typing import List, Optional
import base64
import hashlib
import hmac
import json
import os
import time
import logging
from collections import defaultdict, deque
from urllib.parse import urlencode
from urllib.request import Request as URLRequest, urlopen
from urllib.error import URLError, HTTPError
from xml.etree import ElementTree as ET
import math

import jwt

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, inspect, text as sql_text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from ml.predictor import predict_risk, model_status as get_ml_model_status, DEFAULT_MODEL_PATH


app = FastAPI(
    title="RAKSHA AI",
    description="AI-powered Disaster Intelligence and Emergency Response System",
    version="4.2.0",
)

# =========================================================
# AUTHENTICATION
# =========================================================

APP_ENV = os.getenv("RAKSHA_ENV", "development").lower()
AUTH_SECRET = os.getenv("RAKSHA_SECRET_KEY", "")
if APP_ENV == "production" and len(AUTH_SECRET) < 32:
    raise RuntimeError("RAKSHA_SECRET_KEY must be set to a strong secret in production")
if not AUTH_SECRET:
    AUTH_SECRET = "raksha-ai-local-development-secret-change-me-2026"
TOKEN_TTL_HOURS = int(os.getenv("RAKSHA_TOKEN_TTL_HOURS", "8"))
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "raksha-ai"
JWT_AUDIENCE = "raksha-command-center"
PBKDF2_ITERATIONS = 310000
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_ATTEMPTS = 8
REGISTRATION_WINDOW_SECONDS = 15 * 60
REGISTRATION_MAX_ATTEMPTS = 5
REGISTRATION_ROLES = {"Dispatcher", "Responder"}
REGISTRATION_ATTEMPTS = defaultdict(deque)
LOCATION_RETENTION_HOURS = int(os.getenv("RAKSHA_LOCATION_RETENTION_HOURS", "24"))
ENV_CACHE_SECONDS = int(os.getenv("RAKSHA_ENV_CACHE_SECONDS", "300"))
WEATHER_CACHE_SECONDS = int(os.getenv("RAKSHA_WEATHER_CACHE_SECONDS", "600"))
SACHET_CAP_URL = os.getenv("SACHET_CAP_URL", "").strip()
SACHET_SOURCE_URL = "https://sachet.ndma.gov.in/"
USGS_EARTHQUAKE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_environment_cache = {}
_weather_cache = {}
_sachet_cache = {"etag": None, "xml": None, "fetched_at": 0.0, "alerts": []}

logger = logging.getLogger("raksha")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
MIN_PASSWORD_LENGTH = 8


def validate_password_strength(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
    if len(password) > 200:
        raise HTTPException(status_code=400, detail="Password is too long")
    if not any(ch.islower() for ch in password):
        raise HTTPException(status_code=400, detail="Password must contain a lowercase letter")
    if not any(ch.isupper() for ch in password):
        raise HTTPException(status_code=400, detail="Password must contain an uppercase letter")
    if not any(ch.isdigit() for ch in password):
        raise HTTPException(status_code=400, detail="Password must contain a number")


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("utf-8"))


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("Password is required")
    salt = salt or os.urandom(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        _b64encode(salt),
        _b64encode(derived),
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iteration_text, salt_text, digest_text = stored_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iteration_text)
        salt = _b64decode(salt_text)
        expected = _b64decode(digest_text)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def create_access_token(user_id: int, username: str, role: str, session_version: int = 0) -> str:
    now = int(datetime.utcnow().timestamp())
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "ver": int(session_version),
        "iat": now,
        "exp": now + TOKEN_TTL_HOURS * 60 * 60,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
    }
    return jwt.encode(payload, AUTH_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            AUTH_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={"require": ["sub", "iat", "exp", "iss", "aud", "ver"]},
        )
    except Exception as error:
        logger.warning("Token validation failed: %s", type(error).__name__)
        raise HTTPException(status_code=401, detail="Authentication required") from error

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials.strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = decode_access_token(token)
    try:
        user_id = int(payload["sub"])
        token_version = int(payload.get("ver", 0))
    except (TypeError, ValueError, KeyError):
        raise HTTPException(status_code=401, detail="Authentication required")

    db = SessionLocal()
    try:
        db_user = db.query(UserDB).filter(UserDB.id == user_id).first()
        if db_user is None or not db_user.is_active:
            raise HTTPException(status_code=401, detail="User account is unavailable")
        if int(db_user.session_version or 0) != token_version:
            raise HTTPException(status_code=401, detail="Session expired; please sign in again")
        return {
            "id": db_user.id,
            "username": db_user.username,
            "full_name": db_user.full_name,
            "role": db_user.role,
        }
    finally:
        db.close()

def require_roles(*roles):
    allowed = {str(role).lower() for role in roles}

    def dependency(user: dict = Depends(get_current_user)):
        if str(user.get("role", "")).lower() not in allowed:
            raise HTTPException(status_code=403, detail="You do not have permission for this action")
        return user

    return dependency


LOGIN_ATTEMPTS = defaultdict(deque)


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


@app.middleware("http")
async def security_and_authentication_middleware(request: Request, call_next):
    path = request.url.path
    public_paths = {"/", "/api/health", "/api/auth/login", "/docs", "/openapi.json", "/redoc"}

    if request.method == "OPTIONS" or path in public_paths or not path.startswith("/api/"):
        response = await call_next(request)
    else:
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.lower().startswith("bearer "):
            response = JSONResponse(status_code=401, content={"detail": "Authentication required"})
        else:
            try:
                token = authorization.split(" ", 1)[1].strip()
                payload = decode_access_token(token)
                user_id = int(payload["sub"])
                token_version = int(payload.get("ver", 0))
                db = SessionLocal()
                try:
                    db_user = db.query(UserDB).filter(UserDB.id == user_id).first()
                    auth_valid = db_user is not None and bool(db_user.is_active) and int(db_user.session_version or 0) == token_version
                finally:
                    db.close()
            except HTTPException as error:
                response = JSONResponse(status_code=error.status_code, content={"detail": error.detail})
                auth_valid = False
            except Exception:
                logger.exception("Authentication middleware failure")
                response = JSONResponse(status_code=500, content={"detail": "Authentication service unavailable"})
                auth_valid = False

            if auth_valid:
                # Do not wrap call_next in the authentication try/except: endpoint
                # errors must remain visible as their actual status rather than
                # being misreported as 401 Unauthorized.
                response = await call_next(request)
            elif "response" not in locals():
                response = JSONResponse(status_code=401, content={"detail": "Authentication required"})

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(self), microphone=(), camera=()"
    response.headers["Cache-Control"] = "no-store" if path.startswith("/api/") else response.headers.get("Cache-Control", "no-cache")
    response.headers["X-Robots-Tag"] = "noindex" if path.startswith("/api/") else "noindex"
    return response


# Keep CORS as the outer middleware so authenticated 401/403 responses
# still include the browser CORS headers.
CORS_ORIGINS = [item.strip() for item in os.getenv("RAKSHA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if item.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# REAL-TIME CONNECTION MANAGER
# =========================================================



# =========================================================
# REAL-TIME CONNECTION MANAGER
# =========================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []

        for websocket in list(self.active_connections):
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        for websocket in disconnected:
            self.disconnect(websocket)


manager = ConnectionManager()


# =========================================================
# SQLITE DATABASE
# =========================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./raksha.db")
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=1800)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    full_name = Column(String(150), nullable=False)
    role = Column(String(50), nullable=False)
    password_hash = Column(Text, nullable=False)
    is_active = Column(Integer, default=1)
    session_version = Column(Integer, default=0, nullable=False)
    created_at = Column(String(50), nullable=False)
    approval_status = Column(String(30), default="approved", nullable=False)


class IncidentDB(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True, index=True)
    disaster_type = Column(String(100), nullable=False)
    location = Column(String(255), nullable=False)
    severity = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    people_affected = Column(Integer, default=0)
    status = Column(String(50), default="Active")
    created_at = Column(String(50), nullable=False)
    ai_score = Column(Integer, default=50)
    ai_score_source = Column(String(50), default="rules_baseline")
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_accuracy = Column(Float, nullable=True)
    source = Column(String(50), default="operator")


class ResourceDB(Base):
    __tablename__ = "resources"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    resource_type = Column(String(100), nullable=False)
    location = Column(String(255), nullable=False)
    status = Column(String(50), default="Available")
    capacity = Column(Integer, default=0)
    assigned_incident_id = Column(Integer, nullable=True)
    eta_minutes = Column(Integer, nullable=True)
    created_at = Column(String(50), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    speed_kmh = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)
    location_accuracy = Column(Float, nullable=True)
    last_seen = Column(String(50), nullable=True)
    assigned_user_id = Column(Integer, nullable=True)


class DispatchDB(Base):
    __tablename__ = "dispatches"
    id = Column(Integer, primary_key=True, index=True)
    resource_id = Column(Integer, nullable=False)
    incident_id = Column(Integer, nullable=False)
    previous_status = Column(String(50), nullable=True)
    new_status = Column(String(50), nullable=False)
    eta_minutes = Column(Integer, nullable=True)
    created_at = Column(String(50), nullable=False)


class AlertDB(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, nullable=True)
    alert_type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(50), nullable=False)
    ai_score = Column(Integer, default=0)
    acknowledged = Column(Integer, default=0)
    created_at = Column(String(50), nullable=False)


class EmergencyDB(Base):
    __tablename__ = "emergencies"

    id = Column(Integer, primary_key=True, index=True)
    emergency_type = Column(String(50), nullable=False)
    status = Column(String(50), default="Active")
    message = Column(Text, nullable=False)
    location = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_accuracy = Column(Float, nullable=True)
    source = Column(String(50), default="operator")
    created_at = Column(String(50), nullable=False)
    resolved_at = Column(String(50), nullable=True)


class LocationPingDB(Base):
    __tablename__ = "location_pings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    resource_id = Column(Integer, nullable=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=True)
    speed_kmh = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)
    created_at = Column(String(50), nullable=False, index=True)


class FusionSnapshotDB(Base):
    __tablename__ = "fusion_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    grid_key = Column(String(80), index=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    fused_risk_score = Column(Integer, nullable=False)
    risk_level = Column(String(30), nullable=False)
    incident_signal = Column(Integer, default=0)
    weather_signal = Column(Integer, default=0)
    seismic_signal = Column(Integer, default=0)
    official_alert_signal = Column(Integer, default=0)
    resource_pressure = Column(Integer, default=0)
    data_coverage = Column(Integer, default=0)
    drivers = Column(Text, nullable=True)
    source_status = Column(Text, nullable=True)
    created_at = Column(String(50), nullable=False, index=True)


class AuditLogDB(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    username = Column(String(80), nullable=True)
    action = Column(String(120), nullable=False, index=True)
    entity_type = Column(String(80), nullable=True)
    entity_id = Column(String(80), nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(80), nullable=True)
    created_at = Column(String(50), nullable=False, index=True)


Base.metadata.create_all(bind=engine)


def ensure_schema_columns():
    """Lightweight additive migration for existing SQLite/Postgres databases."""
    additions = {
        "users": {
            "session_version": "INTEGER NOT NULL DEFAULT 0",
            "approval_status": "VARCHAR(30) NOT NULL DEFAULT 'approved'",
        },
        "incidents": {
            "latitude": "FLOAT",
            "longitude": "FLOAT",
            "location_accuracy": "FLOAT",
            "source": "VARCHAR(50) DEFAULT 'operator'",
            "ai_score_source": "VARCHAR(50) DEFAULT 'rules_baseline'",
        },
        "resources": {
            "latitude": "FLOAT",
            "longitude": "FLOAT",
            "speed_kmh": "FLOAT",
            "heading": "FLOAT",
            "location_accuracy": "FLOAT",
            "last_seen": "VARCHAR(50)",
            "assigned_user_id": "INTEGER",
        },
        "emergencies": {
            "latitude": "FLOAT",
            "longitude": "FLOAT",
            "location_accuracy": "FLOAT",
            "source": "VARCHAR(50) DEFAULT 'operator'",
        },
    }
    inspector = inspect(engine)
    for table, columns in additions.items():
        existing = {col["name"] for col in inspector.get_columns(table)}
        for name, ddl in columns.items():
            if name not in existing:
                try:
                    with engine.begin() as connection:
                        connection.execute(sql_text(f'ALTER TABLE {table} ADD COLUMN {name} {ddl}'))
                except Exception as error:
                    logger.warning("Schema column migration skipped for %s.%s: %s", table, name, error)


ensure_schema_columns()


def incident_dict(x):
    return {
        "id": x.id,
        "disaster_type": x.disaster_type,
        "location": x.location,
        "severity": x.severity,
        "description": x.description,
        "people_affected": x.people_affected or 0,
        "status": x.status,
        "created_at": x.created_at,
        "ai_score": x.ai_score or 0,
        "ai_score_source": x.ai_score_source or "rules_baseline",
        "latitude": x.latitude,
        "longitude": x.longitude,
        "location_accuracy": x.location_accuracy,
        "source": x.source or "operator",
    }


def resource_dict(x):
    return {
        "id": x.id,
        "name": x.name,
        "resource_type": x.resource_type,
        "location": x.location,
        "status": x.status,
        "capacity": x.capacity or 0,
        "assigned_incident_id": x.assigned_incident_id,
        "eta_minutes": x.eta_minutes,
        "created_at": x.created_at,
        "latitude": x.latitude,
        "longitude": x.longitude,
        "speed_kmh": x.speed_kmh,
        "heading": x.heading,
        "location_accuracy": x.location_accuracy,
        "last_seen": x.last_seen,
        "assigned_user_id": x.assigned_user_id,
    }


# Backward-compatible helper names used by newer endpoints.
def get_db() -> Session:
    return SessionLocal()


def incident_to_dict(x):
    return incident_dict(x)


def resource_to_dict(x):
    return resource_dict(x)


def alert_dict(x):
    return {
        "id": x.id,
        "incident_id": x.incident_id,
        "alert_type": x.alert_type,
        "title": x.title,
        "message": x.message,
        "severity": x.severity,
        "ai_score": x.ai_score or 0,
        "acknowledged": bool(x.acknowledged),
        "created_at": x.created_at,
    }


def emergency_dict(x):
    return {
        "id": x.id,
        "emergency_type": x.emergency_type,
        "status": x.status,
        "message": x.message,
        "location": x.location,
        "latitude": x.latitude,
        "longitude": x.longitude,
        "location_accuracy": x.location_accuracy,
        "source": x.source or "operator",
        "created_at": x.created_at,
        "resolved_at": x.resolved_at,
    }


async def create_alert(
    db,
    incident_id,
    alert_type,
    title,
    message,
    severity,
    ai_score=0,
):
    alert = AlertDB(
        incident_id=incident_id,
        alert_type=alert_type,
        title=title,
        message=message,
        severity=severity,
        ai_score=ai_score,
        acknowledged=0,
        created_at=datetime.utcnow().isoformat(),
    )
    db.add(alert)
    db.flush()
    data = alert_dict(alert)
    await manager.broadcast({
        "type": "alert_created",
        "alert": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return data


def write_audit(db, user: Optional[dict], action: str, entity_type: Optional[str] = None, entity_id: Optional[str] = None, details: Optional[dict] = None, ip_address: Optional[str] = None):
    entry = AuditLogDB(
        user_id=(user or {}).get("id"),
        username=(user or {}).get("username"),
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        details=json.dumps(details or {}, ensure_ascii=False),
        ip_address=ip_address,
        created_at=datetime.utcnow().isoformat(),
    )
    db.add(entry)
    db.flush()
    return entry


def utc_now_iso():
    return datetime.utcnow().isoformat()


def validate_coordinates(latitude: float, longitude: float):
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise HTTPException(status_code=422, detail="Invalid GPS coordinates")


def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def seed_db():
    db=SessionLocal()
    try:
        default_users = [
            ("admin", "System Administrator", "Admin", "Admin@123"),
            ("dispatcher", "Emergency Dispatcher", "Dispatcher", "Dispatcher@123"),
            ("responder", "Field Responder", "Responder", "Responder@123"),
        ]
        for username, full_name, role, password in default_users:
            existing_user = db.query(UserDB).filter(UserDB.username == username).first()
            if existing_user is None:
                db.add(UserDB(
                    username=username,
                    full_name=full_name,
                    role=role,
                    password_hash=hash_password(password),
                    is_active=1,
                    session_version=0,
                    created_at=datetime.utcnow().isoformat(),
                    approval_status="approved",
                ))

        if db.query(IncidentDB).count()==0:
            now=datetime.utcnow().isoformat()
            db.add_all([
                IncidentDB(id=1,disaster_type="Flood",location="Sector 14",severity="Critical",description="People trapped in flooded building",people_affected=120,status="Active",created_at=now,ai_score=94),
                IncidentDB(id=2,disaster_type="Flood",location="Sector 12",severity="High",description="Main road becoming inaccessible",people_affected=80,status="Active",created_at=now,ai_score=87),
                IncidentDB(id=3,disaster_type="Fire",location="Sector 21",severity="Critical",description="Large fire reported near residential buildings",people_affected=350,status="Active",created_at=now,ai_score=95),
            ])
        if db.query(ResourceDB).count()==0:
            now=datetime.utcnow().isoformat()
            db.add_all([
                ResourceDB(id=1,name="Ambulance Alpha",resource_type="Ambulance",location="Sector 10",status="Available",capacity=4,created_at=now),
                ResourceDB(id=2,name="Fire Unit 01",resource_type="Fire Truck",location="Sector 18",status="Available",capacity=8,created_at=now),
                ResourceDB(id=3,name="Rescue Team Alpha",resource_type="Rescue Team",location="Sector 8",status="Available",capacity=12,created_at=now),
                ResourceDB(id=4,name="RAKSHA Drone 01",resource_type="Drone",location="Command Center",status="Available",capacity=0,created_at=now),
                ResourceDB(id=5,name="Rescue Boat 01",resource_type="Rescue Boat",location="Sector 11",status="Deployed",capacity=20,assigned_incident_id=1,eta_minutes=5,created_at=now),
            ])
        if db.query(AlertDB).count() == 0:
            for incident in db.query(IncidentDB).filter(IncidentDB.severity.in_(["Critical", "High"])).all():
                alert_type = "CRITICAL_INCIDENT" if incident.severity == "Critical" else "HIGH_PRIORITY_INCIDENT"
                title = f"{incident.severity} {incident.disaster_type} Alert"
                message = f"{incident.disaster_type} reported near {incident.location}: {incident.description}"
                db.add(AlertDB(
                    incident_id=incident.id,
                    alert_type=alert_type,
                    title=title,
                    message=message,
                    severity=incident.severity,
                    ai_score=incident.ai_score or 0,
                    acknowledged=0,
                    created_at=datetime.utcnow().isoformat(),
                ))
        db.commit()
    finally: db.close()


seed_db()


# =========================================================
# MODELS
# =========================================================

class IncidentCreate(BaseModel):
    disaster_type: str = Field(min_length=2, max_length=100)
    location: str = Field(min_length=2, max_length=255)
    severity: str
    description: str = Field(min_length=3, max_length=5000)
    people_affected: int = Field(default=0, ge=0, le=1000000000)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    location_accuracy: Optional[float] = Field(default=None, ge=0)
    source: str = Field(default="operator", max_length=50)


class Incident(IncidentCreate):
    id: int
    status: str
    created_at: str
    ai_score: int
    ai_score_source: str = "rules_baseline"


class ResourceCreate(BaseModel):
    name: str
    resource_type: str
    location: str
    status: str = "Available"
    capacity: int = Field(default=0, ge=0)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)


class Resource(ResourceCreate):
    id: int
    assigned_incident_id: Optional[int] = None
    eta_minutes: Optional[int] = None
    created_at: str
    speed_kmh: Optional[float] = None
    heading: Optional[float] = None
    location_accuracy: Optional[float] = None
    last_seen: Optional[str] = None
    assigned_user_id: Optional[int] = None


class Alert(BaseModel):
    id: int
    incident_id: Optional[int] = None
    alert_type: str
    title: str
    message: str
    severity: str
    ai_score: int
    acknowledged: bool
    created_at: str


# Persistent data lives in SQLite (raksha.db).


# =========================================================
# AUTH ROUTES
# =========================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class RegistrationRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    full_name: str = Field(min_length=2, max_length=150)
    password: str = Field(min_length=8, max_length=200)
    role: str


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    full_name: str = Field(min_length=2, max_length=150)
    password: str = Field(min_length=8, max_length=200)
    role: str


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, min_length=2, max_length=150)
    role: Optional[str] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=200)
    is_active: Optional[bool] = None


def public_user(user: UserDB) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "approval_status": getattr(user, "approval_status", "approved") or "approved",
    }


@app.post("/api/auth/register")
def register(payload: RegistrationRequest, request: Request):
    """Submit a Dispatcher/Responder account for Admin approval."""
    role = payload.role.strip().title()
    username = payload.username.strip().lower()
    full_name = payload.full_name.strip()

    if role not in REGISTRATION_ROLES:
        raise HTTPException(status_code=400, detail="Only Dispatcher and Responder accounts can be registered here")
    if not all(ch.isalnum() or ch in "._-" for ch in username):
        raise HTTPException(status_code=400, detail="Username may contain only letters, numbers, dots, underscores or hyphens")
    if not full_name:
        raise HTTPException(status_code=400, detail="Full name is required")
    validate_password_strength(payload.password)

    key = client_key(request)
    now = time.time()
    attempts = REGISTRATION_ATTEMPTS[key]
    while attempts and now - attempts[0] > REGISTRATION_WINDOW_SECONDS:
        attempts.popleft()
    if len(attempts) >= REGISTRATION_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many registration attempts. Try again later.")
    attempts.append(now)

    db = SessionLocal()
    try:
        existing = db.query(UserDB).filter(UserDB.username == username).first()
        if existing is not None:
            raise HTTPException(status_code=409, detail="Username already exists")

        new_user = UserDB(
            username=username,
            full_name=full_name,
            role=role,
            password_hash=hash_password(payload.password),
            is_active=0,
            session_version=0,
            created_at=datetime.utcnow().isoformat(),
            approval_status="pending",
        )
        db.add(new_user)
        db.flush()
        write_audit(
            db,
            {"id": new_user.id, "username": username, "role": role},
            "auth.register_request",
            "user",
            new_user.id,
            {"role": role, "approval_status": "pending"},
            key,
        )
        db.commit()
        db.refresh(new_user)
        return {
            "status": "pending",
            "message": "Registration submitted. Admin approval is required before login.",
            "user": {
                **public_user(new_user),
                "is_active": False,
                "created_at": new_user.created_at,
            },
        }
    finally:
        db.close()


@app.post("/api/auth/login")
def login(payload: LoginRequest, request: Request):
    key = client_key(request)
    now = time.time()
    attempts = LOGIN_ATTEMPTS[key]
    while attempts and now - attempts[0] > LOGIN_WINDOW_SECONDS:
        attempts.popleft()
    if len(attempts) >= LOGIN_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")

    db = SessionLocal()
    try:
        username = payload.username.strip().lower()
        user = db.query(UserDB).filter(UserDB.username == username).first()
        if user is None or not verify_password(payload.password, user.password_hash):
            attempts.append(now)
            raise HTTPException(status_code=401, detail="Invalid username or password")
        if not user.is_active:
            if getattr(user, "approval_status", "approved") == "pending":
                raise HTTPException(status_code=403, detail="Account pending Admin approval")
            raise HTTPException(status_code=403, detail="This account is disabled")

        attempts.clear()
        user_data = public_user(user)
        token = create_access_token(user.id, user.username, user.role, int(user.session_version or 0))
        write_audit(db, user_data, "auth.login", "user", user.id, {"success": True}, key)
        db.commit()
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": TOKEN_TTL_HOURS * 60 * 60,
            "user": user_data,
        }
    finally:
        db.close()


@app.post("/api/auth/logout")
def logout(user: dict = Depends(get_current_user), request: Request = None):
    db = SessionLocal()
    try:
        write_audit(db, user, "auth.logout", "user", user["id"], {}, client_key(request) if request else None)
        db.commit()
        return {"message": "Signed out"}
    finally:
        db.close()


@app.get("/api/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        db_user = db.query(UserDB).filter(UserDB.id == user["id"]).first()
        if db_user is None or not db_user.is_active:
            raise HTTPException(status_code=401, detail="User account is unavailable")
        return public_user(db_user)
    finally:
        db.close()


# =========================================================
# USER MANAGEMENT (ADMIN)
# =========================================================

ALLOWED_ROLES = {"Admin", "Dispatcher", "Responder"}


@app.get("/api/users")
def list_users(user: dict = Depends(require_roles("Admin"))):
    db = SessionLocal()
    try:
        users = db.query(UserDB).order_by(UserDB.id.asc()).all()
        return [
            {
                **public_user(item),
                "is_active": bool(item.is_active),
                "created_at": item.created_at,
                "approval_status": getattr(item, "approval_status", "approved") or "approved",
            }
            for item in users
        ]
    finally:
        db.close()


@app.post("/api/users")
def create_user(payload: UserCreateRequest, user: dict = Depends(require_roles("Admin"))):
    username = payload.username.strip().lower()
    full_name = payload.full_name.strip()
    role = payload.role.strip().title()

    if not username or not full_name:
        raise HTTPException(status_code=400, detail="Username and full name are required")
    if role not in ALLOWED_ROLES - {"Admin"}:
        raise HTTPException(status_code=400, detail="Admin can create Dispatcher or Responder accounts")
    validate_password_strength(payload.password)

    db = SessionLocal()
    try:
        existing = db.query(UserDB).filter(UserDB.username == username).first()
        if existing is not None:
            raise HTTPException(status_code=409, detail="Username already exists")

        new_user = UserDB(
            username=username,
            full_name=full_name,
            role=role,
            password_hash=hash_password(payload.password),
            is_active=1,
            created_at=datetime.utcnow().isoformat(),
            approval_status="approved",
        )
        db.add(new_user)
        write_audit(db, user, "users.create", "user", new_user.id, {"role": new_user.role, "username": new_user.username})
        db.commit()
        db.refresh(new_user)
        return {**public_user(new_user), "is_active": True, "created_at": new_user.created_at}
    finally:
        db.close()


@app.patch("/api/users/{user_id}")
def update_user(user_id: int, payload: UserUpdateRequest, user: dict = Depends(require_roles("Admin"))):
    db = SessionLocal()
    try:
        target = db.query(UserDB).filter(UserDB.id == user_id).first()
        if target is None:
            raise HTTPException(status_code=404, detail="User not found")

        if user_id == int(user["id"]) and payload.is_active is False:
            raise HTTPException(status_code=400, detail="You cannot deactivate your own account")

        if payload.full_name is not None:
            target.full_name = payload.full_name.strip()
        if payload.role is not None:
            role = payload.role.strip().title()
            if role not in ALLOWED_ROLES:
                raise HTTPException(status_code=400, detail="Role must be Admin, Dispatcher, or Responder")
            if role != target.role:
                invalidate_session = True
            # Never remove the last active administrator.
            if target.role == "Admin" and role != "Admin" and target.is_active:
                active_admins = db.query(UserDB).filter(UserDB.role == "Admin", UserDB.is_active == 1).count()
                if active_admins <= 1:
                    raise HTTPException(status_code=400, detail="At least one active Admin account is required")
            target.role = role
        invalidate_session = False
        if payload.password is not None:
            validate_password_strength(payload.password)
            target.password_hash = hash_password(payload.password)
            invalidate_session = True
        if payload.is_active is not None:
            if target.role == "Admin" and payload.is_active is False:
                active_admins = db.query(UserDB).filter(UserDB.role == "Admin", UserDB.is_active == 1).count()
                if active_admins <= 1:
                    raise HTTPException(status_code=400, detail="At least one active Admin account is required")
            target.is_active = 1 if payload.is_active else 0
            target.approval_status = "approved" if payload.is_active else "disabled"
            invalidate_session = True

        if "invalidate_session" in locals() and invalidate_session:
            target.session_version = int(target.session_version or 0) + 1
        write_audit(db, user, "users.update", "user", target.id, {"role": target.role, "is_active": bool(target.is_active)})
        db.commit()
        db.refresh(target)
        return {**public_user(target), "is_active": bool(target.is_active), "created_at": target.created_at, "approval_status": getattr(target, "approval_status", "approved") or "approved"}
    finally:
        db.close()


@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, user: dict = Depends(require_roles("Admin"))):
    if user_id == int(user["id"]):
        raise HTTPException(status_code=400, detail="You cannot delete your own account")

    db = SessionLocal()
    try:
        target = db.query(UserDB).filter(UserDB.id == user_id).first()
        if target is None:
            raise HTTPException(status_code=404, detail="User not found")

        if target.role == "Admin" and target.is_active:
            active_admins = db.query(UserDB).filter(UserDB.role == "Admin", UserDB.is_active == 1).count()
            if active_admins <= 1:
                raise HTTPException(status_code=400, detail="At least one active Admin account is required")

        write_audit(db, user, "users.delete", "user", target.id, {"username": target.username, "role": target.role})
        db.delete(target)
        db.commit()
        return {"message": "User deleted", "user_id": user_id}
    finally:
        db.close()


# =========================================================
# ROOT / HEALTH
# =========================================================

@app.get("/")
def root():
    return {
        "message": "RAKSHA AI Backend is running",
        "version": "4.0.0",
        "status": "online",
    }


@app.get("/api/health")
def health():
    db = SessionLocal()
    try:
        return {
            "status": "online",
            "service": "RAKSHA AI",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "incidents": db.query(IncidentDB).count(),
            "resources": db.query(ResourceDB).count(),
        }
    finally:
        db.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    try:
        payload = decode_access_token(token or "")
        user_id = int(payload["sub"])
        db = SessionLocal()
        try:
            db_user = db.query(UserDB).filter(UserDB.id == user_id).first()
            if db_user is None or not db_user.is_active or int(db_user.session_version or 0) != int(payload.get("ver", 0)):
                raise HTTPException(status_code=401, detail="Authentication required")
            user = {"id": db_user.id, "username": db_user.username, "full_name": db_user.full_name, "role": db_user.role}
        finally:
            db.close()
    except Exception:
        await websocket.close(code=1008, reason="Authentication required")
        return

    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "RAKSHA real-time channel connected",
            "user": {"username": user.get("username"), "role": user.get("role")},
            "timestamp": utc_now_iso(),
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)



# =========================================================
# MULTI-SOURCE DATA FUSION
# =========================================================

def _risk_level(score: int) -> str:
    score = int(max(0, min(100, score)))
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 35:
        return "MODERATE"
    return "LOW"


def _weighted_weather_signal(weather: Optional[dict]) -> tuple:
    if not weather:
        return 0, []
    next24 = weather.get("next_24h") or {}
    precip = float(next24.get("precipitation_mm") or 0)
    precip_prob = float(next24.get("max_precipitation_probability_pct") or 0)
    gust = float(next24.get("max_wind_gusts_kmh") or 0)

    precip_component = min(100.0, precip / 60.0 * 70.0)
    probability_component = min(100.0, precip_prob / 100.0 * 20.0)
    gust_component = min(100.0, max(0.0, gust - 40.0) / 80.0 * 10.0)
    score = int(round(min(100.0, precip_component + probability_component + gust_component)))
    drivers = []
    if precip >= 20:
        drivers.append(f"Next 24h precipitation is {precip:.1f} mm.")
    if precip_prob >= 60:
        drivers.append(f"Peak precipitation probability is {precip_prob:.0f}%.")
    if gust >= 50:
        drivers.append(f"Peak forecast wind gusts are {gust:.0f} km/h.")
    return score, drivers


def _weighted_seismic_signal(earthquakes: Optional[dict]) -> tuple:
    if not earthquakes:
        return 0, []
    strongest = 0
    nearest = None
    for event in earthquakes.get("events") or []:
        mag = _safe_float(event.get("magnitude"), 0.0) or 0.0
        distance = _safe_float(event.get("distance_km"), None)
        if distance is None:
            continue
        distance_factor = max(0.0, 1.0 - min(distance, 1000.0) / 1000.0)
        event_score = mag * 14.0 * (0.25 + 0.75 * distance_factor)
        if event_score > strongest:
            strongest = event_score
            nearest = event
    score = int(round(min(100.0, strongest)))
    drivers = []
    if nearest and (nearest.get("magnitude") or 0) >= 4.5 and (nearest.get("distance_km") or 99999) <= 500:
        drivers.append(f"Magnitude {nearest.get('magnitude'):.1f} earthquake at ~{nearest.get('distance_km'):.0f} km.")
    return score, drivers


def _incident_signal(db: Session, latitude: float, longitude: float) -> tuple:
    incidents = db.query(IncidentDB).filter(IncidentDB.status.notin_(["Resolved", "Closed"])).all()
    contributions = []
    for incident in incidents:
        score = int(incident.ai_score or 0)
        if incident.latitude is not None and incident.longitude is not None:
            distance = haversine_km(latitude, longitude, incident.latitude, incident.longitude)
            if distance > 100:
                continue
            proximity = max(0.2, 1.0 - distance / 100.0)
            contribution = score * proximity
            contributions.append((contribution, incident, distance))
        else:
            # Keep coordinate-less operator incidents as a low-weight regional signal.
            contributions.append((score * 0.25, incident, None))

    contributions.sort(key=lambda item: item[0], reverse=True)
    top = contributions[:5]
    if not top:
        return 0, []
    aggregate = min(100.0, sum(item[0] for item in top) / max(1, min(3, len(top))))
    drivers = []
    for contribution, incident, distance in top[:3]:
        if distance is None:
            drivers.append(f"{incident.disaster_type} incident at {incident.location} (score {incident.ai_score}).")
        else:
            drivers.append(f"{incident.disaster_type} incident ~{distance:.0f} km away (score {incident.ai_score}).")
    return int(round(aggregate)), drivers


def _official_alert_signal(sachet: Optional[dict]) -> tuple:
    if not sachet or not sachet.get("configured"):
        return 0, []
    alerts = sachet.get("alerts") or []
    if not alerts:
        return 0, []
    score = min(100, 55 + min(30, len(alerts) * 5))
    drivers = [f"{len(alerts)} official alert item(s) are available from the configured NDMA SACHET feed."]
    return score, drivers


def _resource_pressure(db: Session) -> tuple:
    resources = db.query(ResourceDB).all()
    if not resources:
        return 100, ["No resources are currently registered."]
    active = [r for r in resources if r.status in {"En Route", "Deployed", "Available"}]
    available = sum(1 for r in active if r.status == "Available")
    total = len(active)
    readiness = (available / total * 100.0) if total else 0.0
    pressure = int(round(100.0 - readiness))
    drivers = []
    if available == 0:
        drivers.append("No resources are currently marked Available.")
    elif pressure >= 60:
        drivers.append(f"Only {available} of {total} active resources are Available.")
    return pressure, drivers


def _data_coverage(source_status: dict) -> int:
    active = 0
    total = len(source_status)
    for status in source_status.values():
        if status in {"connected", "configured", "available"}:
            active += 1
    return int(round(active / total * 100)) if total else 0


def build_fusion_overview(latitude: float, longitude: float, save_snapshot: bool = True) -> dict:
    environment = _get_environment_overview(latitude, longitude)
    weather = environment.get("weather")
    earthquakes = environment.get("earthquakes")
    sachet = environment.get("sachet")

    db = SessionLocal()
    try:
        incident_score, incident_drivers = _incident_signal(db, latitude, longitude)
        weather_score, weather_drivers = _weighted_weather_signal(weather)
        seismic_score, seismic_drivers = _weighted_seismic_signal(earthquakes)
        alert_score, alert_drivers = _official_alert_signal(sachet)
        resource_pressure, resource_drivers = _resource_pressure(db)

        # Weighted situational index. This is an operational index, not a probability.
        hazard_score = (
            0.40 * incident_score +
            0.25 * weather_score +
            0.15 * seismic_score +
            0.20 * alert_score
        )
        fused_score = int(round(min(100.0, 0.85 * hazard_score + 0.15 * resource_pressure)))
        risk_level = _risk_level(fused_score)

        source_status = {
            "incidents": "available",
            "weather": "connected" if weather else "unavailable",
            "earthquakes": "connected" if earthquakes else "unavailable",
            "official_alerts": "configured" if sachet and sachet.get("configured") else "not_configured",
            "resources": "available" if db.query(ResourceDB).count() else "unavailable",
        }
        drivers = incident_drivers[:2] + weather_drivers[:2] + seismic_drivers[:1] + alert_drivers[:1] + resource_drivers[:1]
        source_errors = list(environment.get("source_errors") or [])

        snapshot = {
            "location": {"latitude": latitude, "longitude": longitude},
            "fused_risk_score": fused_score,
            "risk_level": risk_level,
            "index_type": "operational situational risk index",
            "signals": {
                "incident": incident_score,
                "weather": weather_score,
                "seismic": seismic_score,
                "official_alerts": alert_score,
                "resource_pressure": resource_pressure,
            },
            "data_coverage_pct": _data_coverage(source_status),
            "drivers": drivers,
            "source_status": source_status,
            "source_errors": source_errors,
            "environment_generated_at": environment.get("generated_at"),
            "generated_at": utc_now_iso(),
            "disclaimer": "This fused score is an operational situational index for decision support, not a probability of disaster or a substitute for official warnings.",
        }

        if save_snapshot:
            grid_key = _environment_cache_key(latitude, longitude)
            last = (db.query(FusionSnapshotDB)
                    .filter(FusionSnapshotDB.grid_key == grid_key)
                    .order_by(FusionSnapshotDB.id.desc())
                    .first())
            now = datetime.utcnow()
            should_save = last is None
            if last is not None:
                try:
                    should_save = (now - datetime.fromisoformat(last.created_at)).total_seconds() >= ENV_CACHE_SECONDS
                except ValueError:
                    should_save = True
            if should_save:
                row = FusionSnapshotDB(
                    grid_key=grid_key, latitude=latitude, longitude=longitude,
                    fused_risk_score=fused_score, risk_level=risk_level,
                    incident_signal=incident_score, weather_signal=weather_score,
                    seismic_signal=seismic_score, official_alert_signal=alert_score,
                    resource_pressure=resource_pressure, data_coverage=snapshot["data_coverage_pct"],
                    drivers=json.dumps(drivers, ensure_ascii=False),
                    source_status=json.dumps(source_status, ensure_ascii=False),
                    created_at=now.isoformat(),
                )
                db.add(row)
                db.commit()
        return snapshot
    finally:
        db.close()


# =========================================================
# ENVIRONMENTAL / EXTERNAL DATA INGESTION
# =========================================================

def _fetch_url(url: str, headers: Optional[dict] = None, timeout: int = 10):
    req = URLRequest(url, headers=headers or {"User-Agent": "RAKSHA-AI/4.1 (+environment-monitoring)"})
    with urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, dict(response.headers.items()), body


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.asin(min(1.0, math.sqrt(a)))


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _environment_cache_key(latitude: float, longitude: float) -> str:
    return f"{round(latitude, 3)}:{round(longitude, 3)}"


def _parse_weather_payload(payload: dict, source: str, source_url: str):
    current = payload.get("current") or {}
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    precipitation = hourly.get("precipitation") or []
    rain_probability = hourly.get("precipitation_probability") or []
    wind_speed = hourly.get("wind_speed_10m") or []
    wind_gusts = hourly.get("wind_gusts_10m") or []
    temperatures = hourly.get("temperature_2m") or []
    weather_codes = hourly.get("weather_code") or []

    next_24_precip = round(sum(_safe_float(x, 0.0) or 0.0 for x in precipitation[:24]), 1)
    max_rain_probability = max([int(_safe_float(x, 0) or 0) for x in rain_probability[:24]] or [0])
    max_wind = max([_safe_float(x, 0.0) or 0.0 for x in wind_speed[:24]] or [0.0])
    max_gust = max([_safe_float(x, 0.0) or 0.0 for x in wind_gusts[:24]] or [0.0])

    return {
        "source": source,
        "source_type": "weather_model",
        "source_url": source_url,
        "timezone": payload.get("timezone"),
        "current": current,
        "next_24h": {
            "precipitation_mm": next_24_precip,
            "max_precipitation_probability_pct": max_rain_probability,
            "max_wind_kmh": round(max_wind, 1),
            "max_wind_gusts_kmh": round(max_gust, 1),
        },
        "forecast_hours": [
            {
                "time": times[i] if i < len(times) else None,
                "precipitation_mm": _safe_float(precipitation[i] if i < len(precipitation) else None),
                "precipitation_probability_pct": _safe_float(rain_probability[i] if i < len(rain_probability) else None),
                "wind_speed_kmh": _safe_float(wind_speed[i] if i < len(wind_speed) else None),
                "wind_gust_kmh": _safe_float(wind_gusts[i] if i < len(wind_gusts) else None),
                "temperature_c": _safe_float(temperatures[i] if i < len(temperatures) else None),
                "weather_code": weather_codes[i] if i < len(weather_codes) else None,
            }
            for i in range(min(12, len(times)))
        ],
        "fetched_at": utc_now_iso(),
    }


def _fetch_wttr_weather(latitude: float, longitude: float):
    url = f"https://wttr.in/{latitude:.4f},{longitude:.4f}?format=j1"
    status, headers, body = _fetch_url(
        url,
        headers={"User-Agent": "RAKSHA-AI/4.2 (+environment-monitoring)"},
        timeout=10,
    )
    if status != 200:
        raise RuntimeError(f"wttr.in returned {status}")

    payload = json.loads(body)
    current_rows = payload.get("current_condition") or []
    current_row = current_rows[0] if current_rows else {}
    weather_days = payload.get("weather") or []

    current = {
        "temperature_2m": _safe_float(current_row.get("temp_C")),
        "relative_humidity_2m": _safe_float(current_row.get("humidity")),
        "precipitation": _safe_float(current_row.get("precipMM")),
        "weather_code": current_row.get("weatherCode"),
        "wind_speed_10m": _safe_float(current_row.get("windspeedKmph")),
        "wind_gusts_10m": _safe_float(current_row.get("WindGustKmph")),
        "pressure_msl": _safe_float(current_row.get("pressure")),
        "time": utc_now_iso(),
        "weather_description": (
            (current_row.get("weatherDesc") or [{}])[0].get("value")
            if current_row.get("weatherDesc")
            else None
        ),
    }

    forecast_hours = []
    for day in weather_days:
        for hour in day.get("hourly") or []:
            time_value = hour.get("time")
            if isinstance(time_value, str) and len(time_value) <= 4:
                hhmm = time_value.zfill(4)
                date_value = day.get("date")
                if date_value:
                    time_value = f"{date_value}T{hhmm[:2]}:{hhmm[2:]}"
            forecast_hours.append({
                "time": time_value,
                "precipitation_mm": _safe_float(hour.get("precipMM")),
                "precipitation_probability_pct": _safe_float(hour.get("chanceofrain")),
                "wind_speed_kmh": _safe_float(hour.get("windspeedKmph")),
                "wind_gust_kmh": _safe_float(hour.get("WindGustKmph")),
                "temperature_c": _safe_float(hour.get("tempC")),
                "weather_code": hour.get("weatherCode"),
            })
            if len(forecast_hours) >= 12:
                break
        if len(forecast_hours) >= 12:
            break

    next_24 = forecast_hours[:12]
    return {
        "source": "wttr.in (fallback)",
        "source_type": "weather_model",
        "source_url": "https://wttr.in/",
        "timezone": payload.get("timezone"),
        "current": current,
        "next_24h": {
            "precipitation_mm": round(
                sum(_safe_float(x.get("precipitation_mm"), 0.0) or 0.0 for x in next_24), 1
            ),
            "max_precipitation_probability_pct": int(
                max([_safe_float(x.get("precipitation_probability_pct"), 0) or 0 for x in next_24] or [0])
            ),
            "max_wind_kmh": round(
                max([_safe_float(x.get("wind_speed_kmh"), 0.0) or 0.0 for x in next_24] or [0.0]), 1
            ),
            "max_wind_gusts_kmh": round(
                max([_safe_float(x.get("wind_gust_kmh"), 0.0) or 0.0 for x in next_24] or [0.0]), 1
            ),
        },
        "forecast_hours": forecast_hours,
        "fetched_at": utc_now_iso(),
    }


def fetch_weather(latitude: float, longitude: float):
    key = _environment_cache_key(latitude, longitude)
    now = time.time()
    cached = _weather_cache.get(key)
    if cached and now - cached["timestamp"] < WEATHER_CACHE_SECONDS:
        result = dict(cached["value"])
        result["cached"] = True
        return result

    params = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m,wind_gusts_10m,pressure_msl",
        "hourly": "precipitation,precipitation_probability,wind_speed_10m,wind_gusts_10m,temperature_2m,weather_code",
        "forecast_days": 2,
        "timezone": "auto",
    })

    try:
        status, headers, body = _fetch_url(
            f"{OPEN_METEO_URL}?{params}",
            headers={"User-Agent": "RAKSHA-AI/4.2 (+environment-monitoring)"},
        )
        if status != 200:
            raise RuntimeError(f"Weather provider returned {status}")
        payload = json.loads(body)
        result = _parse_weather_payload(payload, "Open-Meteo", "https://open-meteo.com/")
    except HTTPError as error:
        if error.code != 429:
            raise
        logger.warning("Open-Meteo rate limited (429); using wttr.in fallback.")
        result = _fetch_wttr_weather(latitude, longitude)
    except (URLError, RuntimeError, json.JSONDecodeError) as error:
        logger.warning("Open-Meteo unavailable (%s); trying wttr.in fallback.", error)
        result = _fetch_wttr_weather(latitude, longitude)

    _weather_cache[key] = {"timestamp": now, "value": result}
    return result


def fetch_earthquakes(latitude: float, longitude: float, radius_km: int = 300):
    """Fetch earthquakes for the same 7-day / 300-km semantics used by the ML features.

    The production model contains features named eq_count_7d_300km,
    eq_max_magnitude_7d_300km, and eq_nearest_km_7d_300km. Querying the
    USGS catalog directly keeps those feature names aligned with the data
    actually supplied at inference time.
    """
    now = datetime.utcnow()
    start_time = now - timedelta(days=7)
    params = urlencode({
        "format": "geojson",
        "starttime": start_time.replace(microsecond=0).isoformat() + "Z",
        "endtime": now.replace(microsecond=0).isoformat() + "Z",
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": radius_km,
        "eventtype": "earthquake",
        "orderby": "time",
        "limit": 2000,
    })
    status, headers, body = _fetch_url(f"{USGS_EARTHQUAKE_URL}?{params}")
    if status != 200:
        raise RuntimeError(f"USGS returned {status}")
    payload = json.loads(body)
    events = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        ev_lon, ev_lat = _safe_float(coordinates[0]), _safe_float(coordinates[1])
        if ev_lat is None or ev_lon is None:
            continue
        props = feature.get("properties") or {}
        distance = _haversine_km(latitude, longitude, ev_lat, ev_lon)
        if distance > radius_km:
            continue
        events.append({
            "id": feature.get("id"),
            "magnitude": _safe_float(props.get("mag")),
            "place": props.get("place"),
            "time": props.get("time"),
            "url": props.get("url"),
            "latitude": ev_lat,
            "longitude": ev_lon,
            "depth_km": _safe_float(coordinates[2]) if len(coordinates) > 2 else None,
            "distance_km": round(distance, 1),
            "significance": props.get("sig"),
        })

    events.sort(key=lambda item: (item.get("time") is None, item.get("time") or 0), reverse=True)
    return {
        "source": "USGS Earthquake Hazards Program",
        "source_type": "earthquake_catalog",
        "source_url": USGS_EARTHQUAKE_URL,
        "radius_km": radius_km,
        "window_days": 7,
        "events": events[:50],
        "fetched_at": utc_now_iso(),
    }


def _local_name(tag: str) -> str:
    return tag.split("}")[-1].lower()


def parse_sachet_alerts(xml_text: str):
    root = ET.fromstring(xml_text)
    alerts = []
    for item in list(root):
        tag = _local_name(item.tag)
        if tag not in {"item", "entry", "alert"}:
            continue
        values = {}
        for child in list(item):
            values[_local_name(child.tag)] = (child.text or "").strip()
        title = values.get("title") or values.get("event") or "Official disaster alert"
        description = values.get("description") or values.get("summary") or ""
        link = values.get("link") or values.get("id")
        alerts.append({
            "title": title,
            "description": description,
            "link": link,
            "published": values.get("pubdate") or values.get("published") or values.get("sent"),
            "source": "NDMA SACHET",
        })
    return alerts[:100]


def fetch_sachet_alerts():
    if not SACHET_CAP_URL:
        return {
            "configured": False,
            "source": "NDMA SACHET",
            "source_type": "official_alert_feed",
            "source_url": SACHET_SOURCE_URL,
            "alerts": [],
            "message": "Official SACHET CAP feed URL is not configured for this deployment.",
            "fetched_at": utc_now_iso(),
        }
    now = time.time()
    if _sachet_cache["xml"] is not None and now - _sachet_cache["fetched_at"] < ENV_CACHE_SECONDS:
        return {
            "configured": True,
            "source": "NDMA SACHET",
            "source_type": "official_alert_feed",
            "source_url": SACHET_SOURCE_URL,
            "alerts": _sachet_cache["alerts"],
            "cached": True,
            "fetched_at": datetime.fromtimestamp(_sachet_cache["fetched_at"]).isoformat(),
        }
    request_headers = {"User-Agent": "RAKSHA-AI/4.1 (+official-alert-consumer)"}
    if _sachet_cache["etag"]:
        request_headers["If-None-Match"] = _sachet_cache["etag"]
    try:
        status, headers, body = _fetch_url(SACHET_CAP_URL, request_headers, timeout=12)
        if status == 304:
            _sachet_cache["fetched_at"] = now
        else:
            _sachet_cache["xml"] = body
            _sachet_cache["alerts"] = parse_sachet_alerts(body)
            _sachet_cache["etag"] = headers.get("ETag") or headers.get("Etag")
            _sachet_cache["fetched_at"] = now
    except HTTPError as error:
        if error.code == 304:
            _sachet_cache["fetched_at"] = now
        else:
            raise RuntimeError(f"SACHET feed returned {error.code}") from error
    return {
        "configured": True,
        "source": "NDMA SACHET",
        "source_type": "official_alert_feed",
        "source_url": SACHET_SOURCE_URL,
        "alerts": _sachet_cache["alerts"],
        "etag_enabled": True,
        "cached": False,
        "fetched_at": datetime.fromtimestamp(_sachet_cache["fetched_at"]).isoformat(),
    }


@app.get("/api/environment/weather")
def environment_weather(latitude: float = Query(..., ge=-90, le=90), longitude: float = Query(..., ge=-180, le=180), user: dict = Depends(get_current_user)):
    return fetch_weather(latitude, longitude)


@app.get("/api/environment/earthquakes")
def environment_earthquakes(latitude: float = Query(..., ge=-90, le=90), longitude: float = Query(..., ge=-180, le=180), radius_km: int = Query(default=1000, ge=50, le=5000), user: dict = Depends(get_current_user)):
    return fetch_earthquakes(latitude, longitude, radius_km)


@app.get("/api/environment/sachet")
def environment_sachet(user: dict = Depends(get_current_user)):
    return fetch_sachet_alerts()


def _get_environment_overview(latitude: float, longitude: float) -> dict:
    key = _environment_cache_key(latitude, longitude)
    now = time.time()
    cached = _environment_cache.get(key)
    if cached and now - cached["timestamp"] < ENV_CACHE_SECONDS:
        result = dict(cached["value"])
        result["cached"] = True
        return result

    weather = None
    earthquakes = None
    source_errors = []
    try:
        weather = fetch_weather(latitude, longitude)
    except Exception as error:
        logger.warning("Weather source unavailable: %s", error)
        source_errors.append({"source": "Open-Meteo", "message": "Weather data temporarily unavailable."})
    try:
        earthquakes = fetch_earthquakes(latitude, longitude, 300)
    except Exception as error:
        logger.warning("Earthquake source unavailable: %s", error)
        source_errors.append({"source": "USGS", "message": "Earthquake feed temporarily unavailable."})
    try:
        sachet = fetch_sachet_alerts()
    except Exception as error:
        logger.warning("SACHET source unavailable: %s", error)
        sachet = {
            "configured": bool(SACHET_CAP_URL),
            "source": "NDMA SACHET",
            "source_url": SACHET_SOURCE_URL,
            "alerts": [],
            "message": "Official alert feed temporarily unavailable.",
        }
        source_errors.append({"source": "NDMA SACHET", "message": "Official alert feed temporarily unavailable."})

    value = {
        "location": {"latitude": latitude, "longitude": longitude},
        "weather": weather,
        "earthquakes": earthquakes,
        "sachet": sachet,
        "source_errors": source_errors,
        "generated_at": utc_now_iso(),
        "cached": False,
        "disclaimer": "Environmental data is informational and source-dependent. Official emergency alerts remain authoritative.",
    }
    _environment_cache[key] = {"timestamp": now, "value": value}
    return value


@app.get("/api/environment/overview")
def environment_overview(latitude: float = Query(..., ge=-90, le=90), longitude: float = Query(..., ge=-180, le=180), user: dict = Depends(get_current_user)):
    return _get_environment_overview(latitude, longitude)


@app.get("/api/fusion/overview")
def fusion_overview(latitude: float = Query(..., ge=-90, le=90), longitude: float = Query(..., ge=-180, le=180), user: dict = Depends(get_current_user)):
    return build_fusion_overview(latitude, longitude, save_snapshot=True)


@app.get("/api/fusion/history")
def fusion_history(latitude: float = Query(..., ge=-90, le=90), longitude: float = Query(..., ge=-180, le=180), hours: int = Query(default=24, ge=1, le=168), user: dict = Depends(get_current_user)):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    grid_key = _environment_cache_key(latitude, longitude)
    db = SessionLocal()
    try:
        rows = (db.query(FusionSnapshotDB)
                .filter(FusionSnapshotDB.grid_key == grid_key, FusionSnapshotDB.created_at >= cutoff.isoformat())
                .order_by(FusionSnapshotDB.created_at.asc())
                .all())
        return [{
            "created_at": row.created_at,
            "fused_risk_score": row.fused_risk_score,
            "risk_level": row.risk_level,
            "signals": {
                "incident": row.incident_signal,
                "weather": row.weather_signal,
                "seismic": row.seismic_signal,
                "official_alerts": row.official_alert_signal,
                "resource_pressure": row.resource_pressure,
            },
            "data_coverage_pct": row.data_coverage,
        } for row in rows]
    finally:
        db.close()


# =========================================================
# INCIDENTS
# =========================================================

@app.get("/api/incidents", response_model=List[Incident])
def get_incidents(user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        return [incident_dict(x) for x in db.query(IncidentDB).order_by(IncidentDB.id.asc()).all()]
    finally:
        db.close()


@app.get("/api/incidents/{incident_id}", response_model=Incident)
def get_incident(incident_id: int, user: dict = Depends(get_current_user)):
    db=SessionLocal()
    try:
        x=db.query(IncidentDB).filter(IncidentDB.id==incident_id).first()
        if x is None: raise HTTPException(status_code=404, detail="Incident not found")
        return incident_dict(x)
    finally: db.close()


ML_MODEL_PATH = os.getenv("RAKSHA_ML_MODEL_PATH", DEFAULT_MODEL_PATH)


def ml_model_available() -> bool:
    status = get_ml_model_status(ML_MODEL_PATH)
    return bool(status.get("model_available"))


def calculate_ml_or_baseline_score(severity: str, people_affected: int, disaster_type: str = "Other"):
    """Use a validated/demo-approved model when configured, otherwise use the transparent baseline."""
    try:
        result = predict_risk(
            disaster_type=disaster_type,
            severity=severity,
            people_affected=people_affected,
            model_path=ML_MODEL_PATH,
        )
        if result is not None:
            return int(result["risk_score"]), "ml", result
    except Exception:
        logger.exception("ML model inference failed; using baseline")
    return calculate_ai_score(severity, people_affected), "rules_baseline", None


def calculate_ai_score(severity: str, people_affected: int) -> int:
    score_map = {
        "Low": 30,
        "Medium": 55,
        "High": 78,
        "Critical": 95,
    }
    score = score_map.get(severity, 50)
    if people_affected >= 300:
        score += 5
    elif people_affected >= 100:
        score += 3
    return min(score, 100)


@app.post("/api/incidents", response_model=Incident)
async def create_incident(incident: IncidentCreate, user: dict = Depends(require_roles("Admin", "Dispatcher"))):
    db=SessionLocal()
    try:
        severity = incident.severity.strip().capitalize()
        if severity not in {"Low", "Medium", "High", "Critical"}:
            severity = "Medium"

        disaster_type = incident.disaster_type.strip()
        location = incident.location.strip()
        description = incident.description.strip()

        ai_score, score_source, ml_result = calculate_ml_or_baseline_score(
            severity=severity,
            people_affected=incident.people_affected,
            disaster_type=disaster_type,
        )

        x = IncidentDB(
            disaster_type=disaster_type,
            location=location,
            severity=severity,
            description=description,
            people_affected=incident.people_affected,
            status="Active",
            created_at=datetime.utcnow().isoformat(),
            ai_score=ai_score,
            ai_score_source=score_source,
        )
        db.add(x)
        write_audit(db, user, "incidents.create", "incident", x.id, {"severity": x.severity, "source": x.source})
        db.commit()
        db.refresh(x)

        data = incident_dict(x)

        await manager.broadcast({
            "type": "incident_created",
            "incident": data,
            "timestamp": datetime.utcnow().isoformat(),
        })

        if severity in {"Critical", "High"}:
            await create_alert(
                db,
                x.id,
                "CRITICAL_INCIDENT" if severity == "Critical" else "HIGH_PRIORITY_INCIDENT",
                f"{severity} {disaster_type} Alert",
                f"{disaster_type} reported near {location}: {description}",
                severity,
                ai_score,
            )
            db.commit()

        return data

    except HTTPException:
        raise
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to create incident. Please try again.",
        )
    finally:
        db.close()


@app.patch("/api/incidents/{incident_id}/status")
async def update_incident_status(incident_id: int, status: str, user: dict = Depends(require_roles("Admin", "Dispatcher", "Responder"))):
    allowed=["Active","Dispatched","In Progress","Responding","Contained","Resolved","Closed"]
    normalized=status.strip().title()
    if normalized not in allowed: raise HTTPException(status_code=400, detail={"message":"Invalid status","allowed_statuses":allowed})
    db=SessionLocal()
    try:
        x=db.query(IncidentDB).filter(IncidentDB.id==incident_id).first()
        if x is None: raise HTTPException(status_code=404, detail="Incident not found")
        previous_status = x.status
        x.status = normalized
        db.commit()
        db.refresh(x)

        data = incident_dict(x)

        await manager.broadcast({
            "type": "incident_updated",
            "incident": data,
            "previous_status": previous_status,
            "timestamp": datetime.utcnow().isoformat(),
        })

        if normalized in {"Dispatched", "In Progress", "Contained"} and normalized != previous_status:
            await create_alert(
                db,
                x.id,
                "INCIDENT_STATUS",
                f"Incident #{x.id} Status Updated",
                f"{x.disaster_type} at {x.location} changed from {previous_status} to {normalized}.",
                x.severity,
                x.ai_score or 0,
            )
            db.commit()

        return data
    finally: db.close()


@app.delete("/api/incidents/{incident_id}")
async def delete_incident(incident_id: int, user: dict = Depends(require_roles("Admin"))):
    db=SessionLocal()
    try:
        x=db.query(IncidentDB).filter(IncidentDB.id==incident_id).first()
        if x is None: raise HTTPException(status_code=404, detail="Incident not found")
        for resource in db.query(ResourceDB).filter(ResourceDB.assigned_incident_id==incident_id).all():
            resource.assigned_incident_id=None; resource.eta_minutes=None; resource.status="Available"
        deleted = incident_dict(x)
        db.delete(x)
        db.commit()

        await manager.broadcast({
            "type": "incident_deleted",
            "incident": deleted,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return {"message":"Incident deleted","incident":deleted}
    finally: db.close()


# =========================================================
# AI RESPONSE ENGINE
# =========================================================

@app.get("/api/ml/status")
def ml_status(user: dict = Depends(get_current_user)):
    status = get_ml_model_status(ML_MODEL_PATH)
    status["model_path_configured"] = bool(ML_MODEL_PATH)
    return status


class MLContextFeatures(BaseModel):
    # Optional location/date fields used by the production ML feature builder.
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    district: Optional[str] = Field(default=None, max_length=120)
    province: Optional[str] = Field(default=None, max_length=120)
    event_date: Optional[str] = Field(default=None, max_length=30)

    rainfall_1h_mm: Optional[float] = Field(default=None, ge=0)
    rainfall_24h_mm: Optional[float] = Field(default=None, ge=0)
    rainfall_probability_pct: Optional[float] = Field(default=None, ge=0, le=100)
    wind_speed_kmh: Optional[float] = Field(default=None, ge=0)
    wind_gust_kmh: Optional[float] = Field(default=None, ge=0)
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = Field(default=None, ge=0, le=100)
    water_level_m: Optional[float] = None
    water_level_change_m: Optional[float] = None
    population_density_km2: Optional[float] = Field(default=None, ge=0)
    elevation_m: Optional[float] = None
    distance_to_hospital_km: Optional[float] = Field(default=None, ge=0)
    road_accessibility_score: Optional[float] = Field(default=None, ge=0, le=100)
    nearby_incidents_24h: Optional[float] = Field(default=None, ge=0)
    earthquake_magnitude: Optional[float] = Field(default=None, ge=0, le=10)
    resource_pressure_pct: Optional[float] = Field(default=None, ge=0, le=100)
    official_alert_level: Optional[float] = Field(default=None, ge=0, le=4)
    fused_risk_signal: Optional[float] = Field(default=None, ge=0, le=100)
    data_coverage_pct: Optional[float] = Field(default=None, ge=0, le=100)


class MLPredictRequest(BaseModel):
    disaster_type: str = Field(min_length=2, max_length=100)
    severity: str
    people_affected: int = Field(default=0, ge=0, le=1000000000)
    environmental_features: Optional[MLContextFeatures] = None


@app.post("/api/ml/predict")
def ml_predict(payload: MLPredictRequest, user: dict = Depends(get_current_user)):
    if str(user.get("role", "")).lower() not in {"admin", "dispatcher"}:
        raise HTTPException(status_code=403, detail="Only Admin and Dispatcher roles can run ML predictions")
    extra = payload.environmental_features.model_dump(exclude_none=True) if payload.environmental_features else {}
    result = predict_risk(
        disaster_type=payload.disaster_type,
        severity=payload.severity,
        people_affected=payload.people_affected,
        model_path=ML_MODEL_PATH,
        extra_features=extra,
    )
    if result is not None:
        return result

    baseline = calculate_ai_score(payload.severity.strip().capitalize(), payload.people_affected)
    return {
        "risk_score": baseline,
        "risk_level": "CRITICAL" if baseline >= 85 else "HIGH" if baseline >= 70 else "MODERATE" if baseline >= 40 else "LOW",
        "source": "rules_baseline",
        "model_status": get_ml_model_status(ML_MODEL_PATH).get("status"),
        "message": "No validated ML model is active; transparent baseline returned.",
    }


@app.get("/api/ml/model-card")
def ml_model_card(user: dict = Depends(get_current_user)):
    status = get_ml_model_status(ML_MODEL_PATH)
    return {
        "model": status,
        "intended_use": "Decision-support for disaster operations; not an autonomous emergency authority.",
        "activation_policy": "Only validated bundles are active operationally. Synthetic/demo bundles require explicit local opt-in.",
        "required_production_controls": [
            "Representative and permissioned training data",
            "Time-ordered holdout evaluation",
            "Human/domain validation",
            "Prediction monitoring and drift checks",
            "Model versioning and rollback",
        ],
    }


@app.get("/api/ml/fused-risk")
def ml_fused_risk(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    user: dict = Depends(get_current_user),
):
    environment = _get_environment_overview(latitude, longitude)
    fusion = build_fusion_overview(latitude, longitude, save_snapshot=False)

    db = SessionLocal()
    try:
        incidents = db.query(IncidentDB).filter(IncidentDB.status.notin_(["Resolved", "Closed"])).all()
        selected = None
        selected_distance = None
        for incident in incidents:
            if incident.latitude is not None and incident.longitude is not None:
                distance = _haversine_km(latitude, longitude, incident.latitude, incident.longitude)
            else:
                distance = None
            if selected is None or (distance is not None and (selected_distance is None or distance < selected_distance)) or (distance is None and selected_distance is None and (incident.ai_score or 0) > (selected.ai_score or 0)):
                selected = incident
                selected_distance = distance

        weather = environment.get("weather") or {}
        current = weather.get("current") or {}
        next24 = weather.get("next_24h") or {}
        earthquakes = environment.get("earthquakes") or {}
        events = earthquakes.get("events") or []
        max_magnitude = max([float(e.get("magnitude")) for e in events if e.get("magnitude") is not None] or [0.0])
        alerts = (environment.get("sachet") or {}).get("alerts") or []
        alert_level = 3 if alerts else 0

        if selected is not None:
            disaster_type = selected.disaster_type
            severity = selected.severity
            people = int(selected.people_affected or 0)
        else:
            disaster_type = "Multi-Hazard"
            fused = int(fusion.get("fused_risk_score", 0))
            severity = "Critical" if fused >= 85 else "High" if fused >= 70 else "Medium" if fused >= 40 else "Low"
            people = 0

        signals = fusion.get("signals") or {}
        earthquake_feed_available = bool(earthquakes)
        earthquake_events_detected = bool(events)
        nearest_eq_km = None

        for event in events:
            event_lat = event.get("latitude")
            event_lon = event.get("longitude")

            if event_lat is not None and event_lon is not None:
                distance = _haversine_km(
                    latitude,
                    longitude,
                    float(event_lat),
                    float(event_lon),
                )

                if nearest_eq_km is None or distance < nearest_eq_km:
                    nearest_eq_km = distance

        extra = {
            # Existing operational context
            "rainfall_1h_mm": current.get("precipitation", 0),
            "rainfall_24h_mm": next24.get("precipitation_mm", 0),
            "rainfall_probability_pct": next24.get(
                "max_precipitation_probability_pct", 0
            ),
            "wind_speed_kmh": current.get("wind_speed_10m", 0),
            "wind_gust_kmh": current.get(
                "wind_gusts_10m",
                next24.get("max_wind_gusts_kmh", 0),
            ),
            "temperature_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "nearby_incidents_24h": signals.get("incident", 0),
            "earthquake_magnitude": max_magnitude,
            "resource_pressure_pct": signals.get("resource_pressure", 0),
            "official_alert_level": alert_level,
            "fused_risk_signal": fusion.get("fused_risk_score", 0),
            "data_coverage_pct": fusion.get("data_coverage_pct", 0),

            # Phase 5D.19 trained ML features
            "latitude": latitude,
            "longitude": longitude,
            # Weather is queried directly at the requested coordinates.
            "weather_match_distance_km": 0.0,
            "temperature_mean_c": current.get("temperature_2m"),
            "temperature_max_c": current.get("temperature_2m"),
            "temperature_min_c": current.get("temperature_2m"),
            "precipitation_mm": next24.get("precipitation_mm", 0),
            "wind_max_kmh": next24.get(
                "max_wind_kmh",
                current.get("wind_speed_10m", 0),
            ),
            "wind_gust_max_kmh": next24.get(
                "max_wind_gusts_kmh",
                current.get("wind_gusts_10m", 0),
            ),
            "eq_count_7d_300km": len(events),
            "eq_max_magnitude_7d_300km": max_magnitude,
            "eq_nearest_km_7d_300km": nearest_eq_km,
            "has_coordinates": 1,
            "has_weather": 1 if environment.get("weather") else 0,
            "has_earthquake_context": 1 if events else 0,
            "earthquake_feed_available": earthquake_feed_available,
            "earthquake_events_detected": earthquake_events_detected,
        }

        # Use the incident timestamp when available so the predictor can
        # derive the event's month/day seasonality instead of always using now.
        if selected is not None and selected.created_at:
            # predictor expects an event date in YYYY-MM-DD format.
            # IncidentDB.created_at is stored as an ISO timestamp, so keep only the date.
            event_date = str(selected.created_at)[:10]
            extra["event_date"] = event_date

        # Make the distinction between a connected earthquake feed and
        # actual nearby earthquake events explicit for the frontend.
        prediction = predict_risk(
            disaster_type=disaster_type,
            severity=severity,
            people_affected=people,
            model_path=ML_MODEL_PATH,
            extra_features=extra,
        )

        if prediction is not None:
            quality = prediction.setdefault("input_quality", {})
            quality["earthquake_feed_available"] = earthquake_feed_available
            quality["earthquake_events_detected"] = earthquake_events_detected
            # Keep this field about usable earthquake context, not merely
            # whether a provider responded. The event count is separately
            # exposed so the UI can say "feed connected, no nearby events".
            quality["earthquake_context_available"] = earthquake_events_detected

            event_date_value = extra.get("event_date")
            temporal_source = (
                "incident_created_at_proxy" if selected is not None and event_date_value
                else "operational_current_date"
            )
            try:
                event_year = int(str(event_date_value)[:4]) if event_date_value else None
            except (TypeError, ValueError):
                event_year = None
            prediction["temporal_context"] = {
                "event_date": event_date_value,
                "source": temporal_source,
                "training_range": "1971-2013",
                "within_training_range": bool(event_year is not None and 1971 <= event_year <= 2013),
            }

        return {
            "location": {"latitude": latitude, "longitude": longitude},
            "incident_context": {
                "incident_id": selected.id if selected else None,
                "disaster_type": disaster_type,
                "severity": severity,
                "people_affected": people,
                "distance_km": round(selected_distance, 2) if selected_distance is not None else None,
            },
            "environment_features": extra,
            "fusion_score": fusion.get("fused_risk_score", 0),
            "prediction": prediction or {
                "source": "rules_baseline",
                "risk_score": fusion.get("fused_risk_score", 0),
                "message": "No active ML model; fusion index returned as decision-support baseline.",
            },
            "generated_at": utc_now_iso(),
        }
    finally:
        db.close()


@app.get("/api/incidents/{incident_id}/response")
def get_incident_response(incident_id: int, user: dict = Depends(get_current_user)):
    db=SessionLocal()
    try:
        x=db.query(IncidentDB).filter(IncidentDB.id==incident_id).first()
        if x is None: raise HTTPException(status_code=404, detail="Incident not found")
        incident=incident_dict(x)

        disaster_type = str(incident["disaster_type"]).lower()
        severity = str(incident["severity"]).lower()
        people = int(incident.get("people_affected", 0))
        recommendations = []

        if disaster_type == "flood":
            recommendations.extend([
                "Deploy rescue boats and water rescue teams to the affected sector.",
                "Evacuate residents from low-lying and flooded buildings.",
                "Establish an emergency shelter and temporary medical camp.",
                "Block unsafe roads and redirect civilian traffic.",
            ])
            if people >= 100:
                recommendations.append("Prioritize evacuation for the large affected population.")
        elif disaster_type == "fire":
            recommendations.extend([
                "Immediately dispatch fire and rescue units to the incident location.",
                "Establish a safety perimeter around the affected buildings.",
                "Evacuate nearby residential buildings and vulnerable civilians.",
                "Deploy medical teams for smoke inhalation and burn injuries.",
            ])
            if people >= 100:
                recommendations.append("Activate mass-casualty emergency preparedness.")
        elif disaster_type == "earthquake":
            recommendations.extend([
                "Deploy search and rescue teams to damaged structures.",
                "Inspect buildings for structural instability before entry.",
                "Establish emergency medical triage zones.",
                "Evacuate civilians from severely damaged structures.",
            ])
        elif disaster_type == "cyclone":
            recommendations.extend([
                "Activate cyclone emergency response teams.",
                "Move residents from vulnerable coastal and low-lying areas.",
                "Secure emergency shelters and evacuation routes.",
                "Deploy medical and rescue teams to high-risk zones.",
            ])
        elif disaster_type == "landslide":
            recommendations.extend([
                "Deploy search and rescue teams to the affected area.",
                "Restrict access to unstable slopes and damaged roads.",
                "Evacuate residents from areas at risk of further landslides.",
                "Inspect nearby infrastructure for structural damage.",
            ])
        else:
            recommendations.extend([
                "Deploy emergency response teams to the reported location.",
                "Assess the incident zone and identify immediate hazards.",
                "Protect civilians and establish an emergency perimeter.",
                "Deploy medical assistance for affected people.",
            ])

        priority = "CRITICAL" if severity == "critical" else "HIGH" if severity == "high" else "MEDIUM" if severity == "medium" else "LOW"
        ai_score = int(incident.get("ai_score", 50))
        confidence = "Very High" if ai_score >= 90 else "High" if ai_score >= 75 else "Moderate" if ai_score >= 50 else "Low"

        return {
            "incident_id": incident_id,
            "priority": priority,
            "ai_score": ai_score,
            "score_source": incident.get("ai_score_source", "rules_baseline"),
            "confidence": confidence,
            "disaster_type": incident["disaster_type"],
            "location": incident["location"],
            "severity": incident["severity"],
            "people_affected": people,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat(),
        }
    finally:
        db.close()

# =========================================================
# AI INCIDENT INTELLIGENCE ENGINE
# =========================================================

@app.get("/api/incidents/{incident_id}/intelligence")
def get_incident_intelligence(incident_id: int, user: dict = Depends(get_current_user)):
    db = SessionLocal()

    try:
        x = db.query(IncidentDB).filter(IncidentDB.id == incident_id).first()

        if x is None:
            raise HTTPException(status_code=404, detail="Incident not found")

        incident = incident_dict(x)
        disaster_type = str(incident["disaster_type"] or "").lower()
        severity = str(incident["severity"] or "").lower()
        people = int(incident.get("people_affected", 0))
        ai_score = int(incident.get("ai_score", 0))
        score_source = incident.get("ai_score_source", "rules_baseline")
        ml_result = None
        if score_source == "ml":
            try:
                ml_result = predict_risk(
                    disaster_type=incident["disaster_type"],
                    severity=incident["severity"],
                    people_affected=people,
                    model_path=ML_MODEL_PATH,
                )
            except Exception:
                logger.exception("Unable to load ML metadata for incident intelligence")
        description = str(incident.get("description", "")).lower()

        severity_scores = {
            "low": 25,
            "medium": 50,
            "high": 75,
            "critical": 100,
        }

        disaster_scores = {
            "fire": 90,
            "flood": 85,
            "earthquake": 100,
            "cyclone": 90,
            "landslide": 95,
        }

        if people >= 300:
            population_score = 100
        elif people >= 200:
            population_score = 85
        elif people >= 100:
            population_score = 70
        elif people >= 50:
            population_score = 50
        elif people > 0:
            population_score = 30
        else:
            population_score = 15

        exposure_words = [
            "residential", "building", "house", "apartment",
            "school", "hospital", "collapse", "trapped"
        ]
        residential_risk = any(
            word in description for word in exposure_words
        )

        risk_factors = [
            {
                "name": "Incident Severity",
                "score": severity_scores.get(severity, 50),
                "reason": f"Reported severity is {incident['severity']}."
            },
            {
                "name": "Population Impact",
                "score": population_score,
                "reason": f"{people} people reported affected."
            },
            {
                "name": "Disaster Type",
                "score": disaster_scores.get(disaster_type, 60),
                "reason": f"{incident['disaster_type']} requires specialized emergency response."
            },
            {
                "name": "Infrastructure Exposure",
                "score": 90 if residential_risk else 55,
                "reason": (
                    "The report indicates exposure to occupied or vulnerable infrastructure."
                    if residential_risk
                    else
                    "No specific high-occupancy infrastructure exposure was identified."
                ),
            },
        ]

        priority = (
            "CRITICAL" if ai_score >= 90 else
            "HIGH" if ai_score >= 75 else
            "MODERATE" if ai_score >= 50 else
            "LOW"
        )

        confidence = None if score_source == "ml" else min(99, max(60, ai_score + 2))

        assessment = (
            f"{incident['disaster_type']} incident reported at "
            f"{incident['location']} with {people} people affected. "
            f"The system classifies this event as {priority} priority."
        )

        if disaster_type == "fire" and residential_risk:
            assessment += (
                " Residential exposure increases the urgency "
                "for evacuation and fire-response deployment."
            )
        elif disaster_type == "flood":
            assessment += (
                " Flooding increases the need for evacuation, "
                "water rescue and medical support."
            )
        elif disaster_type == "earthquake":
            assessment += (
                " Structural instability and search-and-rescue "
                "requirements significantly increase response risk."
            )
        elif disaster_type == "cyclone":
            assessment += (
                " Weather exposure may increase infrastructure "
                "damage and evacuation requirements."
            )

        recommended_actions = []

        if severity in {"critical", "high"}:
            recommended_actions.append(
                "Prioritize this incident in the emergency command queue."
            )

        if people >= 100:
            recommended_actions.append(
                "Activate large-population evacuation and shelter planning."
            )

        action_map = {
            "fire": [
                "Dispatch fire-response resources immediately.",
                "Establish a safety perimeter around the affected area.",
                "Prepare medical support for smoke inhalation and burn injuries.",
            ],
            "flood": [
                "Deploy water-rescue resources.",
                "Identify evacuation routes and temporary shelters.",
                "Prepare medical evacuation for trapped civilians.",
            ],
            "earthquake": [
                "Deploy search-and-rescue teams.",
                "Inspect structures for collapse hazards.",
                "Establish emergency medical triage.",
            ],
            "cyclone": [
                "Activate cyclone response teams.",
                "Secure evacuation shelters.",
                "Monitor vulnerable infrastructure.",
            ],
            "landslide": [
                "Deploy search-and-rescue teams.",
                "Restrict access to unstable slopes and damaged roads.",
                "Evacuate residents from areas at risk of further landslides.",
            ],
        }

        recommended_actions.extend(
            action_map.get(
                disaster_type,
                [
                    "Deploy appropriate emergency response resources.",
                    "Establish a safe response perimeter.",
                    "Assess medical and evacuation requirements.",
                ],
            )
        )

        resource_requirements = {
            "fire": ["Fire Truck", "Ambulance", "Rescue Team"],
            "flood": ["Rescue Boat", "Rescue Team", "Ambulance"],
            "earthquake": ["Rescue Team", "Ambulance", "Drone"],
            "cyclone": ["Rescue Team", "Drone", "Ambulance"],
            "landslide": ["Rescue Team", "Ambulance", "Drone"],
        }.get(
            disaster_type,
            ["Rescue Team", "Ambulance"],
        )

        return {
            "incident_id": incident_id,
            "risk_score": ai_score,
            "score_source": score_source,
            "priority": priority,
            "confidence": confidence,
            "prediction_interval": (ml_result or {}).get("prediction_interval") if ml_result else None,
            "model_metrics": (ml_result or {}).get("model_metrics", {}) if ml_result else {},
            "assessment": assessment,
            "risk_factors": risk_factors,
            "recommended_actions": recommended_actions,
            "resource_requirements": resource_requirements,
            "generated_at": datetime.utcnow().isoformat(),
        }

    finally:
        db.close()

# =========================================================
# RESOURCES
# =========================================================

@app.get("/api/resources", response_model=List[Resource])
def get_resources(user: dict = Depends(get_current_user)):
    db=SessionLocal()
    try: return [resource_dict(x) for x in db.query(ResourceDB).order_by(ResourceDB.id.asc()).all()]
    finally: db.close()


@app.get("/api/resources/{resource_id}", response_model=Resource)
def get_resource(resource_id: int, user: dict = Depends(get_current_user)):
    db=SessionLocal()
    try:
        x=db.query(ResourceDB).filter(ResourceDB.id==resource_id).first()
        if x is None: raise HTTPException(status_code=404, detail="Resource not found")
        return resource_dict(x)
    finally: db.close()


@app.post("/api/resources", response_model=Resource)
async def create_resource(resource: ResourceCreate, user: dict = Depends(require_roles("Admin"))):
    db=SessionLocal()
    try:
        x = ResourceDB(
            name=resource.name.strip(),
            resource_type=resource.resource_type.strip(),
            location=resource.location.strip(),
            status=resource.status.strip(),
            capacity=resource.capacity,
            created_at=datetime.utcnow().isoformat(),
        )
        db.add(x)
        db.commit()
        db.refresh(x)

        data = resource_dict(x)
        await manager.broadcast({
            "type": "resource_created",
            "resource": data,
            "timestamp": datetime.utcnow().isoformat(),
        })
        return data
    finally: db.close()


@app.post("/api/resources/{resource_id}/dispatch", response_model=Resource)
async def dispatch_resource(resource_id: int, incident_id: int, eta_minutes: int = 10, user: dict = Depends(require_roles("Admin", "Dispatcher"))):
    db=SessionLocal()
    try:
        resource=db.query(ResourceDB).filter(ResourceDB.id==resource_id).first()
        if resource is None: raise HTTPException(status_code=404, detail="Resource not found")
        incident=db.query(IncidentDB).filter(IncidentDB.id==incident_id).first()
        if incident is None: raise HTTPException(status_code=404, detail="Incident not found")
        if resource.status!="Available": raise HTTPException(status_code=409, detail="Resource is not available")
        prev=resource.status
        resource.status="En Route"; resource.assigned_incident_id=incident_id; resource.eta_minutes=max(1,int(eta_minutes))
        if incident.status=="Active": incident.status="Dispatched"
        db.add(DispatchDB(
            resource_id=resource.id,
            incident_id=incident_id,
            previous_status=prev,
            new_status="En Route",
            eta_minutes=resource.eta_minutes,
            created_at=datetime.utcnow().isoformat(),
        ))
        db.commit()
        db.refresh(resource)

        resource_data = resource_dict(resource)
        incident_data = incident_dict(incident)

        await manager.broadcast({
            "type": "resource_dispatched",
            "resource": resource_data,
            "incident": incident_data,
            "timestamp": datetime.utcnow().isoformat(),
        })

        await create_alert(
            db,
            incident.id,
            "RESOURCE_DISPATCHED",
            f"Resource Dispatched to Incident #{incident.id}",
            f"{resource.name} is en route to {incident.location}. ETA: {resource.eta_minutes} minutes.",
            incident.severity,
            incident.ai_score or 0,
        )
        db.commit()

        return resource_data
    finally: db.close()


@app.post("/api/resources/{resource_id}/release", response_model=Resource)
async def release_resource(resource_id: int, user: dict = Depends(require_roles("Admin", "Dispatcher", "Responder"))):
    db=SessionLocal()
    try:
        resource=db.query(ResourceDB).filter(ResourceDB.id==resource_id).first()
        if resource is None: raise HTTPException(status_code=404, detail="Resource not found")
        incident_id=resource.assigned_incident_id
        resource.status="Available"; resource.assigned_incident_id=None; resource.eta_minutes=None
        if incident_id is not None:
            incident=db.query(IncidentDB).filter(IncidentDB.id==incident_id).first()
            if incident and incident.status=="Dispatched": incident.status="In Progress"
        db.commit()
        db.refresh(resource)

        resource_data = resource_dict(resource)

        await manager.broadcast({
            "type": "resource_released",
            "resource": resource_data,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return resource_data
    finally: db.close()


@app.patch("/api/resources/{resource_id}/status", response_model=Resource)
async def update_resource_status(resource_id: int, status: str, user: dict = Depends(require_roles("Admin", "Dispatcher", "Responder"))):
    allowed=["Available","En Route","Deployed","Maintenance","Standby"]
    normalized=status.strip().title()
    if normalized not in allowed: raise HTTPException(status_code=400, detail={"message":"Invalid status","allowed_statuses":allowed})
    db=SessionLocal()
    try:
        resource=db.query(ResourceDB).filter(ResourceDB.id==resource_id).first()
        if resource is None: raise HTTPException(status_code=404, detail="Resource not found")
        resource.status=normalized
        if normalized=="Available": resource.assigned_incident_id=None; resource.eta_minutes=None
        db.commit()
        db.refresh(resource)

        resource_data = resource_dict(resource)

        await manager.broadcast({
            "type": "resource_status_updated",
            "resource": resource_data,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return resource_data
    finally: db.close()


# =========================================================
# AI RESOURCE RECOMMENDATION ENGINE
# =========================================================

def resource_recommendation_score(resource: dict, incident: dict):
    disaster_type = str(incident.get("disaster_type", "")).lower()
    severity = str(incident.get("severity", "")).lower()
    people = int(incident.get("people_affected", 0))
    resource_type = str(resource.get("resource_type", "")).lower()
    score = 0
    reasons = []

    if disaster_type == "fire":
        if "fire" in resource_type:
            score += 50
            reasons.append("Specialized for fire response")
        if "rescue" in resource_type:
            score += 25
            reasons.append("Useful for civilian rescue")
        if "ambulance" in resource_type:
            score += 20
            reasons.append("Useful for fire-related injuries")
    elif disaster_type == "flood":
        if "boat" in resource_type:
            score += 50
            reasons.append("Specialized for flood rescue")
        if "rescue" in resource_type:
            score += 30
            reasons.append("Useful for water rescue")
        if "ambulance" in resource_type:
            score += 15
            reasons.append("Useful for medical evacuation")
    elif disaster_type == "earthquake":
        if "rescue" in resource_type:
            score += 45
            reasons.append("Suitable for search and rescue")
        if "ambulance" in resource_type:
            score += 25
            reasons.append("Suitable for casualty transport")
        if "drone" in resource_type:
            score += 20
            reasons.append("Useful for aerial assessment")
    elif disaster_type == "cyclone":
        if "rescue" in resource_type:
            score += 40
            reasons.append("Suitable for emergency rescue")
        if "drone" in resource_type:
            score += 30
            reasons.append("Useful for aerial damage assessment")
        if "ambulance" in resource_type:
            score += 20
            reasons.append("Useful for medical response")
    else:
        if "rescue" in resource_type:
            score += 30
            reasons.append("General emergency response capability")

    if severity == "critical" and any(x in resource_type for x in ["ambulance", "fire", "rescue"]):
        score += 20
        reasons.append("Priority response for critical incident")
    elif severity == "high":
        score += 10

    if people >= 300 and any(x in resource_type for x in ["ambulance", "rescue"]):
        score += 15
        reasons.append("Large affected population")
    elif people >= 100 and any(x in resource_type for x in ["ambulance", "rescue"]):
        score += 8

    capacity = int(resource.get("capacity", 0))
    if people >= 100 and capacity >= 10:
        score += 10
        reasons.append("High response capacity")
    elif people >= 50 and capacity >= 5:
        score += 5

    if resource.get("location") == incident.get("location"):
        score += 25
        reasons.append("Located at incident sector")

    return min(score, 100), reasons


@app.get("/api/incidents/{incident_id}/resources")
def recommend_resources(incident_id: int, user: dict = Depends(get_current_user)):
    db=SessionLocal()
    try:
        incident_obj=db.query(IncidentDB).filter(IncidentDB.id==incident_id).first()
        if incident_obj is None: raise HTTPException(status_code=404, detail="Incident not found")
        incident=incident_dict(incident_obj)
        recommendations=[]
        for obj in db.query(ResourceDB).filter(ResourceDB.status=="Available").all():
            resource=resource_dict(obj)
            score,reasons=resource_recommendation_score(resource,incident)
            if score>0:
                recommendations.append({"resource_id":resource["id"],"name":resource["name"],"resource_type":resource["resource_type"],"location":resource["location"],"capacity":resource.get("capacity",0),"score":score,"reasons":reasons})
        recommendations.sort(key=lambda item:item["score"], reverse=True)
        return {"incident_id":incident_id,"incident_type":incident["disaster_type"],"severity":incident["severity"],"people_affected":incident.get("people_affected",0),"recommendations":recommendations[:5],"total_available":len(recommendations)}
    finally:
        db.close()


# =========================================================
# ALERTS
# =========================================================

@app.get("/api/alerts", response_model=List[Alert])
def get_alerts(acknowledged: Optional[bool] = None, user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        query = db.query(AlertDB).order_by(AlertDB.id.desc())
        if acknowledged is not None:
            query = query.filter(AlertDB.acknowledged == (1 if acknowledged else 0))
        return [alert_dict(x) for x in query.all()]
    finally:
        db.close()


@app.get("/api/alerts/{alert_id}", response_model=Alert)
def get_alert(alert_id: int, user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        x = db.query(AlertDB).filter(AlertDB.id == alert_id).first()
        if x is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        return alert_dict(x)
    finally:
        db.close()


@app.patch("/api/alerts/{alert_id}/acknowledge", response_model=Alert)
async def acknowledge_alert(alert_id: int, user: dict = Depends(require_roles("Admin", "Dispatcher"))):
    db = SessionLocal()
    try:
        x = db.query(AlertDB).filter(AlertDB.id == alert_id).first()
        if x is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        x.acknowledged = 1
        db.commit()
        db.refresh(x)
        data = alert_dict(x)
        await manager.broadcast({
            "type": "alert_acknowledged",
            "alert": data,
            "timestamp": datetime.utcnow().isoformat(),
        })
        return data
    finally:
        db.close()


# =========================================================
# RESOURCE TRACKING ASSIGNMENT
# =========================================================

@app.patch("/api/resources/{resource_id}/assign-user", response_model=Resource)
async def assign_resource_user(
    resource_id: int,
    user_id: Optional[int] = Query(default=None),
    user: dict = Depends(require_roles("Admin", "Dispatcher")),
):
    db = SessionLocal()
    try:
        resource = db.query(ResourceDB).filter(ResourceDB.id == resource_id).first()
        if resource is None:
            raise HTTPException(status_code=404, detail="Resource not found")

        target = None
        if user_id is not None:
            target = db.query(UserDB).filter(UserDB.id == user_id, UserDB.is_active == 1).first()
            if target is None:
                raise HTTPException(status_code=404, detail="Active user not found")
            if str(target.role).lower() != "responder":
                raise HTTPException(status_code=400, detail="Only Responder accounts can be assigned to field resources")

        previous_user_id = resource.assigned_user_id
        resource.assigned_user_id = user_id
        db.commit()
        db.refresh(resource)
        write_audit(
            db,
            user,
            "resource.assign_user" if user_id is not None else "resource.unassign_user",
            "resource",
            resource.id,
            {"previous_user_id": previous_user_id, "user_id": user_id},
        )
        db.commit()
        data = resource_dict(resource)
        await_result = {
            "type": "resource_assignment_updated",
            "resource": data,
            "timestamp": utc_now_iso(),
        }
        await manager.broadcast(await_result)
        return data
    finally:
        db.close()


# =========================================================
# LIVE GPS TRACKING
# =========================================================

class LocationUpdate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy: Optional[float] = Field(default=None, ge=0)
    speed_kmh: Optional[float] = Field(default=None, ge=0, le=500)
    heading: Optional[float] = Field(default=None, ge=0, le=360)
    resource_id: Optional[int] = None


@app.post("/api/tracking/location")
async def update_live_location(payload: LocationUpdate, user: dict = Depends(get_current_user)):
    validate_coordinates(payload.latitude, payload.longitude)
    db = SessionLocal()
    try:
        resource = None
        if payload.resource_id is not None:
            resource = db.query(ResourceDB).filter(ResourceDB.id == payload.resource_id).first()
            if resource is None:
                raise HTTPException(status_code=404, detail="Resource not found")
            role = str(user.get("role", "")).lower()
            if role not in {"admin", "dispatcher"} and resource.assigned_user_id != user["id"]:
                raise HTTPException(status_code=403, detail="You can only share location for your assigned resource")

        now = utc_now_iso()
        ping = LocationPingDB(
            user_id=user["id"],
            resource_id=payload.resource_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            accuracy=payload.accuracy,
            speed_kmh=payload.speed_kmh,
            heading=payload.heading,
            created_at=now,
        )
        db.add(ping)
        retention_cutoff = (datetime.utcnow() - timedelta(hours=LOCATION_RETENTION_HOURS)).isoformat()
        db.query(LocationPingDB).filter(LocationPingDB.created_at < retention_cutoff).delete(synchronize_session=False)

        if resource is not None:
            resource.latitude = payload.latitude
            resource.longitude = payload.longitude
            resource.speed_kmh = payload.speed_kmh
            resource.heading = payload.heading
            resource.location_accuracy = payload.accuracy
            resource.last_seen = now

        write_audit(db, user, "tracking.location", "resource" if resource else "user", resource.id if resource else user["id"], {"latitude": payload.latitude, "longitude": payload.longitude})
        db.commit()

        resource_data = resource_dict(resource) if resource is not None else None
        event = {
            "type": "tracking_location_updated",
            "location": {
                "user_id": user["id"],
                "username": user["username"],
                "full_name": user.get("full_name"),
                "role": user["role"],
                "resource_id": payload.resource_id,
                "resource_name": resource.name if resource is not None else None,
                "resource_type": resource.resource_type if resource is not None else None,
                "assigned_incident_id": resource.assigned_incident_id if resource is not None else None,
                "latitude": payload.latitude,
                "longitude": payload.longitude,
                "accuracy": payload.accuracy,
                "speed_kmh": payload.speed_kmh,
                "heading": payload.heading,
                "last_seen": now,
            },
            "resource": resource_data,
            "timestamp": now,
        }
        await manager.broadcast(event)
        return event["location"]
    finally:
        db.close()


@app.get("/api/tracking/live")
def get_live_locations(minutes: int = Query(default=10, ge=1, le=1440), user: dict = Depends(get_current_user)):
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    cutoff_text = cutoff.isoformat()
    db = SessionLocal()
    try:
        pings = db.query(LocationPingDB).filter(LocationPingDB.created_at >= cutoff_text).order_by(LocationPingDB.created_at.desc()).limit(2000).all()
        latest = {}
        for ping in pings:
            key = (ping.user_id, ping.resource_id)
            if key in latest:
                continue
            latest[key] = {
                "user_id": ping.user_id,
                "resource_id": ping.resource_id,
                "latitude": ping.latitude,
                "longitude": ping.longitude,
                "accuracy": ping.accuracy,
                "speed_kmh": ping.speed_kmh,
                "heading": ping.heading,
                "last_seen": ping.created_at,
            }
        users = {u.id: u for u in db.query(UserDB).filter(UserDB.is_active == 1).all()}
        resources = {r.id: r for r in db.query(ResourceDB).all()}
        for item in latest.values():
            u = users.get(item["user_id"])
            r = resources.get(item["resource_id"]) if item.get("resource_id") is not None else None
            item.update({
                "username": u.username if u else None,
                "full_name": u.full_name if u else None,
                "role": u.role if u else None,
                "resource_name": r.name if r else None,
                "resource_type": r.resource_type if r else None,
                "assigned_incident_id": r.assigned_incident_id if r else None,
            })
        return list(latest.values())
    finally:
        db.close()


@app.delete("/api/tracking/location")
def stop_live_location(user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        # No data is deleted from the audit trail; this simply records the user's stop action.
        write_audit(db, user, "tracking.stop", "user", user["id"], {})
        db.commit()
        return {"message": "Location sharing stopped"}
    finally:
        db.close()


# =========================================================
# EMERGENCY SOS
# =========================================================

class SOSCreate(BaseModel):
    emergency_type: str = Field(default="SOS", min_length=2, max_length=50)
    message: str = Field(default="Emergency SOS activated", min_length=1, max_length=2000)
    location: Optional[str] = Field(default=None, max_length=255)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    location_accuracy: Optional[float] = Field(default=None, ge=0)
    source: str = Field(default="operator", max_length=50)


@app.post("/api/emergency/sos")
async def activate_emergency_sos(payload: SOSCreate, user: dict = Depends(require_roles("Admin", "Dispatcher", "Responder"))):
    db = SessionLocal()
    try:
        if (payload.latitude is None) != (payload.longitude is None):
            raise HTTPException(status_code=422, detail="Latitude and longitude must be provided together")
        if payload.latitude is not None and payload.longitude is not None:
            validate_coordinates(payload.latitude, payload.longitude)

        emergency = EmergencyDB(
            emergency_type=payload.emergency_type.strip() or "SOS",
            status="Active",
            message=payload.message.strip() or "Emergency SOS activated",
            location=(payload.location.strip() if payload.location else None),
            latitude=payload.latitude,
            longitude=payload.longitude,
            location_accuracy=payload.location_accuracy,
            source=("device_gps" if payload.latitude is not None else payload.source.strip() or "operator"),
            created_at=datetime.utcnow().isoformat(),
            resolved_at=None,
        )
        db.add(emergency)
        db.flush()
        write_audit(db, user, "sos.activate", "emergency", emergency.id, {"has_location": payload.latitude is not None})
        db.commit()
        db.refresh(emergency)
        data = emergency_dict(emergency)

        await create_alert(
            db,
            None,
            "EMERGENCY_SOS",
            "EMERGENCY SOS ACTIVATED",
            data["message"],
            "Critical",
            100,
        )
        db.commit()

        await manager.broadcast({
            "type": "sos_activated",
            "emergency": data,
            "timestamp": datetime.utcnow().isoformat(),
        })
        return data
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to activate SOS. Please try again.",
        )
    finally:
        db.close()


@app.get("/api/emergency/sos")
def get_emergencies(user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        items = db.query(EmergencyDB).order_by(EmergencyDB.id.desc()).all()
        return [emergency_dict(item) for item in items]
    finally:
        db.close()


@app.patch("/api/emergency/sos/{emergency_id}/resolve")
async def resolve_emergency_sos(emergency_id: int, user: dict = Depends(require_roles("Admin", "Dispatcher"))):
    db = SessionLocal()
    try:
        emergency = db.query(EmergencyDB).filter(EmergencyDB.id == emergency_id).first()
        if emergency is None:
            raise HTTPException(status_code=404, detail="Emergency SOS not found")
        emergency.status = "Resolved"
        emergency.resolved_at = datetime.utcnow().isoformat()
        db.commit()
        db.refresh(emergency)
        data = emergency_dict(emergency)
        await manager.broadcast({
            "type": "sos_resolved",
            "emergency": data,
            "timestamp": datetime.utcnow().isoformat(),
        })
        return data
    finally:
        db.close()


# =========================================================
# AUDIT LOGS
# =========================================================

@app.get("/api/audit-logs")
def get_audit_logs(limit: int = Query(default=200, ge=1, le=1000), user: dict = Depends(require_roles("Admin"))):
    db = SessionLocal()
    try:
        logs = db.query(AuditLogDB).order_by(AuditLogDB.id.desc()).limit(limit).all()
        return [{
            "id": item.id,
            "user_id": item.user_id,
            "username": item.username,
            "action": item.action,
            "entity_type": item.entity_type,
            "entity_id": item.entity_id,
            "details": json.loads(item.details or "{}"),
            "ip_address": item.ip_address,
            "created_at": item.created_at,
        } for item in logs]
    finally:
        db.close()


@app.get("/api/system/readiness")
def system_readiness(user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        db.execute(sql_text("SELECT 1"))
        return {
            "status": "ready",
            "environment": APP_ENV,
            "database": "connected",
            "authentication": "active",
            "websocket": "active",
            "live_tracking": True,
            "data_fusion": True,
            "audit_logging": True,
            "live_location_retention_hours": LOCATION_RETENTION_HOURS,
            "ml_mode": "ml" if ml_model_available() else "rules_baseline",
            "timestamp": utc_now_iso(),
        }
    finally:
        db.close()


# =========================================================
# STATISTICS
# =========================================================

@app.get("/api/statistics")
def get_statistics(user: dict = Depends(get_current_user)):
    db=SessionLocal()
    try:
        incidents=db.query(IncidentDB).all(); resources=db.query(ResourceDB).all()
        total=len(incidents); active=sum(1 for i in incidents if str(i.status).lower() in {"active","dispatched","responding","in progress"})
        critical=sum(1 for i in incidents if i.severity=="Critical"); high=sum(1 for i in incidents if i.severity=="High")
        people=sum(int(i.people_affected or 0) for i in incidents)
        avg=round(sum(int(i.ai_score or 0) for i in incidents)/total,1) if total else 0
        alerts_total = db.query(AlertDB).count()
        unacknowledged_alerts = db.query(AlertDB).filter(AlertDB.acknowledged == 0).count()
        emergency_total = db.query(EmergencyDB).count()
        active_sos = db.query(EmergencyDB).filter(EmergencyDB.status == "Active").count()
        return {"total_incidents":total,"active_incidents":active,"critical_incidents":critical,"high_incidents":high,"people_affected":people,"average_ai_score":avg,"total_resources":len(resources),"available_resources":sum(1 for r in resources if r.status=="Available"),"en_route_resources":sum(1 for r in resources if r.status=="En Route"),"deployed_resources":sum(1 for r in resources if r.status=="Deployed"),"total_alerts":alerts_total,"unacknowledged_alerts":unacknowledged_alerts,"total_sos":emergency_total,"active_sos":active_sos}
    finally: db.close()


@app.get("/api/dispatches")
def get_dispatches(user: dict = Depends(get_current_user)):
    db=SessionLocal()
    try:
        return [{"id":d.id,"resource_id":d.resource_id,"incident_id":d.incident_id,"previous_status":d.previous_status,"new_status":d.new_status,"eta_minutes":d.eta_minutes,"created_at":d.created_at} for d in db.query(DispatchDB).order_by(DispatchDB.id.desc()).all()]
    finally: db.close()


@app.get("/api/stats")
def get_stats_alias(user: dict = Depends(get_current_user)):
    return get_statistics()