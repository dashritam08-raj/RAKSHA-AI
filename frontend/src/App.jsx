import React, { useEffect, useMemo, useState } from "react";
import {
  LayoutDashboard,
  Map as MapIcon,
  TriangleAlert,
  AlertTriangle,
  Ambulance,
  Bell,
  FileText,
  ShieldAlert,
  Menu,
  X,
  Flame,
  CloudRain,
  Users,
  Activity,
  Radio,
  ChevronRight,
  RefreshCw,
  Search,
  Filter,
  Eye,
  CheckCircle,
  Check,
  Phone,
  Siren,
  Clock,
  MapPin,
  Brain,
  ArrowLeft,
  Zap,
  Navigation,
  Truck,
  Package,
  Send,
  RotateCcw,
  Plane,
  Ship,
  UserRound,
  CircleDot,
  BarChart3,
  LogIn,
  LogOut,
  ShieldCheck,
  LockKeyhole,
  UserCog,
  LocateFixed,
  StopCircle,
  Play,
  CircleUserRound,
  Wifi,
  Shield,
  ListChecks,
  CloudSun,
  Droplets,
  Wind,
  Thermometer,
  Globe2,
  ExternalLink,
  Layers3,
  Gauge,
  DatabaseZap,
} from "lucide-react";

import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Circle,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const WS_URL = import.meta.env.VITE_WS_URL || API_URL.replace(/^http/, "ws") + "/ws";
const AUTH_TOKEN_KEY = "raksha_token";
const AUTH_USER_KEY = "raksha_user";
const DEFAULT_MONITORING_POINT = { latitude: 22.5726, longitude: 88.3639 };

function readStoredUser() {
  try {
    const value = localStorage.getItem(AUTH_USER_KEY);
    return value ? JSON.parse(value) : null;
  } catch {
    return null;
  }
}

async function apiFetch(url, options = {}) {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  const headers = new Headers(options.headers || {});

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(url, { ...options, headers });

  if (response.status === 401) {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
    window.dispatchEvent(new Event("raksha-auth-expired"));
  }

  return response;
}


function formatApiError(detail, fallback = "Request failed") {
  if (detail == null || detail === "") return fallback;

  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object") {
        const message = item.msg || item.message || item.detail;
        const location = Array.isArray(item.loc) ? item.loc.filter(Boolean).join(" → ") : "";
        if (message) return location ? `${location}: ${message}` : message;
        try {
          return JSON.stringify(item);
        } catch {
          return "Validation error";
        }
      }
      return String(item);
    }).filter(Boolean);

    return messages.length ? messages.join(" • ") : fallback;
  }

  if (typeof detail === "object") {
    const message = detail.message || detail.error || detail.detail;
    if (typeof message === "string") return message;

    try {
      return JSON.stringify(detail);
    } catch {
      return fallback;
    }
  }

  return String(detail);
}

function hasPermission(user, action) {
  const role = String(user?.role || "").toLowerCase();
  const permissions = {
    admin: [
      "updateIncidentStatus",
      "dispatchResource",
      "releaseResource",
      "acknowledgeAlert",
      "activateSOS",
      "resolveSOS",
      "manageUsers",
    ],
    dispatcher: [
      "updateIncidentStatus",
      "dispatchResource",
      "releaseResource",
      "acknowledgeAlert",
      "activateSOS",
      "resolveSOS",
    ],
    responder: [
      "updateIncidentStatus",
      "releaseResource",
      "activateSOS",
    ],
  };
  return (permissions[role] || []).includes(action);
}

function userInitials(user) {
  const name = String(user?.full_name || user?.username || "U").trim();
  return name
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

const SECTOR_COORDINATES = {
  "Sector 12": [22.5150, 88.3650],
  "Sector 14": [22.5200, 88.3700],
  "Sector 21": [22.5350, 88.3800],
};

const DEFAULT_CENTER = [22.5250, 88.3720];

function createLiveLocationIcon(role) {
  const colorClass = String(role || "responder").toLowerCase();
  return L.divIcon({
    className: "raksha-live-marker",
    html: `<div class="live-location-marker ${colorClass}"><div class="live-location-core"></div><div class="live-location-ring"></div></div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
}

function createIncidentIcon(severity, disasterType) {
  const severityClass = String(severity || "Medium").toLowerCase();
  const symbol = String(disasterType || "").toLowerCase() === "fire" ? "🔥" : "💧";
  return L.divIcon({
    className: "raksha-map-marker",
    html: `
      <div class="map-pin ${severityClass}">
        <div class="map-pin-icon">${symbol}</div>
        <div class="map-pin-pulse"></div>
      </div>
    `,
    iconSize: [46, 46],
    iconAnchor: [23, 23],
    popupAnchor: [0, -23],
  });
}

function getIncidentCoordinates(incident) {
  if (incident && incident.latitude != null && incident.longitude != null) {
    return [Number(incident.latitude), Number(incident.longitude)];
  }
  const location = String(incident?.location || "").trim();
  return SECTOR_COORDINATES[location] || null;
}

function MapController({ selectedIncident }) {
  const map = useMap();
  useEffect(() => {
    if (!selectedIncident) return;
    const position = getIncidentCoordinates(selectedIncident);
    if (position) map.flyTo(position, 15, { duration: 1.2 });
  }, [selectedIncident, map]);
  return null;
}


/* =========================================================
   RAKSHA AI — CINEMATIC STARTUP EXPERIENCE
   Self-contained so App.jsx upgrade does not require
   changing the existing App.css.
========================================================= */

function StartupSplash({ onFinished }) {
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    const exitTimer = window.setTimeout(() => setExiting(true), 3100);
    const finishTimer = window.setTimeout(() => onFinished(), 3800);

    return () => {
      window.clearTimeout(exitTimer);
      window.clearTimeout(finishTimer);
    };
  }, [onFinished]);

  const startupStyles = `
    .raksha-startup {
      --rs-bg: #030712;
      --rs-text: #e5f7ff;
      --rs-muted: #8aa3b5;
      --rs-blue: #35a7ff;
      --rs-cyan: #39e6d0;
      --rs-green: #57e389;
      position: fixed;
      inset: 0;
      z-index: 99999;
      overflow: hidden;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 50% 46%, rgba(53,167,255,.15), transparent 24%),
        radial-gradient(circle at 50% 54%, rgba(57,230,208,.08), transparent 38%),
        linear-gradient(180deg, #02050b 0%, var(--rs-bg) 100%);
      color: var(--rs-text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      opacity: 1;
      transform: scale(1);
      transition: opacity .7s ease, transform .7s cubic-bezier(.22,1,.36,1);
    }

    .raksha-startup::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(rgba(76,137,168,.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(76,137,168,.07) 1px, transparent 1px);
      background-size: 54px 54px;
      mask-image: linear-gradient(to bottom, transparent 0%, black 25%, black 75%, transparent 100%);
      animation: rsGridDrift 8s linear infinite;
      pointer-events: none;
    }

    .raksha-startup::after {
      content: "";
      position: absolute;
      inset: -30%;
      background: conic-gradient(
        from 0deg,
        transparent 0deg,
        rgba(57,230,208,.045) 40deg,
        transparent 78deg,
        rgba(53,167,255,.04) 150deg,
        transparent 210deg,
        rgba(57,230,208,.035) 300deg,
        transparent 360deg
      );
      animation: rsAmbientSpin 16s linear infinite;
      pointer-events: none;
    }

    .raksha-startup.exiting {
      opacity: 0;
      transform: scale(1.035);
      pointer-events: none;
    }

    .rs-noise {
      position: absolute;
      inset: 0;
      opacity: .045;
      background-image:
        repeating-linear-gradient(0deg, rgba(255,255,255,.15) 0 1px, transparent 1px 3px);
      mix-blend-mode: screen;
      pointer-events: none;
    }

    .rs-vignette {
      position: absolute;
      inset: 0;
      background: radial-gradient(circle, transparent 35%, rgba(0,0,0,.58) 100%);
      pointer-events: none;
    }

    .rs-core {
      position: relative;
      z-index: 3;
      width: min(92vw, 760px);
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      padding: 28px;
    }

    .rs-orbit {
      position: absolute;
      left: 50%;
      top: 50%;
      width: min(52vw, 450px);
      height: min(52vw, 450px);
      min-width: 270px;
      min-height: 270px;
      transform: translate(-50%, -54%);
      border: 1px solid rgba(86, 208, 255, .15);
      border-radius: 50%;
      box-shadow:
        0 0 60px rgba(53,167,255,.08),
        inset 0 0 50px rgba(57,230,208,.035);
      animation: rsOrbitIn 1.1s cubic-bezier(.16,1,.3,1) forwards;
      opacity: 0;
    }

    .rs-orbit::before,
    .rs-orbit::after {
      content: "";
      position: absolute;
      inset: 7%;
      border-radius: 50%;
      border: 1px solid rgba(57,230,208,.08);
    }

    .rs-orbit::after {
      inset: 16%;
      border-style: dashed;
      border-color: rgba(53,167,255,.08);
      animation: rsOrbitRotate 18s linear infinite reverse;
    }

    .rs-orbit-dot {
      position: absolute;
      top: 50%;
      left: -5px;
      width: 10px;
      height: 10px;
      margin-top: -5px;
      border-radius: 50%;
      background: var(--rs-cyan);
      box-shadow: 0 0 18px rgba(57,230,208,.9);
      animation: rsOrbitRotate 6s linear infinite;
      transform-origin: calc((min(52vw, 450px) / 2) + 5px) 0;
    }

    .rs-logo-wrap {
      position: relative;
      width: min(38vw, 330px);
      min-width: 235px;
      aspect-ratio: 1;
      display: grid;
      place-items: center;
      animation: rsLogoReveal 1.05s cubic-bezier(.2,.9,.2,1) .15s forwards;
      opacity: 0;
      transform: translateY(20px) scale(.78);
    }

    .rs-logo-ring {
      position: absolute;
      inset: 2%;
      border-radius: 50%;
      border: 1px solid rgba(84,205,255,.24);
      box-shadow:
        0 0 22px rgba(53,167,255,.08),
        inset 0 0 28px rgba(57,230,208,.055);
      animation: rsPulse 2.6s ease-in-out infinite;
    }

    .rs-logo-ring::before,
    .rs-logo-ring::after {
      content: "";
      position: absolute;
      border-radius: 50%;
      inset: 8%;
      border: 1px solid rgba(57,230,208,.1);
    }

    .rs-logo-ring::after {
      inset: 17%;
      border-color: rgba(53,167,255,.11);
    }

    .rs-logo {
      position: relative;
      z-index: 2;
      width: 76%;
      height: 76%;
      object-fit: contain;
      filter:
        drop-shadow(0 0 13px rgba(53,167,255,.38))
        drop-shadow(0 0 25px rgba(57,230,208,.15));
      animation: rsLogoFloat 3s ease-in-out infinite;
    }

    .rs-logo-fallback {
      position: relative;
      z-index: 2;
      width: 44%;
      height: 44%;
      display: grid;
      place-items: center;
      border-radius: 28%;
      border: 2px solid rgba(84,205,255,.35);
      background: linear-gradient(145deg, rgba(53,167,255,.18), rgba(57,230,208,.07));
      color: #dffbff;
      box-shadow:
        0 0 28px rgba(53,167,255,.18),
        inset 0 0 22px rgba(57,230,208,.08);
    }

    .rs-scan {
      position: absolute;
      left: 10%;
      right: 10%;
      top: 24%;
      height: 2px;
      z-index: 4;
      background: linear-gradient(90deg, transparent, rgba(104,240,255,.9), transparent);
      box-shadow: 0 0 14px rgba(57,230,208,.75);
      animation: rsScan 2.15s ease-in-out infinite;
      opacity: .75;
      pointer-events: none;
    }

    .rs-title {
      margin-top: 10px;
      font-size: clamp(44px, 7vw, 76px);
      line-height: .9;
      font-weight: 900;
      letter-spacing: .22em;
      padding-left: .22em;
      color: #edfaff;
      text-shadow:
        0 0 20px rgba(53,167,255,.3),
        0 0 40px rgba(57,230,208,.12);
      animation: rsTitleReveal 1s ease .6s forwards;
      opacity: 0;
      transform: translateY(12px);
    }

    .rs-tagline {
      margin-top: 17px;
      color: #9bb0bd;
      font-size: clamp(10px, 1.3vw, 13px);
      font-weight: 800;
      letter-spacing: .34em;
      padding-left: .34em;
      text-transform: uppercase;
      animation: rsTitleReveal .9s ease .9s forwards;
      opacity: 0;
      transform: translateY(10px);
    }

    .rs-status-row {
      width: min(92vw, 600px);
      margin-top: 28px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      animation: rsTitleReveal .9s ease 1.1s forwards;
      opacity: 0;
      transform: translateY(10px);
    }

    .rs-status {
      min-height: 55px;
      padding: 11px 10px;
      border: 1px solid rgba(104,174,206,.13);
      border-radius: 12px;
      background: rgba(5,13,22,.54);
      backdrop-filter: blur(12px);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      color: #a9c2ce;
      font-size: 10px;
      font-weight: 900;
      letter-spacing: .09em;
      text-transform: uppercase;
    }

    .rs-status span {
      width: 7px;
      height: 7px;
      flex: 0 0 7px;
      border-radius: 50%;
      background: var(--rs-green);
      box-shadow: 0 0 10px rgba(87,227,137,.85);
      animation: rsBlink 1.35s ease-in-out infinite;
    }

    .rs-status:nth-child(2) span { animation-delay: .22s; }
    .rs-status:nth-child(3) span { animation-delay: .44s; }

    .rs-loader {
      width: min(82vw, 480px);
      margin-top: 27px;
      animation: rsTitleReveal .9s ease 1.25s forwards;
      opacity: 0;
      transform: translateY(8px);
    }

    .rs-loader-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: #6d8998;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: .17em;
      text-transform: uppercase;
    }

    .rs-loader-track {
      margin-top: 9px;
      width: 100%;
      height: 4px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(131,169,184,.11);
      box-shadow: inset 0 0 0 1px rgba(145,195,215,.06);
    }

    .rs-loader-bar {
      width: 0;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--rs-blue), var(--rs-cyan), var(--rs-green));
      box-shadow: 0 0 16px rgba(57,230,208,.42);
      animation: rsLoad 3.05s cubic-bezier(.16,.78,.19,1) .25s forwards;
    }

    .rs-footer {
      margin-top: 18px;
      color: #58717f;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: .14em;
      text-transform: uppercase;
      animation: rsTitleReveal .9s ease 1.4s forwards;
      opacity: 0;
    }

    .rs-corner {
      position: absolute;
      width: 68px;
      height: 68px;
      border-color: rgba(93,190,222,.16);
      z-index: 3;
      pointer-events: none;
    }

    .rs-corner.tl { left: 22px; top: 22px; border-left: 1px solid; border-top: 1px solid; }
    .rs-corner.tr { right: 22px; top: 22px; border-right: 1px solid; border-top: 1px solid; }
    .rs-corner.bl { left: 22px; bottom: 22px; border-left: 1px solid; border-bottom: 1px solid; }
    .rs-corner.br { right: 22px; bottom: 22px; border-right: 1px solid; border-bottom: 1px solid; }

    @keyframes rsGridDrift {
      from { transform: translate3d(0,0,0); }
      to { transform: translate3d(54px,54px,0); }
    }

    @keyframes rsAmbientSpin {
      to { transform: rotate(360deg); }
    }

    @keyframes rsOrbitIn {
      from { opacity: 0; transform: translate(-50%, -54%) scale(.78) rotate(-16deg); }
      to { opacity: 1; transform: translate(-50%, -54%) scale(1) rotate(0deg); }
    }

    @keyframes rsOrbitRotate {
      to { transform: rotate(360deg); }
    }

    @keyframes rsLogoReveal {
      0% { opacity: 0; transform: translateY(20px) scale(.76); filter: blur(9px); }
      72% { opacity: 1; transform: translateY(-3px) scale(1.015); filter: blur(0); }
      100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
    }

    @keyframes rsLogoFloat {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-5px); }
    }

    @keyframes rsPulse {
      0%, 100% { transform: scale(1); opacity: .72; }
      50% { transform: scale(1.035); opacity: 1; }
    }

    @keyframes rsScan {
      0% { transform: translateY(-20px); opacity: 0; }
      15% { opacity: .75; }
      50% { opacity: 1; }
      85% { opacity: .6; }
      100% { transform: translateY(160px); opacity: 0; }
    }

    @keyframes rsTitleReveal {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes rsLoad {
      0% { width: 0; }
      35% { width: 43%; }
      72% { width: 78%; }
      100% { width: 100%; }
    }

    @keyframes rsBlink {
      0%, 100% { opacity: .3; transform: scale(.8); }
      50% { opacity: 1; transform: scale(1); }
    }

    @media (max-width: 620px) {
      .rs-core { padding: 18px; }
      .rs-logo-wrap { width: min(72vw, 300px); min-width: 220px; }
      .rs-status-row { grid-template-columns: 1fr; width: min(78vw, 360px); }
      .rs-status { min-height: 43px; }
      .rs-tagline { line-height: 1.6; letter-spacing: .18em; padding-left: .18em; }
      .rs-corner { width: 44px; height: 44px; }
      .rs-corner.tl { left: 14px; top: 14px; }
      .rs-corner.tr { right: 14px; top: 14px; }
      .rs-corner.bl { left: 14px; bottom: 14px; }
      .rs-corner.br { right: 14px; bottom: 14px; }
    }

    @media (prefers-reduced-motion: reduce) {
      .raksha-startup *,
      .raksha-startup::before,
      .raksha-startup::after {
        animation-duration: 1ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 1ms !important;
      }
    }
  `;

  return (
    <div className={`raksha-startup ${exiting ? "exiting" : ""}`} aria-label="RAKSHA AI startup">
      <style>{startupStyles}</style>

      <div className="rs-noise" />
      <div className="rs-vignette" />
      <div className="rs-corner tl" />
      <div className="rs-corner tr" />
      <div className="rs-corner bl" />
      <div className="rs-corner br" />

      <div className="rs-orbit" aria-hidden="true">
        <div className="rs-orbit-dot" />
      </div>

      <div className="rs-core">
        <div className="rs-logo-wrap">
          <div className="rs-logo-ring" />
          <div className="rs-scan" />
          <img
            className="rs-logo"
            src="/raksha-symbol-square.png"
            alt="RAKSHA AI"
            onError={(event) => {
              event.currentTarget.style.display = "none";
              event.currentTarget.nextElementSibling?.style.setProperty("display", "grid");
            }}
          />
          <div className="rs-logo-fallback" style={{ display: "none" }}>
            <Shield size={74} strokeWidth={1.25} />
          </div>
        </div>

        <div className="rs-title">RAKSHA</div>
        <div className="rs-tagline">Disaster Intelligence &amp; Response</div>

        <div className="rs-status-row">
          <div className="rs-status"><span />AI ENGINE</div>
          <div className="rs-status"><span />LIVE NETWORK</div>
          <div className="rs-status"><span />COMMAND CENTER</div>
        </div>

        <div className="rs-loader">
          <div className="rs-loader-top">
            <span>Initializing emergency intelligence</span>
            <span>ONLINE</span>
          </div>
          <div className="rs-loader-track">
            <div className="rs-loader-bar" />
          </div>
        </div>

        <div className="rs-footer">SECURE OPERATIONS PLATFORM · REAL-TIME RESPONSE SYSTEM</div>
      </div>
    </div>
  );
}


function App() {
  const [user, setUser] = useState(readStoredUser);
  const [authReady, setAuthReady] = useState(false);
  const [startupComplete, setStartupComplete] = useState(false);
  const [selectedRole, setSelectedRole] = useState(null);
  const [authMode, setAuthMode] = useState("login");

  useEffect(() => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const storedUser = readStoredUser();

    if (!token) {
      setUser(null);
      setAuthReady(true);
      return;
    }

    fetch(`${API_URL}/api/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(async (response) => {
        if (response.ok) {
          const currentUser = await response.json();
          localStorage.setItem(AUTH_USER_KEY, JSON.stringify(currentUser));
          setUser(currentUser);
          return;
        }

        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(AUTH_USER_KEY);
        setUser(null);
      })
      .catch(() => {
        // Keep the stored session during a temporary backend outage.
        setUser(storedUser);
      })
      .finally(() => setAuthReady(true));
  }, []);

  const handleLogin = (session) => {
    localStorage.setItem(AUTH_TOKEN_KEY, session.access_token);
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(session.user));
    setUser(session.user);
  };

  const handleLogout = async () => {
    try {
      await apiFetch(`${API_URL}/api/auth/logout`, { method: "POST" });
    } catch (error) {
      console.warn("Backend logout failed; clearing local session.", error);
    } finally {
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_USER_KEY);
      setUser(null);
    }
  };

  if (!startupComplete) {
    return <StartupSplash onFinished={() => setStartupComplete(true)} />;
  }

  if (!authReady) {
    return (
      <div className="auth-loading">
        <div className="auth-loading-card">
          <ShieldAlert size={32} />
          <strong>RAKSHA AI</strong>
          <span>Secure command center loading...</span>
        </div>
      </div>
    );
  }

  if (!user) {
    if (!selectedRole) {
      return (
        <RoleSelectionPage
          onSelectRole={(role) => {
            setSelectedRole(role);
            setAuthMode("login");
          }}
        />
      );
    }

    if (authMode === "register" && selectedRole !== "Admin") {
      return (
        <RegisterPage
          role={selectedRole}
          onBack={() => setAuthMode("login")}
        />
      );
    }

    return (
      <LoginPage
        role={selectedRole}
        onBack={() => {
          setSelectedRole(null);
          setAuthMode("login");
        }}
        onLogin={handleLogin}
        onRegister={() => setAuthMode("register")}
      />
    );
  }

  return <MainApp user={user} onLogout={handleLogout} />;
}

const ROLE_META = {
  Admin: {
    icon: ShieldCheck,
    short: "ADMINISTRATOR",
    title: "Admin Command",
    description: "Full command-center control, user management, audit visibility and system oversight.",
    access: "Full system access",
    accent: "cyan",
  },
  Dispatcher: {
    icon: Radio,
    short: "DISPATCH OPERATIONS",
    title: "Dispatcher",
    description: "Coordinate incidents, acknowledge alerts, dispatch resources and manage response operations.",
    access: "Dispatch & coordination access",
    accent: "blue",
  },
  Responder: {
    icon: Navigation,
    short: "FIELD RESPONSE",
    title: "Responder",
    description: "Field-focused access for live tracking, incident updates, resource status and emergency response.",
    access: "Field response access",
    accent: "green",
  },
};

const DEMO_ACCOUNTS = {
  Admin: { username: "admin", password: "Admin@123" },
  Dispatcher: { username: "dispatcher", password: "Dispatcher@123" },
  Responder: { username: "responder", password: "Responder@123" },
};

function RoleSelectionPage({ onSelectRole }) {
  return (
    <div className="auth-shell role-select-shell">
      <div className="auth-atmosphere auth-atmosphere-one" />
      <div className="auth-atmosphere auth-atmosphere-two" />
      <div className="auth-grid-overlay" />
      <div className="auth-scanline" />

      <div className="role-select-container">
        <div className="role-select-brand">
          <div className="role-select-logo-frame">
            <img
              src="/raksha-symbol-square.png"
              alt="RAKSHA AI logo"
              className="role-select-logo"
            />
          </div>
          <div className="role-select-brand-copy">
            <strong>RAKSHA AI</strong>
            <span>DISASTER INTELLIGENCE &amp; RESPONSE</span>
          </div>
          <div className="role-select-status">
            <span className="status-pulse-dot" />
            SECURE ACCESS PORTAL
          </div>
        </div>

        <div className="role-select-heading">
          <div className="role-select-eyebrow">
            <ShieldCheck size={16} />
            IDENTITY &amp; ROLE VERIFICATION
          </div>
          <h1>Choose your command role</h1>
          <p>Select the operational role that matches your account before signing in.</p>
        </div>

        <div className="role-card-grid">
          {Object.entries(ROLE_META).map(([role, meta], index) => {
            const Icon = meta.icon;
            return (
              <button
                key={role}
                type="button"
                className={`role-card role-${meta.accent}`}
                style={{ "--role-delay": `${index * 90}ms` }}
                onClick={() => onSelectRole(role)}
              >
                <div className="role-card-topline">
                  <span>{meta.short}</span>
                  <span className="role-card-index">0{index + 1}</span>
                </div>

                <div className="role-card-icon-wrap">
                  <div className="role-card-icon-ring" />
                  <Icon size={30} strokeWidth={1.8} />
                </div>

                <div className="role-card-body">
                  <h2>{meta.title}</h2>
                  <p>{meta.description}</p>
                </div>

                <div className="role-card-footer">
                  <span>{meta.access}</span>
                  <ChevronRight size={18} />
                </div>

                <div className="role-card-shine" />
              </button>
            );
          })}
        </div>

        <div className="role-select-footer">
          <Shield size={15} />
          <span>Access is enforced by the RAKSHA backend after authentication.</span>
        </div>
      </div>
    </div>
  );
}

function LoginPage({ onLogin, role, onBack, onRegister }) {
  const meta = ROLE_META[role] || ROLE_META.Responder;
  const demo = DEMO_ACCOUNTS[role] || DEMO_ACCOUNTS.Responder;
  const RoleIcon = meta.icon;

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const submitLogin = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(formatApiError(data.detail, "Invalid username or password"));
      }

      const actualRole = String(data?.user?.role || "").trim().toLowerCase();
      const expectedRole = String(role).trim().toLowerCase();

      if (actualRole !== expectedRole) {
        throw new Error(
          `This account is registered as ${data?.user?.role || "another role"}. Please select the correct role and try again.`
        );
      }

      onLogin(data);
    } catch (loginError) {
      setError(loginError.message || "Unable to sign in");
    } finally {
      setLoading(false);
    }
  };

  const useDemo = () => {
    setUsername(demo.username);
    setPassword(demo.password);
    setError("");
  };

  return (
    <div className="auth-shell role-login-shell">
      <div className="auth-atmosphere auth-atmosphere-one" />
      <div className="auth-atmosphere auth-atmosphere-two" />
      <div className="auth-grid-overlay" />
      <div className="auth-scanline" />

      <div className="login-layout">
        <button className="auth-back-button" type="button" onClick={onBack}>
          <ArrowLeft size={17} />
          Change role
        </button>

        <div className="login-card login-card-ultimate">
          <div className="login-card-glow" />
          <div className="login-brand">
            <div className="login-brand-icon">
              <img
                className="login-brand-logo"
                src="/raksha-symbol-square.png"
                alt="RAKSHA AI logo"
              />
            </div>
            <div className="login-brand-copy">
              <strong>RAKSHA AI</strong>
              <span>Disaster Intelligence &amp; Response</span>
            </div>
            <div className={`login-role-chip login-role-${meta.accent}`}>
              <RoleIcon size={14} />
              {role}
            </div>
          </div>

          <div className="login-heading">
            <div className="eyebrow">
              <ShieldCheck size={15} />
              SECURE COMMAND ACCESS
            </div>
            <h1>Welcome back</h1>
            <p>Sign in to access the {role.toLowerCase()} command workspace.</p>
          </div>

          <form className="login-form" onSubmit={submitLogin}>
            <label htmlFor="raksha-username">Username</label>
            <div className="login-input-wrap">
              <UserCog size={18} />
              <input
                id="raksha-username"
                type="text"
                value={username}
                onChange={(event) => {
                  setUsername(event.target.value);
                  setError("");
                }}
                placeholder={`Enter ${role.toLowerCase()} username`}
                autoComplete="username"
                autoFocus
                required
              />
            </div>

            <label htmlFor="raksha-password">Password</label>
            <div className="login-input-wrap">
              <LockKeyhole size={18} />
              <input
                id="raksha-password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(event) => {
                  setPassword(event.target.value);
                  setError("");
                }}
                placeholder="Enter your password"
                autoComplete="current-password"
                required
              />
              <button
                className="login-password-toggle"
                type="button"
                onClick={() => setShowPassword((value) => !value)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                title={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <Eye size={17} /> : <Eye size={17} />}
              </button>
            </div>

            {error && (
              <div className="login-error">
                <TriangleAlert size={17} />
                <span>{error}</span>
              </div>
            )}

            <button className="login-button" type="submit" disabled={loading}>
              {loading ? (
                <>
                  <RefreshCw size={18} className="spin" />
                  Authenticating...
                </>
              ) : (
                <>
                  <LogIn size={18} />
                  Sign In Securely
                </>
              )}
            </button>
          </form>

          <div className="login-security-strip">
            <div>
              <ShieldCheck size={15} />
              <span>Encrypted session</span>
            </div>
            <div>
              <DatabaseZap size={15} />
              <span>Backend verified</span>
            </div>
            <div>
              <Wifi size={15} />
              <span>Live network</span>
            </div>
          </div>

          {role !== "Admin" && (
            <div className="login-register-cta">
              <div>
                <strong>New operational account?</strong>
                <span>Request {role} access from this portal.</span>
              </div>
              <button type="button" className="login-register-button" onClick={onRegister}>
                <UserRound size={16} />
                Create Account
              </button>
            </div>
          )}

          <div className="demo-access demo-access-role">
            <div className="demo-access-heading">
              <span>LOCAL DEMO ACCESS</span>
              <small>{role} test account for local development</small>
            </div>
            <button className="demo-role-button" type="button" onClick={useDemo}>
              <span className="demo-role-icon"><RoleIcon size={17} /></span>
              <span className="demo-role-copy">
                <strong>Use {role} demo account</strong>
                <small>{demo.username}</small>
              </span>
              <ChevronRight size={17} />
            </button>
          </div>
        </div>

        <div className="login-page-footer">
          <span>RAKSHA AI</span>
          <span>•</span>
          <span>SECURE OPERATIONS PLATFORM</span>
          <span>•</span>
          <span>REAL-TIME RESPONSE SYSTEM</span>
        </div>
      </div>
    </div>
  );
}


function RegisterPage({ role, onBack }) {
  const meta = ROLE_META[role] || ROLE_META.Responder;
  const RoleIcon = meta.icon;

  const [form, setForm] = useState({
    full_name: "",
    username: "",
    password: "",
    confirmPassword: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [createdUsername, setCreatedUsername] = useState("");

  const password = form.password;
  const checks = [
    [password.length >= 8, "At least 8 characters"],
    [/[a-z]/.test(password), "One lowercase letter"],
    [/[A-Z]/.test(password), "One uppercase letter"],
    [/[0-9]/.test(password), "One number"],
  ];

  const submitRegister = async (event) => {
    event.preventDefault();
    setError("");

    const fullName = form.full_name.trim();
    const username = form.username.trim().toLowerCase();

    if (!fullName) {
      setError("Please enter your full name.");
      return;
    }
    if (!/^[a-z0-9._-]{3,80}$/.test(username)) {
      setError("Username must be 3–80 characters and use only letters, numbers, dots, underscores or hyphens.");
      return;
    }
    if (!checks.every(([ok]) => ok)) {
      setError("Please meet all password requirements.");
      return;
    }
    if (password !== form.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          full_name: fullName,
          username,
          password,
          role,
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(formatApiError(data.detail, "Unable to submit registration"));
      }

      setCreatedUsername(data?.user?.username || username);
      setSubmitted(true);
      setForm({ full_name: "", username: "", password: "", confirmPassword: "" });
    } catch (registerError) {
      setError(registerError.message || "Unable to submit registration");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-shell role-login-shell register-shell">
      <div className="auth-atmosphere auth-atmosphere-one" />
      <div className="auth-atmosphere auth-atmosphere-two" />
      <div className="auth-grid-overlay" />
      <div className="auth-scanline" />

      <div className="login-layout">
        <button className="auth-back-button" type="button" onClick={onBack}>
          <ArrowLeft size={17} />
          Back to {role} login
        </button>

        <div className="login-card login-card-ultimate register-card">
          <div className="login-card-glow" />

          {!submitted ? (
            <>
              <div className="login-brand">
                <div className="login-brand-icon">
                  <img
                    className="login-brand-logo"
                    src="/raksha-symbol-square.png"
                    alt="RAKSHA AI logo"
                  />
                </div>
                <div className="login-brand-copy">
                  <strong>RAKSHA AI</strong>
                  <span>Disaster Intelligence &amp; Response</span>
                </div>
                <div className={`login-role-chip login-role-${meta.accent}`}>
                  <RoleIcon size={14} />
                  {role}
                </div>
              </div>

              <div className="login-heading register-heading">
                <div className="eyebrow">
                  <UserCog size={15} />
                  OPERATIONAL ACCESS REQUEST
                </div>
                <h1>Create your account</h1>
                <p>
                  Register as a {role.toLowerCase()}. Your request will be reviewed by a RAKSHA Admin before the account can be used.
                </p>
              </div>

              <form className="login-form register-form" onSubmit={submitRegister}>
                <label htmlFor="raksha-register-name">Full Name</label>
                <div className="login-input-wrap">
                  <UserRound size={18} />
                  <input
                    id="raksha-register-name"
                    type="text"
                    value={form.full_name}
                    onChange={(event) => {
                      setForm((prev) => ({ ...prev, full_name: event.target.value }));
                      setError("");
                    }}
                    placeholder="Enter your full name"
                    autoComplete="name"
                    autoFocus
                    required
                  />
                </div>

                <label htmlFor="raksha-register-username">Username</label>
                <div className="login-input-wrap">
                  <UserCog size={18} />
                  <input
                    id="raksha-register-username"
                    type="text"
                    value={form.username}
                    onChange={(event) => {
                      setForm((prev) => ({ ...prev, username: event.target.value }));
                      setError("");
                    }}
                    placeholder="e.g. rahul.responder"
                    autoComplete="username"
                    required
                  />
                </div>

                <label htmlFor="raksha-register-password">Password</label>
                <div className="login-input-wrap">
                  <LockKeyhole size={18} />
                  <input
                    id="raksha-register-password"
                    type="password"
                    value={form.password}
                    onChange={(event) => {
                      setForm((prev) => ({ ...prev, password: event.target.value }));
                      setError("");
                    }}
                    placeholder="Create a strong password"
                    autoComplete="new-password"
                    required
                  />
                </div>

                <div className="register-password-checks">
                  {checks.map(([ok, label]) => (
                    <span key={label} className={ok ? "valid" : ""}>
                      <CheckCircle size={13} />
                      {label}
                    </span>
                  ))}
                </div>

                <label htmlFor="raksha-register-confirm">Confirm Password</label>
                <div className="login-input-wrap">
                  <ShieldCheck size={18} />
                  <input
                    id="raksha-register-confirm"
                    type="password"
                    value={form.confirmPassword}
                    onChange={(event) => {
                      setForm((prev) => ({ ...prev, confirmPassword: event.target.value }));
                      setError("");
                    }}
                    placeholder="Re-enter your password"
                    autoComplete="new-password"
                    required
                  />
                </div>

                {error && (
                  <div className="login-error">
                    <TriangleAlert size={17} />
                    <span>{error}</span>
                  </div>
                )}

                <button className="login-button" type="submit" disabled={loading}>
                  {loading ? (
                    <>
                      <RefreshCw size={18} className="spin" />
                      Submitting Request...
                    </>
                  ) : (
                    <>
                      <Send size={18} />
                      Submit Access Request
                    </>
                  )}
                </button>
              </form>

              <div className="register-trust-note">
                <ShieldCheck size={16} />
                <div>
                  <strong>Admin approval required</strong>
                  <span>Registration never grants operational access automatically.</span>
                </div>
              </div>
            </>
          ) : (
            <div className="register-success">
              <div className="register-success-icon">
                <CheckCircle size={44} />
              </div>
              <div className="eyebrow"><ShieldCheck size={15} />REQUEST RECEIVED</div>
              <h1>Registration submitted</h1>
              <p>
                The account <strong>@{createdUsername}</strong> has been created as a pending {role.toLowerCase()} account.
              </p>
              <div className="register-success-card">
                <div><span>STATUS</span><strong>Pending Admin Approval</strong></div>
                <div><span>ROLE</span><strong>{role}</strong></div>
                <div><span>ACCESS</span><strong>Locked until approved</strong></div>
              </div>
              <button className="login-button" type="button" onClick={onBack}>
                <LogIn size={18} />
                Return to Login
              </button>
            </div>
          )}
        </div>

        <div className="login-page-footer">
          <span>RAKSHA AI</span>
          <span>•</span>
          <span>SECURE OPERATIONS PLATFORM</span>
          <span>•</span>
          <span>ADMIN-APPROVED ACCESS</span>
        </div>
      </div>
    </div>
  );
}

function wsUrlWithToken(token) {
  return `${WS_URL}?token=${encodeURIComponent(token || "")}`;
}

function updateLiveLocationList(previous, location) {
  const key = `${location.user_id}:${location.resource_id || "user"}`;
  const next = previous.filter((item) => `${item.user_id}:${item.resource_id || "user"}` !== key);
  return [...next, location].sort((a, b) => String(b.last_seen || "").localeCompare(String(a.last_seen || "")));
}

function MainApp({ user, onLogout }) {
  const [activePage, setActivePage] = useState("Dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [incidents, setIncidents] = useState([]);
  const [resources, setResources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [resourcesLoading, setResourcesLoading] = useState(true);
  const [apiError, setApiError] = useState("");
  const [resourceError, setResourceError] = useState("");
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [liveLocations, setLiveLocations] = useState([]);
  const [dashboardEnvironment, setDashboardEnvironment] = useState(null);
  const [dashboardMlStatus, setDashboardMlStatus] = useState(null);
  const [activeAlertCount, setActiveAlertCount] = useState(0);


  useEffect(() => {
    if (typeof Notification !== "undefined" && Notification.permission === "default") Notification.requestPermission().catch(() => {});
    const handleAuthExpired = () => onLogout();
    window.addEventListener("raksha-auth-expired", handleAuthExpired);
    return () => window.removeEventListener("raksha-auth-expired", handleAuthExpired);
  }, [onLogout]);

  const fetchIncidents = async () => {
    setLoading(true);
    setApiError("");
    try {
      const response = await apiFetch(`${API_URL}/api/incidents`);
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      const data = await response.json();
      setIncidents(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error("RAKSHA API error:", error);
      setApiError(error.message || "Unable to connect to RAKSHA backend");
    } finally {
      setLoading(false);
    }
  };

  const fetchResources = async () => {
    setResourcesLoading(true);
    setResourceError("");
    try {
      const response = await apiFetch(`${API_URL}/api/resources`);
      if (!response.ok) throw new Error(`Resource API returned ${response.status}`);
      const data = await response.json();
      setResources(Array.isArray(data) ? data : []);
      return Array.isArray(data) ? data : [];
    } catch (error) {
      console.error("Resource API error:", error);
      setResourceError(error.message || "Unable to load resources");
      return [];
    } finally {
      setResourcesLoading(false);
    }
  };

  const fetchLiveLocations = async () => {
    try {
      const response = await apiFetch(`${API_URL}/api/tracking/live?minutes=15`);
      if (!response.ok) return;
      const data = await response.json();
      setLiveLocations(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error("Live tracking API error:", error);
    }
  };

  const fetchDashboardSupport = async () => {
    try {
      const [environmentResponse, mlResponse, alertsResponse] = await Promise.all([
        apiFetch(
          `${API_URL}/api/environment/overview?latitude=${DEFAULT_MONITORING_POINT.latitude}&longitude=${DEFAULT_MONITORING_POINT.longitude}`
        ),
        apiFetch(`${API_URL}/api/ml/status`),
        apiFetch(`${API_URL}/api/alerts?acknowledged=false`),
      ]);

      if (environmentResponse.ok) {
        setDashboardEnvironment(await environmentResponse.json());
      }

      if (mlResponse.ok) {
        setDashboardMlStatus(await mlResponse.json());
      }

      if (alertsResponse.ok) {
        const alerts = await alertsResponse.json();
        setActiveAlertCount(Array.isArray(alerts) ? alerts.length : 0);
      }
    } catch (error) {
      console.error("Dashboard support data error:", error);
    }
  };

  const refreshCommandCenter = async () => {
    await Promise.all([
      fetchIncidents(),
      fetchResources(),
      fetchLiveLocations(),
      fetchDashboardSupport(),
    ]);
  };

  useEffect(() => {
    refreshCommandCenter();
  }, []);
  
  // =======================================================
  // REAL-TIME WEBSOCKET WITH RECONNECT
  // =======================================================
  useEffect(() => {
    let stopped = false;
    let reconnectTimer = null;
    let socket = null;
    let retry = 0;

    const connect = () => {
      if (stopped) return;
      const token = localStorage.getItem(AUTH_TOKEN_KEY);
      if (!token) return;

      socket = new WebSocket(wsUrlWithToken(token));
      socket.onopen = () => {
        retry = 0;
        console.log("RAKSHA real-time connection established");
      };
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === "incident_created" && message.incident) {
            setIncidents((previous) => previous.some((i) => i.id === message.incident.id) ? previous : [...previous, message.incident]);
          } else if (message.type === "incident_updated" && message.incident) {
            setIncidents((previous) => previous.map((i) => i.id === message.incident.id ? message.incident : i));
          } else if (message.type === "incident_deleted") {
            const deletedId = message.incident_id ?? message.incident?.id;
            if (deletedId != null) {
              setIncidents((previous) => previous.filter((i) => i.id !== deletedId));
            }
          } else if (["resource_dispatched", "resource_released", "resource_created", "resource_updated", "resource_status_updated"].includes(message.type)) {
            fetchResources();
          } else if (message.type === "resource_assignment_updated" && message.resource) {
            fetchResources();
          } else if (message.type === "tracking_location_updated" && message.location) {
            setLiveLocations((previous) => updateLiveLocationList(previous, message.location));
          } else if (message.type === "alert_created" || message.type === "alert_acknowledged") {
            setActiveAlertCount((previous) =>
              message.type === "alert_created"
                ? previous + 1
                : Math.max(0, previous - 1)
            );
            window.dispatchEvent(new CustomEvent("raksha-alert-update", { detail: message }));
            if (message.type === "alert_created" && typeof Notification !== "undefined" && Notification.permission === "granted") {
              try { new Notification(`RAKSHA: ${message.alert?.title || "New alert"}`, { body: message.alert?.message || "New emergency alert" }); } catch (_) {}
            }
          }
        } catch (error) {
          console.error("RAKSHA realtime message error:", error);
        }
      };
      socket.onerror = () => {
        try { socket.close(); } catch (_) {}
      };
      socket.onclose = () => {
        if (stopped) return;
        const delay = Math.min(30000, 1000 * Math.pow(2, retry++));
        reconnectTimer = setTimeout(connect, delay);
      };
    };

    connect();
    const heartbeat = setInterval(() => {
      try { if (socket && socket.readyState === WebSocket.OPEN) socket.send("ping"); } catch (_) {}
    }, 20000);

    return () => {
      stopped = true;
      clearInterval(heartbeat);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      try { if (socket) socket.close(); } catch (_) {}
    };
  }, []);

  const navigateTo = (page) => {
    setActivePage(page);
    setSelectedIncident(null);
  };

  const menuItems = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "Risk Map", icon: MapIcon },
    { name: "Environmental Intel", icon: CloudSun },
    { name: "Risk Intelligence", icon: Layers3 },
    { name: "Live Tracking", icon: Radio },
    { name: "Incidents", icon: TriangleAlert },
    { name: "Resources", icon: Ambulance },
    { name: "Alerts", icon: Bell },
    { name: "Reports", icon: FileText },
    { name: "Emergency SOS", icon: ShieldAlert },
    ...(String(user?.role || "").toLowerCase() === "admin"
      ? [{ name: "User Management", icon: Users }, { name: "Audit Logs", icon: ListChecks }]
      : []),
  ];

  const activeIncidents = incidents.filter(
    (incident) => !incident.status || String(incident.status).toLowerCase() === "active"
  );
  const criticalIncidents = incidents.filter(
    (incident) => String(incident.severity || "").toLowerCase() === "critical"
  );
  const peopleAffected = incidents.reduce(
    (total, incident) => total + Number(incident.people_affected || 0),
    0
  );
  const averageAIScore = incidents.length
    ? Math.round(
        incidents.reduce((total, incident) => total + Number(incident.ai_score || 0), 0) /
          incidents.length
      )
    : 0;
  const availableResources = resources.filter((r) => r.status === "Available").length;
  const enRouteResources = resources.filter((r) => r.status === "En Route").length;
  const deployedResources = resources.filter((r) => r.status === "Deployed").length;

  return (
    <div className="app">
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="logo-section">
          <div className="logo-icon">
            <img
              src="/raksha-symbol-square.png"
              alt="RAKSHA AI"
              className="sidebar-logo"
            />
          </div>
          {sidebarOpen && (
            <div>
              <h2>RAKSHA</h2>
              <span>AI Disaster Response</span>
            </div>
          )}
        </div>
        <div className="nav-title">{sidebarOpen && "MAIN MENU"}</div>
        <nav>
          {menuItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.name}
                className={`nav-item ${activePage === item.name ? "active" : ""}`}
                onClick={() => navigateTo(item.name)}
              >
                <Icon size={21} />
                {sidebarOpen && <span>{item.name}</span>}
                {sidebarOpen && activePage === item.name && (
                  <ChevronRight size={17} className="nav-arrow" />
                )}
              </button>
            );
          })}
        </nav>
        {sidebarOpen && (
          <div className="sidebar-bottom">
            <div className="system-status">
              <span className="status-dot"></span>
              <div>
                <strong>System Operational</strong>
                <small>All services online</small>
              </div>
            </div>
            <div className="version">RAKSHA AI v5 · {user.role}</div>
          </div>
        )}
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="menu-button" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
          <div className="page-heading">
            <h1>{activePage}</h1>
            <p>AI-powered disaster management and emergency response</p>
          </div>
          <div className="top-actions">
            <div className="live-status"><span></span>LIVE</div>
            <button className="notification-button" onClick={() => navigateTo("Alerts")}>
              <Bell size={21} /><b>{activeAlertCount}</b>
            </button>
            <div className="profile">
              <div className="profile-avatar">{userInitials(user)}</div>
              <div><strong>{user.full_name || user.username}</strong><small>{user.role} · Emergency Control</small></div>
              <button className="logout-button" title="Sign out" onClick={onLogout}><LogOut size={18} /></button>
            </div>
          </div>
        </header>

        <section className="content">
          {activePage === "Dashboard" && (
            <Dashboard
              incidents={incidents}
              resources={resources}
              loading={loading}
              apiError={apiError}
              activeIncidents={activeIncidents}
              criticalIncidents={criticalIncidents}
              peopleAffected={peopleAffected}
              averageAIScore={averageAIScore}
              availableResources={availableResources}
              enRouteResources={enRouteResources}
              deployedResources={deployedResources}
              fetchIncidents={fetchIncidents}
              setActivePage={setActivePage}
              setSelectedIncident={setSelectedIncident}
              user={user}
              dashboardEnvironment={dashboardEnvironment}
              dashboardMlStatus={dashboardMlStatus}
              activeAlertCount={activeAlertCount}
            />
          )}

          {activePage === "Environmental Intel" && <EnvironmentalIntelPage user={user} />}
          {activePage === "Risk Intelligence" && <RiskIntelligencePage user={user} />}

          {activePage === "Risk Map" && (
            <RiskMapPage
              incidents={incidents}
              loading={loading}
              apiError={apiError}
              fetchIncidents={fetchIncidents}
              selectedIncident={selectedIncident}
              setSelectedIncident={setSelectedIncident}
              liveLocations={liveLocations}
            />
          )}

          {activePage === "Live Tracking" && <LiveTrackingPage user={user} liveLocations={liveLocations} setLiveLocations={setLiveLocations} resources={resources} />}
          {activePage === "Incidents" && (
            <IncidentCommandCenter
              incidents={incidents}
              loading={loading}
              apiError={apiError}
              fetchIncidents={fetchIncidents}
              selectedIncident={selectedIncident}
              setSelectedIncident={setSelectedIncident}
              refreshCommandCenter={refreshCommandCenter}
              user={user}
            />
          )}

          {activePage === "Resources" && (
            <ResourcesPage
              incidents={incidents}
              resources={resources}
              setResources={setResources}
              loading={resourcesLoading}
              error={resourceError}
              fetchResources={fetchResources}
              refreshCommandCenter={refreshCommandCenter}
              user={user}
            />
          )}

          {activePage === "Alerts" && (
            <AlertsPage incidents={incidents} setActivePage={setActivePage} user={user} />
          )}

          {activePage === "Reports" && <ReportsPage incidents={incidents} resources={resources} />}

          {activePage === "Emergency SOS" && (
            <EmergencySOSPage
              setActivePage={setActivePage}
              fetchIncidents={fetchIncidents}
              user={user}
            />
          )}

          {activePage === "User Management" && String(user?.role || "").toLowerCase() === "admin" && (
            <UserManagementPage user={user} />
          )}

          {activePage === "Audit Logs" && String(user?.role || "").toLowerCase() === "admin" && (
            <AuditLogsPage />
          )}
        </section>
      </main>
    </div>
  );
}


function MiniDashboardMap({ incidents, setActivePage, setSelectedIncident }) {
  const mappedIncidents = (incidents || []).filter(getIncidentCoordinates);
  const incidentCounts = mappedIncidents.reduce((acc, incident) => {
    const key = String(incident.severity || "Medium").toLowerCase();
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="dashboard-map-shell">
      <MapContainer center={DEFAULT_CENTER} zoom={13} scrollWheelZoom={false} className="dashboard-mini-map">
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {mappedIncidents.map((incident) => {
          const position = getIncidentCoordinates(incident);
          const score = Number(incident.ai_score || 0);
          return (
            <React.Fragment key={incident.id}>
              <Circle
                center={position}
                radius={score >= 90 ? 500 : score >= 75 ? 350 : 220}
                pathOptions={{ fillOpacity: 0.09, weight: 1 }}
              />
              <Marker
                position={position}
                icon={createIncidentIcon(incident.severity, incident.disaster_type)}
                eventHandlers={{
                  click: () => {
                    setSelectedIncident(incident);
                    setActivePage("Incidents");
                  },
                }}
              >
                <Popup>
                  <div className="map-popup">
                    <div className="popup-title">
                      <strong>{incident.disaster_type}</strong>
                      <SeverityBadge severity={incident.severity} />
                    </div>
                    <div className="popup-location"><MapPin size={14} />{incident.location}</div>
                    <div className="popup-stats">
                      <span><Users size={14} />{incident.people_affected || 0}</span>
                      <AIScore score={incident.ai_score} />
                    </div>
                    <button className="popup-view-button" onClick={() => {
                      setSelectedIncident(incident);
                      setActivePage("Incidents");
                    }}>
                      <Eye size={14} />View Incident
                    </button>
                  </div>
                </Popup>
              </Marker>
            </React.Fragment>
          );
        })}
      </MapContainer>
      <div className="dashboard-map-overlay dashboard-map-overlay-top">
        <span className="map-live-indicator"><span />LIVE OPERATIONS</span>
        <button onClick={() => setActivePage("Risk Map")}>Open full map <ChevronRight size={14} /></button>
      </div>
      <div className="dashboard-map-overlay dashboard-map-overlay-bottom">
        <div><strong>{mappedIncidents.length}</strong><span>mapped</span></div>
        <div><strong>{incidentCounts.critical || 0}</strong><span>critical</span></div>
        <div><strong>{incidentCounts.high || 0}</strong><span>high</span></div>
      </div>
      {mappedIncidents.length === 0 && (
        <div className="dashboard-map-empty">
          <MapIcon size={22} />
          <strong>No mapped incidents</strong>
          <span>Incident coordinates will appear here when available.</span>
        </div>
      )}
    </div>
  );
}

function Dashboard({
  incidents,
  resources,
  loading,
  apiError,
  activeIncidents,
  criticalIncidents,
  peopleAffected,
  averageAIScore,
  availableResources,
  enRouteResources,
  deployedResources,
  fetchIncidents,
  setActivePage,
  setSelectedIncident,
  user,
  dashboardEnvironment,
  dashboardMlStatus,
  activeAlertCount,
}) {
  const currentWeather = dashboardEnvironment?.weather?.current || {};
  const next24Weather = dashboardEnvironment?.weather?.next_24h || {};
  const todayLabel = new Date().toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).toUpperCase();
  const roleLabel = user?.role || "Operator";
  const displayName = user?.full_name || user?.username || "Operator";
  const temperature = currentWeather.temperature_2m;
  const humidity = currentWeather.relative_humidity_2m;
  const wind = currentWeather.wind_speed_10m;
  const rain = next24Weather.precipitation_mm;

  return (
    <>
      <div className="welcome">
        <div>
          <h2>Welcome, {displayName} 👋</h2>
          <p>Here's what's happening across your disaster response network.</p>
        </div>
        <div className="date-box">
          <strong>{todayLabel}</strong>
          <span>RAKSHA COMMAND CENTER</span>
        </div>
      </div>

      <div className="stats-grid">
        <StatCard title="Active Incidents" value={activeIncidents.length} subtitle={`${criticalIncidents.length} Critical`} icon={TriangleAlert} danger />
        <StatCard title="People Affected" value={peopleAffected} subtitle="Across reported incidents" icon={Users} />
        <StatCard title="Resources Ready" value={availableResources} subtitle={`${enRouteResources} en route · ${deployedResources} deployed`} icon={Ambulance} />
        <StatCard title="AI Risk Index" value={`${averageAIScore}/100`} subtitle="Current incident average" icon={Brain} warning />
      </div>

      <div className="dashboard-grid">
        <div className="card map-card">
          <div className="card-header">
            <div><h3>Live Disaster Map</h3><p>Real-time incident monitoring</p></div>
            <button className="outline-button" onClick={() => setActivePage("Risk Map")}><MapIcon size={17} />View Full Map</button>
          </div>
          <MiniDashboardMap
            incidents={incidents}
            setActivePage={setActivePage}
            setSelectedIncident={setSelectedIncident}
          />
        </div>

        <div className="card incidents-card">
          <div className="card-header">
            <div><h3>Recent Incidents</h3><p>AI-prioritized emergencies</p></div>
            <button className="text-button" onClick={() => setActivePage("Incidents")}>View All</button>
          </div>
          <div className="incident-list">
            {loading && <LoadingState />}
            {!loading && apiError && <ErrorState error={apiError} retry={fetchIncidents} />}
            {!loading && !apiError && incidents.length === 0 && <EmptyState />}
            {!loading && !apiError && incidents.slice(0, 6).map((incident, index) => (
              <IncidentRow
                key={incident.id || index}
                incident={incident}
                onClick={() => {
                  setSelectedIncident(incident);
                  setActivePage("Incidents");
                }}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="bottom-grid">
        <div className="card weather-card">
          <div className="small-card-title"><CloudRain size={20} />Live Weather</div>
          <div className="weather-main">
            <strong>{temperature != null ? `${Number(temperature).toFixed(1)}°C` : "—"}</strong>
            <span>{dashboardEnvironment?.weather ? `Weather code ${currentWeather.weather_code ?? "—"}` : "Unavailable"}</span>
          </div>
          <div className="weather-stats">
            <span>Humidity {humidity != null ? `${Math.round(Number(humidity))}%` : "—"}</span>
            <span>Wind {wind != null ? `${Number(wind).toFixed(1)} km/h` : "—"}</span>
          </div>
          <small className="dashboard-weather-note">
            Next 24h rain: {rain != null ? `${Number(rain).toFixed(1)} mm` : "—"} · {DEFAULT_MONITORING_POINT.latitude.toFixed(4)}, {DEFAULT_MONITORING_POINT.longitude.toFixed(4)}
          </small>
        </div>
        <div className="card response-card">
          <div className="small-card-title"><Activity size={20} />Response Readiness</div>
          <div className="progress-wrapper">
            <div className="progress-value">{resources.length ? Math.round((availableResources / resources.length) * 100) : 0}%</div>
            <div className="progress-bar"><div className="progress-fill" style={{ width: `${resources.length ? Math.round((availableResources / resources.length) * 100) : 0}%` }} /></div>
          </div>
          <p>{availableResources} resources ready for deployment.</p>
        </div>
        <div className="card ai-card">
          <div className="small-card-title"><Brain size={20} />AI Intelligence</div>
          <div className="ai-overview">
            <strong>{dashboardMlStatus?.model_available ? "ML ACTIVE" : "BASELINE"}</strong>
            <span>
              {dashboardMlStatus?.model_available
                ? `${dashboardMlStatus.model_version || "RAKSHA-ML"} decision-support model is online.`
                : "Rules-based decision support is active while the ML model is unavailable."}
            </span>
          </div>
        </div>
      </div>
    </>
  );
}



function RiskIntelligencePage({ user }) {
  const [coords, setCoords] = useState({ latitude: 22.5726, longitude: 88.3639 });
  const [fusion, setFusion] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");
  const [locationLabel, setLocationLabel] = useState("Kolkata (default)");
  const [mlStatus, setMlStatus] = useState(null);
  const [mlPrediction, setMlPrediction] = useState(null);
  const [mlLoading, setMlLoading] = useState(true);

  const loadMlIntelligence = async (lat = coords.latitude, lon = coords.longitude) => {
    setMlLoading(true);
    try {
      const [statusResponse, predictionResponse] = await Promise.all([
        apiFetch(`${API_URL}/api/ml/status`),
        apiFetch(`${API_URL}/api/ml/fused-risk?latitude=${lat}&longitude=${lon}`),
      ]);
      const statusData = await statusResponse.json().catch(() => null);
      const predictionData = await predictionResponse.json().catch(() => null);
      if (statusResponse.ok) setMlStatus(statusData);
      if (predictionResponse.ok) setMlPrediction(predictionData);
    } catch (err) {
      console.error("ML intelligence error:", err);
    } finally {
      setMlLoading(false);
    }
  };

  const loadFusion = async (lat = coords.latitude, lon = coords.longitude) => {
    setLoading(true);
    setError("");
    try {
      const response = await apiFetch(`${API_URL}/api/fusion/overview?latitude=${lat}&longitude=${lon}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(data.detail, "Unified risk data unavailable"));
      setFusion(data);
      setCoords({ latitude: lat, longitude: lon });
    } catch (err) {
      setError(err.message || "Unable to load unified risk intelligence");
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async (lat = coords.latitude, lon = coords.longitude) => {
    setHistoryLoading(true);
    try {
      const response = await apiFetch(`${API_URL}/api/fusion/history?latitude=${lat}&longitude=${lon}&hours=24`);
      const data = await response.json().catch(() => []);
      if (response.ok) setHistory(Array.isArray(data) ? data : []);
    } finally {
      setHistoryLoading(false);
    }
  };

  const refreshAll = async () => {
    await Promise.all([loadFusion(), loadHistory(), loadMlIntelligence()]);
  };

  const useMyLocation = () => {
    if (!navigator.geolocation) {
      setError("This browser does not support GPS location.");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        setLocationLabel("Current device location");
        await Promise.all([loadFusion(lat, lon), loadHistory(lat, lon), loadMlIntelligence(lat, lon)]);
      },
      () => setError("Location permission was not granted. Showing the default monitoring area."),
      { enableHighAccuracy: true, maximumAge: 120000, timeout: 10000 }
    );
  };

  useEffect(() => {
    refreshAll();
    const timer = setInterval(refreshAll, 60000);
    return () => clearInterval(timer);
  }, []);

  const signals = fusion?.signals || {};
  const sourceStatus = fusion?.source_status || {};
  const drivers = fusion?.drivers || [];
  const score = Number(fusion?.fused_risk_score || 0);
  const riskLevel = fusion?.risk_level || "—";
  const coverage = Number(fusion?.data_coverage_pct || 0);

  const signalItems = [
    ["Incident Pressure", signals.incident || 0, TriangleAlert],
    ["Weather Signal", signals.weather || 0, CloudSun],
    ["Seismic Signal", signals.seismic || 0, Activity],
    ["Official Alerts", signals.official_alerts || 0, Bell],
    ["Resource Pressure", signals.resource_pressure || 0, Ambulance],
  ];

  const statusLabel = (value) => {
    if (value === "connected") return "Connected";
    if (value === "configured") return "Configured";
    if (value === "available") return "Available";
    if (value === "not_configured") return "Not configured";
    return "Unavailable";
  };

  return (
    <div className="fusion-page">
      <div className="fusion-header">
        <div>
          <div className="eyebrow"><Layers3 size={15} />MULTI-SOURCE DATA FUSION</div>
          <h2>Risk Intelligence</h2>
          <p>Combines active incidents, environmental signals, official alerts and response capacity into one explainable operational index.</p>
        </div>
        <div className="fusion-actions">
          <button className="outline-button" onClick={useMyLocation}><LocateFixed size={17} />Use My Location</button>
          <button className="refresh-button" onClick={refreshAll} disabled={loading}><RefreshCw size={17} className={loading ? "spin" : ""} />Refresh Intelligence</button>
        </div>
      </div>

      <div className="fusion-location-bar">
        <div><MapPin size={17} /><span>{locationLabel}</span><small>{coords.latitude.toFixed(4)}, {coords.longitude.toFixed(4)}</small></div>
        <div className="fusion-live-badge"><span></span>FUSION ENGINE LIVE</div>
      </div>

      {error && <div className="environment-error"><TriangleAlert size={18} /><span>{error}</span></div>}

      <div className="fusion-hero-grid">
        <div className={`card fusion-score-card ${String(riskLevel).toLowerCase()}`}>
          <div className="fusion-score-top"><div><div className="eyebrow"><Gauge size={15} />UNIFIED SITUATIONAL RISK</div><h3>Operational Risk Index</h3></div><ShieldCheck size={24} /></div>
          {loading && !fusion ? <LoadingState /> : (
            <>
              <div className="fusion-score-value">{score}<small>/100</small></div>
              <div className="fusion-risk-row"><span className={`fusion-risk-badge ${String(riskLevel).toLowerCase()}`}>{riskLevel}</span><span>{coverage}% data coverage</span></div>
              <div className="fusion-score-bar"><span style={{ width: `${score}%` }} /></div>
              <p>This is a decision-support index derived from currently available signals; it is not a probability of disaster.</p>
            </>
          )}
        </div>

        <div className="card fusion-snapshot-card">
          <div className="small-card-title"><DatabaseZap size={20} />Source Snapshot</div>
          <div className="fusion-source-list">
            {Object.entries(sourceStatus).map(([name, status]) => (
              <div className="fusion-source-row" key={name}>
                <span>{name.replaceAll("_", " ")}</span>
                <b className={`fusion-source-status ${String(status).toLowerCase()}`}>{statusLabel(status)}</b>
              </div>
            ))}
          </div>
          <div className="fusion-updated">Generated {fusion?.generated_at ? new Date(fusion.generated_at).toLocaleString() : "—"}</div>
        </div>
      </div>

      <div className="card fusion-ml-card">
        <div className="environment-card-heading">
          <div>
            <div className="eyebrow"><Brain size={14} />PREDICTIVE ML LAYER</div>
            <h3>Production ML Readiness</h3>
          </div>
          <span className={`fusion-ml-status ${mlStatus?.mode || "rules_baseline"}`}>
            {mlStatus?.mode === "ml" ? "ML ACTIVE" : "RULES BASELINE"}
          </span>
        </div>
        {mlLoading ? <LoadingState /> : (
          <>
            <div className="fusion-ml-grid">
              <div><span>Model</span><strong>{mlStatus?.model_family || "—"}</strong></div>
              <div><span>Version</span><strong>{mlStatus?.model_version || "—"}</strong></div>
              <div><span>ML Impact</span><strong>{mlPrediction?.prediction?.impact_score != null ? Number(mlPrediction.prediction.impact_score).toFixed(2) : "—"}/100</strong></div>
              <div><span>Feature Quality</span><strong>{mlPrediction?.prediction?.feature_completeness_pct ?? "—"}%</strong></div>
            </div>
            <div className="fusion-ml-band-row">
              <span>Risk band</span>
              <strong>{mlPrediction?.prediction?.risk_band || mlPrediction?.prediction?.risk_level || "—"}</strong>
            </div>
          </>
        )}
        <p className="fusion-ml-note">{mlStatus?.message || "Model metadata unavailable."}</p>
        {mlPrediction?.prediction?.warnings?.length > 0 && (
          <div className="fusion-ml-warning">
            <TriangleAlert size={15} />
            <span>{mlPrediction.prediction.warnings[0]}</span>
          </div>
        )}
      </div>

      <div className="fusion-signal-grid">
        {signalItems.map(([label, value, Icon]) => (
          <div className="card fusion-signal-card" key={label}>
            <div className="fusion-signal-icon"><Icon size={19} /></div>
            <div><span>{label}</span><strong>{value}</strong></div>
            <div className="fusion-mini-bar"><span style={{ width: `${Math.min(100, Number(value) || 0)}%` }} /></div>
          </div>
        ))}
      </div>

      <div className="fusion-detail-grid">
        <div className="card fusion-drivers-card">
          <div className="environment-card-heading"><div><div className="eyebrow"><Zap size={14} />EXPLANATION</div><h3>Why the index changed</h3></div><span className="fusion-count">{drivers.length} drivers</span></div>
          {drivers.length ? (
            <div className="fusion-drivers-list">
              {drivers.map((driver, index) => <div key={`${driver}-${index}`}><span>{index + 1}</span><p>{driver}</p></div>)}
            </div>
          ) : <EmptyState />}
        </div>

        <div className="card fusion-history-card">
          <div className="environment-card-heading"><div><div className="eyebrow"><Activity size={14} />TREND</div><h3>Last 24 hours</h3></div><span className="fusion-count">{historyLoading ? "Updating" : `${history.length} snapshots`}</span></div>
          {history.length ? (
            <div className="fusion-history-list">
              {history.slice(-8).map((item, index) => (
                <div className="fusion-history-row" key={`${item.created_at}-${index}`}>
                  <div><strong>{item.fused_risk_score}</strong><span>{item.risk_level}</span></div>
                  <div className="fusion-history-line"><span style={{ width: `${item.fused_risk_score}%` }} /></div>
                  <small>{new Date(item.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small>
                </div>
              ))}
            </div>
          ) : <div className="fusion-empty-history">History will populate as the fusion engine samples the selected location.</div>}
        </div>
      </div>

      <div className="environment-disclaimer">
        <ShieldCheck size={20} />
        <div><strong>Operational safety note</strong><span>RAKSHA combines current source data for situational awareness. Official agency warnings remain authoritative, and critical actions should be verified by authorized operators.</span></div>
      </div>
    </div>
  );
}


function EnvironmentalIntelPage({ user }) {
  const [coords, setCoords] = useState({ latitude: 22.5726, longitude: 88.3639 });
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [locationLabel, setLocationLabel] = useState("Kolkata (default)");

  const loadOverview = async (lat = coords.latitude, lon = coords.longitude) => {
    setLoading(true);
    setError("");
    try {
      const response = await apiFetch(`${API_URL}/api/environment/overview?latitude=${lat}&longitude=${lon}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(data.detail, "Environmental data unavailable"));
      setOverview(data);
      setCoords({ latitude: lat, longitude: lon });
    } catch (err) {
      setError(err.message || "Unable to load environmental data");
    } finally {
      setLoading(false);
    }
  };

  const useMyLocation = () => {
    if (!navigator.geolocation) {
      setError("This browser does not support GPS location.");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocationLabel("Current device location");
        loadOverview(position.coords.latitude, position.coords.longitude);
      },
      () => setError("Location permission was not granted. Showing the default monitoring area."),
      { enableHighAccuracy: true, maximumAge: 120000, timeout: 10000 }
    );
  };

  useEffect(() => {
    loadOverview();
  }, []);

  const weather = overview?.weather;
  const current = weather?.current || {};
  const next24 = weather?.next_24h || {};
  const earthquakes = overview?.earthquakes?.events || [];
  const officialAlerts = overview?.sachet?.alerts || [];

  const metric = (value, unit = "") => value == null ? "—" : `${value}${unit}`;

  return (
    <div className="environment-page">
      <div className="environment-header">
        <div>
          <div className="eyebrow"><CloudSun size={15} />LIVE ENVIRONMENTAL INTELLIGENCE</div>
          <h2>Environmental Intelligence</h2>
          <p>Live weather, earthquake activity and official disaster-alert sources for the selected monitoring point.</p>
        </div>
        <div className="environment-actions">
          <button className="outline-button" onClick={useMyLocation}><LocateFixed size={17} />Use My Location</button>
          <button className="refresh-button" onClick={() => loadOverview()} disabled={loading}><RefreshCw size={17} className={loading ? "spin" : ""} />Refresh Data</button>
        </div>
      </div>

      <div className="environment-location-bar">
        <div><MapPin size={17} /><span>{locationLabel}</span><small>{coords.latitude.toFixed(4)}, {coords.longitude.toFixed(4)}</small></div>
        <span className="source-live-pill"><span></span>LIVE SOURCES</span>
      </div>

      {error && <div className="environment-error"><TriangleAlert size={18} /><span>{error}</span></div>}

      <div className="environment-source-grid">
        <div className="source-card"><div className="source-icon"><CloudSun size={21} /></div><div><strong>Open-Meteo</strong><span>Live weather model data</span></div><div className="source-status">{weather ? "Connected" : "Unavailable"}</div></div>
        <div className="source-card"><div className="source-icon"><Globe2 size={21} /></div><div><strong>USGS</strong><span>Recent earthquake feed</span></div><div className="source-status">{overview?.earthquakes ? "Connected" : "Unavailable"}</div></div>
        <div className="source-card"><div className="source-icon"><Bell size={21} /></div><div><strong>NDMA SACHET</strong><span>Official alert feed</span></div><div className="source-status">{overview?.sachet?.configured ? "Configured" : "Not configured"}</div></div>
      </div>

      {loading && !overview ? <LoadingState /> : (
        <>
          <div className="environment-weather-grid">
            <div className="card environment-main-weather">
              <div className="small-card-title"><CloudSun size={20} />Current Weather</div>
              <div className="weather-main-row">
                <div className="weather-temp">{metric(current.temperature_2m, "°C")}</div>
                <div className="weather-summary"><strong>Weather code {current.weather_code ?? "—"}</strong><span>Updated {current.time ? new Date(current.time).toLocaleString() : "—"}</span></div>
              </div>
              <div className="environment-metric-grid">
                <div><Thermometer size={17} /><span>Temperature</span><strong>{metric(current.temperature_2m, "°C")}</strong></div>
                <div><Droplets size={17} /><span>Humidity</span><strong>{metric(current.relative_humidity_2m, "%")}</strong></div>
                <div><CloudRain size={17} /><span>Precipitation</span><strong>{metric(current.precipitation, " mm")}</strong></div>
                <div><Wind size={17} /><span>Wind</span><strong>{metric(current.wind_speed_10m, " km/h")}</strong></div>
              </div>
            </div>

            <div className="card environment-main-weather">
              <div className="small-card-title"><Activity size={20} />Next 24 Hours</div>
              <div className="next24-highlight"><div><span>Forecast precipitation</span><strong>{metric(next24.precipitation_mm, " mm")}</strong></div><div><span>Max rain probability</span><strong>{metric(next24.max_precipitation_probability_pct, "%")}</strong></div></div>
              <div className="next24-highlight"><div><span>Max wind</span><strong>{metric(next24.max_wind_kmh, " km/h")}</strong></div><div><span>Max gust</span><strong>{metric(next24.max_wind_gusts_kmh, " km/h")}</strong></div></div>
              <p className="environment-note">These are environmental indicators, not official warnings. RAKSHA keeps official alerts visually separate from computed signals.</p>
            </div>
          </div>

          <div className="environment-data-grid">
            <div className="card environment-alert-card">
              <div className="environment-card-heading"><div className="small-card-title"><Bell size={20} />Official Disaster Alerts</div><a href={overview?.sachet?.source_url || "https://sachet.ndma.gov.in/"} target="_blank" rel="noreferrer"><ExternalLink size={15} />SACHET</a></div>
              {officialAlerts.length === 0 ? <EmptyState /> : <div className="environment-alert-list">{officialAlerts.slice(0, 8).map((alert, index) => <a className="environment-alert-item" href={alert.link || overview.sachet.source_url} target="_blank" rel="noreferrer" key={`${alert.link || alert.title}-${index}`}><strong>{alert.title}</strong><span>{alert.description}</span><small>{alert.published || "Official feed"}</small></a>)}</div>}
              {!overview?.sachet?.configured && <div className="environment-feed-note">The NDMA SACHET portal publishes geo-targeted, near-real-time alerts and exposes an India CAP RSS/CAP feed for agencies. Configure <code>SACHET_CAP_URL</code> on the backend when an authorized feed URL is available.</div>}
            </div>

            <div className="card environment-earthquake-card">
              <div className="environment-card-heading"><div className="small-card-title"><Globe2 size={20} />Earthquakes Within 1,000 km</div><a href="https://earthquake.usgs.gov/earthquakes/feed/" target="_blank" rel="noreferrer"><ExternalLink size={15} />USGS</a></div>
              {earthquakes.length === 0 ? <EmptyState /> : <div className="environment-earthquake-list">{earthquakes.slice(0, 10).map((event) => <div className="earthquake-item" key={event.id}><div className="magnitude">{event.magnitude ?? "—"}</div><div className="earthquake-body"><strong>{event.place || "Unknown location"}</strong><span>{event.distance_km} km away · depth {event.depth_km ?? "—"} km</span></div><small>{event.time ? new Date(event.time).toLocaleString() : "—"}</small></div>)}</div>}
            </div>
          </div>

          <div className="environment-disclaimer"><ShieldCheck size={17} /><div><strong>Source-aware monitoring</strong><span>Weather and earthquake data come from external providers. Official government alerts are displayed as official-source information; computed environmental indicators are not substitutes for official warnings.</span></div></div>
        </>
      )}
    </div>
  );
}

function LiveTrackingPage({ user, liveLocations, setLiveLocations, resources }) {
  const [sharing, setSharing] = useState(false);
  const [position, setPosition] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [speed, setSpeed] = useState(null);
  const [heading, setHeading] = useState(null);
  const [error, setError] = useState("");
  const [trackingResourceId, setTrackingResourceId] = useState("");
  const watchIdRef = React.useRef(null);
  const lastSentRef = React.useRef(0);

  const normalizedRole = String(user?.role || "").toLowerCase();
  const trackableResources = (resources || []).filter((resource) => {
    if (normalizedRole === "responder") return Number(resource.assigned_user_id) === Number(user?.id);
    return true;
  });

  const sendPosition = async (coords) => {
    const now = Date.now();
    if (now - lastSentRef.current < 3000) return;
    lastSentRef.current = now;
    try {
      const resourceId = trackingResourceId ? Number(trackingResourceId) : null;
      const response = await apiFetch(`${API_URL}/api/tracking/location`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: coords.latitude,
          longitude: coords.longitude,
          accuracy: coords.accuracy,
          speed_kmh: coords.speed == null ? null : Math.max(0, coords.speed * 3.6),
          heading: coords.heading,
          resource_id: resourceId,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(data.detail, "Location update failed"));
      setLiveLocations((previous) => updateLiveLocationList(previous, data));
    } catch (err) {
      setError(err.message || "Unable to send location");
    }
  };

  const startSharing = () => {
    setError("");
    if (!navigator.geolocation) {
      setError("This browser does not provide GPS location services.");
      return;
    }
    if (sharing) return;
    const watchId = navigator.geolocation.watchPosition(
      (geo) => {
        const coords = geo.coords;
        setPosition({ latitude: coords.latitude, longitude: coords.longitude });
        setAccuracy(coords.accuracy);
        setSpeed(coords.speed == null ? null : coords.speed * 3.6);
        setHeading(coords.heading);
        sendPosition(coords);
      },
      (geoError) => {
        setSharing(false);
        setError(geoError.message || "GPS permission was denied.");
      },
      { enableHighAccuracy: true, maximumAge: 3000, timeout: 15000 }
    );
    watchIdRef.current = watchId;
    setSharing(true);
  };

  const stopSharing = async () => {
    if (watchIdRef.current != null && navigator.geolocation) navigator.geolocation.clearWatch(watchIdRef.current);
    watchIdRef.current = null;
    lastSentRef.current = 0;
    setSharing(false);
    try { await apiFetch(`${API_URL}/api/tracking/location`, { method: "DELETE" }); } catch (_) {}
  };

  useEffect(() => () => {
    if (watchIdRef.current != null && navigator.geolocation) navigator.geolocation.clearWatch(watchIdRef.current);
  }, []);

  const activeTrackers = liveLocations.filter((item) => item.last_seen && Date.now() - new Date(item.last_seen).getTime() < 10 * 60 * 1000);
  const selectedResource = trackableResources.find((item) => Number(item.id) === Number(trackingResourceId));

  return (
    <div className="tracking-page">
      <div className="tracking-header">
        <div>
          <div className="eyebrow"><Radio size={15} />REAL-TIME FIELD TELEMETRY</div>
          <h2>Live Tracking</h2>
          <p>Share a responder or field-device GPS position and monitor active locations in real time.</p>
        </div>
        <div className="tracking-status-pill"><span className="status-dot"></span>{activeTrackers.length} active trackers</div>
      </div>

      {error && <div className="tracking-error"><AlertTriangle size={18} /><span>{error}</span></div>}

      <div className="tracking-grid">
        <div className="card tracking-control-card">
          <div className="small-card-title"><LocateFixed size={20} />My Field Location</div>
          <div className={`tracking-state ${sharing ? "on" : ""}`}>
            <div className="tracking-state-icon">{sharing ? <Wifi size={25} /> : <LocateFixed size={25} />}</div>
            <div><strong>{sharing ? "Location sharing active" : "Location sharing is off"}</strong><span>{sharing ? "The command center is receiving this device's GPS updates." : "Enable this on a responder phone or field device."}</span></div>
          </div>

          <div className="tracking-target-control">
            <label>Track this device as</label>
            <select value={trackingResourceId} onChange={(event) => setTrackingResourceId(event.target.value)} disabled={sharing}>
              <option value="">My user/device</option>
              {trackableResources.map((resource) => (
                <option key={resource.id} value={resource.id}>
                  {resource.name} · {resource.resource_type}{resource.assigned_user_id ? " · assigned" : ""}
                </option>
              ))}
            </select>
            {selectedResource && <small>{selectedResource.location} · Status: {selectedResource.status}</small>}
          </div>

          <div className="tracking-actions">
            {!sharing ? <button className="primary-button" onClick={startSharing}><Play size={17} />Start Location Sharing</button> : <button className="danger-button" onClick={stopSharing}><StopCircle size={17} />Stop Sharing</button>}
          </div>
          <div className="tracking-metrics">
            <div><span>Latitude</span><strong>{position ? position.latitude.toFixed(6) : "—"}</strong></div>
            <div><span>Longitude</span><strong>{position ? position.longitude.toFixed(6) : "—"}</strong></div>
            <div><span>Accuracy</span><strong>{accuracy != null ? `${Math.round(accuracy)} m` : "—"}</strong></div>
            <div><span>Speed</span><strong>{speed != null ? `${speed.toFixed(1)} km/h` : "—"}</strong></div>
          </div>
          <div className="tracking-note"><Shield size={15} />Location is shared only while you keep sharing enabled. GPS data is retained according to the server retention policy.</div>
        </div>

        <div className="card tracking-list-card">
          <div className="small-card-title"><ListChecks size={20} />Live Field Devices</div>
          {activeTrackers.length === 0 ? <EmptyState /> : <div className="tracking-list">{activeTrackers.map((item) => (
            <div className="tracking-row" key={`${item.user_id}-${item.resource_id || "user"}`}>
              <div className="tracking-avatar"><CircleUserRound size={19} /></div>
              <div className="tracking-row-main">
                <strong>{item.resource_name || item.full_name || item.username}</strong>
                <span>{item.resource_id ? `${item.resource_type || "Resource"} #${item.resource_id}` : `${item.role || "User"} device`}</span>
                {item.assigned_incident_id && <small>Incident #{item.assigned_incident_id}</small>}
              </div>
              <div className="tracking-row-right">
                <strong>{item.speed_kmh != null && Number(item.speed_kmh) > 0.5 ? `${Number(item.speed_kmh).toFixed(1)} km/h` : "Stationary"}</strong>
                <small>{new Date(item.last_seen).toLocaleTimeString()}</small>
              </div>
            </div>
          ))}</div>}
        </div>
      </div>

      <div className="card tracking-map-card">
        <div className="small-card-title"><MapIcon size={20} />Live Location Layer</div>
        <div className="tracking-mini-map">
          <MapContainer center={DEFAULT_CENTER} zoom={13} scrollWheelZoom className="real-map">
            <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" />
            {activeTrackers.map((item) => (
              <Marker key={`${item.user_id}-${item.resource_id || "user"}`} position={[Number(item.latitude), Number(item.longitude)]} icon={createLiveLocationIcon(item.role)}>
                <Popup>
                  <strong>{item.resource_name || item.full_name || item.username}</strong><br />
                  {item.resource_id ? `${item.resource_type || "Resource"} #${item.resource_id}` : `${item.role || "User"} device`}<br />
                  Accuracy: {item.accuracy != null ? `${Math.round(item.accuracy)} m` : "—"}<br />
                  Last update: {new Date(item.last_seen).toLocaleString()}
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
      </div>

      <div className="tracking-disclaimer">
        <ShieldCheck size={17} />
        <div><strong>Operational note</strong><span>GPS availability, accuracy and update frequency depend on the device, browser permissions and network connection. RAKSHA marks field locations as active only when a recent GPS update has been received.</span></div>
      </div>
    </div>
  );
}

function RiskMapPage({ incidents, loading, apiError, fetchIncidents, selectedIncident, setSelectedIncident, liveLocations = [] }) {
  const [severityFilter, setSeverityFilter] = useState("All");
  const [typeFilter, setTypeFilter] = useState("All");
  const [search, setSearch] = useState("");

  const filteredIncidents = useMemo(() => incidents.filter((incident) => {
    const query = search.toLowerCase().trim();
    const matchesSearch = !query || String(incident.location || "").toLowerCase().includes(query) || String(incident.disaster_type || "").toLowerCase().includes(query) || String(incident.description || "").toLowerCase().includes(query);
    const matchesSeverity = severityFilter === "All" || incident.severity === severityFilter;
    const matchesType = typeFilter === "All" || incident.disaster_type === typeFilter;
    return matchesSearch && matchesSeverity && matchesType;
  }), [incidents, search, severityFilter, typeFilter]);

  const mappedIncidents = filteredIncidents.filter(getIncidentCoordinates);
  const criticalCount = incidents.filter((i) => i.severity === "Critical").length;
  const highCount = incidents.filter((i) => i.severity === "High").length;
  const people = incidents.reduce((sum, i) => sum + Number(i.people_affected || 0), 0);

  return (
    <div className="risk-map-page">
      <div className="risk-map-header">
        <div>
          <div className="eyebrow"><Navigation size={15} />LIVE GEOSPATIAL INTELLIGENCE</div>
          <h2>Disaster Risk Map</h2>
          <p>Real-time visualization of incidents and AI risk zones.</p>
        </div>
        <button className="refresh-button" onClick={fetchIncidents}><RefreshCw size={17} />Refresh Map</button>
      </div>
      <div className="map-stat-bar">
        <div className="map-stat"><span>ACTIVE INCIDENTS</span><strong>{incidents.length}</strong></div>
        <div className="map-stat critical"><span>CRITICAL ZONES</span><strong>{criticalCount}</strong></div>
        <div className="map-stat high"><span>HIGH RISK</span><strong>{highCount}</strong></div>
        <div className="map-stat people"><span>PEOPLE AFFECTED</span><strong>{people}</strong></div>
      </div>
      <div className="map-toolbar">
        <div className="map-search"><Search size={18} /><input placeholder="Search sector or incident..." value={search} onChange={(e) => setSearch(e.target.value)} /></div>
        <div className="map-select"><Filter size={16} /><select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)}><option>All</option><option>Critical</option><option>High</option><option>Medium</option><option>Low</option></select></div>
        <div className="map-select"><select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}><option>All</option><option>Fire</option><option>Flood</option><option>Earthquake</option><option>Cyclone</option><option>Landslide</option></select></div>
      </div>
      <div className="map-container-wrapper">
        {apiError && <div className="map-error"><TriangleAlert size={18} />{apiError}<button onClick={fetchIncidents}>Retry</button></div>}
        <MapContainer center={DEFAULT_CENTER} zoom={14} scrollWheelZoom className="real-map">
          <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <MapController selectedIncident={selectedIncident} />
          {mappedIncidents.map((incident) => {
            const position = getIncidentCoordinates(incident);
            const score = Number(incident.ai_score || 0);
            return (
              <React.Fragment key={incident.id}>
                <Circle center={position} radius={score >= 90 ? 650 : score >= 75 ? 450 : 300} pathOptions={{ fillOpacity: 0.08, weight: 1 }} />
                <Marker position={position} icon={createIncidentIcon(incident.severity, incident.disaster_type)} eventHandlers={{ click: () => setSelectedIncident(incident) }}>
                  <Popup>
                    <div className="map-popup">
                      <div className="popup-title"><strong>{incident.disaster_type}</strong><SeverityBadge severity={incident.severity} /></div>
                      <div className="popup-location"><MapPin size={14} />{incident.location}</div>
                      <p>{incident.description}</p>
                      <div className="popup-stats"><span><Users size={14} />{incident.people_affected || 0}</span><AIScore score={incident.ai_score} /></div>
                      <button className="popup-view-button" onClick={() => setSelectedIncident(incident)}><Eye size={14} />View Incident</button>
                    </div>
                  </Popup>
                </Marker>
              </React.Fragment>
            );
          })}
          {liveLocations.map((item) => (
            <Marker key={`live-${item.user_id}-${item.resource_id || "user"}`} position={[Number(item.latitude), Number(item.longitude)]} icon={createLiveLocationIcon(item.role)}>
              <Popup>
                <div className="map-popup">
                  <div className="popup-title"><strong><CircleUserRound size={14} /> {item.full_name || item.username || "Live responder"}</strong></div>
                  <div className="popup-location"><Radio size={14} />LIVE LOCATION</div>
                  <p>{item.resource_id ? `Resource #${item.resource_id}` : `${item.role || "User"} device`}</p>
                  <div className="popup-stats"><span><Navigation size={14} />{item.speed_kmh ? `${Number(item.speed_kmh).toFixed(1)} km/h` : "Stationary"}</span><span>{item.last_seen ? new Date(item.last_seen).toLocaleTimeString() : ""}</span></div>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
        <div className="map-overlay-top"><div className="map-live-indicator"><span></span>LIVE MAP OPERATIONS</div></div>
        <div className="map-overlay-bottom"><div><strong>{mappedIncidents.length}</strong><span>mapped incidents</span></div><div><strong>AI</strong><span>risk analysis active</span></div></div>
      </div>
      <div className="map-incidents-section">
        <div className="map-section-title"><div><h3>Detected Incident Zones</h3><p>Click an incident to focus the map.</p></div></div>
        {loading && <LoadingState />}
        {!loading && mappedIncidents.length === 0 && <EmptyState />}
        <div className="map-incident-grid">
          {mappedIncidents.map((incident) => (
            <button className={`map-incident-card ${selectedIncident?.id === incident.id ? "selected" : ""}`} key={incident.id} onClick={() => setSelectedIncident(incident)}>
              <IncidentTypeIcon type={incident.disaster_type} />
              <div><strong>{incident.disaster_type} — {incident.location}</strong><span>{incident.description}</span></div>
              <AIScore score={incident.ai_score} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function IncidentCommandCenter({ incidents, loading, apiError, fetchIncidents, selectedIncident, setSelectedIncident, refreshCommandCenter, user }) {
  const [search, setSearch] = useState("");
  const [severityFilter, setSeverityFilter] = useState("All");

  const filteredIncidents = useMemo(() => incidents.filter((incident) => {
    const query = search.toLowerCase().trim();
    const matchesSearch = !query || `${incident.disaster_type || ""} ${incident.location || ""} ${incident.description || ""}`.toLowerCase().includes(query);
    const matchesSeverity = severityFilter === "All" || incident.severity === severityFilter;
    return matchesSearch && matchesSeverity;
  }), [incidents, search, severityFilter]);

  if (selectedIncident) {
    return <IncidentDetails incident={selectedIncident} goBack={() => setSelectedIncident(null)} refresh={refreshCommandCenter} user={user} />;
  }

  return (
    <div className="command-center">
      <div className="command-header">
        <div><div className="eyebrow"><Radio size={15} />LIVE INCIDENT MONITORING</div><h2>Incident Command Center</h2><p>Monitor, prioritize and coordinate disaster response.</p></div>
        <button className="refresh-button" onClick={fetchIncidents}><RefreshCw size={17} />Refresh</button>
      </div>
      <div className="incident-kpis">
        <MiniKPI title="Total Incidents" value={incidents.length} icon={TriangleAlert} />
        <MiniKPI title="Critical" value={incidents.filter((i) => i.severity === "Critical").length} icon={ShieldAlert} danger />
        <MiniKPI title="Fire Events" value={incidents.filter((i) => i.disaster_type === "Fire").length} icon={Flame} />
        <MiniKPI title="Flood Events" value={incidents.filter((i) => i.disaster_type === "Flood").length} icon={CloudRain} />
      </div>
      <div className="incident-toolbar">
        <div className="search-box"><Search size={18} /><input placeholder="Search incidents..." value={search} onChange={(e) => setSearch(e.target.value)} /></div>
        <div className="filter-control"><Filter size={17} /><select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)}><option>All</option><option>Critical</option><option>High</option><option>Medium</option><option>Low</option></select></div>
      </div>
      <div className="incident-table-card">
        <div className="table-heading"><div><h3>Live Incident Feed</h3><span>Showing {filteredIncidents.length} of {incidents.length}</span></div><div className="feed-status"><span></span>LIVE FEED</div></div>
        {loading && <LoadingState />}
        {!loading && apiError && <ErrorState error={apiError} retry={fetchIncidents} />}
        {!loading && !apiError && filteredIncidents.length === 0 && <EmptyState />}
        {!loading && !apiError && filteredIncidents.length > 0 && (
          <div className="table-wrapper">
            <table>
              <thead><tr><th>INCIDENT</th><th>LOCATION</th><th>SEVERITY</th><th>AFFECTED</th><th>AI SCORE</th><th>STATUS</th><th>ACTION</th></tr></thead>
              <tbody>
                {filteredIncidents.map((incident) => (
                  <tr key={incident.id}>
                    <td><div className="table-incident"><IncidentTypeIcon type={incident.disaster_type} /><div><strong>{incident.disaster_type}</strong><small>ID #{incident.id}</small></div></div></td>
                    <td><div className="location-cell"><MapPin size={15} />{incident.location}</div></td>
                    <td><SeverityBadge severity={incident.severity} /></td>
                    <td>{incident.people_affected || 0}</td>
                    <td><AIScore score={incident.ai_score} /></td>
                    <td><StatusBadge status={incident.status} /></td>
                    <td><button className="view-button" onClick={() => setSelectedIncident(incident)}><Eye size={15} />View</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function IncidentDetails({ incident, goBack, refresh, user }) {
  const [responseData, setResponseData] = useState(null);
  const [responseLoading, setResponseLoading] = useState(true);
  const [resourceRecommendations, setResourceRecommendations] = useState([]);
  const [resourceLoading, setResourceLoading] = useState(true);
  const [resourceDispatching, setResourceDispatching] = useState(false);
  const [selectedResourceId, setSelectedResourceId] = useState(null);
  const [statusUpdating, setStatusUpdating] = useState(false);
  const [
   intelligence,
   setIntelligence
  ] = useState(null);

  const [
   intelligenceLoading,
   setIntelligenceLoading
  ] = useState(true);

  useEffect(() => {

  const loadIntelligence =
    async () => {

      setIntelligenceLoading(
        true
      );

      try {

        const response =
          await apiFetch(
            `${API_URL}/api/incidents/${incident.id}/intelligence`
          );

        if (!response.ok) {

          setIntelligence(
            null
          );

          return;

        }

        const data =
          await response.json();

        setIntelligence(
          data
        );

      } catch (error) {

        console.error(
          "AI intelligence error:",
          error
        );

        setIntelligence(
          null
        );

      } finally {

        setIntelligenceLoading(
          false
        );

      }

    };

  loadIntelligence();

}, [
  incident.id
]);
  useEffect(() => {
    const load = async () => {
      setResponseLoading(true);
      try {
        const response = await apiFetch(`${API_URL}/api/incidents/${incident.id}/response`);
        setResponseData(response.ok ? await response.json() : null);
      } catch (error) {
        console.error(error);
        setResponseData(null);
      } finally {
        setResponseLoading(false);
      }
    };
    load();
  }, [incident.id]);

  const loadRecommendedResources = async () => {
    setResourceLoading(true);
    try {
      const response = await apiFetch(`${API_URL}/api/incidents/${incident.id}/resources`);
      const data = response.ok ? await response.json() : null;
      setResourceRecommendations(Array.isArray(data?.recommendations) ? data.recommendations : []);
    } catch (error) {
      console.error(error);
      setResourceRecommendations([]);
    } finally {
      setResourceLoading(false);
    }
  };

  useEffect(() => {
    loadRecommendedResources();
  }, [incident.id]);

  const dispatchRecommendedResource = async (resource) => {
    setResourceDispatching(true);
    try {
      const response = await apiFetch(
        `${API_URL}/api/resources/${resource.resource_id}/dispatch?incident_id=${incident.id}&eta_minutes=10`,
        { method: "POST" }
      );
      const data = await response.json();
      if (!response.ok) throw new Error(formatApiError(data.detail, "Unable to dispatch resource"));
      setSelectedResourceId(resource.resource_id);
      alert(`${data.name} dispatched successfully. ETA: ${data.eta_minutes} minutes.`);
      await loadRecommendedResources();
      if (refresh) await refresh();
    } catch (error) {
      console.error(error);
      alert(error.message);
    } finally {
      setResourceDispatching(false);
    }
  };

  const updateStatus = async (status) => {
    setStatusUpdating(true);
    try {
      const response = await apiFetch(`${API_URL}/api/incidents/${incident.id}/status?status=${encodeURIComponent(status)}`, { method: "PATCH" });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(formatApiError(data.detail, "Unable to update incident status"));
      }
      if (refresh) await refresh();
      alert(`Incident status updated to ${status}.`);
    } catch (error) {
      alert(error.message);
    } finally {
      setStatusUpdating(false);
    }
  };

  return (
    <div className="incident-details">
      <button className="back-button" onClick={goBack}><ArrowLeft size={18} />Back to Incidents</button>
      <div className="details-header">
        <div>
          <div className="eyebrow"><Radio size={15} />INCIDENT #{incident.id}</div>
          <h2>{incident.disaster_type}<span> — {incident.location}</span></h2>
          <p>{incident.description}</p>
        </div>
        <SeverityBadge severity={incident.severity} />
      </div>
      <div className="details-grid">
        <div className="details-card"><Brain size={21} /><span>AI RISK SCORE</span><strong>{incident.ai_score}<small>/100</small></strong></div>
        <div className="details-card"><Users size={21} /><span>PEOPLE AFFECTED</span><strong>{incident.people_affected || 0}</strong></div>
        <div className="details-card"><Clock size={21} /><span>STATUS</span><strong>{incident.status || "Active"}</strong></div>
        <div className="details-card"><MapPin size={21} /><span>LOCATION</span><strong>{incident.location}</strong></div>
      </div>

      <div className="incident-status-actions">
        {[
          ["Active", "Active"],
          ["Dispatched", "Dispatched"],
          ["In Progress", "In Progress"],
          ["Contained", "Contained"],
          ["Resolved", "Resolved"],
        ].map(([value, label]) => (
          <button
            key={value}
            className={`status-button status-${value.toLowerCase().replace(/\s+/g, "-")} ${incident.status === value ? "current" : ""}`}
            disabled={!hasPermission(user, "updateIncidentStatus") || statusUpdating || incident.status === value}
            onClick={() => updateStatus(value)}
            title={!hasPermission(user, "updateIncidentStatus") ? "Your role is read-only for incident status" : "Update incident status"}
          >
            {incident.status === value ? <CheckCircle size={16} /> : <CircleDot size={16} />}{label}
          </button>
        ))}
      </div>

      <div className="response-panel">
        <div className="response-panel-header">
          <div><div className="eyebrow"><Zap size={15} />AI RESPONSE ENGINE</div><h3>Recommended Emergency Response</h3></div>
          {responseData && <div className="priority-pill">AI PRIORITY: {responseData.priority}</div>}
        </div>
        {responseLoading && <LoadingState />}
        {!responseLoading && responseData && <div className="recommendations">{(responseData.recommendations || []).map((recommendation, index) => (
          <div className="recommendation" key={index}><div className="recommendation-number">{index + 1}</div><span>{recommendation}</span><CheckCircle size={18} /></div>
        ))}</div>}
        {!responseLoading && !responseData && <div className="ai-fallback"><Brain size={25} /><div><strong>AI response unavailable</strong><p>Backend response recommendation endpoint is not available for this incident.</p></div></div>}
      </div>

      
      <div className="intelligence-panel">

  <div className="response-panel-header">

    <div>

      <div className="eyebrow">

        <Brain size={15} />

        AI INCIDENT INTELLIGENCE

      </div>

      <h3>
        Risk Analysis & Assessment
      </h3>

    </div>

    {intelligence && (

      <div className="priority-pill">

        {intelligence.priority}

      </div>

    )}

  </div>


  {intelligenceLoading && (
    <LoadingState />
  )}


  {!intelligenceLoading &&
   intelligence && (

    <>

      <div className="intelligence-summary">

        <div className="intelligence-score">

          <span>
            AI RISK SCORE
          </span>

          <strong>
            {
              intelligence.risk_score
            }
          </strong>

          <small>
            /100
          </small>

          <em className={`score-source ${intelligence.score_source === "ml" ? "ml" : "baseline"}`}>
            {intelligence.score_source === "ml" ? "ML MODEL" : "RULES BASELINE"}
          </em>

        </div>


        <div className="intelligence-confidence">

          <span>
            {intelligence.score_source === "ml" ? "PREDICTION RANGE" : "CONFIDENCE"}
          </span>

          <strong>
            {intelligence.score_source === "ml"
              ? (intelligence.prediction_interval ? `${intelligence.prediction_interval[0]}–${intelligence.prediction_interval[1]}` : "—")
              : `${intelligence.confidence ?? "—"}%`}
          </strong>

        </div>


        <div className="intelligence-assessment">

          <span>
            AI ASSESSMENT
          </span>

          <p>
            {
              intelligence.assessment
            }
          </p>

        </div>

      </div>


      <div className="intelligence-section">

        <h4>
          Risk Factors
        </h4>


        <div className="risk-factor-list">

          {
            intelligence.risk_factors.map(
              factor => (

                <div
                  className="risk-factor"
                  key={
                    factor.name
                  }
                >

                  <div className="risk-factor-heading">

                    <strong>
                      {
                        factor.name
                      }
                    </strong>

                    <span>
                      {
                        factor.score
                      }%
                    </span>

                  </div>


                  <div className="risk-factor-bar">

                    <div
                      style={{
                        width:
                          `${factor.score}%`
                      }}
                    ></div>

                  </div>


                  <p>
                    {
                      factor.reason
                    }
                  </p>

                </div>

              )
            )
          }

        </div>

      </div>


      <div className="intelligence-columns">

        <div>

          <h4>
            Recommended Actions
          </h4>

          <div className="intel-list">

            {
              intelligence
                .recommended_actions
                .map(
                  (
                    action,
                    index
                  ) => (

                    <div
                      key={
                        index
                      }
                    >

                      <CheckCircle
                        size={16}
                      />

                      <span>
                        {
                          action
                        }
                      </span>

                    </div>

                  )
                )
            }

          </div>

        </div>


        <div>

          <h4>
            Required Resources
          </h4>

          <div className="intel-list">

            {
              intelligence
                .resource_requirements
                .map(
                  resource => (

                    <div
                      key={
                        resource
                      }
                    >

                      <Truck
                        size={16}
                      />

                      <span>
                        {
                          resource
                        }
                      </span>

                    </div>

                  )
                )
            }

          </div>

        </div>

      </div>

    </>

  )}

</div>
      <div className="resource-recommendation-panel">
        <div className="response-panel-header">
          <div><div className="eyebrow"><Brain size={15} />AI RESOURCE ALLOCATION</div><h3>Recommended Emergency Resources</h3></div>
          <div className="priority-pill">AI MATCHING</div>
        </div>
        {resourceLoading && <LoadingState />}
        {!resourceLoading && resourceRecommendations.length === 0 && <div className="ai-fallback"><Package size={24} /><div><strong>No suitable resources available</strong><p>All matching resources may currently be deployed.</p></div></div>}
        {!resourceLoading && resourceRecommendations.length > 0 && (
          <div className="recommended-resource-list">
            {resourceRecommendations.map((resource) => {
              const type = String(resource.resource_type || "").toLowerCase();
              const Icon = type.includes("fire") ? Flame : type.includes("ambulance") ? Ambulance : type.includes("drone") ? Plane : type.includes("boat") ? Ship : type.includes("rescue") ? UserRound : Truck;
              return (
                <div className={`recommended-resource ${selectedResourceId === resource.resource_id ? "selected" : ""}`} key={resource.resource_id}>
                  <div className="recommended-resource-icon"><Icon size={22} /></div>
                  <div className="recommended-resource-info">
                    <strong>{resource.name}</strong>
                    <span>{resource.resource_type} · {resource.location}</span>
                    <small>{(resource.reasons || []).slice(0, 2).join(" • ")}</small>
                  </div>
                  <div className="resource-match-score"><strong>{resource.score}%</strong><span>AI Match</span></div>
                  {hasPermission(user, "dispatchResource") && (
                    <button className="resource-dispatch-button" disabled={resourceDispatching} onClick={() => dispatchRecommendedResource(resource)}>
                      {resourceDispatching ? <RefreshCw size={15} className="spin" /> : <Send size={15} />}Dispatch
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="details-action-grid">
        <button className="action-card" onClick={loadRecommendedResources}><Truck size={22} /><strong>Refresh Resources</strong><span>Recalculate the best available units</span></button>
        <button className="action-card" onClick={() => window.open(`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(incident.location)}`, "_blank")}><Navigation size={22} /><strong>Navigate</strong><span>Open incident location</span></button>
        <button className="action-card" onClick={() => alert("Emergency control contact workflow ready.")}><Phone size={22} /><strong>Contact Emergency</strong><span>Connect with emergency control</span></button>
      </div>
    </div>
  );
}


function AuditLogsPage() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadLogs = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await apiFetch(`${API_URL}/api/audit-logs?limit=200`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(data.detail, "Unable to load audit logs"));
      setLogs(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err.message || "Unable to load audit logs");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadLogs(); }, []);

  return (
    <div className="audit-page">
      <div className="audit-header">
        <div>
          <div className="eyebrow"><ListChecks size={15} />ACCOUNTABILITY & SECURITY</div>
          <h2>Audit Logs</h2>
          <p>Immutable operational history for user and system actions.</p>
        </div>
        <button className="refresh-button" onClick={loadLogs}><RefreshCw size={16} />Refresh</button>
      </div>
      {error && <div className="tracking-error"><AlertTriangle size={18} />{error}</div>}
      <div className="card audit-card">
        <div className="audit-summary"><span><strong>{logs.length}</strong> recent events</span><small>Administrative actions are recorded by the backend.</small></div>
        {loading ? <LoadingState /> : logs.length === 0 ? <EmptyState /> : (
          <div className="audit-table-wrap">
            <table className="audit-table">
              <thead><tr><th>Time</th><th>User</th><th>Action</th><th>Entity</th><th>Details</th></tr></thead>
              <tbody>{logs.map((item) => <tr key={item.id}><td>{item.created_at ? new Date(item.created_at).toLocaleString() : "—"}</td><td><strong>{item.username || "system"}</strong></td><td><span className="audit-action">{item.action}</span></td><td>{item.entity_type ? `${item.entity_type}${item.entity_id ? ` #${item.entity_id}` : ""}` : "—"}</td><td><code>{JSON.stringify(item.details || {})}</code></td></tr>)}</tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function UserManagementPage({ user }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState({ full_name: "", username: "", password: "", role: "Responder" });
  const [editForm, setEditForm] = useState({ full_name: "", role: "Responder", password: "" });

  const loadUsers = async () => {
    setLoading(true); setError("");
    try {
      const response = await apiFetch(`${API_URL}/api/users`);
      const data = await response.json().catch(() => []);
      if (!response.ok) throw new Error(formatApiError(data.detail, `User API returned ${response.status}`));
      setUsers(Array.isArray(data) ? data : []);
    } catch (err) { setError(err.message || "Unable to load users"); }
    finally { setLoading(false); }
  };

  useEffect(() => { loadUsers(); }, []);

  const submitCreate = async (event) => {
    event.preventDefault();
    setError("");
    setNotice("");

    const password = String(form.password || "");
    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }
    if (!/[a-z]/.test(password)) {
      setError("Password must contain a lowercase letter.");
      return;
    }
    if (!/[A-Z]/.test(password)) {
      setError("Password must contain an uppercase letter.");
      return;
    }
    if (!/[0-9]/.test(password)) {
      setError("Password must contain a number.");
      return;
    }

    setSaving(true);
    try {
      const response = await apiFetch(`${API_URL}/api/users`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(form),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(data.detail, "Unable to create user"));
      setForm({ full_name: "", username: "", password: "", role: "Responder" });
      setNotice(`User ${data.username} created successfully.`); await loadUsers();
    } catch (err) { setError(err.message || "Unable to create user"); }
    finally { setSaving(false); }
  };

  const startEdit = (target) => {
    setEditingId(target.id); setEditForm({ full_name: target.full_name, role: target.role, password: "" });
    setError(""); setNotice("");
  };

  const saveEdit = async (userId) => {
    setError("");
    setNotice("");

    const newPassword = String(editForm.password || "").trim();
    if (newPassword) {
      if (newPassword.length < 8) {
        setError("New password must be at least 8 characters long.");
        return;
      }
      if (!/[a-z]/.test(newPassword)) {
        setError("New password must contain a lowercase letter.");
        return;
      }
      if (!/[A-Z]/.test(newPassword)) {
        setError("New password must contain an uppercase letter.");
        return;
      }
      if (!/[0-9]/.test(newPassword)) {
        setError("New password must contain a number.");
        return;
      }
    }

    setSaving(true);
    try {
      const payload = { full_name: editForm.full_name, role: editForm.role };
      if (newPassword) payload.password = newPassword;
      const response = await apiFetch(`${API_URL}/api/users/${userId}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(data.detail, "Unable to update user"));
      setEditingId(null); setNotice(`User ${data.username} updated successfully.`); await loadUsers();
    } catch (err) { setError(err.message || "Unable to update user"); }
    finally { setSaving(false); }
  };

  const toggleUser = async (target) => {
    if (target.id === user.id && target.is_active) return;
    setSaving(true); setError(""); setNotice("");
    try {
      const response = await apiFetch(`${API_URL}/api/users/${target.id}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ is_active: !target.is_active }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(data.detail, "Unable to change user status"));
      setNotice(`${data.username} is now ${data.is_active ? "active" : "disabled"}.`); await loadUsers();
    } catch (err) { setError(err.message || "Unable to change user status"); }
    finally { setSaving(false); }
  };

  const removeUser = async (target) => {
    if (target.id === user.id) return;
    if (!window.confirm(`Delete the account '${target.username}'?`)) return;
    setSaving(true); setError(""); setNotice("");
    try {
      const response = await apiFetch(`${API_URL}/api/users/${target.id}`, { method: "DELETE" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(data.detail, "Unable to delete user"));
      setNotice(`${target.username} was deleted.`); await loadUsers();
    } catch (err) { setError(err.message || "Unable to delete user"); }
    finally { setSaving(false); }
  };

  return (
    <div className="users-page">
      <div className="page-hero users-hero">
        <div><div className="eyebrow"><UserCog size={16} />ADMINISTRATION</div><h2>User Management</h2><p>Create and control Dispatcher and Responder accounts for the RAKSHA command center.</p></div>
        <button className="outline-button" onClick={loadUsers} disabled={loading || saving}><RefreshCw size={16} className={loading ? "spin" : ""} />Refresh Users</button>
      </div>
      {(error || notice) && <div className={`user-banner ${error ? "error" : "success"}`}>{error ? <TriangleAlert size={18} /> : <CheckCircle size={18} />}<span>{error || notice}</span></div>}
      <div className="user-management-grid">
        <section className="panel user-create-card">
          <div className="panel-heading"><div><span className="section-kicker">NEW ACCOUNT</span><h3>Create User</h3><p>Provision a secure operational account.</p></div><div className="panel-icon"><UserCog size={20} /></div></div>
          <form className="user-form" onSubmit={submitCreate}>
            <label>Full Name<input value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} placeholder="e.g. Rahul Sharma" required /></label>
            <label>Username<input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} placeholder="e.g. rahul.responder" required /></label>
            <label>
              Temporary Password
              <input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} placeholder="Minimum 8 characters (Aa1...)" autoComplete="new-password" required />
              <small className="user-password-hint">8+ characters with uppercase, lowercase and a number.</small>
            </label>
            <label>Role<select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}><option value="Dispatcher">Dispatcher</option><option value="Responder">Responder</option></select></label>
            <button className="primary-button user-create-button" type="submit" disabled={saving}><Users size={17} />{saving ? "Creating..." : "Create Account"}</button>
          </form>
          <div className="security-note"><ShieldCheck size={17} /><span>Only Admin accounts can provision active users. Self-registered operational accounts remain pending until approved.</span></div>
        </section>
        <section className="panel user-list-card">
          <div className="panel-heading"><div><span className="section-kicker">ACCOUNT DIRECTORY</span><h3>Operational Users</h3><p>{users.length} account{users.length === 1 ? "" : "s"} registered</p></div><div className="user-count-pill"><Users size={15} />{users.length}</div></div>
          {loading ? <LoadingState /> : users.length === 0 ? <EmptyState /> : (
            <div className="user-table-wrap"><table className="user-table"><thead><tr><th>User</th><th>Role</th><th>Status</th><th>Actions</th></tr></thead><tbody>
              {users.map((target) => editingId === target.id ? (
                <tr key={target.id} className="editing-row"><td><div className="user-edit-fields"><input value={editForm.full_name} onChange={(e) => setEditForm({ ...editForm, full_name: e.target.value })} /><input type="password" value={editForm.password} onChange={(e) => setEditForm({ ...editForm, password: e.target.value })} placeholder="New password (8+ chars)" /></div><small>@{target.username}</small></td><td><select value={editForm.role} onChange={(e) => setEditForm({ ...editForm, role: e.target.value })}><option>Admin</option><option>Dispatcher</option><option>Responder</option></select></td><td><StatusBadge status={target.approval_status === "pending" ? "Pending" : target.is_active ? "Active" : "Disabled"} /></td><td><div className="user-actions"><button className="icon-action success" onClick={() => saveEdit(target.id)} disabled={saving}><Check size={16} /></button><button className="icon-action" onClick={() => setEditingId(null)}><X size={16} /></button></div></td></tr>
              ) : (
                <tr key={target.id}><td><div className="managed-user"><div className="managed-avatar">{userInitials(target)}</div><div><strong>{target.full_name}{target.id === user.id ? " (You)" : ""}</strong><small>@{target.username}</small></div></div></td><td><span className={`role-chip ${String(target.role).toLowerCase()}`}>{target.role}</span></td><td><span className={`account-status ${target.approval_status === "pending" ? "pending" : target.is_active ? "active" : "disabled"}`}><span></span>{target.approval_status === "pending" ? "Pending approval" : target.is_active ? "Active" : "Disabled"}</span></td><td><div className="user-actions"><button className="icon-action" title="Edit" onClick={() => startEdit(target)}><UserCog size={16} /></button><button className="icon-action" title={target.approval_status === "pending" ? "Approve" : target.is_active ? "Disable" : "Enable"} onClick={() => toggleUser(target)} disabled={target.id === user.id && target.is_active}><LockKeyhole size={16} /></button><button className="icon-action danger" title="Delete" onClick={() => removeUser(target)} disabled={target.id === user.id}><X size={16} /></button></div></td></tr>
              ))}
            </tbody></table></div>
          )}
        </section>
      </div>
    </div>
  );
}

function ResourcesPage({ incidents, resources, setResources, loading, error, fetchResources, refreshCommandCenter, user }) {
  const [resourceFilter, setResourceFilter] = useState("All");
  const [selectedResource, setSelectedResource] = useState(null);
  const [selectedIncident, setSelectedIncident] = useState("");
  const [dispatching, setDispatching] = useState(false);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [responders, setResponders] = useState([]);
  const [assigningResourceId, setAssigningResourceId] = useState(null);

  useEffect(() => {
    if (!hasPermission(user, "dispatchResource")) return;
    (async () => {
      try {
        const response = await apiFetch(`${API_URL}/api/users`);
        const data = await response.json().catch(() => []);
        if (response.ok) setResponders((Array.isArray(data) ? data : []).filter((item) => String(item.role || "").toLowerCase() === "responder" && item.is_active !== false));
      } catch (_) {}
    })();
  }, [user]);

  const assignResponder = async (resourceId, userId) => {
    setAssigningResourceId(resourceId);
    try {
      const query = userId ? `?user_id=${encodeURIComponent(userId)}` : "";
      const response = await apiFetch(`${API_URL}/api/resources/${resourceId}/assign-user${query}`, { method: "PATCH" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(data.detail, "Unable to assign responder"));
      setResources((previous) => previous.map((item) => item.id === data.id ? data : item));
    } catch (err) {
      alert(err.message || "Unable to assign responder");
    } finally {
      setAssigningResourceId(null);
    }
  };

  const filteredResources = useMemo(() => resources.filter((resource) => {
    const query = search.toLowerCase().trim();
    const matchesSearch = !query || `${resource.name || ""} ${resource.resource_type || ""} ${resource.location || ""}`.toLowerCase().includes(query);
    const matchesType = resourceFilter === "All" || resource.resource_type === resourceFilter;
    const matchesStatus = statusFilter === "All" || resource.status === statusFilter;
    return matchesSearch && matchesType && matchesStatus;
  }), [resources, search, resourceFilter, statusFilter]);

  const availableCount = resources.filter((r) => r.status === "Available").length;
  const enRouteCount = resources.filter((r) => r.status === "En Route").length;
  const deployedCount = resources.filter((r) => r.status === "Deployed").length;

  const dispatchResource = async () => {
    if (!selectedResource || !selectedIncident) return;
    setDispatching(true);
    try {
      const response = await apiFetch(`${API_URL}/api/resources/${selectedResource.id}/dispatch?incident_id=${selectedIncident}&eta_minutes=10`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(formatApiError(data.detail, "Dispatch failed"));
      setResources((previous) => previous.map((resource) => resource.id === data.id ? data : resource));
      setSelectedResource(null);
      setSelectedIncident("");
      await refreshCommandCenter();
      alert(`${data.name} is now en route. ETA: ${data.eta_minutes} minutes.`);
    } catch (error) {
      alert(error.message);
    } finally {
      setDispatching(false);
    }
  };

  const releaseResource = async (resource) => {
    try {
      const response = await apiFetch(`${API_URL}/api/resources/${resource.id}/release`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(formatApiError(data.detail, "Release failed"));
      setResources((previous) => previous.map((item) => item.id === data.id ? data : item));
      await refreshCommandCenter();
    } catch (error) {
      alert(error.message);
    }
  };

  return (
    <div className="resources-page">
      <div className="command-header">
        <div><div className="eyebrow"><Truck size={15} />LIVE RESOURCE CONTROL</div><h2>Emergency Resource Command</h2><p>Monitor, deploy and coordinate emergency response resources.</p></div>
        <button className="refresh-button" onClick={fetchResources}><RefreshCw size={17} />Refresh Resources</button>
      </div>
      <div className="incident-kpis">
        <MiniKPI title="Total Resources" value={resources.length} icon={Truck} />
        <MiniKPI title="Available" value={availableCount} icon={CheckCircle} />
        <MiniKPI title="En Route" value={enRouteCount} icon={Navigation} />
        <MiniKPI title="Deployed" value={deployedCount} icon={Send} danger />
      </div>
      <div className="resource-toolbar">
        <div className="search-box"><Search size={18} /><input placeholder="Search resources..." value={search} onChange={(e) => setSearch(e.target.value)} /></div>
        <div className="filter-control"><Filter size={17} /><select value={resourceFilter} onChange={(e) => setResourceFilter(e.target.value)}><option>All</option><option>Ambulance</option><option>Fire Truck</option><option>Rescue Team</option><option>Drone</option><option>Rescue Boat</option></select></div>
        <div className="filter-control"><select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}><option>All</option><option>Available</option><option>En Route</option><option>Deployed</option><option>Maintenance</option></select></div>
      </div>
      {loading && <LoadingState />}
      {!loading && error && <ErrorState error={error} retry={fetchResources} />}
      {!loading && !error && filteredResources.length === 0 && <EmptyState />}
      {!loading && !error && filteredResources.length > 0 && (
        <div className="resource-grid">
          {filteredResources.map((resource) => {
            const type = String(resource.resource_type || "").toLowerCase();
            const Icon = type.includes("ambulance") ? Ambulance : type.includes("fire") ? Flame : type.includes("drone") ? Plane : type.includes("boat") ? Ship : type.includes("rescue") ? UserRound : Truck;
            return (
              <div className="resource-card" key={resource.id}>
                <div className="resource-card-top"><div className="resource-icon"><Icon size={24} /></div><StatusBadge status={resource.status} /></div>
                <h3>{resource.name}</h3>
                <p><MapPin size={14} />{resource.location}</p>
                <div className="resource-meta"><span>Capacity: {resource.capacity}</span><span>{resource.eta_minutes ? `${resource.eta_minutes} min ETA` : "No ETA"}</span></div>
                {resource.assigned_incident_id && <div className="assigned-incident"><TriangleAlert size={14} />Incident #{resource.assigned_incident_id}</div>}
                {hasPermission(user, "dispatchResource") && (
                  <div className="resource-tracking-assignment">
                    <div className="resource-tracking-label"><LocateFixed size={14} />GPS field assignment</div>
                    <select
                      value={resource.assigned_user_id || ""}
                      onChange={(event) => assignResponder(resource.id, event.target.value)}
                      disabled={assigningResourceId === resource.id}
                    >
                      <option value="">No responder assigned</option>
                      {responders.map((responder) => <option key={responder.id} value={responder.id}>{responder.full_name} · {responder.username}</option>)}
                    </select>
                    {resource.last_seen && <small>Last GPS: {new Date(resource.last_seen).toLocaleTimeString()}</small>}
                  </div>
                )}
                {resource.status === "Available" ? (
                  hasPermission(user, "dispatchResource") ? (
                    <button className="resource-action" onClick={() => { setSelectedResource(resource); setSelectedIncident(""); }}><Send size={15} />Dispatch</button>
                  ) : (
                    <button className="resource-action role-readonly" disabled><LockKeyhole size={15} />Dispatch restricted</button>
                  )
                ) : (
                  hasPermission(user, "releaseResource") && (resource.status === "En Route" || resource.status === "Deployed") ? (
                    <button className="resource-action" onClick={() => releaseResource(resource)}><RotateCcw size={15} />Release</button>
                  ) : (
                    <button className="resource-action role-readonly" disabled>{resource.status}</button>
                  )
                )}
              </div>
            );
          })}
        </div>
      )}
      {selectedResource && (
        <div className="modal-backdrop" onClick={() => setSelectedResource(null)}>
          <div className="dispatch-modal" onClick={(e) => e.stopPropagation()}>
            <div className="dispatch-modal-header"><div><div className="eyebrow"><Send size={15} />RESOURCE DISPATCH</div><h3>{selectedResource.name}</h3></div><button className="modal-close" onClick={() => setSelectedResource(null)}><X size={20} /></button></div>
            <p>Select the active incident to receive this resource.</p>
            <select className="dispatch-select" value={selectedIncident} onChange={(e) => setSelectedIncident(e.target.value)}><option value="">Select active incident</option>{incidents.filter((incident) => !incident.status || incident.status === "Active").map((incident) => <option key={incident.id} value={incident.id}>#{incident.id} — {incident.disaster_type} — {incident.location}</option>)}</select>
            <div className="dispatch-actions"><button className="outline-button" onClick={() => setSelectedResource(null)}>Cancel</button><button className="resource-action" disabled={!selectedIncident || dispatching || !hasPermission(user, "dispatchResource")} onClick={dispatchResource}>{dispatching ? <><RefreshCw size={16} className="spin" />Dispatching...</> : <><Send size={16} />Confirm Dispatch</>}</button></div>
          </div>
        </div>
      )}
    </div>
  );
}

function AlertsPage({ incidents, setActivePage, user }) {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchAlerts = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await apiFetch(`${API_URL}/api/alerts?acknowledged=false`);
      if (!response.ok) {
        throw new Error(`Alerts API returned ${response.status}`);
      }

      const data = await response.json();
      setAlerts(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error("Alerts API error:", err);
      setError(err.message || "Unable to load alerts");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, []);

  useEffect(() => {
    const handleAlertUpdate = () => {
      fetchAlerts();
    };

    window.addEventListener("raksha-alert-update", handleAlertUpdate);

    return () => {
      window.removeEventListener("raksha-alert-update", handleAlertUpdate);
    };
  }, []);

  const acknowledgeAlert = async (alertId) => {
    try {
      const response = await apiFetch(
        `${API_URL}/api/alerts/${alertId}/acknowledge`,
        { method: "PATCH" }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(formatApiError(data.detail, "Unable to acknowledge alert"));
      }

      setAlerts((previous) =>
        previous.filter((alert) => alert.id !== alertId)
      );
    } catch (err) {
      console.error("Alert acknowledgement error:", err);
      alert(err.message);
    }
  };

  const criticalAlerts = alerts.filter(
    (item) => item.severity === "Critical"
  ).length;

  const highAlerts = alerts.filter(
    (item) => item.severity === "High"
  ).length;

  return (
    <div className="alerts-page">
      <div className="command-header">
        <div>
          <div className="eyebrow">
            <Bell size={15} />
            REAL-TIME ALERT NETWORK
          </div>
          <h2>Emergency Alerts</h2>
          <p>AI-prioritized warnings requiring attention.</p>
        </div>
        <div className="alert-live-pill">
          <span></span>
          ALERT NETWORK ONLINE
        </div>
      </div>

      <div className="alerts-summary">
        <div>
          <strong>{alerts.length}</strong>
          <span>active alerts</span>
        </div>
        <div>
          <strong>{criticalAlerts}</strong>
          <span>critical</span>
        </div>
        <div>
          <strong>{highAlerts}</strong>
          <span>high priority</span>
        </div>
      </div>

      {loading && <LoadingState />}

      {!loading && error && (
        <ErrorState error={error} retry={fetchAlerts} />
      )}

      {!loading && !error && alerts.length === 0 && <EmptyState />}

      {!loading && !error && alerts.length > 0 && (
        <div className="alerts-list">
          {alerts.map((alertItem) => (
            <div
              className={`alert-card ${String(alertItem.severity || "Medium").toLowerCase()}`}
              key={alertItem.id}
            >
              <div className="alert-icon">
                <Siren size={24} />
              </div>

              <div className="alert-content">
                <div className="alert-title-row">
                  <h3>{alertItem.title}</h3>
                  <SeverityBadge severity={alertItem.severity} />
                </div>

                <p>{alertItem.message}</p>

                <div className="alert-meta">
                  <span>
                    <Brain size={14} />
                    AI Score: {alertItem.ai_score || 0}
                  </span>

                  {alertItem.incident_id && (
                    <span>
                      <CircleDot size={14} />
                      Incident #{alertItem.incident_id}
                    </span>
                  )}
                </div>
              </div>

              {hasPermission(user, "acknowledgeAlert") ? (
                <button
                  className="dismiss-button"
                  onClick={() => acknowledgeAlert(alertItem.id)}
                >
                  <Check size={16} />
                  Acknowledge
                </button>
              ) : (
                <span className="role-restricted-badge"><LockKeyhole size={14} />View only</span>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="alert-footer">
        <button
          className="outline-button"
          onClick={() => setActivePage("Incidents")}
        >
          <Eye size={17} />
          Open Incident Center
        </button>

        <button
          className="outline-button"
          onClick={fetchAlerts}
        >
          <RefreshCw size={17} />
          Refresh Alerts
        </button>
      </div>
    </div>
  );
}

function ReportsPage({ incidents, resources }) {
  const total = incidents.length;
  const critical = incidents.filter((i) => i.severity === "Critical").length;
  const high = incidents.filter((i) => i.severity === "High").length;
  const fire = incidents.filter((i) => i.disaster_type === "Fire").length;
  const flood = incidents.filter((i) => i.disaster_type === "Flood").length;
  const affected = incidents.reduce((sum, i) => sum + Number(i.people_affected || 0), 0);
  const avgRisk = total ? Math.round(incidents.reduce((sum, i) => sum + Number(i.ai_score || 0), 0) / total) : 0;
  const readiness = resources.length ? Math.round((resources.filter((r) => r.status === "Available").length / resources.length) * 100) : 0;
  return (
    <div className="reports-page">
      <div className="command-header"><div><div className="eyebrow"><BarChart3 size={15} />DISASTER INTELLIGENCE REPORTING</div><h2>Response Reports</h2><p>Operational overview generated from current incident and resource data.</p></div><button className="refresh-button" onClick={() => window.print()}><FileText size={17} />Export Report</button></div>
      <div className="report-stat-grid">
        <ReportStat title="Total Incidents" value={total} icon={TriangleAlert} /><ReportStat title="Critical Events" value={critical} icon={ShieldAlert} /><ReportStat title="People Affected" value={affected} icon={Users} /><ReportStat title="Fire Events" value={fire} icon={Flame} /><ReportStat title="Flood Events" value={flood} icon={CloudRain} /><ReportStat title="Resource Readiness" value={`${readiness}%`} icon={Truck} />
      </div>
      <div className="report-grid">
        <div className="card report-card"><div className="small-card-title"><BarChart3 size={20} />Incident Distribution</div><div className="bar-report"><ReportBar label="Critical" value={critical} total={total} /><ReportBar label="High" value={high} total={total} /><ReportBar label="Fire" value={fire} total={total} /><ReportBar label="Flood" value={flood} total={total} /></div></div>
        <div className="card report-card"><div className="small-card-title"><Brain size={20} />AI Intelligence Summary</div><div className="report-ai-summary"><div className="report-ai-score"><strong>{avgRisk}</strong><span>average AI risk score</span></div><p>RAKSHA AI continuously ranks emergencies using severity, affected population and incident characteristics.</p></div></div>
      </div>
    </div>
  );
}

function EmergencySOSPage({ setActivePage, user }) {
  const [activated, setActivated] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [emergencyId, setEmergencyId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (countdown <= 0) return;
    const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => clearTimeout(timer);
  }, [countdown]);

  const activateSOS = async () => {
    if (activated) {
      setActivated(false);
      setCountdown(0);
      return;
    }

    setLoading(true);
    setError("");

    try {
      const geo = await new Promise((resolve) => {
        if (!navigator.geolocation) return resolve(null);
        navigator.geolocation.getCurrentPosition(
          (position) => resolve(position.coords),
          () => resolve(null),
          { enableHighAccuracy: true, maximumAge: 3000, timeout: 10000 }
        );
      });
      const response = await apiFetch(`${API_URL}/api/emergency/sos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          emergency_type: "SOS",
          message: geo ? "Emergency SOS activated with device location" : "Emergency SOS activated without device location",
          latitude: geo?.latitude ?? null,
          longitude: geo?.longitude ?? null,
          location_accuracy: geo?.accuracy ?? null,
          source: geo ? "device_gps" : "operator",
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(formatApiError(data.detail, "Unable to activate emergency SOS"));

      setEmergencyId(data.id);
      setActivated(true);
      setCountdown(5);
    } catch (err) {
      console.error("SOS activation error:", err);
      setError(err.message || "Unable to activate SOS");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="sos-page">
      <div className="sos-header">
        <div className="eyebrow"><ShieldAlert size={16} />EMERGENCY CONTROL</div>
        <h2>Emergency SOS</h2>
        <p>Rapid emergency escalation interface for critical situations.</p>
      </div>
      <div className="sos-panel">
        <div className={`sos-circle ${activated ? "activated" : ""} ${!hasPermission(user, "activateSOS") ? "role-disabled" : ""}`} onClick={hasPermission(user, "activateSOS") && !loading ? activateSOS : undefined}>
          <Siren size={55} />
          <strong>{activated ? (countdown > 0 ? countdown : "ACTIVE") : "SOS"}</strong>
          <span>{loading ? "Activating..." : activated ? `Emergency protocol${emergencyId ? ` #${emergencyId}` : ""}` : "Press to activate"}</span>
        </div>
        {error && <div className="sos-warning"><AlertTriangle size={20} /><div><strong>SOS activation failed</strong><p>{error}</p></div></div>}
        <div className="sos-warning"><AlertTriangle size={20} /><div><strong>Use only for genuine emergencies</strong><p>Activation creates a persistent emergency event, generates a Critical alert, and notifies connected RAKSHA command centers.</p></div></div>
        <div className="sos-actions">
          <button onClick={() => alert("Emergency hotline workflow ready.")}><Phone size={18} />Emergency Hotline</button>
          <button onClick={() => setActivePage("Resources")}><Ambulance size={18} />Request Medical Response</button>
          <button onClick={() => setActivePage("Resources")}><Truck size={18} />Request Rescue Team</button>
        </div>
      </div>
    </div>
  );
}

function IncidentTypeIcon({ type }) {
  const normalized = String(type || "").toLowerCase();
  const Icon = normalized === "fire" ? Flame : normalized === "earthquake" ? Activity : normalized === "cyclone" ? RotateCcw : normalized === "landslide" ? TriangleAlert : CloudRain;
  return <div className={`incident-icon ${normalized}`}><Icon size={20} /></div>;
}

function SeverityBadge({ severity }) {
  const value = severity || "Unknown";
  return <span className={`severity-badge ${String(value).toLowerCase()}`}><span></span>{value}</span>;
}

function StatusBadge({ status }) {
  const value = status || "Active";
  return <span className={`status-badge ${String(value).toLowerCase().replace(/\s+/g, "-")}`}><span></span>{value}</span>;
}

function AIScore({ score }) {
  const value = Number(score || 0);
  const className = value >= 90 ? "critical" : value >= 75 ? "high" : value >= 50 ? "medium" : "low";
  return <div className={`ai-score-new ${className}`}><Brain size={14} /><strong>{value}</strong></div>;
}

function IncidentRow({ incident, onClick }) {
  return <div className="incident" onClick={onClick}><IncidentTypeIcon type={incident.disaster_type} /><div className="incident-info"><div className="incident-title"><strong>{incident.disaster_type}</strong><SeverityBadge severity={incident.severity} /></div><span>{incident.location}</span><small>{incident.people_affected || 0} people affected</small></div><AIScore score={incident.ai_score} /></div>;
}

function MiniKPI({ title, value, icon: Icon, danger }) {
  return <div className="mini-kpi"><div className={`mini-kpi-icon ${danger ? "danger" : ""}`}><Icon size={20} /></div><div><span>{title}</span><strong>{value}</strong></div></div>;
}

function StatCard({ title, value, subtitle, icon: Icon, danger, warning }) {
  return <div className="stat-card"><div className={`stat-icon ${danger ? "danger" : warning ? "warning" : ""}`}><Icon size={22} /></div><div className="stat-info"><span>{title}</span><strong>{value}</strong><small>{subtitle}</small></div></div>;
}

function ReportStat({ title, value, icon: Icon }) {
  return <div className="report-stat"><div className="report-stat-icon"><Icon size={21} /></div><span>{title}</span><strong>{value}</strong></div>;
}

function ReportBar({ label, value, total }) {
  const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
  return <div className="report-bar-row"><div><span>{label}</span><strong>{value}</strong></div><div className="report-bar-track"><div className="report-bar-fill" style={{ width: `${percentage}%` }}></div></div></div>;
}

function LoadingState() { return <div className="state-message"><RefreshCw size={25} className="spin" /><span>Loading live data...</span></div>; }
function ErrorState({ error, retry }) { return <div className="state-message error"><TriangleAlert size={28} /><strong>Backend connection failed</strong><small>{error}</small><button className="outline-button" onClick={retry}>Retry</button></div>; }
function EmptyState() { return <div className="state-message"><ShieldAlert size={30} /><span>No data found.</span></div>; }

export default App;