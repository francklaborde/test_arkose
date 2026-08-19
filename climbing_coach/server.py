import inspect
import logging
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi import HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
from typing import Optional

from src.climbing_coach import ClimbingCoach, CoachMode, PlanBlock
from src.climber_profile import ClimberProfile
from src.sboulder_collector import setup_logging
from src.auth import AuthStore, COOKIE_NAME, send_magic_link_email

setup_logging()
app = FastAPI()
logger = logging.getLogger("climbing_coach")

STREAM_ERROR_MARKER = "[[STREAM_ERROR]]"

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

async def _safe_stream(gen):
    """Wrap a coach stream so a mid-stream crash (e.g. Mistral API failure)
    ends with a detectable marker instead of just cutting the connection,
    which the client can't distinguish from a normal end of stream."""
    try:
        if inspect.isasyncgen(gen):
            async for chunk in gen:
                yield chunk
        else:
            for chunk in gen:
                yield chunk
    except Exception:
        logger.exception("stream failed")
        yield f"\n\n{STREAM_ERROR_MARKER}"

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = os.environ.get("CLIMBING_DB_PATH", str(_REPO_ROOT / "database" / "climbing.db"))
DEFAULT_PROFILE_PATH = os.environ.get("CLIMBING_PROFILE_PATH", str(_REPO_ROOT / "database" / "franck.json"))
PROFILES_DIR = os.environ.get("CLIMBING_PROFILES_DIR", str(_REPO_ROOT / "database" / "profiles"))

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set. Create a .env file (see .env.example) "
        "or set it in your environment before starting the server."
    )

RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
if not RESEND_API_KEY:
    raise RuntimeError(
        "RESEND_API_KEY is not set. Create a .env file (see .env.example) "
        "or set it in your environment before starting the server."
    )
# Until you verify your own domain in Resend, you can only send from this
# sandbox address, and ONLY to the email tied to your own Resend account —
# fine for testing solo, but friends won't receive anything until you add
# and verify a real domain at resend.com/domains and set FROM_EMAIL.
FROM_EMAIL = os.environ.get("FROM_EMAIL", "Arkose Coach <onboarding@resend.dev>")
# Base URL used to build the link in the magic-link email — set this in
# .env once the app moves off localhost (e.g. to the VPS domain, over HTTPS).
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8000")

auth_store = AuthStore(DEFAULT_DB_PATH, PROFILES_DIR)

# In-memory session store: session_id -> ClimbingCoach instance
sessions: dict[str, ClimbingCoach] = {}
session_accounts: dict[str, str] = {}  # session_id -> account_id

class StartSessionRequest(BaseModel):
    mode: str = "coaching"  # "coaching" | "onboarding"

class ChatRequest(BaseModel):
    session_id: str
    message: str

class OnboardingFinishRequest(BaseModel):
    session_id: str
    profile_path: str

class SyncRequest(BaseModel):
    session_id: str

class PlanBlockEdit(BaseModel):
    name: str
    sets: Optional[int] = None
    reps: Optional[str] = None
    duration_min: Optional[int] = None
    rest_sec: Optional[int] = None
    notes: str = ""

class PlanEditRequest(BaseModel):
    title: Optional[str] = None
    warmup: Optional[list[PlanBlockEdit]] = None
    blocks: Optional[list[PlanBlockEdit]] = None
    cooldown: Optional[list[PlanBlockEdit]] = None

class ProfileUpdateRequest(BaseModel):
    session_id: str
    profile: dict

class MagicLinkRequest(BaseModel):
    email: str

def _get_last_sync(coach: ClimbingCoach) -> Optional[str]:
    stats = coach.get_stats()
    if not stats or not stats.last_sync:
        return None
    timestamps = [ts for ts in stats.last_sync.values() if ts]
    return max(timestamps) if timestamps else None

def _get_current_account(request: Request) -> sqlite3.Row:
    """Resolve the logged-in account from the session cookie, or 401."""
    token = request.cookies.get(COOKIE_NAME)
    account = auth_store.resolve_session(token) if token else None
    if not account:
        raise HTTPException(status_code=401, detail="Not logged in")
    return account

@app.post("/auth/request-link")
def request_magic_link(req: MagicLinkRequest):
    email = req.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    # Always create the account up front (not just on verify) so the
    # profiles/ path exists even if the user never finishes clicking the
    # link — harmless, and simpler than deferring account creation.
    auth_store.get_or_create_account(email)
    token = auth_store.create_magic_link(email)
    verify_url = f"{APP_BASE_URL}/auth/verify?token={token}"
    try:
        send_magic_link_email(RESEND_API_KEY, FROM_EMAIL, email, verify_url)
    except Exception as e:
        logger.exception("failed to send magic link email")
        raise HTTPException(status_code=502, detail=f"Email non envoyé : {e}")
    return {"status": "ok"}

@app.get("/auth/verify")
def verify_magic_link(token: str):
    email = auth_store.consume_magic_link(token)
    if not email:
        raise HTTPException(status_code=400, detail="Lien invalide ou expiré")
    account = auth_store.get_or_create_account(email)
    session_token = auth_store.create_session(account["account_id"])
    response = RedirectResponse(url="/")
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_token,
        max_age=90 * 24 * 3600,
        httponly=True,
        samesite="lax",
        # secure=True should be added once this is served over HTTPS.
    )
    return response

@app.get("/auth/me")
def auth_me(request: Request):
    account = _get_current_account(request)
    return {"email": account["email"], "sboulder_user_id": account["sboulder_user_id"]}

@app.post("/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        auth_store.delete_session(token)
    response.delete_cookie(COOKIE_NAME)
    return {"status": "ok"}

@app.post("/session/start")
def start_session(req: StartSessionRequest, request: Request):
    account = _get_current_account(request)
    session_id = str(uuid.uuid4())
    mode = CoachMode.ONBOARDING if req.mode == "onboarding" else CoachMode.COACHING

    profile_path = account["profile_path"]
    if not Path(profile_path).exists():
        # First session for this account — seed a blank profile, carrying
        # over sboulder_user_id if it was already linked via /session/connect-sboulder
        # before this first session (e.g. a returning magic-link login).
        blank = ClimberProfile(sboulder_user_id=account["sboulder_user_id"] or "")
        blank.save(profile_path)

    coach = ClimbingCoach.from_mistral(  # adjust to your actual factory method name
        api_key=MISTRAL_API_KEY,
        db_path=DEFAULT_DB_PATH,
        profile_path=profile_path,
    )

    if mode == CoachMode.COACHING:
        coach.profile = ClimberProfile.load(profile_path)

    sessions[session_id] = coach
    session_accounts[session_id] = account["account_id"]
    last_sync = _get_last_sync(coach) if mode == CoachMode.COACHING else None
    headers = {
        "X-Session-Id": session_id,
        "X-Last-Sync": last_sync or "",
        "X-Mode": mode.value,
    }
    return StreamingResponse(
        _safe_stream(coach.start_session_stream(mode)),
        media_type="text/plain",
        headers=headers,
    )

@app.post("/session/chat")
def chat(req: ChatRequest):
    coach = sessions.get(req.session_id)
    if coach is None:
        return {"error": "unknown session, call /session/start first"}
    reply = coach.chat(req.message)
    return {"reply": reply}

@app.post("/session/chat/stream")
def chat_stream(req: ChatRequest, background_tasks: BackgroundTasks):
    coach = sessions.get(req.session_id)
    if coach is None:
        raise HTTPException(status_code=404, detail="unknown session, call /session/start first")
    background_tasks.add_task(coach.maybe_generate_plan)
    return StreamingResponse(
        _safe_stream(coach.chat_stream(req.message)),
        media_type="text/plain",
        background=background_tasks,
    )

@app.post("/session/sync")
async def sync_session(req: SyncRequest):
    coach = sessions.get(req.session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        coach.sync_now()
    except Exception as e:
        logger.exception("sync failed")
        raise HTTPException(status_code=502, detail=f"Sync échouée : {e}")
    return {"status": "ok", "last_sync": _get_last_sync(coach)}

@app.post("/session/onboarding/finish")
def finish_onboarding(req: OnboardingFinishRequest):
    coach = sessions.get(req.session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    if coach.mode != CoachMode.ONBOARDING:
        raise HTTPException(status_code=400, detail="Session is not in onboarding mode")
    profile_path = req.profile_path.strip()
    if not profile_path:
        raise HTTPException(status_code=400, detail="profile_path is required")
    try:
        profile = coach.extract_profile_from_history()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {e}")
    coach.save_profile(profile_path)
    return {"status": "ok", "profile_path": profile_path, "profile": profile.to_dict()}

# --- Serve the minimal chat page ---
@app.get("/", response_class=HTMLResponse)
def index():
    with open("chat.html", encoding="utf-8") as f:
        return f.read()

@app.get("/session/stats")
def get_stats(session_id: str):
    coach = sessions.get(session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    stats = coach.get_stats()
    if not stats:
        raise HTTPException(status_code=404, detail="No stats available")
    return {
        "current_level": stats.current_level,
        "current_flash_level": stats.current_flash_level,
        "total_sends": stats.total_sends,
        "total_flashes": stats.total_flashes,
        "sends_by_grade": stats.sends_by_grade,
        "flashes_by_grade": stats.flashes_by_grade,
        "last_sync": _get_last_sync(coach),
    }

@app.get("/session/plan")
def get_plan(session_id: str):
    coach = sessions.get(session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"plans": [p.to_dict() for p in coach.plans]}

@app.delete("/session/plan/{plan_id}")
def delete_plan(plan_id: str, session_id: str):
    coach = sessions.get(session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    before = len(coach.plans)
    coach.plans = [p for p in coach.plans if p.id != plan_id]
    if len(coach.plans) == before:
        raise HTTPException(status_code=404, detail="Plan not found")
    return {"status": "ok"}

@app.patch("/session/plan/{plan_id}")
def edit_plan(plan_id: str, req: PlanEditRequest, session_id: str):
    coach = sessions.get(session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    plan = next((p for p in coach.plans if p.id == plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    if req.title is not None:
        plan.title = req.title
    if req.warmup is not None:
        plan.warmup = [PlanBlock(**b.model_dump()) for b in req.warmup]
    if req.blocks is not None:
        plan.blocks = [PlanBlock(**b.model_dump()) for b in req.blocks]
    if req.cooldown is not None:
        plan.cooldown = [PlanBlock(**b.model_dump()) for b in req.cooldown]
    return {"status": "ok", "plan": plan.to_dict()}

@app.get("/session/profile")
def get_profile(session_id: str):
    coach = sessions.get(session_id)
    if not coach or not coach.profile:
        raise HTTPException(status_code=404, detail="Session or profile not found")
    return coach.profile.to_dict()

@app.get("/session/gyms")
def get_available_gyms(session_id: str):
    coach = sessions.get(session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"gyms": coach.available_gyms()}

@app.put("/session/profile")
def update_profile(req: ProfileUpdateRequest):
    coach = sessions.get(req.session_id)
    if not coach or not coach.profile:
        raise HTTPException(status_code=404, detail="Session or profile not found")
    # sboulder_user_id links the profile to the DB — never let the edit form
    # overwrite it, even if the submitted payload includes it.
    incoming = dict(req.profile)
    incoming["sboulder_user_id"] = coach.profile.sboulder_user_id
    # gyms must stay within the DB's known slugs (checkboxes on the frontend
    # enforce this already; re-check server-side against free-form input).
    valid_gyms = set(coach.available_gyms())
    submitted_gyms = incoming.get("gyms") or []
    unknown = [g for g in submitted_gyms if g not in valid_gyms]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown gym(s): {', '.join(unknown)}")
    coach.profile = ClimberProfile.from_dict(incoming)
    coach.save_profile()
    return {"status": "ok"}

class SboulderIdRequest(BaseModel):
    session_id: str
    sboulder_user_id: str

def _set_sboulder_user_id(coach: ClimbingCoach, session_id: str, sboulder_user_id: str) -> None:
    if not coach.profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    sboulder_user_id = sboulder_user_id.strip()
    coach.profile.sboulder_user_id = sboulder_user_id
    coach.save_profile()
    account_id = session_accounts.get(session_id)
    if account_id:
        auth_store.set_sboulder_user_id(account_id, sboulder_user_id)

@app.put("/session/profile/sboulder-id")
def update_sboulder_id(req: SboulderIdRequest):
    """Manual fallback: paste a sboulder_user_id directly (e.g. found via
    DevTools) instead of using the Arkose+ bookmarklet."""
    coach = sessions.get(req.session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    _set_sboulder_user_id(coach, req.session_id, req.sboulder_user_id)
    return {"status": "ok"}

def _get_db_stats() -> dict:
    """Read-only snapshot of the sqlite DB for the /developer page: route
    counts and last-sync time per gym, plus overall totals."""
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        boulders_by_gym = {
            row["gym"]: row["count"]
            for row in conn.execute("SELECT gym, COUNT(*) AS count FROM boulders GROUP BY gym")
        }
        active_by_gym = {
            row["gym"]: row["count"]
            for row in conn.execute(
                "SELECT gym, COUNT(*) AS count FROM boulders WHERE (closed_at IS NULL OR closed_at > ?) GROUP BY gym", 
                (_now_iso(),)
            )
        }
        sync_by_gym = {
            row["gym"]: {"last_sync": row["last_sync"], "sync_count": row["sync_count"]}
            for row in conn.execute(
                "SELECT gym, MAX(synced_at) AS last_sync, COUNT(*) AS sync_count FROM sync_log GROUP BY gym"
            )
        }
        # boulders and sync_log don't always list the exact same gyms (e.g. a
        # gym that's been synced but has no routes yet), so union both sets.
        all_gyms = sorted(set(boulders_by_gym) | set(sync_by_gym))
        gyms = [
            {
                "gym": gym,
                "route_count": boulders_by_gym.get(gym, 0),
                "active_route_count": active_by_gym.get(gym, 0),
                "last_sync": sync_by_gym.get(gym, {}).get("last_sync"),
                "sync_count": sync_by_gym.get(gym, {}).get("sync_count", 0),
            }
            for gym in all_gyms
        ]
        totals = dict(
            conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM boulders) AS boulders, "
                "(SELECT COUNT(*) FROM ascents) AS ascents, "
                "(SELECT COUNT(*) FROM comments) AS comments, "
                "(SELECT COUNT(DISTINCT user_id) FROM ascents) AS distinct_climbers"
            ).fetchone()
        )
        return {"gyms": gyms, "totals": totals}
    finally:
        conn.close()

@app.get("/developer/db-stats")
def developer_db_stats():
    try:
        return _get_db_stats()
    except Exception as e:
        logger.exception("failed to compute db stats")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/developer", response_class=HTMLResponse)
def developer_page():
    with open("developer.html", encoding="utf-8") as f:
        return f.read()

@app.get("/session/connect-sboulder", response_class=HTMLResponse)
def connect_sboulder(session_id: str, sboulder_user_id: str):
    """Hit by the 'Connect Arkose+' bookmarklet via a plain page navigation
    (javascript: bookmarklets can't do fetch() across origins cleanly),
    so this returns a small confirmation page instead of JSON."""
    coach = sessions.get(session_id)
    if not coach:
        return HTMLResponse(
            "<p>Session introuvable — retourne sur l'app et réessaie.</p>",
            status_code=404,
        )
    try:
        _set_sboulder_user_id(coach, session_id, sboulder_user_id)
    except HTTPException as e:
        return HTMLResponse(f"<p>❌ {e.detail}</p>", status_code=e.status_code)
    return HTMLResponse(
        """
        <html><body style="font-family:sans-serif;text-align:center;padding:40px;">
          <h2>✅ Compte Arkose+ connecté</h2>
          <p id="sboulderMsg">Retour à l'app en cours...</p>
          <a href="/">Retourner à l'app</a>
          <script>
            try {
              const ch = new BroadcastChannel("sboulder-connect");
              ch.postMessage("connected");
            } catch (e) {}
            // If the original tab didn't pick this up (e.g. it was closed,
            // or BroadcastChannel isn't supported), fall back to a manual
            // close instruction instead of forcing a redirect that would
            // just open the app a second time in this tab.
            setTimeout(() => {
              document.getElementById("sboulderMsg").textContent =
                "Tu peux fermer cet onglet et retourner sur l'app.";
            }, 800);
          </script>
        </body></html>
        """
    )


app.mount("/", StaticFiles(directory=Path(__file__).parent), name="static")