import inspect
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
from typing import Optional

from src.climbing_coach import ClimbingCoach, CoachMode, PlanBlock
from src.climber_profile import ClimberProfile
from src.sboulder_collector import setup_logging

setup_logging()
app = FastAPI()
logger = logging.getLogger("climbing_coach")

STREAM_ERROR_MARKER = "[[STREAM_ERROR]]"

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

DEFAULT_DB_PATH = r"C:\Users\Franc\OneDrive\Documents\GitHub\test_arkose\database\climbing.db"
DEFAULT_PROFILE_PATH = r"C:\Users\Franc\OneDrive\Documents\GitHub\test_arkose\database\franck.json"

# In-memory session store: session_id -> ClimbingCoach instance
sessions: dict[str, ClimbingCoach] = {}

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

def _get_last_sync(coach: ClimbingCoach) -> Optional[str]:
    stats = coach.get_stats()
    if not stats or not stats.last_sync:
        return None
    timestamps = [ts for ts in stats.last_sync.values() if ts]
    return max(timestamps) if timestamps else None

@app.post("/session/start")
def start_session(req: StartSessionRequest):
    API_KEY = "REMOVED_API_KEY"
    session_id = str(uuid.uuid4())
    mode = CoachMode.ONBOARDING if req.mode == "onboarding" else CoachMode.COACHING

    coach = ClimbingCoach.from_mistral(  # adjust to your actual factory method name
        api_key=API_KEY,
        db_path=DEFAULT_DB_PATH,
        profile_path=DEFAULT_PROFILE_PATH,
    )

    if mode == CoachMode.COACHING:
        coach.profile = ClimberProfile.load(DEFAULT_PROFILE_PATH)

    sessions[session_id] = coach
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

def _set_sboulder_user_id(coach: ClimbingCoach, sboulder_user_id: str) -> None:
    if not coach.profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    coach.profile.sboulder_user_id = sboulder_user_id.strip()
    coach.save_profile()

@app.put("/session/profile/sboulder-id")
def update_sboulder_id(req: SboulderIdRequest):
    """Manual fallback: paste a sboulder_user_id directly (e.g. found via
    DevTools) instead of using the Arkose+ bookmarklet."""
    coach = sessions.get(req.session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    _set_sboulder_user_id(coach, req.sboulder_user_id)
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
        _set_sboulder_user_id(coach, sboulder_user_id)
    except HTTPException as e:
        return HTMLResponse(f"<p>❌ {e.detail}</p>", status_code=e.status_code)
    return HTMLResponse(
        """
        <html><body style="font-family:sans-serif;text-align:center;padding:40px;">
          <h2>✅ Compte Arkose+ connecté</h2>
          <p>Tu peux retourner sur l'app.</p>
          <a href="/">Retourner à l'app</a>
          <script>setTimeout(() => location.href = '/', 1500);</script>
        </body></html>
        """
    )


app.mount("/", StaticFiles(directory=Path(__file__).parent), name="static")