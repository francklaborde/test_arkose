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

# In-memory session store: session_id -> ClimbingCoach instance
sessions: dict[str, ClimbingCoach] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

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
def start_session():
    API_KEY = "cglowaxE1vSmixJcBun7lKmi71qsw79E"
    session_id = str(uuid.uuid4())
    coach = ClimbingCoach.from_mistral(  # adjust to your actual factory method name
        api_key=API_KEY,
        db_path=r"C:\Users\Franc\OneDrive\Documents\GitHub\test_arkose\database\climbing.db",
        profile_path=r"C:\Users\Franc\OneDrive\Documents\GitHub\test_arkose\database\franck.json",
    )
    coach.profile = ClimberProfile.load(r"C:\Users\Franc\OneDrive\Documents\GitHub\test_arkose\database\franck.json")
    sessions[session_id] = coach
    last_sync = _get_last_sync(coach)
    headers = {
        "X-Session-Id": session_id,
        "X-Last-Sync": last_sync or "",
    }
    return StreamingResponse(
        coach.start_session_stream(CoachMode.COACHING),
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
        coach.chat_stream(req.message),
        media_type="text/plain",
        background=background_tasks,
    )

@app.post("/session/sync")
async def sync_session(req: SyncRequest):
    coach = sessions.get(req.session_id)
    if not coach:
        raise HTTPException(status_code=404, detail="Session not found")
    coach.sync_now()
    return {"status": "ok", "last_sync": _get_last_sync(coach)}

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

@app.put("/session/profile")
def update_profile(req: ProfileUpdateRequest):
    coach = sessions.get(req.session_id)
    if not coach or not coach.profile:
        raise HTTPException(status_code=404, detail="Session or profile not found")
    # sboulder_user_id / gyms link the profile to the DB — never let the
    # edit form overwrite them, even if the submitted payload includes them.
    incoming = dict(req.profile)
    incoming["sboulder_user_id"] = coach.profile.sboulder_user_id
    incoming["gyms"] = coach.profile.gyms
    coach.profile = ClimberProfile.from_dict(incoming)
    coach.save_profile()
    return {"status": "ok"}

# --- Serve manifest.json, sw.js, icons, etc. at root paths ---
app.mount("/", StaticFiles(directory=Path(__file__).parent), name="static")