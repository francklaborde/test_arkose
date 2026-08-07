from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
from typing import Optional

from src.climbing_coach import ClimbingCoach, CoachMode
from src.climber_profile import ClimberProfile
from src.sboulder_collector import setup_logging

setup_logging()
app = FastAPI()

# In-memory session store: session_id -> ClimbingCoach instance
sessions: dict[str, ClimbingCoach] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class StartResponse(BaseModel):
    session_id: str
    reply: str
    last_sync: Optional[str] = None

class SyncRequest(BaseModel):
    session_id: str

def _get_last_sync(coach: ClimbingCoach) -> Optional[str]:
    stats = coach.get_stats()
    if not stats or not stats.last_sync:
        return None
    timestamps = [ts for ts in stats.last_sync.values() if ts]
    return max(timestamps) if timestamps else None

@app.post("/session/start", response_model=StartResponse)
def start_session():
    API_KEY = "REMOVED_API_KEY"
    session_id = str(uuid.uuid4())
    coach = ClimbingCoach.from_mistral(  # adjust to your actual factory method name
        api_key=API_KEY,
        db_path=r"C:\Users\Franc\OneDrive\Documents\GitHub\test_arkose\database\climbing.db",
        profile_path=r"C:\Users\Franc\OneDrive\Documents\GitHub\test_arkose\database\franck.json",
    )
    coach.profile = ClimberProfile.load(r"C:\Users\Franc\OneDrive\Documents\GitHub\test_arkose\database\franck.json")
    opening_message = coach.start_session(CoachMode.COACHING)
    sessions[session_id] = coach
    last_sync = _get_last_sync(coach)
    return StartResponse(session_id=session_id, reply=opening_message, last_sync=last_sync)

@app.post("/session/chat")
def chat(req: ChatRequest):
    coach = sessions.get(req.session_id)
    if coach is None:
        return {"error": "unknown session, call /session/start first"}
    reply = coach.chat(req.message)
    return {"reply": reply}

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

# --- Serve manifest.json, sw.js, icons, etc. at root paths ---
app.mount("/", StaticFiles(directory=Path(__file__).parent), name="static")