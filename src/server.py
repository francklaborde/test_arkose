from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid

from climbing_coach import ClimbingCoach, CoachMode
from climber_profile import ClimberProfile

app = FastAPI()

# In-memory session store: session_id -> ClimbingCoach instance
sessions: dict[str, ClimbingCoach] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class StartResponse(BaseModel):
    session_id: str
    reply: str

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
    return StartResponse(session_id=session_id, reply=opening_message)

@app.post("/session/chat")
def chat(req: ChatRequest):
    coach = sessions.get(req.session_id)
    if coach is None:
        return {"error": "unknown session, call /session/start first"}
    reply = coach.chat(req.message)
    return {"reply": reply}

# --- Serve the minimal chat page ---
@app.get("/", response_class=HTMLResponse)
def index():
    with open("chat.html", encoding="utf-8") as f:
        return f.read()