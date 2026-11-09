"""
Tests for the error-handling paths in server.py.

A fake ClimbingCoach is injected directly into server.sessions so these run
with no real DB, no Mistral call, and no sboulder connection — cheap and
fast, focused on "does the API behave correctly when something fails"
rather than the coaching logic itself.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

import climbing_coach.server as server
from climbing_coach.src.climbing_coach import ClimbingStats
from climbing_coach.src.climber_profile import ClimberProfile


class FakeCoach:
    """Minimal stand-in for ClimbingCoach, implementing just what server.py
    touches. Tests override individual attributes/methods as needed."""

    def __init__(self, mode=None, profile=None):
        self.mode = mode
        self.profile = profile
        self.plans = []
        self._sync_error = None

    def sync_now(self):
        if self._sync_error:
            raise self._sync_error

    def get_stats(self):
        return ClimbingStats(user_id="u1", gyms=["arkose/nation"], last_sync={"arkose/nation": None})

    def start_session_stream(self):
        yield "Bonjour"
        yield " !"

    def chat_stream(self, message):
        yield "Réponse"

    def maybe_generate_plan(self):
        return []

    def available_gyms(self):
        return ["arkose/nation", "arkose/montmartre"]

    def save_profile(self, path=None):
        pass


def failing_stream(*_args, **_kwargs):
    yield "partiel"
    raise RuntimeError("Mistral API unreachable")


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def clear_sessions():
    # server.sessions is a module-level dict shared across tests — reset it
    # so tests can't leak state into each other.
    server.sessions.clear()
    yield
    server.sessions.clear()


def register_session(coach) -> str:
    session_id = str(uuid.uuid4())
    server.sessions[session_id] = coach
    return session_id


# --- /session/sync -----------------------------------------------------

def test_sync_unknown_session_returns_404(client):
    res = client.post("/session/sync", json={"session_id": "does-not-exist"})
    assert res.status_code == 404


def test_sync_success_returns_last_sync(client):
    session_id = register_session(FakeCoach())
    res = client.post("/session/sync", json={"session_id": session_id})
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_sync_failure_returns_502_with_detail(client):
    coach = FakeCoach()
    coach._sync_error = ConnectionError("sboulder websocket closed")
    session_id = register_session(coach)

    res = client.post("/session/sync", json={"session_id": session_id})

    assert res.status_code == 502
    assert "sboulder websocket closed" in res.json()["detail"]


# --- /session/start & /session/chat/stream ------------------------------

def test_start_session_streams_normally(client, monkeypatch):
    coach = FakeCoach()
    monkeypatch.setattr(server.ClimbingCoach, "from_mistral", classmethod(lambda cls, **kw: coach))
    monkeypatch.setattr(
        server.ClimberProfile, "load_or_create",
        classmethod(lambda cls, path: ClimberProfile(sboulder_user_id="u1")),
    )

    res = client.post("/session/start")

    assert res.status_code == 200
    assert res.text == "Bonjour !"
    assert server.STREAM_ERROR_MARKER not in res.text


def test_start_session_stream_crash_emits_error_marker(client, monkeypatch):
    coach = FakeCoach()
    coach.start_session_stream = failing_stream
    monkeypatch.setattr(server.ClimbingCoach, "from_mistral", classmethod(lambda cls, **kw: coach))
    monkeypatch.setattr(
        server.ClimberProfile, "load_or_create",
        classmethod(lambda cls, path: ClimberProfile(sboulder_user_id="u1")),
    )

    res = client.post("/session/start")

    # The connection must not just cut out silently — the client needs a
    # detectable signal that something broke mid-stream.
    assert res.status_code == 200
    assert "partiel" in res.text
    assert server.STREAM_ERROR_MARKER in res.text


def test_chat_stream_unknown_session_returns_404(client):
    res = client.post("/session/chat/stream", json={"session_id": "nope", "message": "salut"})
    assert res.status_code == 404


def test_chat_stream_crash_emits_error_marker(client):
    coach = FakeCoach()
    coach.chat_stream = failing_stream
    session_id = register_session(coach)

    res = client.post("/session/chat/stream", json={"session_id": session_id, "message": "salut"})

    assert res.status_code == 200
    assert server.STREAM_ERROR_MARKER in res.text


# --- Other unknown-session 404s -----------------------------------------

def test_get_stats_unknown_session_returns_404(client):
    res = client.get("/session/stats", params={"session_id": "nope"})
    assert res.status_code == 404


def test_delete_plan_not_found_returns_404(client):
    session_id = register_session(FakeCoach())
    res = client.delete(f"/session/plan/does-not-exist", params={"session_id": session_id})
    assert res.status_code == 404


# --- /session/profile gym validation ------------------------------------

def test_update_profile_rejects_unknown_gym(client):
    profile = ClimberProfile(sboulder_user_id="u1", gyms=["arkose/nation"])
    session_id = register_session(FakeCoach(profile=profile))

    res = client.put("/session/profile", json={
        "session_id": session_id,
        "profile": {**profile.to_dict(), "gyms": ["arkose/not-a-real-gym"]},
    })

    assert res.status_code == 400
    assert "not-a-real-gym" in res.json()["detail"]


def test_update_profile_preserves_sboulder_user_id(client):
    profile = ClimberProfile(sboulder_user_id="original-id", gyms=["arkose/nation"])
    session_id = register_session(FakeCoach(profile=profile))

    incoming = {**profile.to_dict(), "sboulder_user_id": "tampered-id"}
    res = client.put("/session/profile", json={"session_id": session_id, "profile": incoming})

    assert res.status_code == 200
    coach = server.sessions[session_id]
    assert coach.profile.sboulder_user_id == "original-id"