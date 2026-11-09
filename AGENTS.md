# AGENTS.md

Climbing-coach PWA (UI in French), single-user: a FastAPI app that streams LLM coaching (OpenRouter) and training plans, fed by climbing data synced from sboulder.com (Arkose+ gyms). No accounts, no login — one profile file (`database/franck.json` by default).

## Layout
- `climbing_coach/server.py` is the real app: FastAPI backend plus a static PWA (`chat.html`, `sw.js`, `manifest.json` in the same dir). Root `main.py` is an unused uv stub; root `*.ipynb` files are scratch notebooks, not part of the app.
- `climbing_coach/src/` is the library: `coach.py` (`ClimbingCoach` orchestrator — the class to touch for behavior changes), `climbing_coach.py` (re-export facade only), `climbing_stats.py` (SQLite stats), `coaching_prompts.py` (coaching/plan prompts), `llm_client.py` (OpenAI-compatible LLM wrapper; swap providers here), `sboulder_collector.py` (WebSocket sync → SQLite), `climber_profile.py` (profile model).
- `database/` holds local data: `climbing.db` (tables: boulders, ascents, comments, sync_log; the users/magic_links/auth_sessions tables are inert leftovers from the removed magic-link auth) and JSON profiles. Gym names are slugs like `arkose/montmartre`.

## Running the server
- `OPENROUTER_API_KEY` is required in `.env` (repo root, see `.env.example`); the server hard-fails at import without it.
- CWD must be `climbing_coach/` (HTML pages are read by relative path and `from src...` imports assume that directory):
  `cd climbing_coach && uv run python -m uvicorn server:app`
- Single process only: chat sessions live in an in-memory dict (`server.sessions`) — no `--workers`, and restarting drops all sessions.

## Running the tests
- From the repo root: `uv run --with pytest python -m pytest tests/ -q` (pytest is not in the pyproject dev group — hence `--with`).
- Single test: `uv run --with pytest python -m pytest "tests/test_server.py::test_sync_unknown_session_returns_404" -q`
- Tests use a `FakeCoach` injected into `server.sessions` (no real LLM/sboulder/DB). They only need `OPENROUTER_API_KEY` set — no other env prerequisite.
- `tests/` is not committed to git yet.

## Gotchas
- The repo is OneDrive-synced and developed on both Windows and WSL. The root `.venv` is a Windows venv (`pyvenv.cfg` points at `C:\`): do NOT run `uv run`/`uv sync` in WSL — it rebuilds `.venv` for Linux and wrecks the Windows environment. In WSL use a separate venv. (`uv lock` is safe — it only rewrites `uv.lock`.)
- Nearly every tracked `.py`/`.html` file shows as modified in `git status` due to CRLF↔LF churn, not content. Use `git diff --ignore-cr-at-eol` to see real diffs.
- `database/` is gitignored but `database/climbing.db` and `database/franck.json` are tracked anyway (contain real data and show as modified); `climbing.db.bak-nanterre-cleanup` is a local backup.
- `sboulder_user_id` links the profile to the DB and is authoritative: the server strips it from any `PUT /session/profile` payload; set it by editing the profile JSON directly (or via `PUT /session/profile/sboulder-id`). Find the ID on plus.arkose.com via `Meteor.userId()` in the browser console.
- Manual sync CLI: from `climbing_coach/`, `uv run python -m src.sboulder_collector --gym arkose/montmartre --db ../database/climbing.db --profile ../database/franck.json` (pass `--profile` or it creates a stray `climber_profile.json` in CWD). `python -m src.climber_profile <path> --show` prints the LLM context block.
- A mid-stream LLM crash ends the response with the `[[STREAM_ERROR]]` marker (`server.STREAM_ERROR_MARKER`) instead of a silent cut — that contract is tested and the PWA relies on it; don't remove it.
- LLM goes through OpenRouter's OpenAI-compatible API (`openrouter.ai/api/v1`); default model is `google/gemma-4-31b-it` — pass `model=` to use any OpenRouter slug.
