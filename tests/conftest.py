import sys
from pathlib import Path

# Two import roots are involved:
# - the repo root, so `import climbing_coach.server` resolves
#   (climbing_coach/ is a namespace package, no __init__.py at its top level)
# - the climbing_coach/ directory itself, because server.py does
#   `from src.climbing_coach import ...`, which assumes its own directory
#   is on the path
# Set up both so the tests work regardless of where pytest is invoked from.

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLIMBING_COACH_DIR = _REPO_ROOT / "climbing_coach"
for _p in (_REPO_ROOT, _CLIMBING_COACH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))