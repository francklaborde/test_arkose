"""
climbing_coach.py
------------------
Backward-compatible facade. The implementation is split across:

  climbing_stats.py    — StatsBuilder, ClimbingStats, RouteTypeStat, RecentAscent
  coaching_prompts.py  — PromptBuilder, PlanBlock, SessionPlan
  llm_client.py        — LLMClient
  coach.py             — ClimbingCoach, CoachMode

Kept as a thin re-export so existing imports keep working unchanged, e.g.:

    from climbing_coach import ClimbingCoach, CoachMode
    from climber_profile import ClimberProfile

    profile = ClimberProfile.load("franck.json")
    coach = ClimbingCoach.from_mistral(profile=profile, db_path="climbing.db")

    coach.start_session(CoachMode.ONBOARDING)
    coach.chat("Bonjour !")
"""

from .climbing_stats import ClimbingStats, RecentAscent, RouteTypeStat, StatsBuilder
from .coaching_prompts import PlanBlock, PromptBuilder, SessionPlan
from .llm_client import LLMClient
from .coach import ClimbingCoach, CoachMode

__all__ = [
    "ClimbingStats", "RouteTypeStat", "RecentAscent", "StatsBuilder",
    "PlanBlock", "SessionPlan", "PromptBuilder",
    "LLMClient",
    "CoachMode", "ClimbingCoach",
]
