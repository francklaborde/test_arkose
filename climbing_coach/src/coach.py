"""
coach.py
--------
CoachMode and ClimbingCoach — the top-level orchestrator and the only
class most callers interact with directly (see climbing_coach.py for
the stable import path).
"""

from __future__ import annotations

import logging
import json
from enum import Enum
from pathlib import Path
from typing import Optional

from .climber_profile import ClimberProfile
from .climbing_stats import ClimbingStats, StatsBuilder
from .coaching_prompts import PlanBlock, PromptBuilder, SessionPlan
from .llm_client import LLMClient
from .sboulder_collector import SBoulderCollector, encode_grade_level

log = logging.getLogger("climbing_coach")
log.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------

class CoachMode(Enum):
    ONBOARDING = "onboarding"   # guided interview to fill ClimberProfile
    COACHING   = "coaching"     # regular coaching session


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ClimbingCoach:
    """
    Top-level orchestrator. The only class you need in the notebook.

    Handles:
    - Session lifecycle (start/reset)
    - Mode switching (onboarding ↔ coaching)
    - Profile extraction after onboarding
    - Stats injection into coaching prompts
    """

    def __init__(
        self,
        llm: LLMClient,
        profile: Optional[ClimberProfile] = None,
        db_path: Optional[str | Path] = None,
        profile_path: str | Path = "climber_profile.json",
        auto_sync: bool = False,

    ):
        self.llm = llm
        self.profile = profile
        self.profile_path = Path(profile_path)
        self.prompt_builder = PromptBuilder()
        self.mode: Optional[CoachMode] = None
        self.auto_sync = auto_sync
        self.plans: list[SessionPlan] = []

        # StatsBuilder is optional — works without a DB
        self._stats_builder: Optional[StatsBuilder] = None
        if db_path and Path(db_path).exists():
            self.collector = SBoulderCollector(
                user_id=self.profile.sboulder_user_id if self.profile else None,
                db_path=db_path
            )
            self._stats_builder = StatsBuilder(db_path)
            log.info("StatsBuilder ready (db=%s)", db_path)
        elif db_path:
            log.warning("DB path %s not found — stats will be unavailable", db_path)

    # ------------------------------------------------------------------
    # Factory methods (mirror LLMClient factories)
    # ------------------------------------------------------------------

    @classmethod
    def from_mistral(
        cls,
        api_key: str,
        model: str = "mistral-small-latest",
        profile: Optional[ClimberProfile] = None,
        db_path: Optional[str | Path] = None,
        profile_path: str | Path = "climber_profile.json",
        auto_sync: bool = False,
        **llm_kwargs,
    ) -> "ClimbingCoach":
        llm = LLMClient.mistral(api_key=api_key, model=model, **llm_kwargs)
        return cls(llm=llm, profile=profile, db_path=db_path, profile_path=profile_path, auto_sync=auto_sync)

    @classmethod
    def from_openai(
        cls,
        api_key: str,
        model: str = "gpt-4o-mini",
        profile: Optional[ClimberProfile] = None,
        db_path: Optional[str | Path] = None,
        profile_path: str | Path = "climber_profile.json",
        auto_sync: bool = False,
        **llm_kwargs,
    ) -> "ClimbingCoach":
        llm = LLMClient.openai(api_key=api_key, model=model, **llm_kwargs)
        return cls(llm=llm, profile=profile, db_path=db_path, profile_path=profile_path, auto_sync=auto_sync)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _prepare_session(self, mode: CoachMode) -> str:
        """
        Reset history, set mode, build the system prompt, and return the
        opening trigger message. Shared by start_session() and start_session_stream().
        """
        self.mode = mode
        self.llm.reset_history()

        if mode == CoachMode.ONBOARDING:
            self.llm.set_system(self.prompt_builder.onboarding_system(self.profile))
            return "Bonjour, je voudrais créer mon profil de coaching."

        elif mode == CoachMode.COACHING:
            if not self.profile:
                raise ValueError(
                    "No profile loaded. Run an onboarding session first, "
                    "or load a profile with ClimberProfile.load()."
                )
            if self._stats_builder and self.auto_sync:
                self.collector.sync_from_profile(self.profile)
            stats = self._build_stats()
            self._sync_profile_from_stats(stats)
            system = self.prompt_builder.coaching_system(self.profile, stats)
            self.llm.set_system(system)
            name = self.profile.name or "grimpeur"
            # Keep the trigger minimal — the system prompt already contains all
            # the stats context. Let the coach decide what to highlight.
            recent_count = len(stats.recent_ascents) if stats else 0
            if recent_count:
                return f"Bonjour coach, c'est {name}."
            return f"Bonjour coach, c'est {name}. Pas encore de stats disponibles."

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def start_session(self, mode: CoachMode) -> str:
        """
        Reset history, set mode, build system prompt, send opening message.
        Returns the coach's opening line.
        """
        opening_trigger = self._prepare_session(mode)
        return self.llm.chat(opening_trigger)

    def start_session_stream(self, mode: CoachMode):
        """Same as start_session(), but yields the opening line incrementally."""
        opening_trigger = self._prepare_session(mode)
        yield from self.llm.chat_stream(opening_trigger)

    def chat(self, message: str) -> str:
        """Send a message and get the coach's reply."""
        if self.mode is None:
            raise RuntimeError("Call start_session() before chat().")
        return self.llm.chat(message)

    def chat_stream(self, message: str):
        """Send a message and yield the coach's reply incrementally."""
        if self.mode is None:
            raise RuntimeError("Call start_session() before chat().")
        yield from self.llm.chat_stream(message)

    def sync_now(self) -> None:
        """Manually trigger a sync, regardless of auto_sync setting."""
        if not self._stats_builder or not self.profile:
            raise RuntimeError("No collector/profile configured — cannot sync.")
        self.collector.sync_from_profile(self.profile)
        self._sync_profile_from_stats(self._build_stats())
        log.info("Manual sync triggered for user %s", self.profile.sboulder_user_id)

    def _sync_profile_from_stats(self, stats: Optional[ClimbingStats]) -> None:
        """
        Copy DB-derived facts (current Arkose redpoint/flash level) into the
        profile — deterministic, no LLM involved. Saves the profile if
        anything actually changed.
        """
        if not stats or not self.profile:
            return
        changed = False
        if stats.current_level and stats.current_level != self.profile.current_redpoint_level_arkose:
            self.profile.current_redpoint_level_arkose = stats.current_level
            changed = True
        if stats.current_flash_level and stats.current_flash_level != self.profile.current_flash_level_arkose:
            self.profile.current_flash_level_arkose = stats.current_flash_level
            changed = True
        if changed:
            self.save_profile()
            log.info("Profile levels auto-updated from stats")

    # ------------------------------------------------------------------
    # Session plan
    # ------------------------------------------------------------------

    def maybe_generate_plan(self) -> list[SessionPlan]:
        """
        One-shot call that inspects the recent conversation and extracts any NEW
        concrete training session(s) as SessionPlan objects, appended to
        self.plans. Already-known plan titles are passed back to the model so
        it doesn't re-extract them — safe to call after every coaching turn.
        """
        if self.mode != CoachMode.COACHING:
            return []

        transcript = self._history_to_transcript(last_n=8)
        known_titles = [
            f"{p.program_title + ' — ' if p.program_title else ''}{p.title}"
            for p in self.plans
        ]
        raw = self.llm.call_once(
            system=self.prompt_builder.plan_system(self.profile, known_plans=known_titles),
            user_message=f"Conversation récente :\n\n{transcript}",
        )

        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            log.warning("Plan extraction returned invalid JSON, ignoring")
            return []

        new_plans = []
        for plan_data in data.get("plans") or []:
            plan = SessionPlan.from_dict(plan_data)
            self.plans.append(plan)
            new_plans.append(plan)
            log.info("Session plan added: %r", plan.title)
        return new_plans

    # ------------------------------------------------------------------
    # Onboarding extraction
    # ------------------------------------------------------------------

    def extract_profile_from_history(self) -> ClimberProfile:
        """
        Run a one-shot extraction call on the current conversation history
        to produce a ClimberProfile. Does not modify history.
        """
        if self.mode != CoachMode.ONBOARDING:
            raise RuntimeError("extract_profile_from_history() only works after an ONBOARDING session.")

        transcript = self._history_to_transcript()
        raw = self.llm.call_once(
            system=self.prompt_builder.extraction_system(),
            user_message=f"Transcription de l'entretien :\n\n{transcript}",
        )

        # Strip markdown fences if model added them
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        data = json.loads(clean)
        extracted = ClimberProfile.from_dict(data)

        # The interview now only covers qualitative fields (styles, strengths/
        # weaknesses, injuries, goals) — physical profile, level, gyms and
        # training schedule are filled via the profile form beforehand and
        # must survive this extraction untouched. Merge onto the existing
        # profile instead of building a fresh one from the transcript alone.
        profile = self.profile or ClimberProfile()
        profile.preferred_styles = extracted.preferred_styles or profile.preferred_styles
        profile.self_strengths = extracted.self_strengths or profile.self_strengths
        profile.self_weaknesses = extracted.self_weaknesses or profile.self_weaknesses
        profile.injuries = extracted.injuries or profile.injuries
        profile.short_term_goals = extracted.short_term_goals or profile.short_term_goals
        profile.long_term_goals = extracted.long_term_goals or profile.long_term_goals

        self.profile = profile
        log.info("Profile extracted: %r", profile)
        return profile

    def available_gyms(self) -> list[str]:
        """Gym slugs known to the DB — the only valid values for profile.gyms."""
        if not self._stats_builder:
            return []
        return self._stats_builder._all_gyms()

    def save_profile(self, path: Optional[str | Path] = None) -> None:
        """Save the current profile to JSON."""
        if not self.profile:
            raise RuntimeError("No profile to save.")
        self.profile.save(path or self.profile_path)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Optional[ClimbingStats]:
        """Return current stats, or None if no DB is configured."""
        return self._build_stats()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_stats(self) -> Optional[ClimbingStats]:
        if not self._stats_builder or not self.profile or not self.profile.sboulder_user_id:
            return None
        
        min_level = None
        if self.profile.current_flash_level_arkose:
            min_level = encode_grade_level(self.profile.current_flash_level_arkose)

        return self._stats_builder.build(
            user_id=self.profile.sboulder_user_id,
            gyms=self.profile.gyms or None,
            min_level=min_level
        )

    def _history_to_transcript(self, last_n: Optional[int] = None) -> str:
        history = self.llm.history
        if last_n:
            history = history[-last_n:]
        return "\n\n".join(
            f"{'Coach' if m['role'] == 'assistant' else 'Grimpeur'} : {m['content']}"
            for m in history
        )

    def close(self):
        if self._stats_builder:
            self._stats_builder.close()