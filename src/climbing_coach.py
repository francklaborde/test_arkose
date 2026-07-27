"""
climbing_coach.py
-----------------
LLM-powered climbing coach built on top of sboulder_collector and climber_profile.

Architecture
------------
LLMClient      — thin OpenAI-compatible API wrapper (model, history, calls)
StatsBuilder   — queries SQLite and returns structured ClimbingStats
PromptBuilder  — assembles system prompts from profile + stats + mode
ClimbingCoach  — top-level orchestrator; the only class you interact with

Usage (notebook)
----------------
from climbing_coach import ClimbingCoach, CoachMode
from climber_profile import ClimberProfile

profile = ClimberProfile.load("franck.json")
coach = ClimbingCoach.from_mistral(profile=profile, db_path="climbing.db")

# Onboarding interview
coach.start_session(CoachMode.ONBOARDING)
coach.chat("Bonjour !")

# Coaching session
coach.start_session(CoachMode.COACHING)
coach.chat("Qu'est-ce que tu me conseilles pour progresser cette semaine ?")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .climber_profile import ClimberProfile, Injury
from .sboulder_collector import ROUTE_TYPES_BY_ID, ROUTE_TYPE_LEAF_IDS

log = logging.getLogger("climbing_coach")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CoachMode(Enum):
    ONBOARDING = "onboarding"   # guided interview to fill ClimberProfile
    COACHING   = "coaching"     # regular coaching session


# ---------------------------------------------------------------------------
# Data structures returned by StatsBuilder
# ---------------------------------------------------------------------------

@dataclass
class RouteTypeStat:
    """Send rate for a single route type."""
    type_id: int
    label_fr: str
    sent: int       # boulders of this type the user has sent
    total: int      # total boulders of this type in the DB (for this gym)

    @property
    def rate(self) -> float:
        return self.sent / self.total if self.total else 0.0

    def __str__(self) -> str:
        pct = f"{self.rate * 100:.0f}%"
        return f"{self.label_fr}: {self.sent}/{self.total} ({pct})"


@dataclass
class RecentAscent:
    """A single recent send/flash, enriched with boulder metadata."""
    boulder_id: str
    grade: Optional[str]
    ascent_type: str        # "send" or "flash"
    detected_at: str
    route_type_labels: list[str] = field(default_factory=list)
    gym: str = ""

    def __str__(self) -> str:
        kind = "flashé" if self.ascent_type == "flash" else "envoyé"
        types = f" [{', '.join(self.route_type_labels)}]" if self.route_type_labels else ""
        return f"{self.grade or '?'} {kind}{types} ({self.detected_at[:10]})"


@dataclass
class ClimbingStats:
    """
    All computed stats for one climber, ready to be rendered into a prompt.
    Built by StatsBuilder, consumed by PromptBuilder.
    """
    user_id: str
    gyms: list[str]

    # Volume
    total_sends: int = 0
    total_flashes: int = 0

    # Grade distribution — {grade_str: count}
    sends_by_grade: dict[str, int] = field(default_factory=dict)
    flashes_by_grade: dict[str, int] = field(default_factory=dict)

    # Route type analysis (leaf types only, sorted by rate desc)
    route_type_stats: list[RouteTypeStat] = field(default_factory=list)

    # Recent activity (last N ascents)
    recent_ascents: list[RecentAscent] = field(default_factory=list)

    # Open boulders the user hasn't sent yet (by grade, for goal-setting)
    # {grade: count}
    unsent_by_grade: dict[str, int] = field(default_factory=dict)

    # Last sync timestamp per gym
    last_sync: dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def weakest_route_types(self) -> list[RouteTypeStat]:
        """Leaf types with at least 3 total boulders, sorted by send rate asc."""
        return sorted(
            [s for s in self.route_type_stats if s.total >= 3],
            key=lambda s: s.rate,
        )

    @property
    def strongest_route_types(self) -> list[RouteTypeStat]:
        """Leaf types with at least 3 total boulders, sorted by send rate desc."""
        return sorted(
            [s for s in self.route_type_stats if s.total >= 3],
            key=lambda s: s.rate,
            reverse=True,
        )

    def to_llm_context(self, max_recent: int = 5) -> str:
        """Render stats as a compact markdown block for prompt injection."""
        lines = ["## Statistiques de grimpe"]

        # Volume
        lines.append(f"- **Total envois** : {self.total_sends} blocs "
                     f"({self.total_flashes} flashés)")

        # Grade distribution
        if self.sends_by_grade:
            grade_str = ", ".join(
                f"{g}: {c}" for g, c in sorted(self.sends_by_grade.items())
            )
            lines.append(f"- **Envois par grade** : {grade_str}")

        if self.flashes_by_grade:
            flash_str = ", ".join(
                f"{g}: {c}" for g, c in sorted(self.flashes_by_grade.items())
            )
            lines.append(f"- **Flashs par grade** : {flash_str}")

        # Route type weaknesses (top 3)
        weak = self.weakest_route_types[:3]
        if weak:
            lines.append("- **Types de bloc les moins envoyés** :")
            for s in weak:
                lines.append(f"  - {s}")

        # Route type strengths (top 3)
        strong = self.strongest_route_types[:3]
        if strong:
            lines.append("- **Types de bloc les mieux maîtrisés** :")
            for s in strong:
                lines.append(f"  - {s}")

        # Unsent boulders still open
        if self.unsent_by_grade:
            unsent_str = ", ".join(
                f"{g}: {c}" for g, c in sorted(self.unsent_by_grade.items())
            )
            lines.append(f"- **Blocs ouverts non envoyés** : {unsent_str}")

        # Recent ascents
        if self.recent_ascents:
            lines.append(f"- **Derniers envois** :")
            for a in self.recent_ascents[:max_recent]:
                lines.append(f"  - {a}")

        # Last sync
        if self.last_sync:
            for gym, ts in self.last_sync.items():
                date = ts[:10] if ts else "jamais"
                lines.append(f"- **Dernière sync {gym}** : {date}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StatsBuilder
# ---------------------------------------------------------------------------

class StatsBuilder:
    """
    Queries the SQLite database and builds a ClimbingStats object.
    Read-only — never writes to the DB.
    """

    def __init__(self, db_path: str | Path = "climbing.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def build(
        self,
        user_id: str,
        gyms: Optional[list[str]] = None,
        recent_n: int = 10,
    ) -> ClimbingStats:
        """
        Build a full ClimbingStats for user_id.
        If gyms is None, includes all gyms found in the DB.
        """
        if not gyms:
            gyms = self._all_gyms()

        stats = ClimbingStats(user_id=user_id, gyms=gyms)

        # Sent boulder IDs for this user
        sent_ids = self._sent_ids(user_id)
        flashed_ids = self._flashed_ids(user_id)

        # Volume
        stats.total_sends   = len(sent_ids)
        stats.total_flashes = len(flashed_ids)

        # Grade distributions
        stats.sends_by_grade   = self._grade_distribution(sent_ids)
        stats.flashes_by_grade = self._grade_distribution(flashed_ids)

        # Route type stats (per gym, then merge)
        stats.route_type_stats = self._route_type_stats(sent_ids, gyms)

        # Recent ascents
        stats.recent_ascents = self._recent_ascents(user_id, recent_n)

        # Unsent open boulders
        stats.unsent_by_grade = self._unsent_by_grade(sent_ids, gyms)

        # Last sync per gym
        stats.last_sync = {gym: self._last_sync(gym) for gym in gyms}

        log.debug("Stats built: %d sends, %d flashes for user %s",
                  stats.total_sends, stats.total_flashes, user_id)
        return stats

    # ------------------------------------------------------------------
    # Private query helpers
    # ------------------------------------------------------------------

    def _all_gyms(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT gym FROM boulders"
        ).fetchall()
        return [r["gym"] for r in rows]

    def _sent_ids(self, user_id: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT boulder_id FROM ascents WHERE user_id=? AND ascent_type='send'",
            (user_id,),
        ).fetchall()
        return {r["boulder_id"] for r in rows}

    def _flashed_ids(self, user_id: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT boulder_id FROM ascents WHERE user_id=? AND ascent_type='flash'",
            (user_id,),
        ).fetchall()
        return {r["boulder_id"] for r in rows}

    def _grade_distribution(self, boulder_ids: set[str]) -> dict[str, int]:
        if not boulder_ids:
            return {}
        placeholders = ",".join("?" * len(boulder_ids))
        rows = self._conn.execute(
            f"SELECT grade, COUNT(*) as c FROM boulders "
            f"WHERE boulder_id IN ({placeholders}) AND grade IS NOT NULL "
            f"GROUP BY grade ORDER BY grade",
            list(boulder_ids),
        ).fetchall()
        return {r["grade"]: r["c"] for r in rows}

    def _route_type_stats(
        self, sent_ids: set[str], gyms: list[str]
    ) -> list[RouteTypeStat]:
        """
        For each leaf route type, count how many boulders exist (total)
        and how many the user has sent.
        """
        gym_placeholders = ",".join("?" * len(gyms))
        all_boulders = self._conn.execute(
            f"SELECT boulder_id, route_types FROM boulders "
            f"WHERE gym IN ({gym_placeholders})",
            gyms,
        ).fetchall()

        # Accumulate counts
        total_by_type: dict[int, int] = {}
        sent_by_type:  dict[int, int] = {}

        for row in all_boulders:
            try:
                type_ids: list[int] = json.loads(row["route_types"] or "[]")
            except (json.JSONDecodeError, TypeError):
                continue

            # Only leaf types
            leaf_ids = [t for t in type_ids if t in ROUTE_TYPE_LEAF_IDS]
            for tid in leaf_ids:
                total_by_type[tid] = total_by_type.get(tid, 0) + 1
                if row["boulder_id"] in sent_ids:
                    sent_by_type[tid] = sent_by_type.get(tid, 0) + 1

        result = []
        for tid, total in total_by_type.items():
            rt = ROUTE_TYPES_BY_ID.get(tid)
            label = rt.label_fr if rt else f"Type {tid}"
            result.append(RouteTypeStat(
                type_id=tid,
                label_fr=label,
                sent=sent_by_type.get(tid, 0),
                total=total,
            ))

        return result

    def _recent_ascents(self, user_id: str, n: int) -> list[RecentAscent]:
        rows = self._conn.execute(
            """SELECT a.boulder_id, a.ascent_type, a.detected_at,
                      b.grade, b.route_types, b.gym
               FROM ascents a
               JOIN boulders b ON a.boulder_id = b.boulder_id
               WHERE a.user_id = ?
               ORDER BY a.detected_at DESC
               LIMIT ?""",
            (user_id, n),
        ).fetchall()

        ascents = []
        for r in rows:
            try:
                type_ids: list[int] = json.loads(r["route_types"] or "[]")
            except (json.JSONDecodeError, TypeError):
                type_ids = []

            labels = [
                ROUTE_TYPES_BY_ID[t].label_fr
                for t in type_ids
                if t in ROUTE_TYPES_BY_ID and not ROUTE_TYPES_BY_ID[t].is_category
            ]
            ascents.append(RecentAscent(
                boulder_id=r["boulder_id"],
                grade=r["grade"],
                ascent_type=r["ascent_type"],
                detected_at=r["detected_at"],
                route_type_labels=labels,
                gym=r["gym"],
            ))
        return ascents

    def _unsent_by_grade(
        self, sent_ids: set[str], gyms: list[str]
    ) -> dict[str, int]:
        """Open boulders (no closed_at) not yet sent by the user, grouped by grade."""
        gym_placeholders = ",".join("?" * len(gyms))
        rows = self._conn.execute(
            f"SELECT boulder_id, grade FROM boulders "
            f"WHERE gym IN ({gym_placeholders}) AND closed_at IS NULL AND grade IS NOT NULL",
            gyms,
        ).fetchall()

        counts: dict[str, int] = {}
        for r in rows:
            if r["boulder_id"] not in sent_ids:
                counts[r["grade"]] = counts.get(r["grade"], 0) + 1
        return counts

    def _last_sync(self, gym: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT synced_at FROM sync_log WHERE gym=? ORDER BY id DESC LIMIT 1",
            (gym,),
        ).fetchone()
        return row["synced_at"] if row else None

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Assembles the system prompt from profile, stats, and session mode.
    Keeping this separate makes it easy to iterate on prompt wording
    without touching conversation or DB logic.
    """

    # ------------------------------------------------------------------
    # Onboarding
    # ------------------------------------------------------------------

    ONBOARDING_PROMPT = """Tu es un coach escalade bienveillant et expérimenté \
qui réalise un entretien d'onboarding avec un nouveau grimpeur. \
Ton objectif est de collecter suffisamment d'informations pour construire \
son profil de coaching personnalisé.

Tu dois couvrir progressivement ces thèmes, dans un ordre naturel :
1. Profil physique (âge, taille, envergure, poids) — demande-les ensemble de façon légère
2. Historique de grimpe (depuis combien de temps, comment il a commencé)
3. Niveau actuel (grade redpoint, grade flash)
4. Styles préférés et points forts ressentis
5. Points faibles ressentis ou identifiés
6. Entraînement actuel (séances/semaine, durée, setup maison, autres activités)
7. Blessures actuelles ou passées importantes
8. Objectifs court terme et long terme

Règles importantes :
- Pose UNE seule question à la fois, ou un groupe logique de 2-3 questions courtes
- Reformule et valide ce que le grimpeur te dit avant de passer au thème suivant
- Adapte ton vocabulaire au niveau détecté
- Si une réponse est vague, creuse avec une question de suivi
- Reste conversationnel — mieux vaut un profil partiel honnête qu'un profil complet inventé
- Quand tu estimes avoir couvert les thèmes essentiels, termine par :
  "J'ai maintenant une bonne image de ton profil. Veux-tu ajouter autre chose avant que je le finalise ?"
"""

    # ------------------------------------------------------------------
    # Coaching base
    # ------------------------------------------------------------------

    COACHING_BASE = """Tu es un coach escalade expert, analytique et bienveillant. \
Tu accompagnes ce grimpeur dans sa progression en t'appuyant sur son profil \
et ses statistiques de grimpe réelles issues de la salle Arkose.

Tes réponses doivent être :
- Concrètes et actionnables (pas de généralités)
- Adaptées au niveau et aux objectifs du grimpeur
- Attentives aux blessures actives (ne jamais recommander ce qui est contre-indiqué)
- Capables de célébrer les progrès récents quand c'est pertinent

Lorsque tu analyses les statistiques, privilégie les types de voies sous-représentés \
dans les envois pour identifier les axes de progression prioritaires.
"""

    # ------------------------------------------------------------------
    # Extraction (used internally after onboarding)
    # ------------------------------------------------------------------

    EXTRACTION_PROMPT = """Tu es un extracteur de données structurées. \
On te donne la transcription d'un entretien entre un coach escalade et un grimpeur.

Ton unique rôle est d'extraire les informations mentionnées et de les retourner \
en JSON pur, sans aucun texte avant ou après, sans balises markdown.

Retourne UNIQUEMENT un objet JSON valide avec les champs suivants \
(omets les champs non mentionnés, ne les invente jamais) :

{
  "name": string,
  "age": int,
  "height_cm": int,
  "wingspan_cm": int,
  "weight_kg": float,
  "years_climbing": float,
  "started_at_grade": string,
  "current_redpoint_grade": string,
  "current_flash_grade": string,
  "preferred_styles": [string],
  "self_strengths": [string],
  "self_weaknesses": [string],
  "gym_sessions_per_week": int,
  "typical_session_duration_min": int,
  "other_activities": [string],
  "home_setup": [string],
  "injuries": [
    { "description": string, "active": bool, "avoid": [string] }
  ],
  "short_term_goals": [string],
  "long_term_goals": [string],
  "coach_tone": string,
  "coach_language": "fr",
  "focus_preference": string
}
"""

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def onboarding_system(self) -> str:
        return self.ONBOARDING_PROMPT

    def coaching_system(
        self,
        profile: ClimberProfile,
        stats: Optional[ClimbingStats] = None,
    ) -> str:
        parts = [self.COACHING_BASE]
        parts.append(profile.to_llm_context())
        if stats:
            parts.append(stats.to_llm_context())
        tone_instruction = self._tone_instruction(profile.coach_tone)
        if tone_instruction:
            parts.append(tone_instruction)
        parts.append(f"\nRéponds toujours en {self._lang_label(profile.coach_language)}.")
        return "\n\n".join(parts)

    def extraction_system(self) -> str:
        return self.EXTRACTION_PROMPT

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tone_instruction(self, tone: str) -> str:
        return {
            "direct":      "Sois direct et va à l'essentiel, sans fioritures.",
            "encouraging": "Adopte un ton encourageant, valorise les efforts.",
            "analytical":  "Adopte un ton analytique, appuie-toi sur les chiffres.",
            "concise":     "Sois très concis : maximum 3-4 phrases par réponse.",
        }.get(tone, "")

    def _lang_label(self, lang: str) -> str:
        return {"fr": "français", "en": "anglais"}.get(lang, lang)


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around an OpenAI-compatible chat API.
    Maintains message history for the current session.
    Model-agnostic: works with Mistral, OpenAI, Groq, etc.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._history: list[dict] = []    # user/assistant turns only
        self._system: str = ""

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def set_system(self, system_prompt: str) -> None:
        """Set (or replace) the system prompt for the current session."""
        self._system = system_prompt

    def reset_history(self) -> None:
        """Clear conversation history (start a new session)."""
        self._history = []

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Send a user message, get the assistant reply, update history.
        Returns the assistant's text response.
        """
        self._history.append({"role": "user", "content": user_message})

        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.extend(self._history)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        reply = response.choices[0].message.content
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def call_once(self, system: str, user_message: str) -> str:
        """
        Single stateless call — does NOT affect history.
        Used for extraction and other one-shot tasks.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.1,   # low temp for structured extraction
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Factory methods for common providers
    # ------------------------------------------------------------------

    @classmethod
    def mistral(cls, api_key: str, model: str = "mistral-small-latest", **kwargs) -> "LLMClient":
        return cls(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
            model=model,
            **kwargs,
        )

    @classmethod
    def openai(cls, api_key: str, model: str = "gpt-4o-mini", **kwargs) -> "LLMClient":
        return cls(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model,
            **kwargs,
        )

    @classmethod
    def groq(cls, api_key: str, model: str = "llama-3.1-8b-instant", **kwargs) -> "LLMClient":
        return cls(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model=model,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# ClimbingCoach
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
    ):
        self.llm = llm
        self.profile = profile
        self.profile_path = Path(profile_path)
        self.prompt_builder = PromptBuilder()
        self.mode: Optional[CoachMode] = None

        # StatsBuilder is optional — works without a DB
        self._stats_builder: Optional[StatsBuilder] = None
        if db_path and Path(db_path).exists():
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
        **llm_kwargs,
    ) -> "ClimbingCoach":
        llm = LLMClient.mistral(api_key=api_key, model=model, **llm_kwargs)
        return cls(llm=llm, profile=profile, db_path=db_path, profile_path=profile_path)

    @classmethod
    def from_openai(
        cls,
        api_key: str,
        model: str = "gpt-4o-mini",
        profile: Optional[ClimberProfile] = None,
        db_path: Optional[str | Path] = None,
        profile_path: str | Path = "climber_profile.json",
        **llm_kwargs,
    ) -> "ClimbingCoach":
        llm = LLMClient.openai(api_key=api_key, model=model, **llm_kwargs)
        return cls(llm=llm, profile=profile, db_path=db_path, profile_path=profile_path)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, mode: CoachMode) -> str:
        """
        Reset history, set mode, build system prompt, send opening message.
        Returns the coach's opening line.
        """
        self.mode = mode
        self.llm.reset_history()

        if mode == CoachMode.ONBOARDING:
            self.llm.set_system(self.prompt_builder.onboarding_system())
            opening_trigger = "Bonjour, je voudrais créer mon profil de coaching."

        elif mode == CoachMode.COACHING:
            if not self.profile:
                raise ValueError(
                    "No profile loaded. Run an onboarding session first, "
                    "or load a profile with ClimberProfile.load()."
                )
            stats = self._build_stats()
            system = self.prompt_builder.coaching_system(self.profile, stats)
            self.llm.set_system(system)
            name = self.profile.name or "grimpeur"
            opening_trigger = f"Bonjour coach, je suis {name}. Je suis prêt pour ma session."

        else:
            raise ValueError(f"Unknown mode: {mode}")

        reply = self.llm.chat(opening_trigger)
        return reply

    def chat(self, message: str) -> str:
        """Send a message and get the coach's reply."""
        if self.mode is None:
            raise RuntimeError("Call start_session() before chat().")
        return self.llm.chat(message)

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
        profile = ClimberProfile.from_dict(data)
        self.profile = profile
        log.info("Profile extracted: %r", profile)
        return profile

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
        return self._stats_builder.build(
            user_id=self.profile.sboulder_user_id,
            gyms=self.profile.gyms or None,
        )

    def _history_to_transcript(self) -> str:
        return "\n\n".join(
            f"{'Coach' if m['role'] == 'assistant' else 'Grimpeur'} : {m['content']}"
            for m in self.llm.history
        )

    def close(self):
        if self._stats_builder:
            self._stats_builder.close()