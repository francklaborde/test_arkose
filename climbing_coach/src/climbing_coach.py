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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .climber_profile import ClimberProfile, Injury
from .sboulder_collector import SBoulderCollector, ROUTE_TYPES_BY_ID, ROUTE_TYPE_LEAF_IDS, decode_grade, sboulder_url, encode_grade_level

log = logging.getLogger("climbing_coach")
log.addHandler(logging.NullHandler())


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

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
    holds_color: Optional[int] = None
    sents_count: int = 0        # gym-wide send count — rarity signal
    flashes_count: int = 0
    comments: list[str] = field(default_factory=list)
    # meaningful text comments from the community (empty-text filtered out)

    def __str__(self) -> str:
        kind = "flashé" if self.ascent_type == "flash" else "envoyé"
        label = decode_grade(self.holds_color, self.grade) if self.holds_color else (self.grade or "?")
        types = f" [{', '.join(self.route_type_labels)}]" if self.route_type_labels else ""
        rarity = f"réussie par {self.sents_count} grimpeur(s) en salle" if self.sents_count else "aucun envoi salle enregistré"
        return f"{label} {kind}{types}{rarity} ({self.detected_at[:10]})"


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

    # Unsent open boulders with community comments (potential projects)
    # list of (boulder_id, grade, sents_count, comment_texts)
    commented_projects: list[dict] = field(default_factory=list)

    # Individual open, unsent boulders with full detail (for direct recommendations)
    unsent_boulders: list[dict] = field(default_factory=list)

    current_level: Optional[str] = None
    current_flash_level: Optional[str] = None

    # Timestamp (ISO) of the single most recent ascent across all gyms —
    # used to surface how long it's been since the climber last climbed.
    last_ascent_at: Optional[str] = None

    @property
    def days_since_last_ascent(self) -> Optional[int]:
        """Whole days elapsed since the last recorded ascent, or None if unknown."""
        if not self.last_ascent_at:
            return None
        try:
            last = datetime.fromisoformat(self.last_ascent_at)
        except ValueError:
            return None
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last).days

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

        # Time since the last ascent — important signal for how the coach
        # should open the conversation (long break vs. active streak).
        days = self.days_since_last_ascent
        if days is not None:
            if days == 0:
                lines.append("- **Dernière séance** : aujourd'hui")
            elif days == 1:
                lines.append("- **Dernière séance** : hier")
            else:
                lines.append(f"- **Dernière séance** : il y a {days} jours")

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

        # # Unsent boulders still open
        # if self.unsent_by_grade:
        #     unsent_str = ", ".join(
        #         f"{g}: {c}" for g, c in sorted(self.unsent_by_grade.items())
        #     )
        #     lines.append(f"- **Blocs ouverts non envoyés** : {unsent_str}")

        # Individual unsent boulders with links (candidate list for recommendations)
        if self.unsent_boulders:
            lines.append("- **Voies ouvertes non envoyées (détail)** :")
            for b in self.unsent_boulders[:25]:
                types = f" [{', '.join(b['route_types'])}]" if b['route_types'] else ""
                rarity = f"réussie par {b['sents_count']} grimpeur(s) en salle" if b['sents_count'] else "aucun envoi salle enregistré"
                lines.append(f"  - {b['grade']}{types} — {b['url']} ({rarity})")
                # for c in b["comments"][:1]:  # 1 seul commentaire ici, info secondaire
                #     lines.append(f'    > commentaire communauté (info, pas un critère) : "{c}"')
    
        # Recent ascents with community context
        if self.recent_ascents:
            lines.append("- **Derniers envois** :")
            for a in self.recent_ascents[:max_recent]:
                lines.append(f"  - {a}")
                for comment in a.comments[:2]:  # max 2 comments per ascent
                    lines.append(f'    > commentaire communauté : "{comment}"')

        # # Commented projects (unsent boulders with community feedback)
        # if self.commented_projects:
        #     lines.append("- **Projets avec commentaires communautaires** :")
        #     for p in self.commented_projects[:5]:
        #         rarity = f"{p['sents_count']} envois salle"
        #         lines.append(f"  - {p['grade'] or '?'} ({rarity}) :")
        #         for c in p["comments"][:2]:
        #             lines.append(f'    > commentaire communauté : "{c}"')

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
        min_level: Optional[tuple[int, int]] = None,
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
        # _recent_ascents is ordered DESC, so index 0 is the single most
        # recent ascent regardless of the recent_n truncation.
        stats.last_ascent_at = stats.recent_ascents[0].detected_at if stats.recent_ascents else None

        # Unsent open boulders
        stats.unsent_by_grade = self._unsent_by_grade(sent_ids, gyms)

        # Unsent open boulders (individual, with url — new candidate for recommendations)
        stats.unsent_boulders = self._unsent_boulders(sent_ids, gyms, min_level=min_level)

        # Commented projects — unsent boulders that have community text comments
        stats.commented_projects = self._commented_projects(sent_ids, gyms)

        # Enrich recent ascents with send counts + comments
        for ascent in stats.recent_ascents:
            row = self._boulder_meta(ascent.boulder_id)
            if row:
                ascent.sents_count   = row["sents_count"] or 0
                ascent.flashes_count = row["flashes_count"] or 0
                ascent.holds_color   = row["holds_color"]
            ascent.comments = self._boulder_comments(ascent.boulder_id)

        stats.current_level       = self._current_level_arkose(user_id, "send")
        stats.current_flash_level = self._current_level_arkose(user_id, "flash")

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
            f"SELECT holds_color, grade, COUNT(*) as c FROM boulders "
            f"WHERE boulder_id IN ({placeholders}) AND grade IS NOT NULL AND holds_color IS NOT NULL "
            f"GROUP BY holds_color, grade ORDER BY holds_color, grade",
            list(boulder_ids),
        ).fetchall()
        return {
            decode_grade(r["holds_color"], r["grade"]): r["c"]
            for r in rows
        }

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
            f"SELECT boulder_id, holds_color, grade FROM boulders "
            f"WHERE gym IN ({gym_placeholders}) AND (closed_at IS NULL OR closed_at > ?)"
            f"AND grade IS NOT NULL AND holds_color IS NOT NULL",
            (*gyms, _now_iso()),
        ).fetchall()

        counts: dict[str, int] = {}
        for r in rows:
            if r["boulder_id"] not in sent_ids:
                label = decode_grade(r["holds_color"], r["grade"])
                counts[label] = counts.get(label, 0) + 1
        return counts

    def _boulder_meta(self, boulder_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT sents_count, flashes_count, holds_color FROM boulders WHERE boulder_id=?",
            (boulder_id,),
        ).fetchone()

    def _boulder_comments(self, boulder_id: str, max_comments: int = 5) -> list[str]:
        """Return meaningful text comments for a boulder, newest first."""
        tables = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='comments'"
        ).fetchone()
        if not tables:
            return []
        rows = self._conn.execute(
            """SELECT text FROM comments
               WHERE boulder_id=? AND text != '' AND text IS NOT NULL
               ORDER BY date DESC LIMIT ?""",
            (boulder_id, max_comments),
        ).fetchall()
        return [r["text"] for r in rows]

    def _commented_projects(
        self, sent_ids: set[str], gyms: list[str]
    ) -> list[dict]:
        """
        Unsent open boulders that have at least one meaningful community comment.
        Sorted by gym-wide send count desc (popular routes = likely good projects).
        """
        gym_placeholders = ",".join("?" * len(gyms))
        rows = self._conn.execute(
            f"""SELECT b.boulder_id, b.grade, b.sents_count, b.flashes_count
               FROM boulders b
               WHERE b.gym IN ({gym_placeholders})
                 AND (closed_at IS NULL OR closed_at > ?)
               ORDER BY b.sents_count DESC""",
            (*gyms, _now_iso()),
        ).fetchall()

        result = []
        for r in rows:
            if r["boulder_id"] in sent_ids:
                continue
            comments = self._boulder_comments(r["boulder_id"])
            if not comments:
                continue
            result.append({
                "boulder_id":   r["boulder_id"],
                "grade":        r["grade"],
                "sents_count":  r["sents_count"] or 0,
                "flashes_count": r["flashes_count"] or 0,
                "comments":     comments,
            })
        return result

    def _unsent_boulders(
        self, sent_ids: set[str], gyms: list[str],
        min_level: Optional[tuple[int, int]] = None,
        limit: int = 25
    ) -> list[dict]:
        gym_placeholders = ",".join("?" * len(gyms))
        rows = self._conn.execute(
            f"""SELECT b.boulder_id, b.gym, b.holds_color, b.grade,
                       b.route_types, b.sents_count, b.flashes_count
               FROM boulders b
               WHERE b.gym IN ({gym_placeholders})
                 AND (closed_at IS NULL OR closed_at > ?)
                 AND b.grade IS NOT NULL AND b.holds_color IS NOT NULL
               ORDER BY b.sents_count DESC""",
            (*gyms, _now_iso()),
        ).fetchall()

        result = []
        for r in rows:
            if r["boulder_id"] in sent_ids:
                continue
            if min_level:
                color, grade = r["holds_color"], int(r["grade"])
                if (color, grade) < min_level:
                    continue

            try:
                type_ids: list[int] = json.loads(r["route_types"] or "[]")
            except (json.JSONDecodeError, TypeError):
                type_ids = []
            labels = [
                ROUTE_TYPES_BY_ID[t].label_fr
                for t in type_ids
                if t in ROUTE_TYPES_BY_ID and not ROUTE_TYPES_BY_ID[t].is_category
            ]

            result.append({
                "boulder_id":    r["boulder_id"],
                "grade":         decode_grade(r["holds_color"], r["grade"]),
                "url":           sboulder_url(r["gym"], r["boulder_id"]),
                "route_types":   labels,
                "sents_count":   r["sents_count"] or 0,
                "flashes_count": r["flashes_count"] or 0,
            })
            if len(result) >= limit:
                break
        return result

    def _current_level_arkose(
        self, user_id: str, ascent_type: str, top_n: int = 5, months: int = 12
    ) -> Optional[str]:
        """
        Among the top_n hardest boulders of ascent_type ('send' or 'flash')
        completed in the last `months` months, return the easiest one (decoded).
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30 * months)).isoformat()
        rows = self._conn.execute(
            """SELECT b.holds_color, b.grade
               FROM ascents a
               JOIN boulders b ON a.boulder_id = b.boulder_id
               WHERE a.user_id = ? AND a.ascent_type = ? AND a.detected_at >= ?
                 AND b.grade IS NOT NULL AND b.holds_color IS NOT NULL""",
            (user_id, ascent_type, cutoff),
        ).fetchall()

        if not rows:
            return None

        graded = sorted(
            [(r["holds_color"], int(r["grade"])) for r in rows],
            reverse=True,
        )
        top = graded[:top_n]
        easiest_color, easiest_grade = top[-1]
        return decode_grade(easiest_color, str(easiest_grade))
    
    def _last_sync(self, gym: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT synced_at FROM sync_log WHERE gym=? ORDER BY id DESC LIMIT 1",
            (gym,),
        ).fetchone()
        return row["synced_at"] if row else None

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Session plan (warm-up / exercises / rest, generated from the conversation)
# ---------------------------------------------------------------------------

@dataclass
class PlanBlock:
    """A single item within a session plan (warmup exercise, main block, cooldown)."""
    name: str
    sets: Optional[int] = None
    reps: Optional[str] = None          # free text, e.g. "8-10" or "max"
    duration_min: Optional[int] = None
    rest_sec: Optional[int] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sets": self.sets,
            "reps": self.reps,
            "duration_min": self.duration_min,
            "rest_sec": self.rest_sec,
            "notes": self.notes,
        }


@dataclass
class SessionPlan:
    """A structured training session plan, extracted on demand from the conversation."""
    title: str
    warmup: list[PlanBlock] = field(default_factory=list)
    blocks: list[PlanBlock] = field(default_factory=list)
    cooldown: list[PlanBlock] = field(default_factory=list)
    total_duration_min: Optional[int] = None
    equipment_used: list[str] = field(default_factory=list)
    # Set when this plan is one session of a multi-session program
    # (e.g. a 3-session build-up toward a specific project).
    program_title: Optional[str] = None
    session_index: Optional[int] = None
    session_total: Optional[int] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    generated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "program_title": self.program_title,
            "session_index": self.session_index,
            "session_total": self.session_total,
            "warmup": [b.to_dict() for b in self.warmup],
            "blocks": [b.to_dict() for b in self.blocks],
            "cooldown": [b.to_dict() for b in self.cooldown],
            "total_duration_min": self.total_duration_min,
            "equipment_used": self.equipment_used,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionPlan":
        def _blocks(key: str) -> list[PlanBlock]:
            return [PlanBlock(**b) for b in (data.get(key) or [])]
        return cls(
            title=data.get("title") or "Séance",
            program_title=data.get("program_title") or None,
            session_index=data.get("session_index"),
            session_total=data.get("session_total"),
            warmup=_blocks("warmup"),
            blocks=_blocks("blocks"),
            cooldown=_blocks("cooldown"),
            total_duration_min=data.get("total_duration_min"),
            equipment_used=data.get("equipment_used") or [],
        )


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
1. Profil physique (sexe, âge, taille, envergure, poids) — demande-les ensemble de façon légère
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

## Ton et posture
- Tu es un coach, pas un assistant — tu guides, tu ne demandes pas la permission
- Ton premier réflexe est de regarder ce que le grimpeur a réussi depuis la dernière fois \
et de le souligner chaleureusement, même brièvement. Un envoi récent mérite d'être reconnu.
- Si les statistiques montrent des nouveaux envois ou flashs depuis la dernière sync, \
commence toujours par les mentionner naturellement avant d'aller plus loin.
- Reste humain : une phrase d'accroche chaleureuse vaut mieux qu'un rapport de données.

## Réponses
- Concrètes et actionnables — pas de généralités
- Adaptées au niveau et aux objectifs du grimpeur
- Toujours attentives aux blessures actives (ne jamais recommander ce qui est contre-indiqué)
- Proportionnées à la question : une question simple appelle une réponse courte

## Analyse des statistiques
- Utilise les types de voies sous-représentés dans les envois pour identifier \
les axes de progression prioritaires
- Tiens compte du nombre d'envois salle sur chaque bloc : un bloc envoyé par peu de grimpeurs \
est un vrai exploit, dis-le
- Les commentaires affichés viennent TOUJOURS de la communauté, jamais du grimpeur \
lui-même. Ne jamais dire "tu soulignais" ou "comme tu l'as dit" à propos d'un commentaire — \
dis plutôt "un grimpeur a noté que..." ou "la communauté mentionne...".

## Message d'accueil
Quand le grimpeur arrive en session, commence par un recap court mais précis de sa situation \
récente (2-3 phrases). Sois FACTUEL et CHIFFRÉ, jamais vague :
- Cite le nombre exact d'envois et leurs cotations précises (ex: "3 voies noires\
(sous entendu couleur difficulté noire) 5 barres (sous entendu 5 barres sur 5 selon les cotations arkose), 1 rouge 3 barres", \
pas "quelques 5 et quelques 3")
- Cite une période précise si elle est disponible dans les stats (ex: "cette semaine", \
"depuis ta dernière session du [date]"), jamais "ces derniers temps"
- Nomme explicitement le ou les types de voie concernés par la tendance (ex: "arquée", \
"tendu", "gainage"), pas "tes points forts" de façon générique
- Si un envoi est rare ou notable (peu de grimpeurs l'ont envoyé, flash sur un bloc dur), \
dis-le avec le chiffre exact si disponible
- Si les statistiques indiquent que la dernière séance remonte à plusieurs semaines, \
mentionne-le avec bienveillance et sans culpabilisation (ex: reprise en douceur), \
jamais sur un ton de reproche. À l'inverse, une reprise rapprochée mérite d'être valorisée.
Interdiction stricte d'inventer ou d'arrondir des chiffres non présents dans les stats : \
si une donnée précise manque, formule la phrase sans elle plutôt que d'être approximatif.
Termine en évoquant, sans les détailler, qu'il y a des pistes de travail possibles pour la suite \
(une phrase suffit, du type "on pourrait creuser deux ou trois pistes aujourd'hui"). \
Ne développe jamais ces pistes toi-même à ce stade — laisse le grimpeur choisir la direction \
qu'il veut prendre.
Évite les formulations "soit... soit..." ou les listes à choix multiples déguisées en phrase. \
Parle comme un humain qui a vraiment regardé les stats, pas comme un menu. Une seule piste \
suggérée avec conviction vaut mieux que deux options neutres jetées côte à côte. \
Pas plus d'un emoji dans tout le message, et seulement s'il apporte vraiment quelque chose.
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
  "sex": "homme" | "femme" | "autre",
  "age": int,
  "height_cm": int,
  "wingspan_cm": int,
  "weight_kg": float,
  "years_climbing": float,
  "started_at_level": string,
  "current_redpoint_grade_fr": string,
  "current_flash_grade_fr": string,
  "current_redpoint_level_arkose": string,
  "current_flash_level_arkose": string,
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
    # Session plan extraction (used internally after each coaching turn)
    # ------------------------------------------------------------------

    PLAN_PROMPT = """Tu es un extracteur qui transforme une conversation de coaching \
escalade en un ou plusieurs plans de séance structurés.

Un "plan" peut être :
- une séance ponctuelle (échauffement / bloc principal / retour au calme)
- une séance faisant partie d'un programme à plusieurs séances (ex: préparer \
un projet précis sur 3 séances) — dans ce cas renseigne "program_title" \
(même titre de programme pour toutes les séances qui le composent), \
"session_index" et "session_total"
- une routine complémentaire à faire en parallèle de l'escalade (ex: tractions, \
gainage à la maison) — dans ce cas laisse "program_title" à null

La liste des plans déjà enregistrés pour ce grimpeur est fournie dans le \
contexte ci-dessous. Ne les ré-extrait JAMAIS. N'extrait que les plans NOUVEAUX \
qui apparaissent dans la conversation ci-dessous et qui ne sont pas déjà dans \
cette liste.

Si aucun plan nouveau n'apparaît, retourne exactement : {"plans": []}

Sinon, retourne UNIQUEMENT un objet JSON valide, sans texte avant/après, sans \
balises markdown, au format :

{
  "plans": [
    {
      "title": string,
      "program_title": string ou null,
      "session_index": int ou null,
      "session_total": int ou null,
      "warmup": [ { "name": string, "duration_min": int, "notes": string } ],
      "blocks": [ { "name": string, "sets": int, "reps": string, "duration_min": int, \
"rest_sec": int, "notes": string } ],
      "cooldown": [ { "name": string, "duration_min": int, "notes": string } ],
      "total_duration_min": int,
      "equipment_used": [string]
    }
  ]
}

Règles :
- Si la conversation décrit un programme sur plusieurs séances en une fois, \
extrait TOUTES les séances du programme d'un coup (un objet par séance dans "plans").
- Adapte les exercices au matériel réellement disponible (donné dans le contexte \
ci-dessous). Sans accès salle : uniquement préparation physique, mobilité, \
doigts (si poutre disponible), gainage, etc. — jamais de bloc ni de voie.
- Respecte strictement les blessures actives listées dans le profil.
- N'invente pas de contraintes non mentionnées.
- Chaque champ numérique manquant doit être omis plutôt qu'inventé au hasard.
- "reps" est une chaîne libre (ex: "8-10", "max", "3x échec").
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

    def plan_system(
        self,
        profile: Optional[ClimberProfile] = None,
        known_plans: Optional[list[str]] = None,
    ) -> str:
        parts = [self.PLAN_PROMPT]
        if profile:
            parts.append(profile.to_llm_context())
        if known_plans:
            listing = "\n".join(f"- {t}" for t in known_plans)
        else:
            listing = "Aucun pour l'instant."
        parts.append(f"## Plans déjà enregistrés (ne pas les ré-extraire)\n{listing}")
        return "\n\n".join(parts)

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

    def chat_stream(self, user_message: str):
        """
        Send a user message, yield the assistant reply incrementally as it is
        generated, and update history once the stream completes.
        """
        self._history.append({"role": "user", "content": user_message})

        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.extend(self._history)

        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            stream=True,
        )

        chunks: list[str] = []
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                chunks.append(delta)
                yield delta

        self._history.append({"role": "assistant", "content": "".join(chunks)})

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
    def mistral(cls, api_key: str, model: str = "mistral-large-latest", **kwargs) -> "LLMClient":
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
            self.llm.set_system(self.prompt_builder.onboarding_system())
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