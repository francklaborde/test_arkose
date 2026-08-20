"""
climbing_stats.py
------------------
Queries SQLite (boulders/ascents/sync_log) and returns structured
ClimbingStats. Split out of climbing_coach.py so DB/stats logic can
be iterated on without touching prompts or the LLM client.
"""

from __future__ import annotations

import logging
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ._utils import _now_iso
from .sboulder_collector import (
    ROUTE_TYPES_BY_ID,
    ROUTE_TYPE_LEAF_IDS,
    decode_grade,
    sboulder_url,
)

log = logging.getLogger("climbing_coach")
log.addHandler(logging.NullHandler())

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

    def to_llm_context(self, max_recent: int = 10) -> str:
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


