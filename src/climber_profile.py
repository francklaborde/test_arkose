"""
climber_profile.py
------------------
Static/semi-static context about a climber, stored as JSON and injected
into every LLM prompt.

Usage
-----
# Create a blank template
profile = ClimberProfile()
profile.save("franck.json")

# Edit franck.json manually, then load it
profile = ClimberProfile.load("franck.json")
print(profile.to_llm_context())
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("sboulder.profile")
log.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Injury
# ---------------------------------------------------------------------------

@dataclass
class Injury:
    """A current or past injury relevant to training."""
    description: str
    # e.g. "Left A2 pulley partial tear", "Légère inflammation coude droit"
    active: bool = True
    # True = still limiting training; False = healed but worth keeping for history
    avoid: list[str] = field(default_factory=list)
    # Hold types / movements to avoid, e.g. ["crimps", "deadhangs"]

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "active": self.active,
            "avoid": self.avoid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Injury":
        return cls(
            description=d["description"],
            active=d.get("active", True),
            avoid=d.get("avoid", []),
        )


# ---------------------------------------------------------------------------
# ClimberProfile
# ---------------------------------------------------------------------------

@dataclass
class ClimberProfile:
    """
    All static/semi-static context about a climber.
    Stored as a JSON file; edit it directly — no code changes needed.

    Every field is optional except sboulder_user_id (needed to match DB ascents).
    to_llm_context() only renders non-empty fields, so partial profiles work fine.
    """

    # ---- Identity --------------------------------------------------------
    sboulder_user_id: str = ""
    # Meteor user ID, e.g. "qQFsxQKYvqRqYJNKa" — links to DB ascents
    name: str = ""
    gyms: list[str] = field(default_factory=list)
    # Gym slugs to sync by default, e.g. ["arkose/montmartre", "arkose/nation"]

    # ---- Physical profile ------------------------------------------------
    age: Optional[int] = None
    height_cm: Optional[int] = None
    # Already available in the sboulder users collection
    wingspan_cm: Optional[int] = None
    # Ape index = wingspan - height (positive = long arms, advantage on big moves)
    weight_kg: Optional[float] = None

    # ---- Climbing history ------------------------------------------------
    years_climbing: Optional[float] = None
    started_at_grade: Optional[str] = None
    # Grade (Fontainebleau) when they first started, e.g. "5b"
    current_redpoint_grade: Optional[str] = None
    # Max grade sent after multiple attempts
    current_flash_grade: Optional[str] = None
    # Max grade sent first try
    preferred_styles: list[str] = field(default_factory=list)
    # Free text, used verbatim in the prompt
    # e.g. ["powerful", "dynamic", "compression", "slab"]

    # ---- Self-assessed strengths / weaknesses ----------------------------
    # These complement the data-driven analysis from routeTypes stats.
    self_strengths: list[str] = field(default_factory=list)
    # e.g. ["contact strength", "reading sequences", "slab balance"]
    self_weaknesses: list[str] = field(default_factory=list)
    # e.g. ["coordination moves", "heel hooks", "maintaining tension on volumes"]

    # ---- Training schedule -----------------------------------------------
    gym_sessions_per_week: Optional[int] = None
    other_activities: list[str] = field(default_factory=list)
    # e.g. ["fingerboard 2x/semaine", "yoga 1x/semaine", "course à pied"]
    typical_session_duration_min: Optional[int] = None
    home_setup: list[str] = field(default_factory=list)
    # e.g. ["fingerboard", "campus board"] — empty list if no home setup

    # ---- Injuries --------------------------------------------------------
    injuries: list[Injury] = field(default_factory=list)

    # ---- Goals -----------------------------------------------------------
    short_term_goals: list[str] = field(default_factory=list)
    # e.g. ["Enchaîner le 7b en zone 3 avant sa fermeture"]
    long_term_goals: list[str] = field(default_factory=list)
    # e.g. ["Atteindre 7c d'ici fin d'année", "Participer au circuit local"]

    # ---- Coaching preferences --------------------------------------------
    coach_tone: str = "direct"
    # "direct" | "encouraging" | "analytical" | "concise"
    coach_language: str = "fr"
    # Language the LLM should reply in — "fr" or "en"
    focus_preference: str = "weaknesses"
    # "weaknesses" | "strengths" | "balanced" | "ask_each_time"

    # ---- Free-form notes -------------------------------------------------
    notes: str = ""
    # Anything that doesn't fit the fields above:
    # past injury context, motivation, scheduling constraints, etc.

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def ape_index(self) -> Optional[int]:
        """wingspan - height. Positive = long arms (advantage on big moves)."""
        if self.height_cm and self.wingspan_cm:
            return self.wingspan_cm - self.height_cm
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "sboulder_user_id": self.sboulder_user_id,
            "name": self.name,
            "gyms": self.gyms,
            "age": self.age,
            "height_cm": self.height_cm,
            "wingspan_cm": self.wingspan_cm,
            "weight_kg": self.weight_kg,
            "years_climbing": self.years_climbing,
            "started_at_grade": self.started_at_grade,
            "current_redpoint_grade": self.current_redpoint_grade,
            "current_flash_grade": self.current_flash_grade,
            "preferred_styles": self.preferred_styles,
            "self_strengths": self.self_strengths,
            "self_weaknesses": self.self_weaknesses,
            "gym_sessions_per_week": self.gym_sessions_per_week,
            "other_activities": self.other_activities,
            "typical_session_duration_min": self.typical_session_duration_min,
            "home_setup": self.home_setup,
            "injuries": [i.to_dict() for i in self.injuries],
            "short_term_goals": self.short_term_goals,
            "long_term_goals": self.long_term_goals,
            "coach_tone": self.coach_tone,
            "coach_language": self.coach_language,
            "focus_preference": self.focus_preference,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClimberProfile":
        return cls(
            sboulder_user_id=d.get("sboulder_user_id", ""),
            name=d.get("name", ""),
            gyms=d.get("gyms", []),
            age=d.get("age"),
            height_cm=d.get("height_cm"),
            wingspan_cm=d.get("wingspan_cm"),
            weight_kg=d.get("weight_kg"),
            years_climbing=d.get("years_climbing"),
            started_at_grade=d.get("started_at_grade"),
            current_redpoint_grade=d.get("current_redpoint_grade"),
            current_flash_grade=d.get("current_flash_grade"),
            preferred_styles=d.get("preferred_styles", []),
            self_strengths=d.get("self_strengths", []),
            self_weaknesses=d.get("self_weaknesses", []),
            gym_sessions_per_week=d.get("gym_sessions_per_week"),
            other_activities=d.get("other_activities", []),
            typical_session_duration_min=d.get("typical_session_duration_min"),
            home_setup=d.get("home_setup", []),
            injuries=[Injury.from_dict(i) for i in d.get("injuries", [])],
            short_term_goals=d.get("short_term_goals", []),
            long_term_goals=d.get("long_term_goals", []),
            coach_tone=d.get("coach_tone", "direct"),
            coach_language=d.get("coach_language", "fr"),
            focus_preference=d.get("focus_preference", "weaknesses"),
            notes=d.get("notes", ""),
        )

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path = "climber_profile.json") -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        log.info("Profile saved → %s", path.resolve())

    @classmethod
    def load(cls, path: str | Path = "climber_profile.json") -> "ClimberProfile":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Profile not found at {path}.\n"
                "Run ClimberProfile().save('<path>') to create a blank template."
            )
        profile = cls.from_dict(json.loads(path.read_text()))
        log.info("Profile loaded ← %s", path.resolve())
        return profile

    @classmethod
    def load_or_create(cls, path: str | Path = "climber_profile.json") -> "ClimberProfile":
        """Load existing profile, or write a blank template if the file is missing."""
        path = Path(path)
        if path.exists():
            return cls.load(path)
        log.info("No profile found at %s — creating blank template", path)
        profile = cls()
        profile.save(path)
        return profile

    # ------------------------------------------------------------------
    # LLM context rendering
    # ------------------------------------------------------------------

    def to_llm_context(self) -> str:
        """
        Render the profile as a compact markdown block ready to be injected
        into an LLM system prompt.

        Only non-empty / non-None fields are included to keep token count low.
        """
        lines: list[str] = ["## Profil grimpeur"]

        def _add(label: str, value) -> None:
            if value is not None and value != "" and value != []:
                lines.append(f"- **{label}** : {value}")

        # Identity
        _add("Nom", self.name or None)
        _add("Salles", ", ".join(self.gyms) if self.gyms else None)

        # Physical
        phys_parts = []
        if self.age:
            phys_parts.append(f"{self.age} ans")
        if self.height_cm:
            phys_parts.append(f"{self.height_cm} cm")
        if self.weight_kg:
            phys_parts.append(f"{self.weight_kg} kg")
        if self.ape_index is not None:
            sign = "+" if self.ape_index >= 0 else ""
            phys_parts.append(f"envergure {sign}{self.ape_index} cm")
        _add("Physique", ", ".join(phys_parts) if phys_parts else None)

        # Climbing level
        _add("Années de grimpe", self.years_climbing)
        _add("Grade de départ", self.started_at_grade)
        level_parts = []
        if self.current_redpoint_grade:
            level_parts.append(f"redpoint {self.current_redpoint_grade}")
        if self.current_flash_grade:
            level_parts.append(f"flash {self.current_flash_grade}")
        _add("Niveau actuel", ", ".join(level_parts) if level_parts else None)
        _add("Styles préférés", ", ".join(self.preferred_styles) if self.preferred_styles else None)

        # Strengths / weaknesses
        _add("Points forts (auto)", ", ".join(self.self_strengths) if self.self_strengths else None)
        _add("Points faibles (auto)", ", ".join(self.self_weaknesses) if self.self_weaknesses else None)

        # Training
        sched_parts = []
        if self.gym_sessions_per_week:
            sched_parts.append(f"{self.gym_sessions_per_week} séances/semaine")
        if self.typical_session_duration_min:
            sched_parts.append(f"~{self.typical_session_duration_min} min/séance")
        _add("Entraînement", ", ".join(sched_parts) if sched_parts else None)
        _add("Autres activités", ", ".join(self.other_activities) if self.other_activities else None)
        _add("Setup maison", ", ".join(self.home_setup) if self.home_setup else None)

        # Injuries (active only)
        active = [i for i in self.injuries if i.active]
        if active:
            lines.append("- **Blessures actives** :")
            for inj in active:
                avoid_str = f" (éviter : {', '.join(inj.avoid)})" if inj.avoid else ""
                lines.append(f"  - {inj.description}{avoid_str}")

        # Goals
        if self.short_term_goals:
            lines.append("- **Objectifs court terme** :")
            for g in self.short_term_goals:
                lines.append(f"  - {g}")
        if self.long_term_goals:
            lines.append("- **Objectifs long terme** :")
            for g in self.long_term_goals:
                lines.append(f"  - {g}")

        # Coaching prefs
        _add("Ton souhaité", self.coach_tone)
        _add("Focus", self.focus_preference)
        _add("Langue", self.coach_language)

        # Free notes
        _add("Notes", self.notes or None)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ClimberProfile(name={self.name!r}, "
            f"redpoint={self.current_redpoint_grade!r}, "
            f"gyms={self.gyms})"
        )


# ---------------------------------------------------------------------------
# CLI — python climber_profile.py [path]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage a climber profile JSON")
    parser.add_argument("path", nargs="?", default="climber_profile.json",
                        help="Path to the JSON profile file")
    parser.add_argument("--show", action="store_true",
                        help="Print the LLM context block")
    args = parser.parse_args()

    profile = ClimberProfile.load_or_create(args.path)
    print(repr(profile))
    if args.show:
        print()
        print(profile.to_llm_context())