"""
SBoulderCollector
-----------------
Connects to sboulder.com via WebSocket (DDP/SockJS), fetches all boulders
for a given gym, and persists them to a local SQLite database.

Features:
  - Full boulder fetch with pagination guard (configurable limit)
  - Snapshot-aware: each sync run is timestamped; boulders are never
    deleted, only marked closed when `closedAt` is set
  - Detects NEW boulders (not seen in previous sync) and NEWLY COMPLETED
    ones (your userId appeared in sentsList / flashesList since last sync)
  - Exposes a clean summary after each sync

Dependencies:
    pip install websocket-client
"""

import json
import random
import sqlite3
import string
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import websocket  # websocket-client

from climber_profile import Injury, ClimberProfile  # noqa: F401

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# We do NOT call basicConfig here — that's the entry point's job.
# Instead we add a NullHandler so log calls are silently dropped if the
# caller hasn't configured logging (standard library best practice).
log = logging.getLogger("sboulder")
log.addHandler(logging.NullHandler())


def setup_logging(level: int = logging.INFO) -> None:
    """
    Call this once from your server startup or notebook to activate logs
    for all arkose modules (sboulder, climbing_coach, sboulder.profile).

    Works correctly whether uvicorn/Jupyter has already configured the
    root logger or not — attaches directly to each module logger so it
    never fights with the host framework.

    Example
    -------
    from sboulder_collector import setup_logging
    setup_logging()               # INFO by default
    setup_logging(logging.DEBUG)  # verbose, shows per-boulder inserts
    """
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)

    # Apply to every arkose logger explicitly — this works even when
    # uvicorn or Jupyter already owns the root logger.
    for name in ("sboulder", "sboulder.profile", "climbing_coach"):
        logger = logging.getLogger(name)
        # Remove the NullHandler that was added at import time
        logger.handlers = [h for h in logger.handlers
                           if not isinstance(h, logging.NullHandler)]
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False   # don't double-print via the root logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOCKJS_INFO_URL = "https://www.sboulder.com/sockjs/info"
WS_URL_TEMPLATE = "wss://www.sboulder.com/sockjs/{server}/{session}/websocket"
ORIGIN_HEADER = "https://www.sboulder.com"

# Route type definitions, sourced directly from _gyms.info → routeTypes payload.
# Structure per entry: id -> (label_fr, label_en, parent_id | None, children_ids)
#
# The raw API payload format is:
#   [id, [label, visible, usedForHand, usedForFoot, usedForPhysical, '', parent_id_or_False, [children...]?]]
#
# Categories (no parent): 30 (hand grips), 32 (foot grips), 41 (physical)
# Standalone (no parent, no children): 1, 31, 4, 39, 42

@dataclass(frozen=True)
class RouteTypeInfo:
    id: int
    label_fr: str
    label_en: str
    parent_id: Optional[int]        # None = top-level category or standalone
    children_ids: list[int] = field(default_factory=list)

    @property
    def is_category(self) -> bool:
        return bool(self.children_ids)


ROUTE_TYPE_DEFINITIONS: list[RouteTypeInfo] = [
    # ---- Hand grip category ----
    RouteTypeInfo(30, "Tenue de prise de main",          "Hand grip (category)",          None,  [28, 13, 11, 18, 34]),
    RouteTypeInfo(28, "Arquée/Semi-arquée/Tendu",        "Crimp/Half-crimp/Open hand",    30,    []),
    RouteTypeInfo(13, "Pince",                           "Pinch",                         30,    []),
    RouteTypeInfo(11, "Plat sans pouce",                 "Flat hold, no thumb",           30,    []),
    RouteTypeInfo(18, "Angle de volume",                 "Volume edge",                   30,    []),
    RouteTypeInfo(34, "Mono/Bidoigt/Coincement/Pommeau", "Mono/Two-finger/Jam/Pommel",    30,    []),
    # ---- Foot grip category ----
    RouteTypeInfo(32, "Tenue de prise de pied",          "Foot grip (category)",          None,  [40, 35, 36]),
    RouteTypeInfo(40, "Petit pied",                      "Small foothold",                32,    []),
    RouteTypeInfo(35, "Adhérence",                       "Friction/Smearing",             32,    []),
    RouteTypeInfo(36, "Crochet talon/pointe",            "Heel/Toe hook",                 32,    []),
    # ---- Physical category ----
    RouteTypeInfo(41, "Physique",                        "Physical (category)",           None,  [2, 37, 38]),
    RouteTypeInfo(2,  "Haut du corps",                   "Upper body",                    41,    []),
    RouteTypeInfo(37, "Bas du corps",                    "Lower body",                    41,    []),
    RouteTypeInfo(38, "Gainage",                         "Core",                          41,    []),
    # ---- Standalone types (no parent category) ----
    RouteTypeInfo(1,  "Placement/Sensation",             "Body positioning/feel",         None,  []),
    RouteTypeInfo(31, "Souplesse",                       "Flexibility",                   None,  []),
    RouteTypeInfo(4,  "Complexe/À méthode",              "Complex/Beta-intensive",        None,  []),
    RouteTypeInfo(39, "Engagement/Prise de décision",    "Commitment/Decision-making",    None,  []),
    RouteTypeInfo(42, "Coordination/Synchronisation",    "Coordination/Synchronization",  None,  []),
]

# Fast lookup by ID
ROUTE_TYPES_BY_ID: dict[int, RouteTypeInfo] = {rt.id: rt for rt in ROUTE_TYPE_DEFINITIONS}

# Convenience: flat label dicts (used for quick display / LLM prompts)
ROUTE_TYPES_FR: dict[int, str] = {rt.id: rt.label_fr for rt in ROUTE_TYPE_DEFINITIONS}
ROUTE_TYPES_EN: dict[int, str] = {rt.id: rt.label_en for rt in ROUTE_TYPE_DEFINITIONS}

# IDs that are leaf types (not categories themselves) — these are the ones
# worth aggregating for coaching analysis
ROUTE_TYPE_LEAF_IDS: set[int] = {
    rt.id for rt in ROUTE_TYPE_DEFINITIONS if not rt.is_category
}


def decode_route_types(ids: list[int], lang: str = "en") -> list[str]:
    """Return human-readable labels for a list of route type IDs."""
    lookup = ROUTE_TYPES_FR if lang == "fr" else ROUTE_TYPES_EN
    return [lookup.get(i, f"Unknown({i})") for i in ids]


def route_type_category(type_id: int) -> Optional["RouteTypeInfo"]:
    """Return the parent category for a leaf type ID, or None if already a category."""
    rt = ROUTE_TYPES_BY_ID.get(type_id)
    if rt is None or rt.parent_id is None:
        return None
    return ROUTE_TYPES_BY_ID.get(rt.parent_id)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _random_id(length: int = 17) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def _parse_date(raw) -> Optional[str]:
    """Turn a Meteor $date (ms epoch) or ISO string into an ISO-8601 string."""
    if raw is None:
        return None
    if isinstance(raw, dict) and "$date" in raw:
        ts = raw["$date"] / 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    if isinstance(raw, str):
        return raw
    return None


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Dataclass for a single boulder (typed, easy to inspect)
# ---------------------------------------------------------------------------

@dataclass
class Boulder:
    boulder_id: str                    # Meteor document _id
    gym: str
    grade: Optional[str] = None
    label: Optional[int] = None
    boulder_num: Optional[int] = None
    zone: Optional[int] = None
    holds_color: Optional[int] = None
    route_setter: list[str] = field(default_factory=list)
    route_types: list[int] = field(default_factory=list)
    created_at: Optional[str] = None
    closed_at: Optional[str] = None
    sents_count: int = 0
    flashes_count: int = 0
    sents_list: list[str] = field(default_factory=list)
    flashes_list: list[str] = field(default_factory=list)

    @classmethod
    def from_ddp(cls, doc_id: str, fields: dict) -> "Boulder":
        if fields.get("closedAt") is not None:
            print(f"DEBUG closedAt for {doc_id}: {fields.get('closedAt')!r}")
        return cls(
            boulder_id=doc_id,
            gym=fields.get("gym", ""),
            grade=fields.get("grade"),
            label=fields.get("label"),
            boulder_num=fields.get("boulderNum"),
            zone=fields.get("zone"),
            holds_color=fields.get("holdsColor"),
            route_setter=fields.get("routeSetter") or [],
            route_types=fields.get("routeTypes") or [],
            created_at=_parse_date(fields.get("createdAt")),
            closed_at=_parse_date(fields.get("closedAt")),
            sents_count=fields.get("sentsCount") or 0,
            flashes_count=fields.get("flashesCount") or 0,
            sents_list=fields.get("sentsList") or [],
            flashes_list=fields.get("flashesList") or [],
        )

    def decode_route_types(self, lang: str = "en") -> list[str]:
        return decode_route_types(self.route_types, lang=lang)

    def leaf_route_types(self) -> list["RouteTypeInfo"]:
        """Return only non-category route type entries (useful for coaching stats)."""
        return [
            ROUTE_TYPES_BY_ID[i] for i in self.route_types
            if i in ROUTE_TYPES_BY_ID and not ROUTE_TYPES_BY_ID[i].is_category
        ]

    @property
    def is_currently_closed(self) -> bool:
        """True only if closed_at is set AND already in the past."""
        if not self.closed_at:
            return False
        return self.closed_at <= _now_iso()

    def is_sent_by(self, user_id: str) -> bool:
        return user_id in self.sents_list

    def is_flashed_by(self, user_id: str) -> bool:
        return user_id in self.flashes_list


# ---------------------------------------------------------------------------
# BoulderComment — a single community comment on a boulder
# ---------------------------------------------------------------------------

@dataclass
class BoulderComment:
    comment_id: str
    boulder_id: str
    user_id: str
    user_name: str
    user_level: Optional[int]   # label score at the gym (e.g. 7 = grade 7 climber)
    text: str
    has_video: bool
    date: Optional[str]

    @classmethod
    def from_ddp(cls, comment_id: str, fields: dict) -> "BoulderComment":
        # Parse nested userProfile Astronomy object
        user_name = ""
        user_level = None
        try:
            up = fields.get("userProfile", {})
            values_str = up.get("$value", {}).get("values", "{}")
            values = json.loads(values_str)
            user_name = values.get("name", "")
            # scores is a dict of gym_slug -> {label: int}
            scores = values.get("scores", {})
            gym = fields.get("boulder", {}).get("gym", "")
            if gym and gym in scores:
                user_level = scores[gym].get("label")
        except Exception:
            pass

        return cls(
            comment_id=comment_id,
            boulder_id=fields.get("boulderId", ""),
            user_id=fields.get("userId", ""),
            user_name=user_name,
            user_level=user_level,
            text=fields.get("text", "").strip(),
            has_video=bool(fields.get("videoId")),
            date=_parse_date(fields.get("date")),
        )

    @property
    def is_meaningful(self) -> bool:
        """True if the comment has actual text (not just a video post)."""
        return bool(self.text)


# ---------------------------------------------------------------------------
# SyncResult: what changed during one run
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    gym: str
    synced_at: str
    total_fetched: int = 0
    new_boulders: list[Boulder] = field(default_factory=list)
    newly_sent: list[Boulder] = field(default_factory=list)
    newly_flashed: list[Boulder] = field(default_factory=list)
    newly_closed: list[Boulder] = field(default_factory=list)

    def summary(self, user_id: Optional[str] = None) -> str:
        lines = [
            f"=== Sync summary for {self.gym} @ {self.synced_at} ===",
            f"  Boulders fetched : {self.total_fetched}",
            f"  New boulders     : {len(self.new_boulders)}",
            f"  Newly closed     : {len(self.newly_closed)}",
        ]
        if user_id:
            lines += [
                f"  Newly sent       : {len(self.newly_sent)}",
                f"  Newly flashed    : {len(self.newly_flashed)}",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

class BoulderDB:
    """
    SQLite-backed store for boulders and sync history.

    Schema
    ------
    boulders          — one row per boulder (upserted on each sync)
    ascents           — one row per (boulder_id, user_id) for sends/flashes
    sync_log          — one row per sync run (gym + timestamp)
    """

    def __init__(self, db_path: str | Path = "climbing.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
        log.info("Database ready at %s", self.db_path.resolve())

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self):
        log.info("Initializing database schema...")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS boulders (
                boulder_id      TEXT PRIMARY KEY,
                gym             TEXT NOT NULL,
                grade           TEXT,
                label           INTEGER,
                boulder_num     INTEGER,
                zone            INTEGER,
                holds_color     INTEGER,
                route_setter    TEXT,        -- JSON list
                route_types     TEXT,        -- JSON list of ints
                created_at      TEXT,
                closed_at       TEXT,
                sents_count     INTEGER DEFAULT 0,
                flashes_count   INTEGER DEFAULT 0,
                first_seen_at   TEXT NOT NULL,
                last_updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ascents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                boulder_id      TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                ascent_type     TEXT NOT NULL CHECK(ascent_type IN ('send','flash')),
                detected_at     TEXT NOT NULL,
                UNIQUE(boulder_id, user_id, ascent_type)
            );

            CREATE TABLE IF NOT EXISTS sync_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                gym         TEXT NOT NULL,
                synced_at   TEXT NOT NULL,
                fetched     INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS comments (
                comment_id      TEXT PRIMARY KEY,
                boulder_id      TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                user_name       TEXT,
                user_level      INTEGER,        -- commenter's grade label at the gym
                text            TEXT,
                has_video       INTEGER DEFAULT 0,
                date            TEXT,
                first_seen_at   TEXT NOT NULL,
                FOREIGN KEY(boulder_id) REFERENCES boulders(boulder_id)
            );

            CREATE INDEX IF NOT EXISTS idx_boulders_gym ON boulders(gym);
            CREATE INDEX IF NOT EXISTS idx_ascents_user ON ascents(user_id);
            CREATE INDEX IF NOT EXISTS idx_comments_boulder ON comments(boulder_id);
        """)
        self._conn.commit()
        log.info("Schema ready (tables: boulders, ascents, comments, sync_log)")

    # ------------------------------------------------------------------
    # Context manager for transactions
    # ------------------------------------------------------------------

    @contextmanager
    def _tx(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def upsert_boulder(self, b: Boulder, now: str) -> bool:
        """
        Insert or update a boulder. Returns True if this is a *new* boulder
        (not seen before).
        """
        existing = self._conn.execute(
            "SELECT boulder_id FROM boulders WHERE boulder_id = ?", (b.boulder_id,)
        ).fetchone()

        is_new = existing is None

        with self._tx() as cur:
            if is_new:
                cur.execute("""
                    INSERT INTO boulders (
                        boulder_id, gym, grade, label, boulder_num, zone,
                        holds_color, route_setter, route_types,
                        created_at, closed_at, sents_count, flashes_count,
                        first_seen_at, last_updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    b.boulder_id, b.gym, b.grade, b.label, b.boulder_num,
                    b.zone, b.holds_color,
                    json.dumps(b.route_setter),
                    json.dumps(b.route_types),
                    b.created_at, b.closed_at,
                    b.sents_count, b.flashes_count,
                    now, now,
                ))
            else:
                cur.execute("""
                    UPDATE boulders SET
                        grade=?, label=?, boulder_num=?, zone=?,
                        holds_color=?, route_setter=?, route_types=?,
                        created_at=?, closed_at=?,
                        sents_count=?, flashes_count=?,
                        last_updated_at=?
                    WHERE boulder_id=?
                """, (
                    b.grade, b.label, b.boulder_num, b.zone,
                    b.holds_color,
                    json.dumps(b.route_setter),
                    json.dumps(b.route_types),
                    b.created_at, b.closed_at,
                    b.sents_count, b.flashes_count,
                    now, b.boulder_id,
                ))

        return is_new

    def close_boulders(self, boulder_ids: list[str], now: str) -> int:
        """Mark specific boulders as closed (only those still open)."""
        if not boulder_ids:
            return 0
        placeholders = ",".join("?" for _ in boulder_ids)
        with self._tx() as cur:
            cur.execute(f"""
                UPDATE boulders
                SET closed_at = ?, last_updated_at = ?
                WHERE boulder_id IN ({placeholders})
                AND closed_at IS NULL
            """, (now, now, *boulder_ids))
            return cur.rowcount

    def record_ascent(self, boulder_id: str, user_id: str,
                      ascent_type: str, now: str) -> bool:
        """
        Insert a send/flash record. Returns True if this is newly detected
        (not already in the DB). Silently ignores duplicates.
        """
        try:
            with self._tx() as cur:
                cur.execute("""
                    INSERT INTO ascents (boulder_id, user_id, ascent_type, detected_at)
                    VALUES (?,?,?,?)
                """, (boulder_id, user_id, ascent_type, now))
            return True
        except sqlite3.IntegrityError:
            return False  # already recorded

    def upsert_comment(self, c: "BoulderComment", now: str) -> bool:
        """Insert a comment if not already stored. Returns True if new."""
        existing = self._conn.execute(
            "SELECT comment_id FROM comments WHERE comment_id=?", (c.comment_id,)
        ).fetchone()
        if existing:
            return False
        with self._tx() as cur:
            cur.execute("""
                INSERT INTO comments
                    (comment_id, boulder_id, user_id, user_name, user_level,
                     text, has_video, date, first_seen_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                c.comment_id, c.boulder_id, c.user_id, c.user_name,
                c.user_level, c.text, int(c.has_video), c.date, now,
            ))
        return True

    def get_comments_for_boulder(self, boulder_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            """SELECT * FROM comments WHERE boulder_id=?
               ORDER BY date DESC""",
            (boulder_id,),
        ).fetchall()

    def get_comments_for_gym(self, gym: str, meaningful_only: bool = True) -> list[sqlite3.Row]:
        """Return all comments for boulders belonging to a gym."""
        query = """
            SELECT c.* FROM comments c
            JOIN boulders b ON c.boulder_id = b.boulder_id
            WHERE b.gym = ?
        """
        if meaningful_only:
            query += " AND c.text != '' AND c.text IS NOT NULL"
        query += " ORDER BY c.date DESC"
        return self._conn.execute(query, (gym,)).fetchall()

    def log_sync(self, gym: str, synced_at: str, fetched: int):
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO sync_log (gym, synced_at, fetched) VALUES (?,?,?)",
                (gym, synced_at, fetched),
            )

    # ------------------------------------------------------------------
    # Public read helpers
    # ------------------------------------------------------------------

    def get_boulders_for_gym(self, gym: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM boulders WHERE gym=? ORDER BY created_at DESC", (gym,)
        ).fetchall()

    def get_ascents_for_user(self, user_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            """SELECT a.*, b.grade, b.gym, b.route_types
               FROM ascents a JOIN boulders b ON a.boulder_id=b.boulder_id
               WHERE a.user_id=?
               ORDER BY a.detected_at DESC""",
            (user_id,),
        ).fetchall()

    def last_sync(self, gym: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT synced_at FROM sync_log WHERE gym=? ORDER BY id DESC LIMIT 1",
            (gym,),
        ).fetchone()
        return row["synced_at"] if row else None

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Main collector class
# ---------------------------------------------------------------------------

class SBoulderCollector:
    """
    Fetches boulder data from sboulder.com and persists it to SQLite.

    Usage
    -----
    collector = SBoulderCollector(
        user_id="qQFsxQKYvqRqYJNKa",  # your Meteor user ID
        db_path="climbing.db",
    )
    result = collector.sync(gym="arkose/montmartre", limit=500)
    print(result.summary(collector.user_id))
    """

    # Timeout waiting for the "ready" signal from the server (seconds)
    READY_TIMEOUT = 30

    def __init__(
        self,
        user_id: Optional[str] = None,
        db_path: str | Path = "climbing.db",
    ):
        self.user_id = user_id
        self.db = BoulderDB(db_path)

        # Internal state reset on each sync() call
        self._raw_boulders: dict[str, dict] = {}
        self._raw_comments: dict[str, dict] = {}
        self._ready = False
        self._sub_id: Optional[str] = None
        self._comment_sub_ids: set[str] = set()
        self._comments_requested = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self, gym: str, limit: int = 500) -> SyncResult:
        """
        Connect to sboulder.com, pull up to `limit` boulders for `gym`,
        persist to DB, and return a SyncResult describing what changed.
        """
        log.info("=" * 60)
        log.info("Starting sync | gym=%s | limit=%d", gym, limit)
        log.info("=" * 60)

        # Reset per-run state
        self._raw_boulders = {}
        self._raw_comments = {}
        self._ready = False
        self._sub_id = _random_id()
        self._comment_sub_ids = set()
        self._comments_requested = False

        ws_url = self._build_ws_url()
        log.info("WebSocket URL: %s", ws_url)
        log.info("Timeout: %ds", self.READY_TIMEOUT)

        ws = websocket.WebSocketApp(
            ws_url,
            on_open=lambda ws: self._on_open(ws, gym, limit),
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            header={"Origin": ORIGIN_HEADER},
        )

        # Run in a thread with a timeout guard
        import threading
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()

        deadline = time.time() + self.READY_TIMEOUT
        while not self._ready and time.time() < deadline:
            time.sleep(0.2)

        ws.close()
        thread.join(timeout=5)

        if not self._ready:
            log.warning("Timed out after %ds — partial data collected "
                        "(%d boulders, %d comments)",
                        self.READY_TIMEOUT,
                        len(self._raw_boulders),
                        len(self._raw_comments))
        else:
            log.info("Data collection complete")

        boulders = [
            Boulder.from_ddp(doc_id, fields)
            for doc_id, fields in self._raw_boulders.items()
        ]
        comments = [
            BoulderComment.from_ddp(cid, fields)
            for cid, fields in self._raw_comments.items()
        ]
        log.info("Fetched %d boulders, %d comments from WebSocket",
                 len(boulders), len(comments))

        return self._persist(gym, boulders, comments)

    def sync_multiple(self, gyms: list[str], limit: int = 500) -> list[SyncResult]:
        """Convenience wrapper: sync several gyms in sequence."""
        return [self.sync(gym, limit) for gym in gyms]

    def close(self):
        self.db.close()

    # ------------------------------------------------------------------
    # WebSocket callbacks
    # ------------------------------------------------------------------

    def _build_ws_url(self) -> str:
        server = str(random.randint(0, 999))
        session = _random_id(8)
        return WS_URL_TEMPLATE.format(server=server, session=session)

    def _on_open(self, ws, gym: str, limit: int):
        log.info("WebSocket connected → %s", ws.url)
        log.info("Sending DDP handshake...")
        # 1. DDP connect handshake
        ws.send(json.dumps([json.dumps({
            "msg": "connect",
            "version": "1",
            "support": ["1", "pre2", "pre1"],
        })]))
        time.sleep(0.8)

        log.info("Subscribing to boulders (gym=%s, limit=%d)", gym, limit)
        # 2. Subscribe to boulders list
        ws.send(json.dumps([json.dumps({
            "msg": "sub",
            "id": self._sub_id,
            "name": "_boulders.list",
            "params": [
                {"gym": gym, "isClosed": None},
                {
                    "isClosed": 1,
                    "createdAt": -1,
                    "boulderNum": -1,
                    "label": -1,
                    "holdsColor": -1,
                },
                limit,
                None,
            ],
        })]))

    def _on_message(self, ws, message: str):
        # SockJS heartbeat
        if message in ("o", "h"):
            return

        if not message.startswith("a["):
            return

        try:
            frames = json.loads(message[1:])  # strip the leading 'a'
        except json.JSONDecodeError:
            log.warning("Could not parse SockJS frame: %s", message[:120])
            return

        for frame in frames:
            try:
                data = json.loads(frame)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("msg")

            if msg_type == "added" and data.get("collection") == "boulders":
                self._raw_boulders[data["id"]] = data.get("fields", {})
                n = len(self._raw_boulders)
                if n % 50 == 0:
                    log.info("  ... %d boulders received so far", n)

            elif msg_type == "changed" and data.get("collection") == "boulders":
                self._raw_boulders.setdefault(data["id"], {}).update(
                    data.get("fields", {})
                )

            elif msg_type == "added" and data.get("collection") == "comments":
                self._raw_comments.setdefault(data["id"], data.get("fields", {}))
                n = len(self._raw_comments)
                if n % 100 == 0:
                    log.info("  ... %d comments received so far", n)

            elif msg_type == "ready":
                subs = data.get("subs", [])
                if self._sub_id in subs:
                    log.info("Boulder subscription ready — %d boulders received",
                             len(self._raw_boulders))
                    if not self._comments_requested:
                        self._comments_requested = True
                        self._fetch_comments(ws, list(self._raw_boulders.keys()))
                # All comment subs returned ready → we're done
                elif self._comment_sub_ids and self._comment_sub_ids.issubset(set(subs)):
                    log.info("All comment subscriptions ready — %d comments received",
                             len(self._raw_comments))
                    self._ready = True

            elif msg_type == "error":
                log.error("DDP error: %s", data)

    def _fetch_comments(self, ws, boulder_ids: list[str], delay: float = 0.15):
        """Subscribe to comments for each boulder individually."""
        log.info("Sending %d comment subscriptions (delay=%.2fs each)...",
                 len(boulder_ids), delay)
        for i, bid in enumerate(boulder_ids, 1):
            sub_id = _random_id()
            self._comment_sub_ids.add(sub_id)
            ws.send(json.dumps([json.dumps({
                "msg": "sub",
                "id": sub_id,
                "name": "_boulders.comments",
                "params": [bid],
            })]))
            if i % 50 == 0:
                log.info("  ... %d/%d comment subs sent", i, len(boulder_ids))
            time.sleep(delay)
        log.info("All comment subscriptions sent — waiting for ready signal")

    def _on_error(self, ws, error):
        log.error("WebSocket error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        log.info("WebSocket closed (status=%s, msg=%s)", close_status_code, close_msg)

    # ------------------------------------------------------------------
    # Persistence + diff logic
    # ------------------------------------------------------------------

    def _persist(
        self,
        gym: str,
        boulders: list[Boulder],
        comments: list["BoulderComment"],
    ) -> SyncResult:
        now = _now_iso()
        result = SyncResult(gym=gym, synced_at=now, total_fetched=len(boulders))

        log.info("Persisting %d boulders to database...", len(boulders))

        # Load existing closed_at values to detect newly-closed routes
        existing_closed: dict[str, Optional[str]] = {
            row["boulder_id"]: row["closed_at"]
            for row in self.db.get_boulders_for_gym(gym)
        }
        log.debug("Loaded %d existing boulder records for diff", len(existing_closed))

        for b in boulders:
            is_new = self.db.upsert_boulder(b, now)

            if is_new:
                log.debug("  [NEW]    %s grade=%s", b.boulder_id, b.grade)
                result.new_boulders.append(b)
            else:
                prev_closed = existing_closed.get(b.boulder_id)
                if b.is_currently_closed and not prev_closed:
                    log.debug("  [CLOSED] %s grade=%s closedAt=%s",
                              b.boulder_id, b.grade, b.closed_at[:10])
                    result.newly_closed.append(b)

            # Track personal ascents if a user_id is configured
            if self.user_id:
                if b.is_sent_by(self.user_id):
                    if self.db.record_ascent(b.boulder_id, self.user_id, "send", now):
                        log.info("  [SEND]   %s grade=%s (newly detected)",
                                 b.boulder_id, b.grade)
                        result.newly_sent.append(b)

                if b.is_flashed_by(self.user_id):
                    if self.db.record_ascent(b.boulder_id, self.user_id, "flash", now):
                        log.info("  [FLASH]  %s grade=%s (newly detected)",
                                 b.boulder_id, b.grade)
                        result.newly_flashed.append(b)

        # Close boulders that were open in DB but absent from this sync's batch
        # Safety guard: skip closing logic if this sync looks incomplete
        MIN_EXPECTED_RATIO = 0.5  # tune as needed
        if existing_closed and len(boulders) < len(existing_closed) * MIN_EXPECTED_RATIO:
            log.warning(
                "Sync fetched only %d boulders vs %d known — skipping closure "
                "logic (likely partial/incomplete sync)",
                len(boulders), len(existing_closed)
            )
        else:
            # Close boulders that were open in DB but absent from this sync's batch
            fetched_ids = {b.boulder_id for b in boulders}
            missing_ids = [
                bid for bid, closed_at in existing_closed.items()
                if closed_at is None and bid not in fetched_ids
            ]
            if missing_ids:
                closed_count = self.db.close_boulders(missing_ids, now)
                log.info("Closed %d boulder(s) no longer present in gym feed: %s",
                        closed_count, missing_ids)
        
        log.info(
            "Boulders persisted: %d new, %d updated, %d newly closed, "
            "%d newly sent, %d newly flashed",
            len(result.new_boulders),
            len(boulders) - len(result.new_boulders),
            len(result.newly_closed),
            len(result.newly_sent),
            len(result.newly_flashed),
        )

        # Persist comments
        log.info("Persisting %d comments...", len(comments))
        new_comments = sum(
            1 for c in comments if self.db.upsert_comment(c, now)
        )
        log.info("Comments persisted: %d new, %d duplicates skipped",
                 new_comments, len(comments) - new_comments)

        self.db.log_sync(gym=gym, synced_at=now, fetched=len(boulders))
        log.info("Sync logged to sync_log table")

        log.info(result.summary(self.user_id))
        return result


# ---------------------------------------------------------------------------
# Quick CLI entrypoint  (python sboulder_collector.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync Arkose/SBoulder data to SQLite")
    parser.add_argument("--gym", default="arkose/montmartre", help="Gym slug")
    parser.add_argument("--user-id", default=None, help="Your SBoulder user ID")
    parser.add_argument("--db", default="climbing.db", help="SQLite file path")
    parser.add_argument("--limit", type=int, default=500, help="Max boulders to fetch")
    parser.add_argument("--profile", default="climber_profile.json", help="Profile JSON path")
    args = parser.parse_args()

    profile = ClimberProfile.load_or_create(args.profile)
    user_id = args.user_id or profile.sboulder_user_id or None

    collector = SBoulderCollector(user_id=user_id, db_path=args.db)
    try:
        result = collector.sync(gym=args.gym, limit=args.limit)
        print(result.summary(user_id))
        if profile.name:
            print(f"\n--- LLM context preview for {profile.name} ---")
            print(profile.to_llm_context())
    finally:
        collector.close()