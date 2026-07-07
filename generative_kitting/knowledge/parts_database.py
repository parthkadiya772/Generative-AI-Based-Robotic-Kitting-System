"""
Parts Knowledge Database for the Generative Kitting System.

SQLite-backed relational store that bridges the semantic gap between
VLM output labels and validated physical manipulation parameters.

Tables
------
PARTS   — part metadata (label, category, weight, material, grip specs)
KITS    — kit definitions with required parts list
ACTION_LOG — execution history for analytics & debugging
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.logger import log


class PartsDatabase:
    """
    SQLite-backed knowledge store for kitting parts, kit definitions,
    and action execution logs.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS parts (
        part_id          TEXT PRIMARY KEY,
        label            TEXT NOT NULL,
        category         TEXT NOT NULL,
        weight_grams     REAL DEFAULT 0.0,
        material         TEXT DEFAULT 'unknown',
        grip_type        TEXT DEFAULT 'parallel',
        grip_force_n     REAL DEFAULT 40.0,
        fragile          INTEGER DEFAULT 0,
        stackable        INTEGER DEFAULT 0,
        max_stack_height INTEGER DEFAULT 1,
        usd_mesh_file    TEXT DEFAULT '',
        description      TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS kits (
        kit_id           TEXT PRIMARY KEY,
        kit_name         TEXT NOT NULL,
        required_parts   TEXT NOT NULL,
        description      TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS action_log (
        log_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT NOT NULL,
        session_id       TEXT NOT NULL,
        action           TEXT NOT NULL,
        object_id        TEXT DEFAULT '',
        params           TEXT DEFAULT '{}',
        success          INTEGER DEFAULT 1,
        error_message    TEXT DEFAULT '',
        duration_seconds REAL DEFAULT 0.0
    );
    """

    def __init__(self, db_path: str = "knowledge/kitting.db"):
        """
        Open or create the SQLite database and ensure schema exists.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.
        """
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        log.info(f"PartsDatabase initialised at {db_path}")

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    # ─── Parts CRUD ──────────────────────────────────────────

    def add_part(
        self,
        part_id: str,
        label: str,
        category: str,
        weight_grams: float = 0.0,
        material: str = "unknown",
        grip_type: str = "parallel",
        grip_force_n: float = 40.0,
        fragile: bool = False,
        stackable: bool = False,
        max_stack_height: int = 1,
        usd_mesh_file: str = "",
        description: str = "",
    ) -> None:
        """Insert or replace a part record."""
        self.conn.execute(
            """INSERT OR REPLACE INTO parts
               (part_id, label, category, weight_grams, material,
                grip_type, grip_force_n, fragile, stackable,
                max_stack_height, usd_mesh_file, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                part_id, label, category, weight_grams, material,
                grip_type, grip_force_n, int(fragile), int(stackable),
                max_stack_height, usd_mesh_file, description,
            ),
        )
        self.conn.commit()

    def get_part(self, part_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single part by ID."""
        row = self.conn.execute(
            "SELECT * FROM parts WHERE part_id = ?", (part_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_part_by_label(self, label: str) -> List[Dict[str, Any]]:
        """Retrieve all parts matching a label."""
        rows = self.conn.execute(
            "SELECT * FROM parts WHERE label = ?", (label,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_parts(self) -> List[Dict[str, Any]]:
        """Retrieve all parts in the database."""
        rows = self.conn.execute("SELECT * FROM parts ORDER BY label").fetchall()
        return [dict(r) for r in rows]

    def delete_part(self, part_id: str) -> None:
        """Delete a part by ID."""
        self.conn.execute("DELETE FROM parts WHERE part_id = ?", (part_id,))
        self.conn.commit()

    # ─── Kits CRUD ───────────────────────────────────────────

    def add_kit(
        self,
        kit_id: str,
        kit_name: str,
        required_parts: List[Dict[str, Any]],
        description: str = "",
    ) -> None:
        """
        Insert or replace a kit definition.

        Parameters
        ----------
        required_parts : list of dict
            Each dict: {"label": str, "quantity": int}
        """
        self.conn.execute(
            """INSERT OR REPLACE INTO kits
               (kit_id, kit_name, required_parts, description)
               VALUES (?, ?, ?, ?)""",
            (kit_id, kit_name, json.dumps(required_parts), description),
        )
        self.conn.commit()

    def get_kit(self, kit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a kit definition by ID."""
        row = self.conn.execute(
            "SELECT * FROM kits WHERE kit_id = ?", (kit_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["required_parts"] = json.loads(d["required_parts"])
            return d
        return None

    def get_all_kits(self) -> List[Dict[str, Any]]:
        """Retrieve all kit definitions."""
        rows = self.conn.execute("SELECT * FROM kits ORDER BY kit_name").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["required_parts"] = json.loads(d["required_parts"])
            result.append(d)
        return result

    # ─── Action Logging ──────────────────────────────────────

    def log_action(
        self,
        session_id: str,
        action: str,
        object_id: str = "",
        params: Optional[Dict] = None,
        success: bool = True,
        error_message: str = "",
        duration_seconds: float = 0.0,
    ) -> int:
        """
        Log an executed action for audit and analytics.

        Returns
        -------
        int
            The log_id of the inserted record.
        """
        cursor = self.conn.execute(
            """INSERT INTO action_log
               (timestamp, session_id, action, object_id, params,
                success, error_message, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                session_id, action, object_id,
                json.dumps(params or {}),
                int(success), error_message, duration_seconds,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_session_log(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all action logs for a session."""
        rows = self.conn.execute(
            "SELECT * FROM action_log WHERE session_id = ? ORDER BY log_id",
            (session_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d["params"])
            d["success"] = bool(d["success"])
            result.append(d)
        return result

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Compute summary statistics for a session."""
        logs = self.get_session_log(session_id)
        total = len(logs)
        successes = sum(1 for l in logs if l["success"])
        failures = total - successes
        total_time = sum(l.get("duration_seconds", 0) for l in logs)
        return {
            "session_id": session_id,
            "total_actions": total,
            "successes": successes,
            "failures": failures,
            "total_time_seconds": round(total_time, 2),
            "success_rate": round(successes / total * 100, 1) if total > 0 else 0.0,
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
