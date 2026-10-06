"""
database.py
------------
All SQLite persistence for the Industrial Machine Failure Intelligence
System. Uses only the standard-library `sqlite3` module (no ORM, no
PostgreSQL). Every query is parameterized — user input is NEVER
concatenated into SQL strings.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils import DB_PATH, get_logger

logger = get_logger("pdm.database")


@contextmanager
def get_connection(db_path: str = DB_PATH):
    """Context manager yielding a sqlite3 connection with row access by name."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Create the database file and all required tables if they do not exist."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS machines (
                machine_id TEXT PRIMARY KEY,
                machine_type TEXT,
                installation_date TEXT,
                operating_hours REAL,
                status TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                temperature REAL,
                vibration REAL,
                pressure REAL,
                rpm REAL,
                voltage REAL,
                current REAL,
                operating_hours REAL,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                health_score REAL,
                health_status TEXT,
                failure_probability REAL,
                anomaly_score REAL,
                risk_level TEXT,
                predicted_failure_type TEXT,
                rul REAL,
                recommendation TEXT,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS maintenance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                maintenance_date TEXT NOT NULL,
                maintenance_type TEXT,
                description TEXT,
                cost REAL,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
            )
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_sensor_machine ON sensor_readings(machine_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_machine ON predictions(machine_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_maint_machine ON maintenance_records(machine_id)")

    logger.info("Database initialized at %s", db_path)


# --------------------------------------------------------------------------
# machines
# --------------------------------------------------------------------------

def upsert_machine(
    machine_id: str,
    machine_type: str = "Unknown",
    installation_date: Optional[str] = None,
    operating_hours: float = 0.0,
    status: str = "UNKNOWN",
    db_path: str = DB_PATH,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO machines (machine_id, machine_type, installation_date, operating_hours, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(machine_id) DO UPDATE SET
                machine_type=excluded.machine_type,
                operating_hours=excluded.operating_hours,
                status=excluded.status
            """,
            (
                machine_id,
                machine_type,
                installation_date,
                operating_hours,
                status,
                datetime.utcnow().isoformat(),
            ),
        )


def get_machines(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM machines ORDER BY machine_id").fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# sensor_readings
# --------------------------------------------------------------------------

def insert_sensor_reading(reading: Dict[str, Any], db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sensor_readings
                (machine_id, timestamp, temperature, vibration, pressure, rpm, voltage, current, operating_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reading.get("machine_id"),
                reading.get("timestamp", datetime.utcnow().isoformat()),
                reading.get("temperature"),
                reading.get("vibration"),
                reading.get("pressure"),
                reading.get("rpm"),
                reading.get("voltage"),
                reading.get("current"),
                reading.get("operating_hours"),
            ),
        )


def insert_sensor_readings_bulk(readings: List[Dict[str, Any]], db_path: str = DB_PATH) -> int:
    if not readings:
        return 0
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO sensor_readings
                (machine_id, timestamp, temperature, vibration, pressure, rpm, voltage, current, operating_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.get("machine_id"),
                    r.get("timestamp", datetime.utcnow().isoformat()),
                    r.get("temperature"),
                    r.get("vibration"),
                    r.get("pressure"),
                    r.get("rpm"),
                    r.get("voltage"),
                    r.get("current"),
                    r.get("operating_hours"),
                )
                for r in readings
            ],
        )
    return len(readings)


def get_sensor_readings(machine_id: Optional[str] = None, limit: int = 1000,
                         db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        if machine_id:
            rows = conn.execute(
                "SELECT * FROM sensor_readings WHERE machine_id = ? ORDER BY timestamp DESC LIMIT ?",
                (machine_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------

def insert_prediction(pred: Dict[str, Any], db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (machine_id, timestamp, health_score, health_status, failure_probability,
                 anomaly_score, risk_level, predicted_failure_type, rul, recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pred.get("machine_id"),
                pred.get("timestamp", datetime.utcnow().isoformat()),
                pred.get("health_score"),
                pred.get("health_status"),
                pred.get("failure_probability"),
                pred.get("anomaly_score"),
                pred.get("risk_level"),
                pred.get("predicted_failure_type"),
                pred.get("rul"),
                pred.get("recommendation"),
            ),
        )
        return cur.lastrowid


def get_predictions(machine_id: Optional[str] = None, limit: int = 500,
                     db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        if machine_id:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE machine_id = ? ORDER BY timestamp DESC LIMIT ?",
                (machine_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_latest_prediction_per_machine(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.* FROM predictions p
            INNER JOIN (
                SELECT machine_id, MAX(timestamp) AS max_ts
                FROM predictions GROUP BY machine_id
            ) latest
            ON p.machine_id = latest.machine_id AND p.timestamp = latest.max_ts
            """
        ).fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# maintenance_records
# --------------------------------------------------------------------------

def insert_maintenance_record(record: Dict[str, Any], db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO maintenance_records
                (machine_id, maintenance_date, maintenance_type, description, cost)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.get("machine_id"),
                record.get("maintenance_date", datetime.utcnow().date().isoformat()),
                record.get("maintenance_type"),
                record.get("description"),
                record.get("cost"),
            ),
        )
        return cur.lastrowid


def get_maintenance_records(
    machine_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM maintenance_records WHERE 1=1"
    params: List[Any] = []
    if machine_id:
        query += " AND machine_id = ?"
        params.append(machine_id)
    if start_date:
        query += " AND maintenance_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND maintenance_date <= ?"
        params.append(end_date)
    query += " ORDER BY maintenance_date DESC"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def is_database_empty(db_path: str = DB_PATH) -> bool:
    with get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM sensor_readings").fetchone()["c"]
        return count == 0
