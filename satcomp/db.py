from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .models import InstanceDef, RunDef, SolverDef


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    tags TEXT,
    notes TEXT,
    parameters TEXT,
    platform_version TEXT,
    host TEXT,
    os TEXT,
    cpu TEXT
);

CREATE TABLE IF NOT EXISTS solvers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    bin TEXT,
    args TEXT,
    command_template TEXT,
    supports_seed INTEGER,
    default_threads INTEGER,
    metadata TEXT,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS instances (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    rel_path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER,
    vars INTEGER,
    clauses INTEGER,
    family TEXT,
    ext TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    solver_id INTEGER NOT NULL,
    instance_id INTEGER NOT NULL,
    status TEXT,
    result TEXT,
    wall_time REAL,
    cpu_time REAL,
    max_rss_mb REAL,
    exit_code INTEGER,
    stdout_path TEXT,
    stderr_path TEXT,
    model_path TEXT,
    started_at TEXT,
    finished_at TEXT,
    seed INTEGER,
    timeout INTEGER,
    mem_mb INTEGER,
    threads INTEGER,
    UNIQUE(run_id, solver_id, instance_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_solver ON results(solver_id);
CREATE INDEX IF NOT EXISTS idx_results_instance ON results(instance_id);
"""


class DB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()

    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
        return self._local.conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            delattr(self._local, "conn")

    def ensure_schema(self) -> None:
        self.conn().executescript(SCHEMA)
        self.conn().commit()

    def upsert_solver(self, solver: SolverDef) -> int:
        conn = self.conn()
        cur = conn.execute(
            "SELECT id FROM solvers WHERE name=? AND version=?",
            (solver.name, solver.version),
        )
        row = cur.fetchone()
        metadata = json.dumps(solver.raw)
        if row:
            conn.execute(
                """
                UPDATE solvers SET bin=?, command_template=?, supports_seed=?,
                    default_threads=?, metadata=?
                WHERE id=?
                """,
                (
                    solver.bin,
                    solver.command_template,
                    1 if solver.supports_seed else 0,
                    solver.default_threads,
                    metadata,
                    row["id"],
                ),
            )
            conn.commit()
            return int(row["id"])

        cur = conn.execute(
            """
            INSERT INTO solvers (name, version, bin, command_template, supports_seed, default_threads, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                solver.name,
                solver.version,
                solver.bin,
                solver.command_template,
                1 if solver.supports_seed else 0,
                solver.default_threads,
                metadata,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def upsert_instance(self, instance: InstanceDef, created_at: str) -> int:
        conn = self.conn()
        cur = conn.execute(
            "SELECT id FROM instances WHERE rel_path=?", (instance.rel_path,)
        )
        row = cur.fetchone()
        if row:
            conn.execute(
                """
                UPDATE instances SET path=?, sha256=?, size_bytes=?, vars=?, clauses=?, family=?, ext=?
                WHERE id=?
                """,
                (
                    str(instance.path),
                    instance.sha256,
                    instance.size_bytes,
                    instance.vars,
                    instance.clauses,
                    instance.family,
                    instance.ext,
                    row["id"],
                ),
            )
            conn.commit()
            return int(row["id"])

        cur = conn.execute(
            """
            INSERT INTO instances (path, rel_path, sha256, size_bytes, vars, clauses, family, ext, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(instance.path),
                instance.rel_path,
                instance.sha256,
                instance.size_bytes,
                instance.vars,
                instance.clauses,
                instance.family,
                instance.ext,
                created_at,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def create_run(self, run: RunDef, created_at: str) -> int:
        conn = self.conn()
        cur = conn.execute(
            """
            INSERT INTO runs (name, created_at, tags, notes, parameters, platform_version, host, os, cpu)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.name,
                created_at,
                run.tags,
                run.notes,
                json.dumps(run.parameters),
                run.platform_version,
                run.host,
                run.os,
                run.cpu,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def list_runs(self) -> List[sqlite3.Row]:
        return list(self.conn().execute("SELECT * FROM runs ORDER BY id DESC"))

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        cur = self.conn().execute("SELECT * FROM runs WHERE id=?", (run_id,))
        return cur.fetchone()

    def get_run_counts(self, run_id: int) -> sqlite3.Row:
        cur = self.conn().execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status IN ('ok','timeout','memout','crash','error') THEN 1 ELSE 0 END) AS finished,
                SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok,
                SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running
            FROM results WHERE run_id=?
            """,
            (run_id,),
        )
        row = cur.fetchone()
        return row

    def reset_running(self, run_id: int) -> None:
        self.conn().execute(
            "UPDATE results SET status='pending' WHERE run_id=? AND status='running'",
            (run_id,),
        )
        self.conn().commit()

    def insert_pending_results(
        self,
        run_id: int,
        solver_ids: Iterable[int],
        instance_ids: Iterable[int],
        timeout: int,
        mem_mb: int,
        threads: int,
        seed: int | None,
    ) -> int:
        conn = self.conn()
        count = 0
        for solver_id in solver_ids:
            for instance_id in instance_ids:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO results
                    (run_id, solver_id, instance_id, status, result, timeout, mem_mb, threads, seed)
                    VALUES (?, ?, ?, 'pending', 'UNKNOWN', ?, ?, ?, ?)
                    """,
                    (run_id, solver_id, instance_id, timeout, mem_mb, threads, seed),
                )
                count += cur.rowcount
        conn.commit()
        return count

    def fetch_pending_tasks(self, run_id: int) -> List[sqlite3.Row]:
        cur = self.conn().execute(
            """
            SELECT r.id AS result_id,
                   s.id AS solver_id,
                   s.name AS solver_name,
                   s.version AS solver_version,
                   s.bin AS solver_bin,
                   s.command_template AS command_template,
                   s.supports_seed AS supports_seed,
                   s.default_threads AS default_threads,
                   i.id AS instance_id,
                   i.path AS instance_path,
                   i.rel_path AS instance_rel_path,
                   i.sha256 AS sha256,
                   i.vars AS vars,
                   i.clauses AS clauses,
                   r.timeout AS timeout,
                   r.mem_mb AS mem_mb,
                   r.threads AS threads,
                   r.seed AS seed
            FROM results r
            JOIN solvers s ON s.id = r.solver_id
            JOIN instances i ON i.id = r.instance_id
            WHERE r.run_id=? AND r.status='pending'
            ORDER BY r.id ASC
            """,
            (run_id,),
        )
        return list(cur)

    def update_result(self, result_id: int, fields: dict[str, Any]) -> None:
        if not fields:
            return
        keys = ", ".join([f"{k}=?" for k in fields])
        values = list(fields.values()) + [result_id]
        self.conn().execute(f"UPDATE results SET {keys} WHERE id=?", values)
        self.conn().commit()

    def get_results_for_run(self, run_id: int) -> List[sqlite3.Row]:
        cur = self.conn().execute(
            """
            SELECT r.*, s.name AS solver_name, s.version AS solver_version,
                   i.rel_path AS instance_rel_path, i.family AS family, i.vars AS vars, i.clauses AS clauses
            FROM results r
            JOIN solvers s ON s.id = r.solver_id
            JOIN instances i ON i.id = r.instance_id
            WHERE r.run_id=?
            """,
            (run_id,),
        )
        return list(cur)

    def get_instances(self, family: Optional[str] = None) -> List[sqlite3.Row]:
        if family:
            cur = self.conn().execute(
                "SELECT * FROM instances WHERE family=? ORDER BY rel_path",
                (family,),
            )
        else:
            cur = self.conn().execute("SELECT * FROM instances ORDER BY rel_path")
        return list(cur)

    def get_solvers(self) -> List[sqlite3.Row]:
        return list(self.conn().execute("SELECT * FROM solvers ORDER BY name"))

    def find_solver_by_name(self, name: str) -> Optional[sqlite3.Row]:
        cur = self.conn().execute("SELECT * FROM solvers WHERE name=?", (name,))
        return cur.fetchone()
