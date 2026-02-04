from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .config import load_config
from .db import DB
from .report import render_cactus_html
from .stats import compute_cactus_data, compute_pairwise, compute_solver_stats

config = load_config()

db = DB(config.db_path)
db.ensure_schema()

app = FastAPI(title="satcomp")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

FINISHED_STATUSES = ("ok", "timeout", "memout", "crash", "error")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _safe_under_dir(path: Path, root: Path) -> bool:
    try:
        path = path.resolve()
        root = root.resolve()
        return root == path or root in path.parents
    except Exception:
        return False


def _read_tail_bytes(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
            return f.read()
    except Exception:
        return b""


def _decode_log(data: bytes) -> str:
    if not data:
        return ""
    # Some Windows tools (e.g. wsl.exe) can emit UTF-16LE into redirected output.
    if b"\x00" in data:
        try:
            return data.decode("utf-16-le", errors="ignore")
        except Exception:
            pass
    return data.decode("utf-8", errors="ignore")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    runs = db.list_runs()
    return templates.TemplateResponse(
        "index.html", {"request": request, "runs": runs}
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: int):
    run = db.get_run(run_id)
    if not run:
        return HTMLResponse("Run not found", status_code=404)

    rows = db.get_results_for_run(run_id)
    solver_stats = compute_solver_stats(rows)["solvers"]
    cactus = compute_cactus_data(rows)
    pairwise = compute_pairwise(rows)
    cactus_html = render_cactus_html(cactus)

    return templates.TemplateResponse(
        "run.html",
        {
            "request": request,
            "run": run,
            "solver_stats": solver_stats,
            "pairwise": pairwise,
            "cactus_html": cactus_html,
            "results": rows,
            "total": len(rows),
        },
    )


@app.get("/runs/{run_id}/live", response_class=HTMLResponse)
async def run_live(request: Request, run_id: int):
    run = db.get_run(run_id)
    if not run:
        return HTMLResponse("Run not found", status_code=404)
    return templates.TemplateResponse(
        "live.html",
        {
            "request": request,
            "run": run,
        },
    )


@app.get("/api/runs/{run_id}/live_state")
async def live_state(
    run_id: int,
    recent_limit: int = Query(50, ge=1, le=500),
    running_limit: int = Query(50, ge=1, le=500),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    counts_row = db.conn().execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status IN ('ok','timeout','memout','crash','error') THEN 1 ELSE 0 END) AS finished,
            SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running,
            SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok,
            SUM(CASE WHEN status='timeout' THEN 1 ELSE 0 END) AS timeout,
            SUM(CASE WHEN status='memout' THEN 1 ELSE 0 END) AS memout,
            SUM(CASE WHEN status='crash' THEN 1 ELSE 0 END) AS crash,
            SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS error
        FROM results WHERE run_id=?
        """,
        (run_id,),
    ).fetchone()
    counts = _row_to_dict(counts_row) if counts_row else {}

    solver_rows = list(
        db.conn().execute(
            """
            SELECT
                s.id AS solver_id,
                s.name AS solver_name,
                s.version AS solver_version,
                COUNT(*) AS total,
                SUM(CASE WHEN r.status='pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN r.status='running' THEN 1 ELSE 0 END) AS running,
                SUM(CASE WHEN r.status='timeout' THEN 1 ELSE 0 END) AS timeout,
                SUM(CASE WHEN r.status='memout' THEN 1 ELSE 0 END) AS memout,
                SUM(CASE WHEN r.status='crash' THEN 1 ELSE 0 END) AS crash,
                SUM(CASE WHEN r.status='error' THEN 1 ELSE 0 END) AS error,
                SUM(CASE WHEN r.status='ok' AND r.result IN ('SAT','UNSAT') THEN 1 ELSE 0 END) AS solved,
                SUM(CASE WHEN r.status='ok' AND r.result='SAT' THEN 1 ELSE 0 END) AS sat,
                SUM(CASE WHEN r.status='ok' AND r.result='UNSAT' THEN 1 ELSE 0 END) AS unsat,
                AVG(CASE WHEN r.status='ok' AND r.result IN ('SAT','UNSAT') THEN r.wall_time END) AS avg_wall_time,
                (
                    SUM(
                        CASE
                            WHEN r.status='ok' AND r.result IN ('SAT','UNSAT') THEN r.wall_time
                            ELSE COALESCE(r.timeout, 0) * 2
                        END
                    ) / COUNT(*)
                ) AS par2
            FROM results r
            JOIN solvers s ON s.id = r.solver_id
            WHERE r.run_id=?
            GROUP BY s.id
            ORDER BY s.name, s.version
            """,
            (run_id,),
        )
    )
    solvers: List[Dict[str, Any]] = []
    for row in solver_rows:
        item = _row_to_dict(row)
        total = int(item.get("total") or 0)
        solved = int(item.get("solved") or 0)
        item["unknown"] = max(0, total - solved)
        item["solve_rate"] = (solved / total) * 100.0 if total else 0.0
        item["solver"] = f"{item.get('solver_name', '')} {item.get('solver_version', '')}".strip()
        solvers.append(item)

    recent_rows = list(
        db.conn().execute(
            f"""
            SELECT
                r.id AS result_id,
                i.rel_path AS instance_rel_path,
                s.name AS solver_name,
                s.version AS solver_version,
                r.status,
                r.result,
                r.wall_time,
                r.exit_code,
                r.finished_at
            FROM results r
            JOIN solvers s ON s.id = r.solver_id
            JOIN instances i ON i.id = r.instance_id
            WHERE r.run_id=?
              AND r.status IN ({",".join(["?"] * len(FINISHED_STATUSES))})
              AND r.finished_at IS NOT NULL
            ORDER BY r.finished_at DESC, r.id DESC
            LIMIT ?
            """,
            (run_id, *FINISHED_STATUSES, recent_limit),
        )
    )
    recent = []
    for row in recent_rows:
        item = _row_to_dict(row)
        item["solver"] = f"{item.get('solver_name', '')} {item.get('solver_version', '')}".strip()
        recent.append(item)

    running_rows = list(
        db.conn().execute(
            """
            SELECT
                r.id AS result_id,
                i.rel_path AS instance_rel_path,
                s.name AS solver_name,
                s.version AS solver_version,
                r.started_at,
                r.timeout
            FROM results r
            JOIN solvers s ON s.id = r.solver_id
            JOIN instances i ON i.id = r.instance_id
            WHERE r.run_id=? AND r.status='running'
            ORDER BY r.started_at DESC, r.id DESC
            LIMIT ?
            """,
            (run_id, running_limit),
        )
    )
    running = []
    for row in running_rows:
        item = _row_to_dict(row)
        item["solver"] = f"{item.get('solver_name', '')} {item.get('solver_version', '')}".strip()
        running.append(item)

    eta_seconds = None
    try:
        params = json.loads(run["parameters"] or "{}")
        jobs = int(params.get("jobs") or 0) or 1
    except Exception:
        jobs = 1

    avg_row = db.conn().execute(
        """
        SELECT AVG(wall_time) AS avg_wall
        FROM results
        WHERE run_id=? AND status='ok' AND result IN ('SAT','UNSAT') AND wall_time IS NOT NULL
        """,
        (run_id,),
    ).fetchone()
    avg_wall = float(avg_row["avg_wall"]) if avg_row and avg_row["avg_wall"] is not None else None
    if avg_wall is not None and counts.get("pending"):
        try:
            eta_seconds = (int(counts["pending"]) / max(1, jobs)) * avg_wall
        except Exception:
            eta_seconds = None

    return {
        "run": _row_to_dict(run),
        "counts": counts,
        "solvers": solvers,
        "recent": recent,
        "running": running,
        "eta_seconds": eta_seconds,
        "server_time": _utc_now_iso(),
    }


@app.get("/api/results/{result_id}/logs")
async def result_logs(
    result_id: int,
    max_bytes: int = Query(65536, ge=1024, le=1024 * 1024),
):
    row = db.conn().execute(
        "SELECT run_id, stdout_path, stderr_path FROM results WHERE id=?",
        (result_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Result not found")

    stdout_raw = str(row["stdout_path"] or "").strip()
    stderr_raw = str(row["stderr_path"] or "").strip()
    stdout_path = Path(stdout_raw) if stdout_raw else None
    stderr_path = Path(stderr_raw) if stderr_raw else None

    run_dir = config.runs_dir / f"run_{row['run_id']}"
    if stdout_path is not None and not _safe_under_dir(stdout_path, run_dir):
        raise HTTPException(status_code=400, detail="stdout_path outside run dir")
    if stderr_path is not None and not _safe_under_dir(stderr_path, run_dir):
        raise HTTPException(status_code=400, detail="stderr_path outside run dir")

    stdout = _decode_log(_read_tail_bytes(stdout_path, max_bytes)) if stdout_path and stdout_path.exists() else ""
    stderr = _decode_log(_read_tail_bytes(stderr_path, max_bytes)) if stderr_path and stderr_path.exists() else ""

    return {
        "result_id": result_id,
        "run_id": row["run_id"],
        "stdout": stdout,
        "stderr": stderr,
        "truncated": (bool(stdout_path and stdout_path.exists() and stdout_path.stat().st_size > max_bytes))
        or (bool(stderr_path and stderr_path.exists() and stderr_path.stat().st_size > max_bytes)),
    }
