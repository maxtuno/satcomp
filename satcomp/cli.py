from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from .config import load_config, write_default_config
from .db import DB
from .models import InstanceDef, RunDef, SolverDef
from .parsers import parse_dimacs_header
from .report import generate_report
from .runner import run_task
from .utils import (
    ensure_dir,
    file_size,
    load_yaml_or_json,
    normalize_rel_path,
    parse_list,
    sha256_cnf,
    to_iso,
)

app = typer.Typer(no_args_is_help=True)
solvers_app = typer.Typer(no_args_is_help=True)
instances_app = typer.Typer(no_args_is_help=True)
run_app = typer.Typer(no_args_is_help=True)

app.add_typer(solvers_app, name="solvers")
app.add_typer(instances_app, name="instances")
app.add_typer(run_app, name="run")


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@app.callback()
def main(debug: bool = typer.Option(False, "--debug", help="Enable debug logging")):
    setup_logging(debug)


def get_db() -> DB:
    config = load_config()
    ensure_dir(config.data_dir)
    db = DB(config.db_path)
    db.ensure_schema()
    return db


def get_platform_version() -> str:
    if (Path.cwd() / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
    return "unknown"


def load_solver_file(path: Path) -> SolverDef:
    data = load_yaml_or_json(path)
    name = str(data.get("name", "")).strip()
    version = str(data.get("version", "")).strip()
    bin_path = str(data.get("bin", "")).strip()
    if not name or not version or not bin_path:
        raise ValueError(f"Missing required fields in {path}")

    supports_seed = bool(data.get("supports_seed", False))
    default_threads = int(data.get("default_threads", 1))

    command_template = data.get("command_template")
    if not command_template:
        args = data.get("args", "")
        if isinstance(args, list):
            args = " ".join(args)
        args = str(args).strip()
        command_template = f"{bin_path} {args} {{cnf}}".strip()
    if "{cnf}" not in command_template:
        command_template = f"{command_template} {{cnf}}"

    return SolverDef(
        name=name,
        version=version,
        bin=bin_path,
        command_template=command_template,
        supports_seed=supports_seed,
        default_threads=default_threads,
        raw=data,
    )


def load_solvers_from_dir(solvers_dir: Path) -> list[SolverDef]:
    solvers: list[SolverDef] = []
    for path in list(solvers_dir.glob("*.yaml")) + list(solvers_dir.glob("*.yml")) + list(
        solvers_dir.glob("*.json")
    ):
        solvers.append(load_solver_file(path))
    return solvers


def index_instances(benchmarks_dir: Path, db: DB) -> int:
    count = 0
    now = to_iso(datetime.utcnow())
    for path in benchmarks_dir.rglob("*.cnf"):
        count += _index_one(path, benchmarks_dir, db, now)
    for path in benchmarks_dir.rglob("*.cnf.gz"):
        count += _index_one(path, benchmarks_dir, db, now)
    return count


def _index_one(path: Path, base_dir: Path, db: DB, now: str) -> int:
    rel_path = normalize_rel_path(base_dir, path)
    family = rel_path.split("/")[0] if "/" in rel_path else ""
    ext = ".cnf.gz" if rel_path.endswith(".cnf.gz") else ".cnf"

    vars_count = None
    clauses_count = None

    # stream to parse header and hash
    import hashlib
    from .utils import open_cnf

    hasher = hashlib.sha256()
    with open_cnf(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            hasher.update(line)
            if vars_count is None:
                try:
                    text = line.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""
                v, c = parse_dimacs_header([text])
                if v is not None:
                    vars_count, clauses_count = v, c

    sha = hasher.hexdigest()

    instance = InstanceDef(
        path=path.resolve(),
        rel_path=rel_path,
        sha256=sha,
        size_bytes=file_size(path),
        vars=vars_count,
        clauses=clauses_count,
        family=family,
        ext=ext,
    )
    db.upsert_instance(instance, created_at=now)
    return 1


@app.command()
def init(force: bool = typer.Option(False, "--force", help="Overwrite existing files")):
    config = load_config()
    for d in [
        config.benchmarks_dir,
        config.solvers_dir,
        config.runs_dir,
        config.reports_dir,
        config.data_dir,
    ]:
        ensure_dir(d)

    cfg_path = Path.cwd() / "config.yaml"
    if force or not cfg_path.exists():
        write_default_config(cfg_path)

    minisat = config.solvers_dir / "minisat.yaml"
    cadical = config.solvers_dir / "cadical.yaml"
    if force or not minisat.exists():
        minisat.write_text(
            "name: minisat\n"
            "version: \"2.2.0\"\n"
            "bin: \"./solvers/bin/minisat\"\n"
            "command_template: \"{bin} -cpu-lim={timeout} {cnf}\"\n"
            "supports_seed: false\n"
            "default_threads: 1\n",
            encoding="utf-8",
        )
    if force or not cadical.exists():
        cadical.write_text(
            "name: cadical\n"
            "version: \"1.5.8\"\n"
            "bin: \"./solvers/bin/cadical\"\n"
            "command_template: \"{bin} -t {timeout} {cnf}\"\n"
            "supports_seed: true\n"
            "default_threads: 1\n",
            encoding="utf-8",
        )

    toy_dir = config.benchmarks_dir / "toy"
    ensure_dir(toy_dir)
    samples = {
        toy_dir / "sat1.cnf": "c toy SAT\np cnf 1 1\n1 0\n",
        toy_dir / "unsat1.cnf": "c toy UNSAT\np cnf 1 2\n1 0\n-1 0\n",
        toy_dir / "sat2.cnf": "c toy SAT\np cnf 2 2\n1 2 0\n-1 2 0\n",
    }
    for path, content in samples.items():
        if force or not path.exists():
            path.write_text(content, encoding="utf-8")

    typer.echo("Initialized satcomp structure")


@solvers_app.command("add")
def solvers_add(path: Path):
    config = load_config()
    ensure_dir(config.solvers_dir)
    if not path.exists():
        raise typer.BadParameter(f"Solver file not found: {path}")

    solver = load_solver_file(path)
    target = config.solvers_dir / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)

    db = get_db()
    db.upsert_solver(solver)
    typer.echo(f"Added solver {solver.name} {solver.version}")


@solvers_app.command("list")
def solvers_list():
    db = get_db()
    solvers = db.get_solvers()
    if not solvers:
        config = load_config()
        solver_defs = load_solvers_from_dir(config.solvers_dir)
        for s in solver_defs:
            db.upsert_solver(s)
        solvers = db.get_solvers()
    if not solvers:
        typer.echo("No solvers found. Add YAML/JSON files to ./solvers")
        return
    for s in solvers:
        typer.echo(f"{s['id']}: {s['name']} {s['version']} -> {s['bin']}")


@solvers_app.command("validate")
def solvers_validate():
    config = load_config()
    errors = 0
    for path in list(config.solvers_dir.glob("*.yaml")) + list(config.solvers_dir.glob("*.yml")) + list(
        config.solvers_dir.glob("*.json")
    ):
        try:
            load_solver_file(path)
            typer.echo(f"OK {path.name}")
        except Exception as exc:
            errors += 1
            typer.echo(f"ERROR {path.name}: {exc}")
    if errors:
        raise typer.Exit(code=1)


@instances_app.command("index")
def instances_index():
    config = load_config()
    db = get_db()
    if not config.benchmarks_dir.exists():
        raise typer.BadParameter(f"Benchmarks dir not found: {config.benchmarks_dir}")
    count = index_instances(config.benchmarks_dir, db)
    typer.echo(f"Indexed {count} instances")


@run_app.command("create")
def run_create(
    name: str = typer.Option("run", "--name"),
    solvers: str = typer.Option("all", "--solvers", help="Comma list or 'all'"),
    family: str = typer.Option("", "--family", help="Comma list of subdirs"),
    timeout: Optional[int] = typer.Option(None, "--timeout"),
    mem_mb: Optional[int] = typer.Option(None, "--mem-mb"),
    threads: Optional[int] = typer.Option(None, "--threads"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    save_model: Optional[bool] = typer.Option(None, "--save-model"),
    jobs: Optional[int] = typer.Option(None, "--jobs"),
    tags: str = typer.Option("", "--tags"),
    notes: str = typer.Option("", "--notes"),
):
    config = load_config()
    db = get_db()

    solver_defs = load_solvers_from_dir(config.solvers_dir)
    if not solver_defs:
        raise typer.BadParameter("No solver definitions found in ./solvers")

    requested = parse_list(solvers)
    if solvers == "all" or not requested:
        selected = solver_defs
    else:
        selected = [s for s in solver_defs if s.name in requested]
    if not selected:
        raise typer.BadParameter("No solvers matched selection")

    solver_ids = [db.upsert_solver(s) for s in selected]

    instances = db.get_instances()
    if not instances:
        raise typer.BadParameter("No instances indexed. Run 'satcomp instances index'")

    families = set(parse_list(family)) if family else set()
    if families:
        instances = [row for row in instances if row["family"] in families]

    if not instances:
        raise typer.BadParameter("No instances matched filters")

    run_params = {
        "timeout": timeout if timeout is not None else config.default_timeout,
        "mem_mb": mem_mb if mem_mb is not None else config.default_mem_mb,
        "threads": threads if threads is not None else config.default_threads,
        "seed": seed,
        "save_model": save_model if save_model is not None else config.default_save_model,
        "jobs": jobs if jobs is not None else config.default_jobs,
        "solvers": [s.name for s in selected],
        "family": list(families),
    }

    run = RunDef(
        name=name,
        tags=tags,
        notes=notes,
        parameters=run_params,
        platform_version=get_platform_version(),
        host=socket.gethostname(),
        os=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or platform.machine(),
    )

    run_id = db.create_run(run, created_at=to_iso(datetime.utcnow()))
    inserted = db.insert_pending_results(
        run_id,
        solver_ids,
        [row["id"] for row in instances],
        run_params["timeout"],
        run_params["mem_mb"],
        run_params["threads"],
        run_params["seed"],
    )

    typer.echo(f"Created run {run_id} with {inserted} tasks")


@run_app.command("start")
def run_start(
    run_id: int = typer.Argument(...),
    jobs: Optional[int] = typer.Option(None, "--jobs"),
    timeout: Optional[int] = typer.Option(None, "--timeout"),
    mem_mb: Optional[int] = typer.Option(None, "--mem-mb"),
    threads: Optional[int] = typer.Option(None, "--threads"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    save_model: Optional[bool] = typer.Option(None, "--save-model"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    config = load_config()
    db = get_db()
    run = db.get_run(run_id)
    if not run:
        raise typer.BadParameter(f"Run {run_id} not found")

    params = json.loads(run["parameters"] or "{}")
    timeout = timeout if timeout is not None else params.get("timeout", config.default_timeout)
    mem_mb = mem_mb if mem_mb is not None else params.get("mem_mb", config.default_mem_mb)
    threads = threads if threads is not None else params.get("threads", config.default_threads)
    seed = seed if seed is not None else params.get("seed")
    save_model = save_model if save_model is not None else params.get("save_model", config.default_save_model)
    jobs = jobs if jobs is not None else params.get("jobs", config.default_jobs)

    db.reset_running(run_id)
    tasks = db.fetch_pending_tasks(run_id)
    if not tasks:
        typer.echo("No pending tasks")
        return

    run_dir = config.runs_dir / f"run_{run_id}"
    ensure_dir(run_dir)

    if dry_run:
        for row in tasks:
            solver = {
                "solver_name": row["solver_name"],
                "solver_bin": row["solver_bin"],
                "command_template": row["command_template"],
                "supports_seed": row["supports_seed"],
                "default_threads": row["default_threads"],
            }
            instance = {
                "instance_path": row["instance_path"],
                "instance_rel_path": row["instance_rel_path"],
            }
            result = run_task(
                row["result_id"],
                solver,
                instance,
                run_dir,
                timeout,
                mem_mb,
                threads,
                seed,
                save_model,
                dry_run=True,
            )
            typer.echo(result["command"])
        return

    def submit(row):
        started = to_iso(datetime.utcnow())
        db.update_result(
            row["result_id"],
            {
                "status": "running",
                "started_at": started,
                "timeout": timeout,
                "mem_mb": mem_mb,
                "threads": threads,
                "seed": seed,
            },
        )
        solver = {
            "solver_name": row["solver_name"],
            "solver_bin": row["solver_bin"],
            "command_template": row["command_template"],
            "supports_seed": row["supports_seed"],
            "default_threads": row["default_threads"],
        }
        instance = {
            "instance_path": row["instance_path"],
            "instance_rel_path": row["instance_rel_path"],
        }
        return run_task(
            row["result_id"],
            solver,
            instance,
            run_dir,
            timeout,
            mem_mb,
            threads,
            seed,
            save_model,
            dry_run=False,
        )

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(submit, row): row for row in tasks}
        for future in as_completed(futures):
            row = futures[future]
            try:
                res = future.result()
                db.update_result(
                    row["result_id"],
                    {
                        "status": res["status"],
                        "result": res["result"],
                        "wall_time": res["wall_time"],
                        "cpu_time": res["cpu_time"],
                        "max_rss_mb": res["max_rss_mb"],
                        "exit_code": res["exit_code"],
                        "stdout_path": res["stdout_path"],
                        "stderr_path": res["stderr_path"],
                        "model_path": res["model_path"],
                        "finished_at": res["finished_at"],
                    },
                )
            except Exception as exc:
                db.update_result(
                    row["result_id"],
                    {
                        "status": "error",
                        "result": "UNKNOWN",
                        "finished_at": to_iso(datetime.utcnow()),
                    },
                )
                typer.echo(f"Task {row['result_id']} failed: {exc}")


@run_app.command("status")
def run_status(run_id: int):
    config = load_config()
    db = get_db()
    run = db.get_run(run_id)
    if not run:
        raise typer.BadParameter(f"Run {run_id} not found")

    counts = db.get_run_counts(run_id)
    params = json.loads(run["parameters"] or "{}")
    jobs = params.get("jobs", config.default_jobs)

    typer.echo(
        f"Run {run_id}: total={counts['total']} finished={counts['finished']} pending={counts['pending']} running={counts['running']}"
    )

    rows = db.get_results_for_run(run_id)
    times = [r["wall_time"] for r in rows if r["status"] == "ok" and r["wall_time"] is not None]
    if times and counts["pending"]:
        avg = sum(times) / len(times)
        eta = (counts["pending"] / max(1, jobs)) * avg
        typer.echo(f"ETA ~ {eta:.1f}s")


@app.command()
def report(run_id: int = typer.Argument(...)):
    config = load_config()
    db = get_db()
    out_dir = config.reports_dir / f"run_{run_id}"
    generate_report(db, run_id, out_dir, Path(__file__).parent / "templates")
    typer.echo(f"Report generated in {out_dir}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    import uvicorn

    uvicorn.run("satcomp.webapp:app", host=host, port=port, reload=False)


@app.command()
def export(
    run_id: int = typer.Argument(...),
    fmt: str = typer.Option("jsonl", "--format"),
    out: Optional[Path] = typer.Option(None, "--out"),
):
    config = load_config()
    db = get_db()
    run = db.get_run(run_id)
    if not run:
        raise typer.BadParameter(f"Run {run_id} not found")

    fmt = fmt.lower()
    if fmt not in {"jsonl", "csv"}:
        raise typer.BadParameter("Format must be jsonl or csv")

    if out is None:
        out = Path.cwd() / f"export_run_{run_id}.{fmt}"

    rows = db.conn().execute(
        """
        SELECT r.*, s.name AS solver_name, s.version AS solver_version, s.bin AS solver_bin,
               s.command_template AS solver_template, s.supports_seed AS solver_supports_seed,
               s.default_threads AS solver_default_threads,
               i.rel_path AS instance_rel_path, i.path AS instance_path, i.sha256 AS instance_sha256,
               i.size_bytes AS instance_size_bytes, i.vars AS instance_vars, i.clauses AS instance_clauses,
               i.family AS instance_family
        FROM results r
        JOIN solvers s ON s.id = r.solver_id
        JOIN instances i ON i.id = r.instance_id
        WHERE r.run_id=?
        """,
        (run_id,),
    )

    if fmt == "jsonl":
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                record = {
                    "run": dict(run),
                    "solver": {
                        "name": row["solver_name"],
                        "version": row["solver_version"],
                        "bin": row["solver_bin"],
                        "command_template": row["solver_template"],
                        "supports_seed": row["solver_supports_seed"],
                        "default_threads": row["solver_default_threads"],
                    },
                    "instance": {
                        "rel_path": row["instance_rel_path"],
                        "path": row["instance_path"],
                        "sha256": row["instance_sha256"],
                        "size_bytes": row["instance_size_bytes"],
                        "vars": row["instance_vars"],
                        "clauses": row["instance_clauses"],
                        "family": row["instance_family"],
                    },
                    "result": {
                        "status": row["status"],
                        "result": row["result"],
                        "wall_time": row["wall_time"],
                        "cpu_time": row["cpu_time"],
                        "max_rss_mb": row["max_rss_mb"],
                        "exit_code": row["exit_code"],
                        "stdout_path": row["stdout_path"],
                        "stderr_path": row["stderr_path"],
                        "model_path": row["model_path"],
                        "started_at": row["started_at"],
                        "finished_at": row["finished_at"],
                        "timeout": row["timeout"],
                        "mem_mb": row["mem_mb"],
                        "threads": row["threads"],
                        "seed": row["seed"],
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        import csv

        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_id",
                    "run_name",
                    "run_created_at",
                    "solver_name",
                    "solver_version",
                    "instance_rel_path",
                    "instance_sha256",
                    "status",
                    "result",
                    "wall_time",
                    "cpu_time",
                    "max_rss_mb",
                    "exit_code",
                    "timeout",
                    "mem_mb",
                    "threads",
                    "seed",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        run["id"],
                        run["name"],
                        run["created_at"],
                        row["solver_name"],
                        row["solver_version"],
                        row["instance_rel_path"],
                        row["instance_sha256"],
                        row["status"],
                        row["result"],
                        row["wall_time"],
                        row["cpu_time"],
                        row["max_rss_mb"],
                        row["exit_code"],
                        row["timeout"],
                        row["mem_mb"],
                        row["threads"],
                        row["seed"],
                    ]
                )

    typer.echo(f"Exported to {out}")


@app.command("import")
def import_results(
    file: Path = typer.Argument(...),
    fmt: Optional[str] = typer.Option(None, "--format"),
):
    if not file.exists():
        raise typer.BadParameter(f"File not found: {file}")

    fmt = (fmt or file.suffix.lstrip(".")).lower()
    if fmt not in {"jsonl", "csv"}:
        raise typer.BadParameter("Format must be jsonl or csv")

    db = get_db()

    if fmt == "jsonl":
        run_id = None
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                run_data = data.get("run", {})
                if run_id is None:
                    run = RunDef(
                        name=run_data.get("name", "import"),
                        tags=run_data.get("tags", "import"),
                        notes=run_data.get("notes", ""),
                        parameters=json.loads(run_data.get("parameters", "{}")) if isinstance(run_data.get("parameters"), str) else run_data.get("parameters", {}),
                        platform_version=run_data.get("platform_version", "import"),
                        host=run_data.get("host", ""),
                        os=run_data.get("os", ""),
                        cpu=run_data.get("cpu", ""),
                    )
                    run_id = db.create_run(run, created_at=run_data.get("created_at", to_iso(datetime.utcnow())))

                solver_data = data.get("solver", {})
                solver = SolverDef(
                    name=solver_data.get("name", "solver"),
                    version=solver_data.get("version", ""),
                    bin=solver_data.get("bin", ""),
                    command_template=solver_data.get("command_template", "{bin} {cnf}"),
                    supports_seed=bool(solver_data.get("supports_seed", False)),
                    default_threads=int(solver_data.get("default_threads", 1)),
                    raw=solver_data,
                )
                solver_id = db.upsert_solver(solver)

                instance_data = data.get("instance", {})
                path_str = instance_data.get("path") or instance_data.get("rel_path") or ""
                instance = InstanceDef(
                    path=Path(path_str).resolve() if path_str else Path(path_str),
                    rel_path=instance_data.get("rel_path", ""),
                    sha256=instance_data.get("sha256", ""),
                    size_bytes=int(instance_data.get("size_bytes", 0)),
                    vars=instance_data.get("vars"),
                    clauses=instance_data.get("clauses"),
                    family=instance_data.get("family", ""),
                    ext=".cnf",
                )
                db.upsert_instance(instance, created_at=to_iso(datetime.utcnow()))

                result = data.get("result", {})
                db.conn().execute(
                    """
                    INSERT OR IGNORE INTO results
                    (run_id, solver_id, instance_id, status, result, wall_time, cpu_time, max_rss_mb,
                     exit_code, stdout_path, stderr_path, model_path, started_at, finished_at,
                     seed, timeout, mem_mb, threads)
                    VALUES (?, ?, (SELECT id FROM instances WHERE rel_path=?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        solver_id,
                        instance.rel_path,
                        result.get("status"),
                        result.get("result"),
                        result.get("wall_time"),
                        result.get("cpu_time"),
                        result.get("max_rss_mb"),
                        result.get("exit_code"),
                        result.get("stdout_path"),
                        result.get("stderr_path"),
                        result.get("model_path"),
                        result.get("started_at"),
                        result.get("finished_at"),
                        result.get("seed"),
                        result.get("timeout"),
                        result.get("mem_mb"),
                        result.get("threads"),
                    ),
                )
                db.conn().commit()
    else:
        import csv

        with file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            run_id = None
            for row in reader:
                if run_id is None:
                    run = RunDef(
                        name=row.get("run_name", "import"),
                        tags="import",
                        notes="",
                        parameters={},
                        platform_version="import",
                        host="",
                        os="",
                        cpu="",
                    )
                    run_id = db.create_run(run, created_at=row.get("run_created_at") or to_iso(datetime.utcnow()))

                solver = SolverDef(
                    name=row.get("solver_name", "solver"),
                    version=row.get("solver_version", ""),
                    bin="",
                    command_template="{bin} {cnf}",
                    supports_seed=False,
                    default_threads=1,
                    raw={},
                )
                solver_id = db.upsert_solver(solver)

                instance = InstanceDef(
                    path=Path(row.get("instance_rel_path", "")),
                    rel_path=row.get("instance_rel_path", ""),
                    sha256=row.get("instance_sha256", ""),
                    size_bytes=0,
                    vars=None,
                    clauses=None,
                    family="",
                    ext=".cnf",
                )
                db.upsert_instance(instance, created_at=to_iso(datetime.utcnow()))

                db.conn().execute(
                    """
                    INSERT OR IGNORE INTO results
                    (run_id, solver_id, instance_id, status, result, wall_time, cpu_time, max_rss_mb,
                     exit_code, seed, timeout, mem_mb, threads)
                    VALUES (?, ?, (SELECT id FROM instances WHERE rel_path=?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        solver_id,
                        instance.rel_path,
                        row.get("status"),
                        row.get("result"),
                        row.get("wall_time"),
                        row.get("cpu_time"),
                        row.get("max_rss_mb"),
                        row.get("exit_code"),
                        row.get("seed"),
                        row.get("timeout"),
                        row.get("mem_mb"),
                        row.get("threads"),
                    ),
                )
                db.conn().commit()

    typer.echo("Import completed")
