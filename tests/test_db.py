import json
from pathlib import Path

from satcomp.db import DB
from satcomp.models import InstanceDef, RunDef, SolverDef
from satcomp.utils import to_iso
from datetime import datetime


def test_db_insert(tmp_path: Path):
    db_path = tmp_path / "results.db"
    db = DB(db_path)
    db.ensure_schema()

    solver = SolverDef(
        name="minisat",
        version="2.2.0",
        bin="/bin/minisat",
        command_template="{bin} {cnf}",
        supports_seed=False,
        default_threads=1,
        raw={},
    )
    solver_id = db.upsert_solver(solver)
    assert solver_id > 0

    instance = InstanceDef(
        path=tmp_path / "a.cnf",
        rel_path="a.cnf",
        sha256="deadbeef",
        size_bytes=10,
        vars=1,
        clauses=1,
        family="",
        ext=".cnf",
    )
    instance_id = db.upsert_instance(instance, created_at=to_iso(datetime.utcnow()))
    assert instance_id > 0

    run = RunDef(
        name="test",
        tags="",
        notes="",
        parameters={"timeout": 1},
        platform_version="test",
        host="host",
        os="os",
        cpu="cpu",
    )
    run_id = db.create_run(run, created_at=to_iso(datetime.utcnow()))
    assert run_id > 0

    inserted = db.insert_pending_results(run_id, [solver_id], [instance_id], 1, 1, 1, None)
    assert inserted == 1

    counts = db.get_run_counts(run_id)
    assert counts["total"] == 1
