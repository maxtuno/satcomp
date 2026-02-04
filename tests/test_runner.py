from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from satcomp.runner import run_task


def _quote_cmd_arg(value: str) -> str:
    if os.name == "nt":
        return f"\"{value}\"" if any(ch.isspace() for ch in value) else value
    return shlex.quote(value)


def test_run_task_accepts_sat_exit_codes(tmp_path: Path):
    cnf = tmp_path / "a.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n", encoding="utf-8")

    solver_script = tmp_path / "solver_sat.py"
    solver_script.write_text(
        "import sys\n"
        "print('s SATISFIABLE')\n"
        "sys.exit(10)\n",
        encoding="utf-8",
    )

    solver = {
        "solver_name": "dummy_sat",
        "solver_bin": f"{_quote_cmd_arg(sys.executable)} {_quote_cmd_arg(str(solver_script))}",
        "command_template": '{bin} "{cnf}"',
        "supports_seed": False,
        "default_threads": 1,
    }
    instance = {"instance_path": str(cnf), "instance_rel_path": "a.cnf"}

    res = run_task(
        result_id=1,
        solver=solver,
        instance=instance,
        run_dir=tmp_path / "run",
        timeout=5,
        mem_mb=0,
        threads=1,
        seed=None,
        save_model=False,
        dry_run=False,
    )

    assert res["exit_code"] == 10
    assert res["status"] == "ok"
    assert res["result"] == "SAT"


def test_run_task_accepts_unsat_exit_codes(tmp_path: Path):
    cnf = tmp_path / "a.cnf"
    cnf.write_text("p cnf 1 2\n1 0\n-1 0\n", encoding="utf-8")

    solver_script = tmp_path / "solver_unsat.py"
    solver_script.write_text(
        "import sys\n"
        "print('s UNSATISFIABLE')\n"
        "sys.exit(20)\n",
        encoding="utf-8",
    )

    solver = {
        "solver_name": "dummy_unsat",
        "solver_bin": f"{_quote_cmd_arg(sys.executable)} {_quote_cmd_arg(str(solver_script))}",
        "command_template": '{bin} "{cnf}"',
        "supports_seed": False,
        "default_threads": 1,
    }
    instance = {"instance_path": str(cnf), "instance_rel_path": "a.cnf"}

    res = run_task(
        result_id=1,
        solver=solver,
        instance=instance,
        run_dir=tmp_path / "run",
        timeout=5,
        mem_mb=0,
        threads=1,
        seed=None,
        save_model=False,
        dry_run=False,
    )

    assert res["exit_code"] == 20
    assert res["status"] == "ok"
    assert res["result"] == "UNSAT"
