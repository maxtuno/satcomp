from __future__ import annotations

import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .parsers import extract_model, parse_solver_output
from .utils import ensure_dir, to_iso, to_wsl_path

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _now_iso() -> str:
    return to_iso(datetime.utcnow())


def render_command(template: str, variables: Dict[str, Any]) -> str:
    return template.format(**variables)


def split_command(command: str) -> list[str]:
    return shlex.split(command, posix=os.name != "nt")


def _is_wsl_command(command: list[str]) -> bool:
    if not command:
        return False
    exe = command[0].strip('"')
    name = Path(exe).name.lower()
    return name in {"wsl", "wsl.exe"}


def kill_process_tree(proc: subprocess.Popen):
    if psutil is None:
        try:
            proc.kill()
        except Exception:
            pass
        return

    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception:
                pass
        try:
            parent.kill()
        except Exception:
            pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_task(
    result_id: int,
    solver: Dict[str, Any],
    instance: Dict[str, Any],
    run_dir: Path,
    timeout: int,
    mem_mb: int,
    threads: int,
    seed: int | None,
    save_model: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    solver_name = solver["solver_name"]
    instance_path = Path(instance["instance_path"])
    rel_path = instance["instance_rel_path"].replace("/", "_")

    solver_dir = run_dir / solver_name
    ensure_dir(solver_dir)

    stdout_path = solver_dir / f"{rel_path}.out"
    stderr_path = solver_dir / f"{rel_path}.err"
    model_path = solver_dir / f"{rel_path}.model"

    supports_seed = bool(solver.get("supports_seed"))
    default_threads = solver.get("default_threads") or 1
    threads_value = threads or default_threads
    seed_value = seed if supports_seed and seed is not None else ""

    template = solver.get("command_template") or "{bin} {cnf}"
    variables = {
        "cnf": str(instance_path),
        "seed": seed_value,
        "timeout": timeout,
        "mem_mb": mem_mb,
        "threads": threads_value,
        "bin": solver.get("solver_bin"),
    }
    command_str = render_command(template, variables)
    command = split_command(command_str)

    # If the solver is executed via WSL, pass the CNF path as a Linux path.
    if _is_wsl_command(command):
        cnf_wsl = to_wsl_path(variables["cnf"])
        if cnf_wsl != variables["cnf"]:
            variables["cnf"] = cnf_wsl
            command_str = render_command(template, variables)
            command = split_command(command_str)

    if dry_run:
        return {
            "status": "ok",
            "result": "UNKNOWN",
            "wall_time": 0.0,
            "cpu_time": None,
            "max_rss_mb": None,
            "exit_code": 0,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "model_path": None,
            "started_at": _now_iso(),
            "finished_at": _now_iso(),
            "command": command_str,
            "skipped": True,
        }

    start_ts = _now_iso()
    start_time = time.perf_counter()
    max_rss = 0.0
    status = "ok"
    exit_code = None

    stdout_file = open(stdout_path, "wb")
    stderr_file = open(stderr_path, "wb")

    preexec = None
    creationflags = 0
    if os.name == "posix":
        def _preexec():  # pragma: no cover
            os.setsid()
            if mem_mb:
                try:
                    import resource

                    limit = int(mem_mb) * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                except Exception:
                    pass

        preexec = _preexec
    else:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        command,
        stdout=stdout_file,
        stderr=stderr_file,
        preexec_fn=preexec,
        creationflags=creationflags,
    )

    ps_proc = None
    if psutil is not None:
        try:
            ps_proc = psutil.Process(proc.pid)
        except Exception:
            ps_proc = None

    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                exit_code = ret
                break

            elapsed = time.perf_counter() - start_time
            if timeout and elapsed > timeout:
                status = "timeout"
                kill_process_tree(proc)
                exit_code = proc.poll()
                break

            if ps_proc is not None:
                try:
                    rss = ps_proc.memory_info().rss
                    max_rss = max(max_rss, rss)
                    if mem_mb and rss > mem_mb * 1024 * 1024:
                        status = "memout"
                        kill_process_tree(proc)
                        exit_code = proc.poll()
                        break
                except Exception:
                    pass

            time.sleep(0.05)
    finally:
        stdout_file.close()
        stderr_file.close()

    wall_time = time.perf_counter() - start_time
    cpu_time = None
    if ps_proc is not None:
        try:
            cpu_times = ps_proc.cpu_times()
            cpu_time = cpu_times.user + cpu_times.system
        except Exception:
            cpu_time = None

    stdout_text = ""
    stderr_text = ""
    try:
        stdout_text = Path(stdout_path).read_text(encoding="utf-8", errors="ignore")
        stderr_text = Path(stderr_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass

    result = parse_solver_output(stdout_text, stderr_text)
    exit_code_to_result = {10: "SAT", 20: "UNSAT"}
    if exit_code in exit_code_to_result and result == "UNKNOWN":
        result = exit_code_to_result[exit_code]

    model_saved = None
    if save_model and result == "SAT":
        model = extract_model(stdout_text)
        if model:
            model_path.write_text(model, encoding="utf-8")
            model_saved = str(model_path)

    if status in {"timeout", "memout"}:
        final_status = status
        result = "UNKNOWN"
    else:
        ok_exit_codes = {0, 10, 20}
        final_status = "ok" if exit_code in ok_exit_codes else "crash"
        if final_status != "ok":
            result = "UNKNOWN"

    finished_ts = _now_iso()
    return {
        "status": final_status,
        "result": result,
        "wall_time": wall_time,
        "cpu_time": cpu_time,
        "max_rss_mb": (max_rss / (1024 * 1024)) if max_rss else None,
        "exit_code": exit_code,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "model_path": model_saved,
        "started_at": start_ts,
        "finished_at": finished_ts,
        "command": command_str,
        "skipped": False,
    }
