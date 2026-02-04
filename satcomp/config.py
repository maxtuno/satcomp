from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    base_dir: Path
    benchmarks_dir: Path
    solvers_dir: Path
    runs_dir: Path
    reports_dir: Path
    data_dir: Path
    db_path: Path
    default_timeout: int
    default_mem_mb: int
    default_threads: int
    default_jobs: int
    default_save_model: bool
    version: str


DEFAULT_CONFIG = {
    "version": "0.1.0",
    "paths": {
        "benchmarks_dir": "./benchmarks",
        "solvers_dir": "./solvers",
        "runs_dir": "./runs",
        "reports_dir": "./reports",
        "data_dir": "./data",
        "db_path": "./data/results.db",
    },
    "defaults": {
        "timeout": 60,
        "mem_mb": 2048,
        "threads": 1,
        "jobs": 2,
        "save_model": False,
    },
}


def load_config(config_path: Path | None = None) -> Config:
    base_dir = Path.cwd()
    if config_path is None:
        config_path = base_dir / "config.yaml"

    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        data = DEFAULT_CONFIG

    paths: Dict[str, Any] = data.get("paths", {})
    defaults: Dict[str, Any] = data.get("defaults", {})

    def _p(key: str, fallback: str) -> Path:
        return (base_dir / paths.get(key, fallback)).resolve()

    return Config(
        base_dir=base_dir,
        benchmarks_dir=_p("benchmarks_dir", "./benchmarks"),
        solvers_dir=_p("solvers_dir", "./solvers"),
        runs_dir=_p("runs_dir", "./runs"),
        reports_dir=_p("reports_dir", "./reports"),
        data_dir=_p("data_dir", "./data"),
        db_path=_p("db_path", "./data/results.db"),
        default_timeout=int(defaults.get("timeout", 60)),
        default_mem_mb=int(defaults.get("mem_mb", 2048)),
        default_threads=int(defaults.get("threads", 1)),
        default_jobs=int(defaults.get("jobs", 2)),
        default_save_model=bool(defaults.get("save_model", False)),
        version=str(data.get("version", "0.1.0")),
    )


def write_default_config(path: Path) -> None:
    path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
