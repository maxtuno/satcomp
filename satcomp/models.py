from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class SolverDef:
    name: str
    version: str
    bin: str
    command_template: str
    supports_seed: bool
    default_threads: int
    raw: Dict[str, Any]


@dataclass
class InstanceDef:
    path: Path
    rel_path: str
    sha256: str
    size_bytes: int
    vars: int | None
    clauses: int | None
    family: str
    ext: str


@dataclass
class RunDef:
    name: str
    tags: str
    notes: str
    parameters: Dict[str, Any]
    platform_version: str
    host: str
    os: str
    cpu: str
