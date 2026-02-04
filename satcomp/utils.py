from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


def load_yaml_or_json(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any]
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported solver definition: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Invalid solver definition format: {path}")
    return data


def dump_json(data: Any) -> str:
    if is_dataclass(data):
        data = asdict(data)
    return json.dumps(data, ensure_ascii=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def open_cnf(path: Path):
    if path.suffix.lower() == ".gz":
        import gzip

        return gzip.open(path, "rb")
    return open(path, "rb")


def sha256_cnf(path: Path) -> str:
    hasher = hashlib.sha256()
    with open_cnf(path) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size


def normalize_rel_path(base: Path, path: Path) -> str:
    rel = path.relative_to(base)
    return rel.as_posix()


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def json_loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def to_iso(dt) -> str:
    return dt.isoformat(timespec="seconds")


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


_WIN_DRIVE_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_WSL_UNC_DOLLAR_RE = re.compile(r"^\\\\wsl\\$\\[^\\]+\\(.*)$", re.IGNORECASE)
_WSL_UNC_LOCALHOST_RE = re.compile(r"^\\\\wsl\\.localhost\\[^\\]+\\(.*)$", re.IGNORECASE)


def to_wsl_path(path: str | Path) -> str:
    """
    Convert a Windows path to a WSL-friendly POSIX path.

    Examples:
      - C:\\Users\\me\\a.cnf -> /mnt/c/Users/me/a.cnf
      - \\\\wsl$\\Ubuntu\\home\\me\\a.cnf -> /home/me/a.cnf
    """
    path_str = str(path)
    if not path_str:
        return path_str

    # Already POSIX.
    if path_str.startswith("/"):
        return path_str

    # Handle extended-length paths like \\?\C:\...
    if path_str.startswith("\\\\?\\"):
        path_str = path_str[4:]
        if path_str.upper().startswith("UNC\\"):
            path_str = "\\" + path_str[3:]

    # Handle WSL UNC paths (Windows view of Linux filesystem).
    match = _WSL_UNC_DOLLAR_RE.match(path_str) or _WSL_UNC_LOCALHOST_RE.match(path_str)
    if match:
        rest = match.group(1).replace("\\", "/").lstrip("/")
        return f"/{rest}"

    # Drive-letter absolute paths.
    if _WIN_DRIVE_RE.match(path_str):
        drive = path_str[0].lower()
        rest = path_str[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"

    return path_str
