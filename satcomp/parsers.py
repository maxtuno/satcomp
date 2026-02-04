from __future__ import annotations

import re
from typing import Iterable, Tuple


HEADER_RE = re.compile(r"^p\s+cnf\s+(\d+)\s+(\d+)", re.IGNORECASE)


def parse_dimacs_header(lines: Iterable[str]) -> Tuple[int | None, int | None]:
    for line in lines:
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        match = HEADER_RE.match(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def parse_dimacs_header_from_bytes(data: bytes) -> Tuple[int | None, int | None]:
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return None, None
    return parse_dimacs_header(text.splitlines())


SAT_PATTERNS = [
    re.compile(r"\bUNSATISFIABLE\b", re.IGNORECASE),
    re.compile(r"\bSATISFIABLE\b", re.IGNORECASE),
    re.compile(r"\bUNSAT\b", re.IGNORECASE),
    re.compile(r"\bSAT\b", re.IGNORECASE),
    re.compile(r"\bUNKNOWN\b", re.IGNORECASE),
    re.compile(r"\bINDETERMINATE\b", re.IGNORECASE),
]


def parse_solver_output(stdout: str, stderr: str) -> str:
    combined = "\n".join([stdout or "", stderr or ""]).strip()
    if not combined:
        return "UNKNOWN"

    if SAT_PATTERNS[0].search(combined):
        return "UNSAT"
    if SAT_PATTERNS[1].search(combined):
        return "SAT"
    if SAT_PATTERNS[4].search(combined) or SAT_PATTERNS[5].search(combined):
        return "UNKNOWN"

    # fallback tokens
    if SAT_PATTERNS[2].search(combined):
        return "UNSAT"
    if SAT_PATTERNS[3].search(combined):
        return "SAT"

    return "UNKNOWN"


def extract_model(stdout: str) -> str | None:
    if not stdout:
        return None
    lines = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("v"):
            value = stripped[1:].strip()
            if value:
                lines.append(value)
    if not lines:
        return None
    return "\n".join(lines)
