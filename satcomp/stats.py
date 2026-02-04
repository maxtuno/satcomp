from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple


def _percentile(values: List[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = int(math.ceil((p / 100.0) * len(values))) - 1
    k = max(0, min(k, len(values) - 1))
    return values[k]


def _solver_key(row) -> str:
    return f"{row['solver_name']} {row['solver_version']}"


def compute_solver_stats(rows: Iterable[dict]) -> Dict[str, Any]:
    per_solver: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        per_solver[_solver_key(row)].append(row)

    metrics = []
    for solver, srows in per_solver.items():
        total = len(srows)
        solved_rows = [r for r in srows if r["status"] == "ok" and r["result"] in {"SAT", "UNSAT"}]
        solved = len(solved_rows)
        sat = sum(1 for r in solved_rows if r["result"] == "SAT")
        unsat = sum(1 for r in solved_rows if r["result"] == "UNSAT")
        unknown = total - solved
        times = [r["wall_time"] for r in solved_rows if r["wall_time"] is not None]
        avg = mean(times) if times else None
        med = median(times) if times else None
        p95 = _percentile(times, 95)

        timeout = None
        if srows:
            timeout = srows[0]["timeout"]
        par2 = None
        par10 = None
        if timeout is not None:
            par2 = sum(
                (r["wall_time"] if r["status"] == "ok" and r["result"] in {"SAT", "UNSAT"} else timeout * 2)
                for r in srows
            ) / total
            par10 = sum(
                (r["wall_time"] if r["status"] == "ok" and r["result"] in {"SAT", "UNSAT"} else timeout * 10)
                for r in srows
            ) / total

        metrics.append(
            {
                "solver": solver,
                "total": total,
                "solved": solved,
                "sat": sat,
                "unsat": unsat,
                "unknown": unknown,
                "solve_rate": (solved / total) * 100 if total else 0.0,
                "avg": avg,
                "median": med,
                "p95": p95,
                "par2": par2,
                "par10": par10,
            }
        )

    return {"solvers": metrics}


def compute_cactus_data(rows: Iterable[dict]) -> Dict[str, List[Tuple[int, float]]]:
    per_solver: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and row["result"] in {"SAT", "UNSAT"}:
            if row["wall_time"] is not None:
                per_solver[_solver_key(row)].append(row["wall_time"])

    cactus: Dict[str, List[Tuple[int, float]]] = {}
    for solver, times in per_solver.items():
        times_sorted = sorted(times)
        cumulative = []
        total = 0.0
        for i, t in enumerate(times_sorted, 1):
            total += t
            cumulative.append((i, total))
        cactus[solver] = cumulative
    return cactus


def compute_pairwise(rows: Iterable[dict]) -> List[Dict[str, Any]]:
    by_instance: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_instance[row["instance_id"]][_solver_key(row)] = row

    solvers = sorted({ _solver_key(r) for r in rows })
    results = []
    for i, a in enumerate(solvers):
        for b in solvers[i + 1 :]:
            a_only = b_only = both = both_same = both_diff = 0
            for inst_results in by_instance.values():
                ra = inst_results.get(a)
                rb = inst_results.get(b)
                if not ra or not rb:
                    continue
                a_solved = ra["status"] == "ok" and ra["result"] in {"SAT", "UNSAT"}
                b_solved = rb["status"] == "ok" and rb["result"] in {"SAT", "UNSAT"}
                if a_solved and not b_solved:
                    a_only += 1
                elif b_solved and not a_solved:
                    b_only += 1
                elif a_solved and b_solved:
                    both += 1
                    if ra["result"] == rb["result"]:
                        both_same += 1
                    else:
                        both_diff += 1
            results.append(
                {
                    "solver_a": a,
                    "solver_b": b,
                    "a_only": a_only,
                    "b_only": b_only,
                    "both": both,
                    "both_same": both_same,
                    "both_diff": both_diff,
                }
            )
    return results


def compute_family_stats(rows: Iterable[dict]) -> Dict[str, Dict[str, Any]]:
    per_family: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        family = row["family"] if "family" in row.keys() else ""
        per_family[family].append(row)

    stats = {}
    for family, frows in per_family.items():
        stats[family] = compute_solver_stats(frows)
    return stats
