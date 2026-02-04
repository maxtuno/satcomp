from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .db import DB
from .stats import compute_cactus_data, compute_family_stats, compute_pairwise, compute_solver_stats


def _env(template_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )


def render_cactus_html(cactus: Dict[str, list[tuple[int, float]]]) -> str:
    fig = go.Figure()
    for solver, series in cactus.items():
        if not series:
            continue
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=solver))
    fig.update_layout(
        title="Cactus Plot",
        xaxis_title="# solved",
        yaxis_title="Cumulative time (s)",
        template="plotly_white",
        height=420,
    )
    return fig.to_html(include_plotlyjs="inline", full_html=False)


def generate_report(db: DB, run_id: int, out_dir: Path, template_dir: Path) -> None:
    run = db.get_run(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")

    rows = db.get_results_for_run(run_id)
    solver_stats = compute_solver_stats(rows)
    cactus = compute_cactus_data(rows)
    pairwise = compute_pairwise(rows)
    family_stats = compute_family_stats(rows)

    env = _env(template_dir)
    html_template = env.get_template("report.html")
    md_template = env.get_template("report.md")

    cactus_html = render_cactus_html(cactus)

    out_dir.mkdir(parents=True, exist_ok=True)

    html = html_template.render(
        run=run,
        solver_stats=solver_stats["solvers"],
        cactus_html=cactus_html,
        pairwise=pairwise,
        family_stats=family_stats,
        total=len(rows),
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    md = md_template.render(
        run=run,
        solver_stats=solver_stats["solvers"],
        pairwise=pairwise,
        family_stats=family_stats,
        total=len(rows),
    )
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    (out_dir / "cactus.json").write_text(json.dumps(cactus, indent=2), encoding="utf-8")
