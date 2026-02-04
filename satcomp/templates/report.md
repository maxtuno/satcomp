# Report - Run {{ run.id }} - {{ run.name }}

Created: {{ run.created_at }}  
Tags: {{ run.tags or "" }}  
Notes: {{ run.notes or "" }}  
Total tasks: {{ total }}

## Solver Metrics

| Solver | Solved | Unknown | Solve Rate | Avg | Median | P95 | PAR2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{% for row in solver_stats -%}
| {{ row.solver }} | {{ row.solved }}/{{ row.total }} | {{ row.unknown }} | {{ "%.1f" % row.solve_rate }}% | {{ row.avg or "" }} | {{ row.median or "" }} | {{ row.p95 or "" }} | {{ row.par2 or "" }} |
{% endfor %}

## Pairwise

| Solver A | Solver B | A Only | B Only | Both | Both Same | Both Diff |
| --- | --- | --- | --- | --- | --- | --- |
{% for row in pairwise -%}
| {{ row.solver_a }} | {{ row.solver_b }} | {{ row.a_only }} | {{ row.b_only }} | {{ row.both }} | {{ row.both_same }} | {{ row.both_diff }} |
{% endfor %}
