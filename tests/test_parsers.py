from satcomp.parsers import parse_dimacs_header, parse_solver_output


def test_parse_dimacs_header():
    lines = ["c comment", "p cnf 10 20", "1 -2 0"]
    v, c = parse_dimacs_header(lines)
    assert v == 10
    assert c == 20


def test_parse_solver_output():
    assert parse_solver_output("s SATISFIABLE", "") == "SAT"
    assert parse_solver_output("s UNSATISFIABLE", "") == "UNSAT"
    assert parse_solver_output("UNKNOWN", "") == "UNKNOWN"
    assert parse_solver_output("SAT", "") == "SAT"
