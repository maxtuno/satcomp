from satcomp.utils import to_wsl_path


def test_to_wsl_path_passthrough():
    assert to_wsl_path("") == ""
    assert to_wsl_path("/mnt/c/Users/me/a.cnf") == "/mnt/c/Users/me/a.cnf"
    assert to_wsl_path("relative/path.cnf") == "relative/path.cnf"


def test_to_wsl_path_drive_letter():
    assert to_wsl_path(r"C:\Users\me\a.cnf") == "/mnt/c/Users/me/a.cnf"
    assert to_wsl_path("D:/benchmarks/toy/a.cnf") == "/mnt/d/benchmarks/toy/a.cnf"


def test_to_wsl_path_wsl_unc():
    assert to_wsl_path(r"\\wsl$\Ubuntu\home\me\a.cnf") == "/home/me/a.cnf"
    assert to_wsl_path(r"\\wsl.localhost\Ubuntu\home\me\a.cnf") == "/home/me/a.cnf"


def test_to_wsl_path_extended_length():
    assert to_wsl_path(r"\\?\C:\Users\me\a.cnf") == "/mnt/c/Users/me/a.cnf"
