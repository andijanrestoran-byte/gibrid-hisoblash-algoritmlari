"""Strukturaviy tahlil va indeksni qisqartirish (Pantelides) testlari."""

from simulator.core.structural import analyze, incidence_matrix, pantelides


PENDULUM = {
    "title": "Mayatnik (1-tartibli ko'rinish)",
    "equations": [
        {"name": "x'=vx", "vars": ["x'", "vx"]},
        {"name": "y'=vy", "vars": ["y'", "vy"]},
        {"name": "vx'=-λx", "vars": ["vx'", "lam", "x"]},
        {"name": "vy'=-λy-g", "vars": ["vy'", "lam", "y"]},
        {"name": "x²+y²=L²", "vars": ["x", "y"], "constraint": True},
    ],
}

INDEX1 = {
    "equations": [
        {"name": "x'=-x+y", "vars": ["x'", "x", "y"]},
        {"name": "0=y-sin(x)", "vars": ["y", "x"], "constraint": True},
    ],
}


def test_pendulum_structural_index_is_3():
    """Dekart mayatnigi — strukturaviy indeks 3 bo'lishi kerak."""
    res = pantelides(PENDULUM["equations"])
    assert res["structural_index"] == 3
    # Geometrik cheklov (5-tenglama) ikki marta differensiallanishi kerak.
    assert res["diff_per_eq"][4] == 2


def test_index1_dae_no_differentiation():
    """Indeks-1 DAE da differensiallash talab etilmaydi."""
    res = pantelides(INDEX1["equations"])
    assert res["structural_index"] == 1
    assert all(d == 0 for d in res["diff_per_eq"])


def test_incidence_matrix_shape():
    """Insidentlik matritsasi o'lchami tenglama×o'zgaruvchi bo'lishi kerak."""
    eq_names, var_names, matrix = incidence_matrix(PENDULUM["equations"])
    assert len(matrix) == len(eq_names) == 5
    assert all(len(row) == len(var_names) for row in matrix)


def test_analyze_returns_display_dict():
    """analyze() ko'rsatish uchun to'liq lug'at qaytarishi kerak."""
    res = analyze(PENDULUM)
    assert res["structural_index"] == 3
    assert res["steps"]
    assert "x²+y²=L²" in res["constraints"]
