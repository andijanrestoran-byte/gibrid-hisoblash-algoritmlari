"""Usullarni qiyoslash va aniqlik moduli testlari (core/accuracy.py)."""

from __future__ import annotations

import numpy as np

from simulator.core.accuracy import (
    COMPARE_METHODS,
    compare_methods,
    sample_on_grid,
)
from simulator.core.solver_manager import SimulationConfig, simulate
from simulator.presets import get_preset


def _config(T=4.0):
    return SimulationConfig(t_span=(0.0, T), rtol=1e-6, atol=1e-9, method="RK45")


def test_sample_on_grid_matches_exact_for_quadrature():
    """Sof kvadratura yechimi to'rda aniq yechimga mos kelishi kerak."""
    preset = get_preset("sharp_peak")
    params = preset.merged_params()
    automaton = preset.build(params)
    exact = preset.exact(params)
    traj = simulate(automaton, _config())

    grid = np.linspace(0.0, 4.0, 50)
    ys = sample_on_grid(traj, grid)
    ref = np.array([exact(t) for t in grid])
    # AUTO emas, RK45 — lekin keskin cho'qqi to'g'ri tugunlarda yechiladi.
    assert np.nanmax(np.abs(ys - ref)) < 0.5


def test_compare_methods_returns_all_methods():
    preset = get_preset("smooth_linear")
    params = preset.merged_params()
    automaton = preset.build(params)
    exact = preset.exact(params)
    cmp = compare_methods(automaton, _config(T=5.0), exact=exact)

    methods = [r["method"] for r in cmp["rows"]]
    # smooth_linear bir rejimli -> adaptiv usullar + doimiy qadamli Euler/RK4.
    for m in COMPARE_METHODS:
        assert m in methods
    assert "Euler" in methods and "RK4" in methods
    assert cmp["reference_label"] == "Aniq yechim"
    assert len(cmp["series"]) == len(cmp["rows"])
    assert len(cmp["grid"]) == len(cmp["reference_series"][0])


def test_smooth_linear_all_methods_correct():
    """Silliq (qat'iy bo'lmagan) modelda barcha usullar to'g'ri ishlashi kerak."""
    preset = get_preset("smooth_linear")
    params = preset.merged_params()
    automaton = preset.build(params)
    exact = preset.exact(params)
    cmp = compare_methods(automaton, _config(T=5.0), exact=exact)

    verdicts = {r["method"]: r["verdict"] for r in cmp["rows"]}
    # Silliq modelda har bir usul aniq yechimga yaqin natija beradi.
    assert all(v == "correct" for v in verdicts.values()), verdicts


def test_compare_without_exact_uses_reference():
    """Aniq yechim bo'lmasa, yuqori aniqlikdagi etalon ishlatiladi."""
    preset = get_preset("van_der_pol")
    params = preset.merged_params()
    automaton = preset.build(params)
    cmp = compare_methods(automaton, _config(T=5.0), exact=None)

    assert cmp["reference_label"].startswith("Etalon")
    # Etalon o'ziga nisbatan o'lchanmaydi, lekin usullar baholanadi.
    assert all(r["verdict"] in ("correct", "wrong", "error") for r in cmp["rows"])


def test_stiff_model_methods_match_exact():
    """Qat'iy (turg'un) chiziqli modelda yopiq usullar aniq yechimga mos.

    Bu model qat'iylik NISBATI katta (AUTO almashinuvini ko'rsatadi), lekin
    barqaror (so'nuvchi) bo'lgani uchun barcha usullar aniq yechimga yaqin
    natija beradi. Radau/BDF (yopiq) albatta to'g'ri bo'lishi shart.
    """
    preset = get_preset("stiff_demo")
    params = preset.merged_params({"n": 10})
    automaton = preset.build(params)
    exact = preset.exact(params)
    cmp = compare_methods(automaton, _config(T=5.0), exact=exact)

    by = {r["method"]: r for r in cmp["rows"]}
    assert by["Radau"]["verdict"] == "correct", by["Radau"]
    assert by["BDF"]["verdict"] == "correct", by["BDF"]
    # Maksimal xato kichik ekanini ham tekshiramiz.
    assert by["Radau"]["rel_error"] < 0.02
