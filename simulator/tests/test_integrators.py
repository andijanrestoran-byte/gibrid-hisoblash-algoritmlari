"""Doimiy qadamli usullar va ularning qiyoslashdagi roli (§2.3, §3.1)."""

from __future__ import annotations

import numpy as np

from simulator.core.integrators import euler, rk4, integrate_on_grid
from simulator.core.accuracy import compare_methods, FIXED_STEP_NAMES
from simulator.core.solver_manager import SimulationConfig
from simulator.presets import get_preset


def test_rk4_more_accurate_than_euler_on_smooth():
    """RK4 silliq tenglamada Euler'dan ancha aniqroq bo'lishi kerak."""
    # x' = x,  x(0)=1,  aniq yechim x(t)=e^t — tugunlarda bevosita solishtiramiz.
    f = lambda t, x: x
    te, xe = euler(f, (0, 1), np.array([1.0]), n_steps=100)
    tr, xr = rk4(f, (0, 1), np.array([1.0]), n_steps=100)
    err_e = np.max(np.abs(xe[:, 0] - np.exp(te)))
    err_r = np.max(np.abs(xr[:, 0] - np.exp(tr)))
    assert err_r < err_e
    assert err_r < 1e-7  # RK4 (h=0.01) tugun xatosi ~h⁴


def test_euler_unstable_on_stiff():
    """Oshkora Euler qat'iy masalada beqarorlashadi (cheksiz o'sadi).

    Aniq yechim x(t)=e^(-1000t) -> 0 ga intiladi, lekin katta qadamda Euler
    barqarorlik sohasidan chiqib, qiymat cheksiz o'sib ketadi.
    """
    f = lambda t, x: -1000.0 * x
    ts, xs = euler(f, (0, 1), np.array([1.0]), n_steps=100)  # h=0.01, h·|λ|=10 ≫ 2
    # Aniq yechim ~0 bo'lishi kerak; Euler esa ulkan qiymatga o'sadi (beqaror).
    assert np.max(np.abs(xs)) > 1e6


def test_compare_includes_fixed_step_for_single_mode():
    """Bir rejimli modelda qiyoslash Euler va RK4 ni ham o'z ichiga oladi."""
    preset = get_preset("smooth_linear")
    params = preset.merged_params()
    automaton = preset.build(params)
    exact = preset.exact(params)
    cfg = SimulationConfig(t_span=(0.0, 5.0), rtol=1e-6, atol=1e-9, method="RK45")
    cmp = compare_methods(automaton, cfg, exact=exact)
    methods = [r["method"] for r in cmp["rows"]]
    for m in FIXED_STEP_NAMES:
        assert m in methods


def test_compare_fixed_step_wrong_on_stiff():
    """Qat'iy masalada doimiy qadamli Euler 'Noto'g'ri' hukm olishi kerak."""
    preset = get_preset("stiff_demo")
    # n=0 -> A=[[2,1],[1,2]], xos qiymatlar -3 va -1 -> Euler h=5/400=0.0125 OK emas?
    # Kuchli qat'iylik uchun emas, balki katta amplituda: x' = -A x da tez moda -3.
    # Aniq qat'iy holat uchun van_der_pol katta mu ishlatamiz.
    preset = get_preset("van_der_pol")
    params = preset.merged_params({"mu": 80.0, "x0": 2.0, "y0": 0.0})
    automaton = preset.build(params)
    cfg = SimulationConfig(t_span=(0.0, 30.0), rtol=1e-6, atol=1e-9, method="RK45")
    cmp = compare_methods(automaton, cfg, exact=None)
    by = {r["method"]: r for r in cmp["rows"]}
    # Doimiy qadamli Euler qat'iy relaksatsion tebranishda beqarorlashadi.
    assert by["Euler"]["verdict"] == "wrong", by["Euler"]
    # Yopiq Radau esa to'g'ri ishlaydi (etalonning o'zi — nisbiy xato ~0).
    assert by["Radau"]["verdict"] == "correct", by["Radau"]


def test_hybrid_preset_no_fixed_step():
    """Ko'p rejimli (gibrid) modelda doimiy qadamli usullar qo'shilmaydi."""
    preset = get_preset("bouncing_ball")
    params = preset.merged_params()
    automaton = preset.build(params)
    cfg = SimulationConfig(t_span=(0.0, 3.0), rtol=1e-6, atol=1e-9, method="RK45")
    cmp = compare_methods(automaton, cfg, exact=None)
    methods = [r["method"] for r in cmp["rows"]]
    assert "Euler" not in methods
    assert "RK4" not in methods
