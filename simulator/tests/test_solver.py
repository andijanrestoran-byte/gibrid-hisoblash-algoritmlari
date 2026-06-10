"""Usulni avtomatik almashtirish (AUTO_ODE) testlari."""

from simulator.core import simulate, SimulationConfig
from simulator.core.solver_manager import estimate_stiffness
from simulator.presets import get_preset

import numpy as np


def test_stiff_demo_switches_method():
    """Qat'iy sistemada AUTO rejim RK45 dan yashirin usulga o'tishi kerak."""
    preset = get_preset("stiff_demo")
    # n katta — aniq qat'iy.
    automaton = preset.build(preset.merged_params({"n": 8}))
    config = SimulationConfig(t_span=(0.0, 5.0), method="AUTO",
                              auto_implicit="Radau")
    traj = simulate(automaton, config)

    methods = {log.method for log in traj.interval_log}
    # Boshida RK45, keyin Radau bo'lishi kerak.
    assert "RK45" in methods
    assert "Radau" in methods
    # Hech bo'lmaganda bitta intervalda almashish belgisi qo'yilishi kerak.
    assert any(log.switched for log in traj.interval_log)


def test_smooth_problem_stays_explicit():
    """Silliq (qat'iy bo'lmagan) masalada AUTO RK45 da qolishi kerak."""
    preset = get_preset("three_state")
    automaton = preset.build(preset.merged_params())
    config = SimulationConfig(t_span=(0.0, 10.0), method="AUTO")
    traj = simulate(automaton, config)

    methods = {log.method for log in traj.interval_log}
    assert methods == {"RK45"}


def test_manual_method_is_respected():
    """Qo'lda tanlangan usul jurnalda saqlanishi kerak."""
    preset = get_preset("stiff_demo")
    automaton = preset.build(preset.merged_params({"n": 6}))
    config = SimulationConfig(t_span=(0.0, 5.0), method="BDF")
    traj = simulate(automaton, config)
    methods = {log.method for log in traj.interval_log}
    assert methods == {"BDF"}


def test_estimate_stiffness_ratio_linear_system():
    """Qat'iylik nisbati R = max|Re λ| / min|Re λ| ga mos kelishi kerak.

    Dissertatsiya matritsasi A_n=[[1+2⁻ⁿ,1],[1,1+2⁻ⁿ]] uchun xos qiymatlar
    2+2⁻ⁿ va 2⁻ⁿ; demak R = (2+2⁻ⁿ)/2⁻ⁿ = 2ⁿ⁺¹ + 1.
    """
    n = 6
    p = 2.0 ** (-n)
    A = np.array([[1.0 + p, 1.0], [1.0, 1.0 + p]])
    ratio = estimate_stiffness(lambda t, x: -(A @ x), 0.0, np.array([1.0, 1.0]))
    expected = 1.0 + 2.0 ** (n + 1)
    assert abs(ratio - expected) / expected < 1e-3


def test_estimate_stiffness_scalar_is_one():
    """Skalyar (bitta moda) tizimda qat'iylik nisbati 1 bo'lishi kerak."""
    ratio = estimate_stiffness(lambda t, x: -3.0 * x, 0.0, np.array([1.0]))
    assert abs(ratio - 1.0) < 1e-9
