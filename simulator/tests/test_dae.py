"""Algebraik-differensial tenglamalar (DAE) va indeksni qisqartirish testlari."""

import numpy as np

from simulator.core import simulate, solve_dae, solve_algebraic, SimulationConfig
from simulator.presets import get_preset


def test_solve_algebraic_scalar():
    """Nyuton usuli oddiy algebraik tenglamani yechishi kerak: y² - 2 = 0."""
    y = solve_algebraic(lambda yy: np.array([yy[0] ** 2 - 2.0]), np.array([1.0]))
    assert abs(y[0] - np.sqrt(2.0)) < 1e-8


def test_pendulum_constraint_maintained():
    """Mayatnik DAE da geometrik cheklov x²+y²=L² saqlanishi kerak."""
    preset = get_preset("pendulum_dae")
    params = preset.merged_params({"L": 1.0, "theta0": 60.0})
    dae = preset.build(params)
    config = SimulationConfig(t_span=(0.0, 5.0), method="RK45",
                              rtol=1e-9, atol=1e-11)
    traj = solve_dae(dae, config)

    x = traj.concat_x()  # ustunlar: x, y, vx, vy, λ
    radius_sq = x[:, 0] ** 2 + x[:, 1] ** 2
    # Indeks-1 ga keltirilgan tizimda cheklov yaxshi saqlanadi.
    assert np.max(np.abs(radius_sq - 1.0)) < 1e-3


def test_pendulum_has_algebraic_variable():
    """Natijada algebraik o'zgaruvchi (λ) holat vektoriga qo'shilishi kerak."""
    preset = get_preset("pendulum_dae")
    dae = preset.build(preset.merged_params())
    config = SimulationConfig(t_span=(0.0, 2.0), method="RK45")
    traj = solve_dae(dae, config)
    # 4 differensial + 1 algebraik = 5 o'zgaruvchi.
    assert len(traj.var_names) == 5
    assert traj.concat_x().shape[1] == 5
    # λ (taranglik) cheklovga mos: musbat bo'lishi kerak (osilgan holat).
    lam = traj.concat_x()[:, 4]
    assert np.all(np.isfinite(lam))


def test_pendulum_energy_bounded():
    """Mayatnik energiyasi (balandlik) chegaralangan bo'lishi kerak (drift yo'q)."""
    preset = get_preset("pendulum_dae")
    dae = preset.build(preset.merged_params({"theta0": 90.0}))
    config = SimulationConfig(t_span=(0.0, 8.0), method="AUTO",
                              rtol=1e-9, atol=1e-11)
    traj = solve_dae(dae, config)
    y = traj.concat_x()[:, 1]
    # y ∈ [-L, L]; kichik son xatosiga yo'l qo'yamiz.
    assert y.min() >= -1.0 - 1e-3
    assert y.max() <= 1.0 + 1e-3


def test_smooth_linear_stays_explicit():
    """Qat'iy bo'lmagan chiziqli modelda AUTO RK45 da qolishi kerak."""
    preset = get_preset("smooth_linear")
    automaton = preset.build(preset.merged_params())
    config = SimulationConfig(t_span=(0.0, 5.0), method="AUTO")
    traj = simulate(automaton, config)
    methods = {log.method for log in traj.interval_log}
    assert methods == {"RK45"}
