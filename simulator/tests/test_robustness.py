"""Mustahkamlik (robustness) testlari — stress-test natijasida qo'shilgan."""

import json
import math

import numpy as np

from simulator.core import simulate, SimulationConfig
from simulator.presets import get_preset
from simulator.results import serialize_trajectory, _json_safe


def test_json_safe_replaces_nonfinite():
    """_json_safe NaN/Inf qiymatlarni None ga almashtirishi kerak."""
    data = {"a": float("nan"), "b": [1.0, float("inf"), -float("inf")], "c": 2.5}
    safe = _json_safe(data)
    assert safe["a"] is None
    assert safe["b"] == [1.0, None, None]
    assert safe["c"] == 2.5
    # Qat'iy JSON (NaN/Inf yo'q) seriyalanishi kerak.
    json.dumps(safe, allow_nan=False)


def test_serialize_is_strict_json():
    """Barcha misollar uchun serialize natijasi qat'iy JSON bo'lishi kerak."""
    for key in ["bouncing_ball", "stiff_demo", "pendulum_dae", "van_der_pol"]:
        preset = get_preset(key)
        from simulator.core import solve_dae
        model = preset.build(preset.merged_params())
        config = SimulationConfig(t_span=(0.0, 3.0), method="RK45")
        traj = solve_dae(model, config) if preset.kind == "dae" else simulate(model, config)
        res = serialize_trajectory(traj, two_dim=preset.two_dim)
        # allow_nan=False bo'lsa NaN/Inf da xato beradi — bermasligi kerak.
        json.dumps(res, allow_nan=False)


def test_wall_clock_budget_triggers():
    """Juda kichik vaqt budjeti simulyatsiyani to'xtatib, xabar berishi kerak."""
    preset = get_preset("stiff_demo")
    automaton = preset.build(preset.merged_params({"n": 10}))
    config = SimulationConfig(t_span=(0.0, 5.0), method="AUTO",
                              max_wall_seconds=1e-12)
    traj = simulate(automaton, config)
    assert traj.success is False
    assert "Vaqt chegarasi" in traj.message
    # Qisman natija ham JSON ga seriyalanishi kerak.
    json.dumps(serialize_trajectory(traj, two_dim=preset.two_dim), allow_nan=False)


def test_cat_mouse_degenerate_is_finite():
    """Mushuk va sichqon bir nuqtada boshlansa ham, natija chekli va tez bo'lishi kerak."""
    preset = get_preset("cat_mouse")
    params = preset.merged_params({"cat_x": 5.0, "cat_y": 4.0,
                                   "mouse_x": 5.0, "mouse_y": 4.0})
    automaton = preset.build(params)
    config = SimulationConfig(t_span=(0.0, 20.0), method="RK45")
    traj = simulate(automaton, config)
    x = traj.concat_x()
    assert np.all(np.isfinite(x))  # singulyar maydon portlamasligi kerak


def test_bouncing_ball_inelastic_graceful():
    """Elastiklik e=0 da simulyatsiya cheksiz tsiklga tushmasligi kerak."""
    preset = get_preset("bouncing_ball")
    automaton = preset.build(preset.merged_params({"e": 0.0}))
    config = SimulationConfig(t_span=(0.0, 4.0), method="RK45")
    traj = simulate(automaton, config)
    # Crash yoki cheksiz tsikl bo'lmasligi — natija chekli.
    assert np.all(np.isfinite(traj.concat_x()))
