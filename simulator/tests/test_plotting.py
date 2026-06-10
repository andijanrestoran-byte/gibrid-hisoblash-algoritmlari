"""Adaptiv grafik nuqtalari testlari."""

import json

import numpy as np

from simulator.core import simulate, SimulationConfig
from simulator.core.plotting import build_series, build_plot_data
from simulator.presets import get_preset


def _value_near(series_dict, t_target):
    """Berilgan t ga eng yaqin nuqtadagi birinchi o'zgaruvchi qiymatini qaytaradi."""
    s = series_dict["series"][0]
    ts = [v for v in s["t"] if v is not None]
    ys = [v for v in s["y"] if v is not None]
    idx = min(range(len(ts)), key=lambda i: abs(ts[i] - t_target))
    return ts[idx], ys[idx]


def test_adaptive_captures_sharp_peak():
    """Adaptiv rejim t=1 atrofidagi keskin cho'qqini (qiymat ~2) yo'qotmasligi kerak.

    Yechim x(t) = t + exp(-(t-1)²/0.01); t=1 da lokal cho'qqi qiymati 2.
    (Global maksimum oxirgi nuqtada — chiziqli t hadi tufayli — bu yerda
    bizni qiziqtirgani lokal cho'qqi.)
    """
    preset = get_preset("sharp_peak")
    automaton = preset.build(preset.merged_params())  # A=1 -> cho'qqi ~2
    config = SimulationConfig(t_span=(0.0, 4.0), method="RK45")
    traj = simulate(automaton, config)

    adaptive = build_series(traj, mode="adaptive")
    t_at, y_at = _value_near(adaptive, 1.0)
    assert abs(t_at - 1.0) < 0.02   # cho'qqi yaqinida nuqta bo'lishi kerak
    assert abs(y_at - 2.0) < 0.05   # va qiymat ~2 bo'lishi kerak


def test_adaptive_denser_than_naive():
    """Adaptiv rejim cho'qqi atrofida naive'dan ko'proq nuqta qo'shishi kerak."""
    preset = get_preset("sharp_peak")
    automaton = preset.build(preset.merged_params())
    config = SimulationConfig(t_span=(0.0, 4.0), method="RK45")
    traj = simulate(automaton, config)

    naive = build_series(traj, mode="naive")
    adaptive = build_series(traj, mode="adaptive")
    assert adaptive["n_points"] > naive["n_points"]


def test_plot_data_is_json_serializable():
    """build_plot_data natijasi JSON ga muammosiz seriyalanishi kerak."""
    preset = get_preset("bouncing_ball")
    automaton = preset.build(preset.merged_params())
    config = SimulationConfig(t_span=(0.0, 2.0), method="RK45")
    traj = simulate(automaton, config)

    data = build_plot_data(traj)
    text = json.dumps(data)  # xato bermasligi kerak (None -> null)
    assert "naive" in text and "adaptive" in text
    # Hodisa chiziqlari mavjud bo'lishi kerak.
    assert len(data["events"]) > 0


def test_phase_portrait_present_for_2d():
    """2 o'zgaruvchili tizimda fazaviy portret ma'lumoti bo'lishi kerak."""
    preset = get_preset("stiff_demo")
    automaton = preset.build(preset.merged_params({"n": 4}))
    config = SimulationConfig(t_span=(0.0, 3.0), method="AUTO")
    traj = simulate(automaton, config)
    data = build_plot_data(traj)
    assert data["adaptive"]["phase"] is not None
    assert "x" in data["adaptive"]["phase"]
