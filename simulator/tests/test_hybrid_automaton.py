"""Gibrid avtomat yadrosi va hodisa lokalizatsiyasi testlari."""

import math

import numpy as np

from simulator.core import simulate, SimulationConfig
from simulator.core.event_detection import bisection_localize
from simulator.presets import get_preset


def test_bouncing_ball_first_impact_analytic():
    """Sakrovchi to'pda birinchi urilish vaqti analitik qiymatga mos kelishi.

    x₀ balandlikdan tinch holatda tushgan to'p t₁ = √(2·x₀/g) da uriladi.
    """
    preset = get_preset("bouncing_ball")
    params = preset.merged_params({"g": 9.81, "x0": 1.0, "v0": 0.0, "e": 0.8})
    automaton = preset.build(params)
    # Aniqlikni tekshirish uchun qat'iy bo'lmagan RK45 yetarli.
    config = SimulationConfig(t_span=(0.0, 4.0), method="RK45",
                              rtol=1e-10, atol=1e-12)
    traj = simulate(automaton, config)

    assert len(traj.events) > 0
    t_first = traj.events[0].time
    t_expected = math.sqrt(2.0 * 1.0 / 9.81)
    assert abs(t_first - t_expected) < 1e-6


def test_bouncing_ball_reset_applied():
    """Reset urilishdan keyin tezlik ishorasini o'zgartirib so'ndirishi kerak."""
    preset = get_preset("bouncing_ball")
    params = preset.merged_params({"e": 0.8})
    automaton = preset.build(params)
    config = SimulationConfig(t_span=(0.0, 2.0), method="RK45")
    traj = simulate(automaton, config)

    ev = traj.events[0]
    v_before = ev.x_before[1]
    v_after = ev.x_after[1]
    assert v_before < 0  # urilishda pastga harakat
    assert math.isclose(v_after, -0.8 * v_before, rel_tol=1e-9)


def test_independent_bisection_matches_scipy():
    """Mustaqil bisektsiya scipy hodisa vaqtiga mos kelishi kerak."""
    preset = get_preset("bouncing_ball")
    automaton = preset.build(preset.merged_params())
    config = SimulationConfig(t_span=(0.0, 2.0), method="RK45",
                              rtol=1e-10, atol=1e-12)
    traj = simulate(automaton, config)

    ev = traj.events[0]
    assert ev.bisection_time is not None
    assert abs(ev.bisection_time - ev.time) < 1e-7


def test_bisection_localize_simple_root():
    """Bisektsiya lokalizatori oddiy funksiyada ildizni 1e-9 da topishi kerak."""
    # g(t) = t - 0.5; ildiz t = 0.5
    root = bisection_localize(lambda t: t - 0.5, 0.0, 1.0, tol=1e-9)
    assert abs(root - 0.5) < 1e-9


def test_zeno_protection_triggers():
    """Sakrovchi to'p chekli vaqtda Zeno chegarasiga yetishi kerak."""
    preset = get_preset("bouncing_ball")
    automaton = preset.build(preset.merged_params({"e": 0.8}))
    config = SimulationConfig(t_span=(0.0, 10.0), method="RK45")
    traj = simulate(automaton, config)
    # max_events = 80; e<1 bo'lgani uchun Zeno himoyasi ishlashi kerak.
    assert traj.zeno is True
    assert len(traj.events) == automaton.max_events
