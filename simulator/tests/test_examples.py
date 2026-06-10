"""Nochiziqli va ta'qib misollari uchun xulq-atvor testlari."""

import numpy as np

from simulator.core import simulate, SimulationConfig
from simulator.presets import get_preset


def test_van_der_pol_stiff_switches():
    """μ katta Van der Pol'da AUTO yashirin usulga o'tishi kerak (nochiziqli qat'iylik)."""
    preset = get_preset("van_der_pol")
    automaton = preset.build(preset.merged_params({"mu": 20.0}))
    config = SimulationConfig(t_span=(0.0, 40.0), method="AUTO")
    traj = simulate(automaton, config)
    methods = {log.method for log in traj.interval_log}
    assert "RK45" in methods and "Radau" in methods


def test_van_der_pol_limit_cycle_bounded():
    """Van der Pol yechimi chegaralangan (chegaraviy sikl) bo'lishi kerak."""
    preset = get_preset("van_der_pol")
    automaton = preset.build(preset.merged_params({"mu": 5.0}))
    config = SimulationConfig(t_span=(0.0, 30.0), method="AUTO")
    traj = simulate(automaton, config)
    x = traj.concat_x()
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x[:, 0])) < 5.0  # amplituda chegaralangan


def test_cat_mouse_terminates_with_event():
    """Mushuk-sichqon ta'qibi hodisa (ushlash yoki qochish) bilan tugashi kerak."""
    preset = get_preset("cat_mouse")
    automaton = preset.build(preset.merged_params())
    config = SimulationConfig(t_span=(0.0, 30.0), method="RK45")
    traj = simulate(automaton, config)
    assert len(traj.events) >= 1
    names = [ev.transition_name for ev in traj.events]
    assert any("ushladi" in n or "qochdi" in n for n in names)


def test_rocket_time_triggered_updates():
    """Raketa diskret vaqt momentlarida (har Δt da) yo'nalishni yangilashi kerak."""
    preset = get_preset("rocket_pursuit")
    automaton = preset.build(preset.merged_params({"period": 0.5}))
    config = SimulationConfig(t_span=(0.0, 6.0), method="RK45")
    traj = simulate(automaton, config)
    # Vaqtga asoslangan yangilash hodisalari (parvoz → parvoz).
    updates = [ev for ev in traj.events
               if ev.from_mode == "parvoz" and ev.to_mode == "parvoz"]
    assert len(updates) >= 3
    # Yangilashlar ~Δt oralig'ida bo'lishi kerak.
    times = sorted(ev.time for ev in updates)
    gaps = np.diff(times)
    assert np.allclose(gaps, 0.5, atol=1e-3)
