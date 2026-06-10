"""Tayyor misollar (presets) testlari."""

import numpy as np

from simulator.core import simulate, solve_dae, SimulationConfig
from simulator.presets import all_presets, get_preset


def _run(preset, params=None):
    """Misolni turiga qarab (ODE/DAE) hisoblaydi."""
    model = preset.build(preset.merged_params(params))
    config = SimulationConfig(
        t_span=(0.0, preset.default_T),
        method=preset.default_method,
        rtol=preset.default_rtol,
        atol=preset.default_atol,
    )
    if preset.kind == "dae":
        return solve_dae(model, config)
    return simulate(model, config)


def test_all_presets_build_and_run():
    """Har bir misol xatosiz qurilib, simulyatsiya qilinishi kerak."""
    for preset in all_presets():
        traj = _run(preset)
        assert traj.success or traj.zeno
        assert len(traj.pieces) > 0
        # Holat qiymatlari chekli bo'lishi kerak (cheksizlikka ketmasligi).
        x_all = traj.concat_x()
        assert np.all(np.isfinite(x_all))
        # var_names soni holat o'lchamiga mos kelishi kerak.
        assert x_all.shape[1] == len(traj.var_names)


def test_three_state_periodicity():
    """Uch tugunli avtomat davriy bo'lishi: A→B o'tishlari teng oraliqda."""
    preset = get_preset("three_state")
    automaton = preset.build(preset.merged_params())
    config = SimulationConfig(t_span=(0.0, 40.0), method="RK45",
                              rtol=1e-9, atol=1e-12)
    traj = simulate(automaton, config)

    # A→B o'tish vaqtlarini ajratamiz.
    ab_times = [ev.time for ev in traj.events
                if ev.from_mode == "A" and ev.to_mode == "B"]
    assert len(ab_times) >= 3

    # Birinchi (o'tkinchi) sikldan keyin davrlar teng bo'lishi kerak.
    periods = np.diff(ab_times)
    stable = periods[1:]  # birinchisini o'tkinchi deb tashlaymiz
    assert np.allclose(stable, stable[0], atol=1e-4)


def test_thermostat_bounds():
    """Termostat harorati [m, M] oralig'ida tebranishi kerak."""
    preset = get_preset("thermostat")
    params = preset.merged_params()
    m, M = params["m"], params["M"]
    automaton = preset.build(params)
    config = SimulationConfig(t_span=(0.0, 40.0), method="RK45")
    traj = simulate(automaton, config)

    x = traj.concat_x()[:, 0]
    # Kichik son xatosiga yo'l qo'yamiz.
    assert x.min() >= m - 1e-3
    assert x.max() <= M + 1e-3
    # Bir necha marta almashishi (davriy rejim) kerak.
    assert len(traj.events) >= 4


def test_preset_param_metadata():
    """Har bir misolda forma uchun parametr metama'lumotlari to'liq bo'lishi kerak."""
    for preset in all_presets():
        assert preset.name and preset.description
        assert preset.variables
        for spec in preset.params:
            assert spec.name and spec.label
            assert spec.default is not None
