"""Termostat — ikki rejimli gibrid tizim.

Ikki uzluksiz rejim:
    "sovish"  (isitgich o'chiq):  T' = -K·T          (harorat pasayadi)
    "isish"   (isitgich yoniq):   T' =  K·(h - T)     (harorat h ga intiladi)

Diskret o'tishlar (gisteretik boshqaruv):
    sovish → isish:   T = m  (harorat pastki chegaragacha tushadi)
    isish  → sovish:  T = M  (harorat yuqori chegaragacha ko'tariladi)

Natija — m va M oralig'ida davriy tebranuvchi harorat (limit sikl).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.hybrid_automaton import HybridAutomaton, Mode, Transition
from .registry import ParamSpec, Preset, register


def build(params: Dict[str, float]) -> HybridAutomaton:
    K = float(params["K"])
    h = float(params["h"])
    m = float(params["m"])
    M = float(params["M"])
    T0 = float(params["T0"])

    def f_off(t, s):
        # Isitgich o'chiq: harorat 0 ga qarab eksponensial pasayadi
        return np.array([-K * s[0]])

    def f_on(t, s):
        # Isitgich yoniq: harorat h ga qarab intiladi
        return np.array([K * (h - s[0])])

    off = Mode(name="sovish", f=f_off, description="Isitgich o'chiq: T'=-K·T")
    on = Mode(name="isish", f=f_on, description="Isitgich yoniq: T'=K·(h-T)")

    to_on = Transition(
        from_mode="sovish",
        to_mode="isish",
        guard=lambda t, s: s[0] - m,  # T = m ga tushganda
        direction=-1.0,
        name="Pastki chegara: isitgich yoqildi",
    )
    to_off = Transition(
        from_mode="isish",
        to_mode="sovish",
        guard=lambda t, s: s[0] - M,  # T = M ga ko'tarilganda
        direction=+1.0,
        name="Yuqori chegara: isitgich o'chirildi",
    )

    return HybridAutomaton(
        modes=[off, on],
        transitions=[to_on, to_off],
        initial_mode="sovish",
        initial_state=[T0],
        var_names=["T (harorat)"],
        max_events=10000,
    )


PRESET = Preset(
    key="thermostat",
    name="Termostat",
    description=(
        "Isitgich pastki (m) va yuqori (M) chegaralar orasida yonib-o'chib, "
        "haroratni davriy ushlab turadi. Ikki rejimli gibrid avtomat."
    ),
    notes=(
        "Sof davriy limit sikl: rejim almashinishi guard chegaralarida aniq "
        "ro'y berishini va gibrid vaqt hisobini ko'rsatadi."
    ),
    variables=["T (harorat)"],
    params=[
        ParamSpec("K", "Issiqlik almashinuv koeffitsienti K", 0.5, 0.01, 10.0, 0.01),
        ParamSpec("h", "Isitgich nishon harorati h", 30.0, 0.0, 200.0, 0.5),
        ParamSpec("m", "Pastki chegara m", 18.0, -50.0, 200.0, 0.5),
        ParamSpec("M", "Yuqori chegara M", 22.0, -50.0, 200.0, 0.5),
        ParamSpec("T0", "Boshlang'ich harorat T₀", 20.0, -50.0, 200.0, 0.5),
    ],
    build=build,
    default_T=20.0,
    default_method="AUTO",
    two_dim=False,
    icon="🌡️",
    category="Boshqaruv",
)

register(PRESET)
