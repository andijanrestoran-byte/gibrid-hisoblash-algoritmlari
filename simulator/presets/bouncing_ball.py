"""Sakrovchi to'p — klassik gibrid tizim namunasi.

Uzluksiz dinamika (erkin tushish):
    x' = v,   v' = -g
Diskret o'tish (yerga urilish):
    guard:  x = 0  (x kamayib nolni kesganda, ya'ni v < 0)
    reset:  x := 0,  v := -e·v     (e — elastiklik koeffitsienti)

Bu misol Zeno hodisasini ham ko'rsatadi: e < 1 bo'lsa, urilishlar soni
chekli vaqtda cheksizlikka intiladi. max_events himoyasi simulyatsiyani
to'xtatadi.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.hybrid_automaton import HybridAutomaton, Mode, Transition
from .registry import ParamSpec, Preset, register


def build(params: Dict[str, float]) -> HybridAutomaton:
    g = float(params["g"])
    e = float(params["e"])
    x0 = float(params["x0"])
    v0 = float(params["v0"])

    def f(t, s):
        # s = [x, v];  x' = v,  v' = -g
        return np.array([s[1], -g])

    def guard_ground(t, s):
        # x balandligi; manfiyga o'tganda (v<0) urilish ro'y beradi
        return s[0]

    def reset_bounce(t, s):
        # Yerda x ni nolga qisamiz, tezlik ishorasini o'zgartiramiz va so'ndiramiz
        return np.array([0.0, -e * s[1]])

    fly = Mode(name="uchish", f=f, description="Erkin tushish: x'=v, v'=-g")
    bounce = Transition(
        from_mode="uchish",
        to_mode="uchish",
        guard=guard_ground,
        reset=reset_bounce,
        direction=-1.0,  # faqat x kamayib nolni kesganda (v<0)
        name="Yerga urilish (v := -e·v)",
    )

    return HybridAutomaton(
        modes=[fly],
        transitions=[bounce],
        initial_mode="uchish",
        initial_state=[x0, v0],
        var_names=["x (balandlik)", "v (tezlik)"],
        max_events=80,  # Zeno himoyasi: amalda chekli vaqtda cheksiz urilish
    )


PRESET = Preset(
    key="bouncing_ball",
    name="Sakrovchi to'p",
    description=(
        "Erkin tushayotgan to'p yerga urilib qaytadi. Har urilishda tezlik "
        "e koeffitsientiga so'nadi. Klassik gibrid tizim va Zeno hodisasi misoli."
    ),
    notes=(
        "Urilish vaqti analitik qiymat t₁ = √(2·x₀/g) ga juda aniq mos kelishi "
        "hodisa lokalizatsiyasi (bisektsiya 1e-9) to'g'riligini ko'rsatadi."
    ),
    variables=["x (balandlik)", "v (tezlik)"],
    params=[
        ParamSpec("g", "Erkin tushish tezlanishi g", 9.81, 0.1, 100.0, 0.01,
                  "m/s²"),
        ParamSpec("e", "Elastiklik koeffitsienti e", 0.8, 0.0, 0.999, 0.01,
                  "0 < e < 1 — har urilishda tezlik shu nisbatda so'nadi"),
        ParamSpec("x0", "Boshlang'ich balandlik x₀", 1.0, 0.0, 100.0, 0.1, "m"),
        ParamSpec("v0", "Boshlang'ich tezlik v₀", 0.0, -50.0, 50.0, 0.1, "m/s"),
    ],
    build=build,
    default_T=4.0,
    default_method="AUTO",
    two_dim=True,
    icon="🏀",
    category="Mexanika",
)

register(PRESET)
