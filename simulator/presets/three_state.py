"""Uch tugunli davriy avtomat.

Bitta skalyar holat s, uchta rejim (a < 0):
    rejim A:  s' = a·s + 1
    rejim B:  s' = a·s - 1
    rejim C:  s' = a·s + 1

Diskret o'tishlar (siklik):
    A → B:  s = +0.9   (s o'sib yuqori chegaraga yetganda)
    B → C:  s = -0.9   (s kamayib pastki chegaraga yetganda)
    C → A:  s =  0      (s o'sib nolga qaytganda)

Boshlang'ich s₀ = 0.1. Natija — qat'iy davriy yechim (limit sikl):
o'tish nuqtalari va sikl davri har takrorda bir xil bo'ladi.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.hybrid_automaton import HybridAutomaton, Mode, Transition
from .registry import ParamSpec, Preset, register


def build(params: Dict[str, float]) -> HybridAutomaton:
    a = float(params["a"])
    s0 = float(params["s0"])

    mode_a = Mode(name="A", f=lambda t, s: np.array([a * s[0] + 1.0]),
                  description="s' = a·s + 1")
    mode_b = Mode(name="B", f=lambda t, s: np.array([a * s[0] - 1.0]),
                  description="s' = a·s - 1")
    mode_c = Mode(name="C", f=lambda t, s: np.array([a * s[0] + 1.0]),
                  description="s' = a·s + 1")

    a_to_b = Transition(
        from_mode="A", to_mode="B",
        guard=lambda t, s: s[0] - 0.9, direction=+1.0,
        name="A→B (s = +0.9)",
    )
    b_to_c = Transition(
        from_mode="B", to_mode="C",
        guard=lambda t, s: s[0] + 0.9, direction=-1.0,
        name="B→C (s = -0.9)",
    )
    c_to_a = Transition(
        from_mode="C", to_mode="A",
        guard=lambda t, s: s[0], direction=+1.0,
        name="C→A (s = 0)",
    )

    return HybridAutomaton(
        modes=[mode_a, mode_b, mode_c],
        transitions=[a_to_b, b_to_c, c_to_a],
        initial_mode="A",
        initial_state=[s0],
        var_names=["s"],
        max_events=10000,
    )


PRESET = Preset(
    key="three_state",
    name="Uch tugunli davriy avtomat",
    description=(
        "Uch rejim siklik almashib, s o'zgaruvchisi qat'iy davriy yechim "
        "hosil qiladi. Gibrid limit sikl namunasi."
    ),
    notes=(
        "Davriylik testi: birinchi to'liq sikldan keyin A→B o'tishlari "
        "orasidagi vaqt o'zgarmas bo'lib qoladi — bu yechimning davriyligini "
        "tasdiqlaydi."
    ),
    variables=["s"],
    params=[
        ParamSpec("a", "Koeffitsient a", -1.0, -10.0, -0.01, 0.01,
                  "a < 0 bo'lishi shart (turg'un rejimlar uchun)"),
        ParamSpec("s0", "Boshlang'ich qiymat s₀", 0.1, -0.89, 0.89, 0.01),
    ],
    build=build,
    default_T=20.0,
    default_method="AUTO",
    two_dim=False,
    icon="🔁",
    category="Avtomat",
)

register(PRESET)
