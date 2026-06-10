"""Van der Pol ostsilyatori — nochiziqli test modeli (§3.2).

Dissertatsiya nochiziqli test modellarini dx/dt = F(x) ko'rinishida, muvozanat
nuqtalari oldindan ma'lum bo'lgan holda qaraydi. Bunday modellar algoritmning:
  - nochiziqlikka moslashuvchanligini,
  - qat'iylikni aniqlash qobiliyatini,
  - usulni avtomatik almashtirishini
baholashga xizmat qiladi.

Van der Pol ostsilyatori shu maqsadlar uchun klassik misol:

    x' = y
    y' = μ·(1 - x²)·y - x

Yagona muvozanat nuqtasi — (0, 0) (beqaror fokus), uning atrofida turg'un
chegaraviy sikl (limit cycle) mavjud. μ katta bo'lganda yechim "relaksatsion
tebranish"ga aylanadi: tez va sekin bosqichlar almashadi — ya'ni sistema
nochiziqli bo'lsa-da QAT'IY bo'lib qoladi. Shu sababli AUTO rejim μ katta
bo'lganda ochiq RK45 dan yashirin Radau ga avtomatik o'tadi.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.hybrid_automaton import HybridAutomaton, Mode
from .registry import ParamSpec, Preset, register


def build(params: Dict[str, float]) -> HybridAutomaton:
    mu = float(params["mu"])
    x0 = float(params["x0"])
    y0 = float(params["y0"])

    def f(t, s):
        # s = [x, y];  x' = y,  y' = μ(1-x²)y - x
        return np.array([s[1], mu * (1.0 - s[0] ** 2) * s[1] - s[0]])

    mode = Mode(name="ostsilyator", f=f,
                description="x'=y, y'=μ(1-x²)y-x (nochiziqli)")

    return HybridAutomaton(
        modes=[mode],
        transitions=[],
        initial_mode="ostsilyator",
        initial_state=[x0, y0],
        var_names=["x", "y"],
        max_events=10000,
    )


PRESET = Preset(
    key="van_der_pol",
    name="Van der Pol ostsilyatori",
    description=(
        "Nochiziqli ostsilyator: muvozanat nuqtasi (0,0), atrofida turg'un "
        "chegaraviy sikl. μ katta bo'lganda relaksatsion (qat'iy) tebranishga "
        "aylanadi va AUTO rejim yashirin usulga o'tadi."
    ),
    notes=(
        "Nochiziqli test modeli (§3.2): nochiziqlik + qat'iylikni aniqlash + "
        "usul almashinuvini birgalikda namoyish etadi. μ ni oshirib hisob "
        "jurnalida RK45 → Radau o'tishini kuzating. Fazaviy portret — chegaraviy sikl."
    ),
    variables=["x", "y"],
    params=[
        ParamSpec("mu", "Nochiziqlik/qat'iylik parametri μ", 8.0, 0.1, 100.0, 0.1,
                  "μ katta — relaksatsion (qat'iy) tebranish"),
        ParamSpec("x0", "Boshlang'ich x₀", 2.0, -5.0, 5.0, 0.1),
        ParamSpec("y0", "Boshlang'ich y₀", 0.0, -5.0, 5.0, 0.1),
    ],
    build=build,
    default_T=25.0,
    default_method="AUTO",
    two_dim=True,
    icon="➰",
    category="Nochiziqli",
)

register(PRESET)
