"""Keskin cho'qqi — naive va adaptiv grafikni taqqoslash misoli.

Funksiya:
    f(t) = t + A·exp(-(t - c)² / w)
Tenglama:
    dx/dt = f'(t),   x(0) = f(0)
Demak aniq yechim x(t) = f(t). c = 1, w = 0.01 da t = 1 atrofida juda
ingichka (kenglik ~0.1) va baland (~A) cho'qqi paydo bo'ladi.

Dissertatsiyaning asosiy natijasi: solver yechimni TO'G'RI hisoblasa ham,
agar grafik faqat solver tugunlari orqali to'g'ri chiziq bilan chizilsa,
cho'qqi yo'qolib ketishi mumkin (Mathematica/Simulink ham shu xatoga yo'l
qo'ygan). Adaptiv rejim zich chiqishdan foydalanib cho'qqini saqlaydi.

Shu sababli bu misolda naive/adaptiv taqqoslash standart ravishda yoqilgan.
T = 2, 4, 12 qiymatlarida farq yaqqol ko'rinadi.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.hybrid_automaton import HybridAutomaton, Mode, Transition
from .registry import ParamSpec, Preset, register


def build(params: Dict[str, float]) -> HybridAutomaton:
    A = float(params["A"])
    c = float(params["c"])
    w = float(params["w"])

    def f_value(t):
        return t + A * np.exp(-((t - c) ** 2) / w)

    def f_prime(t):
        # f'(t) = 1 + A·exp(-(t-c)²/w)·(-2(t-c)/w)
        gauss = A * np.exp(-((t - c) ** 2) / w)
        return 1.0 + gauss * (-2.0 * (t - c) / w)

    def f(t, s):
        # Holat x ga bog'liq emas — sof kvadratura: x' = f'(t)
        return np.array([f_prime(t)])

    x0 = f_value(0.0)
    mode = Mode(name="yagona", f=f, description="dx/dt = f'(t)")

    return HybridAutomaton(
        modes=[mode],
        transitions=[],  # diskret o'tishlar yo'q — sof uzluksiz masala
        initial_mode="yagona",
        initial_state=[x0],
        var_names=["x"],
        max_events=10000,
    )


def make_exact(params: Dict[str, float]):
    """Aniq yechim: x(t) = f(t) = t + A·exp(-(t-c)²/w) (sof kvadratura)."""
    A = float(params["A"])
    c = float(params["c"])
    w = float(params["w"])

    def exact(t):
        return np.array([t + A * np.exp(-((t - c) ** 2) / w)])

    return exact


PRESET = Preset(
    key="sharp_peak",
    name="Keskin cho'qqi (naive vs adaptiv)",
    description=(
        "Yechim t = 1 atrofida juda ingichka va baland cho'qqiga ega. "
        "Naive grafik (faqat solver tugunlari) cho'qqini yo'qotishi mumkin, "
        "adaptiv grafik esa uni saqlaydi."
    ),
    notes=(
        "Dissertatsiyaning markaziy natijasi: to'g'ri yechim ≠ to'g'ri grafik. "
        "T = 2, 4, 12 da naive va adaptiv qatorlarni yonma-yon solishtiring."
    ),
    variables=["x"],
    params=[
        ParamSpec("A", "Cho'qqi balandligi A", 1.0, 0.0, 100.0, 0.1),
        ParamSpec("c", "Cho'qqi markazi c", 1.0, 0.0, 50.0, 0.1),
        ParamSpec("w", "Cho'qqi kengligi w", 0.01, 1e-4, 5.0, 0.001,
                  "Kichik w — ingichka cho'qqi (qiyinroq holat)"),
    ],
    build=build,
    exact=make_exact,
    default_T=4.0,
    default_method="AUTO",
    compare_default=True,  # naive/adaptiv taqqoslash standart yoniq
    two_dim=False,
    icon="📈",
    category="Vizualizatsiya",
)

register(PRESET)
