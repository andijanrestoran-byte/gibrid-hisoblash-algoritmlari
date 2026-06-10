"""Qat'iy (stiff) sistema — AUTO rejimning usul almashtirishini namoyish etadi.

Dissertatsiya (§3.2) dagi qat'iy model matritsasi:
    A_n = [[1 + 2⁻ⁿ,  1      ],
           [ 1,        1 + 2⁻ⁿ]]

A_n ning xos qiymatlari: λ₁ = 2 + 2⁻ⁿ,  λ₂ = 2⁻ⁿ.
n ortishi bilan bu xos qiymatlar orasidagi farq (qat'iylik nisbati
R = λ₁/λ₂ = 2ⁿ⁺¹ + 1) keskin ortadi — yechimda tez va sekin o'zgaruvchi
komponentlar paydo bo'ladi. Bu — dissertatsiyada ta'riflangan qat'iylik.

Turg'un (so'nuvchi) qat'iy tizim olish uchun tenglama dx/dt = -A_n·x
ko'rinishida qo'llaniladi: xos qiymatlar -(2+2⁻ⁿ) va -(2⁻ⁿ) bo'lib, tez
komponenta darhol so'nadi, sekin komponenta esa uzoq saqlanadi.

AUTO rejim (AUTO_ODE) avval ochiq RK45 dan boshlaydi, qat'iylik nisbati R
chegaradan oshganini aniqlaydi va avtomatik ravishda yashirin Radau (yoki
BDF) usuliga o'tadi. Bu hisob jurnalida ko'rinadi. n ni oshirib qat'iylikni
kuchaytiring (default n=6 da R≈129, n=8 da R≈513).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from scipy.linalg import expm

from ..core.hybrid_automaton import HybridAutomaton, Mode
from .registry import ParamSpec, Preset, register


def build(params: Dict[str, float]) -> HybridAutomaton:
    n = int(round(float(params["n"])))
    x0 = float(params["x0"])
    x1 = float(params["x1"])

    # Dissertatsiyadagi matritsa: A_n = [[1+2^-n, 1], [1, 1+2^-n]].
    p = 2.0 ** (-n)
    A = np.array([[1.0 + p, 1.0], [1.0, 1.0 + p]])

    def f(t, s):
        # Turg'un so'nuvchi qat'iy tizim: dx/dt = -A_n · x
        return -(A @ s)

    mode = Mode(name="qat'iy", f=f, description="x' = -A_n · x (stiff)")

    return HybridAutomaton(
        modes=[mode],
        transitions=[],
        initial_mode="qat'iy",
        initial_state=[x0, x1],
        var_names=["x₀", "x₁"],
        max_events=10000,
    )


def make_exact(params: Dict[str, float]):
    """Aniq yechim: x(t) = exp(-A_n·t)·x₀ (turg'un qat'iy chiziqli sistema)."""
    n = int(round(float(params["n"])))
    x_init = np.array([float(params["x0"]), float(params["x1"])])
    p = 2.0 ** (-n)
    A = np.array([[1.0 + p, 1.0], [1.0, 1.0 + p]])

    def exact(t):
        return expm(-A * t) @ x_init

    return exact


PRESET = Preset(
    key="stiff_demo",
    name="Qat'iy sistema (AUTO usul almashinuvi)",
    description=(
        "Xos qiymatlari keskin farq qiladigan chiziqli sistema. n ortgan "
        "sari qat'iylik kuchayadi va AUTO rejim ochiq RK45 dan yashirin "
        "Radau usuliga avtomatik o'tadi."
    ),
    notes=(
        "Dissertatsiya §3.2 qat'iy modeli: A_n=[[1+2⁻ⁿ,1],[1,1+2⁻ⁿ]]. Hisob "
        "jurnalida (4-tab) RK45 → Radau o'tishini va qat'iylik nisbati R ni "
        "kuzating. n ni oshirib qat'iylikni kuchaytiring."
    ),
    variables=["x₀", "x₁"],
    params=[
        ParamSpec("n", "Qat'iylik darajasi n", 6, 0, 16, 1,
                  "xos qiymatlar 2+2⁻ⁿ va 2⁻ⁿ; nisbati R≈2ⁿ⁺¹ — n katta, qat'iyroq",
                  integer=True),
        ParamSpec("x0", "Boshlang'ich x₀", 1.0, -10.0, 10.0, 0.1),
        ParamSpec("x1", "Boshlang'ich x₁", 0.0, -10.0, 10.0, 0.1),
    ],
    build=build,
    exact=make_exact,
    default_T=5.0,
    default_method="AUTO",
    two_dim=True,
    icon="⚡",
    category="Qat'iylik",
)

register(PRESET)
