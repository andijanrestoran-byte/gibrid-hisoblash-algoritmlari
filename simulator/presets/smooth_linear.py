"""Qat'iy bo'lmagan chiziqli model — AUTO ochiq usulda qolishini ko'rsatadi.

Dissertatsiya (§3.2) qat'iy modelga (stiff_demo) qarama-qarshi misol sifatida
quyidagi qat'iy bo'lmagan chiziqli sistemani keltiradi:

    dx/dt = A·x,    A = [[-2,  1],
                        [ 1, -2]]

A ning xos qiymatlari: λ₁ = -1,  λ₂ = -3. Ular bir xil tartibda bo'lgani
uchun (qat'iylik nisbati R = 3) yechim vaqt bo'yicha silliq o'zgaradi va
masala qat'iylik xususiyatiga ega emas. Shu sababli AUTO rejim butun davomida
ochiq RK45 usulida qoladi — buni hisob jurnalida (4-tab) ko'rish mumkin.

Bu misol stiff_demo bilan birgalikda AUTO_ODE algoritmining ikki xil
xulq-atvorini (silliq → RK45, qat'iy → Radau) yonma-yon namoyish etadi.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from scipy.linalg import expm

from ..core.hybrid_automaton import HybridAutomaton, Mode
from .registry import ParamSpec, Preset, register


def build(params: Dict[str, float]) -> HybridAutomaton:
    d = float(params["d"])
    c = float(params["c"])
    x0 = float(params["x0"])
    x1 = float(params["x1"])

    # A = [[-d, c], [c, -d]];  xos qiymatlar -d+c va -d-c.
    A = np.array([[-d, c], [c, -d]])

    def f(t, s):
        return A @ s

    mode = Mode(name="silliq", f=f, description="x' = A·x (qat'iy emas)")

    return HybridAutomaton(
        modes=[mode],
        transitions=[],
        initial_mode="silliq",
        initial_state=[x0, x1],
        var_names=["x₀", "x₁"],
        max_events=10000,
    )


def make_exact(params: Dict[str, float]):
    """Aniq yechim: x(t) = exp(A·t)·x₀ (chiziqli sistema matritsa eksponentasi)."""
    d = float(params["d"])
    c = float(params["c"])
    x_init = np.array([float(params["x0"]), float(params["x1"])])
    A = np.array([[-d, c], [c, -d]])

    def exact(t):
        return expm(A * t) @ x_init

    return exact


PRESET = Preset(
    key="smooth_linear",
    name="Qat'iy bo'lmagan chiziqli model",
    description=(
        "Xos qiymatlari bir xil tartibda (λ=-1, -3) bo'lgan silliq chiziqli "
        "sistema. AUTO rejim butun davomida ochiq RK45 usulida qoladi "
        "(stiff_demo ga qarama-qarshi misol)."
    ),
    notes=(
        "Dissertatsiya §3.2: qat'iy modelga qarshi qo'yilgan qat'iy bo'lmagan "
        "test modeli. Hisob jurnalida usul RK45 da o'zgarmay qolishini kuzating."
    ),
    variables=["x₀", "x₁"],
    params=[
        ParamSpec("d", "Diagonal d", 2.0, 0.1, 20.0, 0.1,
                  "A=[[-d,c],[c,-d]]; xos qiymatlar -d±c"),
        ParamSpec("c", "Bog'lanish c", 1.0, 0.0, 19.0, 0.1),
        ParamSpec("x0", "Boshlang'ich x₀", 1.0, -10.0, 10.0, 0.1),
        ParamSpec("x1", "Boshlang'ich x₁", 0.0, -10.0, 10.0, 0.1),
    ],
    build=build,
    exact=make_exact,
    default_T=5.0,
    default_method="AUTO",
    two_dim=True,
    icon="〰️",
    category="Qat'iylik",
)

register(PRESET)
