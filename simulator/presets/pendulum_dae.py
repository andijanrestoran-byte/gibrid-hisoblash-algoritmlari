"""Cheklovli mayatnik — DAE va indeksni qisqartirish namunasi (§2.1).

Dekart koordinatalarida ipga osilgan mayatnik klassik algebraik-differensial
tenglamalar (DAE) misoli bo'lib, uning indeksi 3 ga teng:

    x'' = -λ·x
    y'' = -λ·y - g
    0   = x² + y² - L²          (geometrik cheklov — indeks-3)

bu yerda λ — ipdagi taranglik (algebraik o'zgaruvchi), L — ip uzunligi.

Indeksni qisqartirish: x² + y² - L² = 0 cheklovini vaqt bo'yicha ikki marta
differensiallaymiz:
    1-marta:  x·x' + y·y' = 0                         (tezlik darajasi)
    2-marta:  x'² + y'² + x·x'' + y·y'' = 0           (tezlanish darajasi)
x'' va y'' ni qo'yib, λ uchun ALGEBRAIK tenglama hosil bo'ladi:
    λ·(x² + y²) = x'² + y'² - g·y
Bu — indeks-1 yarim-oshkor DAE. Endi λ ni har bir qadamda bevosita (Nyuton
usuli bilan) topib, sistema differensial o'zgaruvchilar [x, y, vx, vy] bo'yicha
yechiladi.

Fazaviy portretda (x vs y) yechim L radiusli aylana bo'ylab harakatlanadi —
bu algebraik cheklov saqlanayotganini ko'rsatadi.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from ..core.dae import DAESystem
from .registry import ParamSpec, Preset, register


def build(params: Dict[str, float]) -> DAESystem:
    L = float(params["L"])
    g = float(params["g"])
    theta0 = math.radians(float(params["theta0"]))  # gradusdan radianga
    omega0 = float(params["omega0"])

    # Boshlang'ich holat: pastdan theta burchakda.
    px = L * math.sin(theta0)
    py = -L * math.cos(theta0)
    # Tezlik cheklovga mos (urinma yo'nalishda): v = omega0·L·(cosθ, sinθ).
    vx = omega0 * L * math.cos(theta0)
    vy = omega0 * L * math.sin(theta0)
    # λ boshlang'ich mos taxminiy qiymati.
    lam0 = (vx * vx + vy * vy - g * py) / (px * px + py * py)

    def f(t, x, y):
        # x = [px, py, vx, vy];  y = [λ]
        lam = y[0]
        return np.array([x[2], x[3], -lam * x[0], -lam * x[1] - g])

    def cons(t, x, y):
        # Algebraik cheklov (indeks-1): λ·(x²+y²) - (vx²+vy² - g·y) = 0
        lam = y[0]
        return np.array([lam * (x[0] ** 2 + x[1] ** 2)
                         - (x[2] ** 2 + x[3] ** 2 - g * x[1])])

    return DAESystem(
        f=f,
        g=cons,
        x0=[px, py, vx, vy],
        y0=[lam0],
        x_names=["x", "y", "vx", "vy"],
        y_names=["λ (taranglik)"],
    )


PRESET = Preset(
    key="pendulum_dae",
    name="Cheklovli mayatnik (DAE)",
    description=(
        "Dekart koordinatalaridagi mayatnik — indeks-3 algebraik-differensial "
        "tenglama. Geometrik cheklov ikki marta differensiallanib indeks-1 ga "
        "keltiriladi, taranglik λ har qadamda Nyuton usuli bilan topiladi."
    ),
    notes=(
        "Indeksni qisqartirish (§2.1) namunasi. Fazaviy portret (x vs y) — "
        "L radiusli aylana; bu cheklov x²+y²=L² saqlanayotganini ko'rsatadi. "
        "λ qiymati 'Hisob jurnali' yonidagi grafikda kuzatiladi."
    ),
    variables=["x", "y", "vx", "vy", "λ"],
    params=[
        ParamSpec("L", "Ip uzunligi L", 1.0, 0.1, 10.0, 0.1, "m"),
        ParamSpec("g", "Erkin tushish tezlanishi g", 9.81, 0.1, 50.0, 0.01, "m/s²"),
        ParamSpec("theta0", "Boshlang'ich burchak θ₀", 60.0, -179.0, 179.0, 1.0,
                  "gradus (pastki vaziyatdan)"),
        ParamSpec("omega0", "Boshlang'ich burchak tezligi ω₀", 0.0, -10.0, 10.0,
                  0.1, "rad/s"),
    ],
    build=build,
    default_T=10.0,
    default_method="AUTO",
    default_rtol=1e-8,
    default_atol=1e-10,
    two_dim=True,
    icon="🕰️",
    category="DAE / indeks",
    kind="dae",
    comparable=False,  # DAE alohida solver bilan yechiladi (ODE qiyoslash mos emas)
    # Strukturaviy tahlil uchun 1-tartibli ko'rinish (indeks-3 → indeks-1).
    structure={
        "title": "Mayatnik DAE — strukturaviy tahlil (1-tartibli ko'rinish)",
        "equations": [
            {"name": "x' = vx", "vars": ["x'", "vx"]},
            {"name": "y' = vy", "vars": ["y'", "vy"]},
            {"name": "vx' = -λ·x", "vars": ["vx'", "lam", "x"]},
            {"name": "vy' = -λ·y - g", "vars": ["vy'", "lam", "y"]},
            {"name": "x² + y² = L²", "vars": ["x", "y"], "constraint": True},
        ],
    },
)

register(PRESET)
