"""Raketaning maqsadni ta'qib qilishi — diskret vaqtli (sample-data) gibrid tizim.

Dissertatsiya kirish qismida keltirilgan misol: raketa maqsadni ta'qib qiladi,
biroq maqsad haqidagi ma'lumotni olish va parvoz parametrlarini (yo'nalishni)
o'zgartirish faqat DISKRET vaqt momentlarida amalga oshiriladi.

Uzluksiz dinamika ("parvoz" rejimi) — raketa doimiy tezlik va yo'nalishda:
    (rx, ry)' = (vx, vy),   (vx, vy)' = 0,   τ' = 1   (τ — ichki soat)

Diskret o'tishlar:
    yo'nalishni yangilash:  τ = Δt  → raketa joriy maqsadga qarab qayta yo'naladi,
                                       tezlik moduli saqlanadi, τ := 0
    maqsadga yetish:        |raketa - maqsad| = r  → "to'xtash" rejimiga

Maqsad doimiy tezlik bilan harakatlanadi. Bu misol VAQTGA asoslangan
(state emas, balki soatga bog'liq) hodisalarni — ya'ni sample-data boshqaruvni
namoyish etadi.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.hybrid_automaton import HybridAutomaton, Mode, Transition
from .registry import ParamSpec, Preset, register

_EPS = 1e-9


def build(params: Dict[str, float]) -> HybridAutomaton:
    speed = float(params["speed"])
    dt = float(params["period"])
    r = float(params["catch_r"])
    txv = float(params["tgt_vx"])
    tyv = float(params["tgt_vy"])
    tx0, ty0 = float(params["tgt_x0"]), float(params["tgt_y0"])
    rx0, ry0 = float(params["rkt_x0"]), float(params["rkt_y0"])

    def target(t):
        return np.array([tx0 + txv * t, ty0 + tyv * t])

    def f_flight(t, s):
        # s = [rx, ry, vx, vy, τ]
        return np.array([s[2], s[3], 0.0, 0.0, 1.0])

    def f_stop(t, s):
        return np.zeros(5)

    def guard_update(t, s):
        return s[4] - dt  # τ Δt ga yetganda

    def reset_aim(t, s):
        rx, ry, vx, vy, tau = s
        tgt = target(t)
        d = max(np.hypot(tgt[0] - rx, tgt[1] - ry), _EPS)
        return np.array([rx, ry, speed * (tgt[0] - rx) / d,
                         speed * (tgt[1] - ry) / d, 0.0])

    def guard_catch(t, s):
        tgt = target(t)
        return np.hypot(s[0] - tgt[0], s[1] - tgt[1]) - r

    # Boshlang'ich yo'nalish — maqsadga qarab
    tgt0 = target(0.0)
    d0 = max(np.hypot(tgt0[0] - rx0, tgt0[1] - ry0), _EPS)
    vx0 = speed * (tgt0[0] - rx0) / d0
    vy0 = speed * (tgt0[1] - ry0) / d0

    flight = Mode(name="parvoz", f=f_flight, description="Doimiy yo'nalishdagi parvoz")
    stop = Mode(name="to'xtash", f=f_stop, description="Maqsadga yetildi")

    update = Transition(
        from_mode="parvoz", to_mode="parvoz",
        guard=guard_update, reset=reset_aim, direction=+1.0,
        name="Yo'nalishni yangilash (diskret moment)",
    )
    catch = Transition(
        from_mode="parvoz", to_mode="to'xtash",
        guard=guard_catch, direction=-1.0,
        name="Maqsadga yetildi",
    )

    return HybridAutomaton(
        modes=[flight, stop],
        transitions=[update, catch],
        initial_mode="parvoz",
        initial_state=[rx0, ry0, vx0, vy0, 0.0],
        var_names=["raketa x", "raketa y", "vx", "vy", "τ (soat)"],
        max_events=10000,
    )


PRESET = Preset(
    key="rocket_pursuit",
    name="Raketa maqsadni ta'qib qiladi (sample-data)",
    description=(
        "Raketa harakatlanuvchi maqsadni ta'qib qiladi, biroq yo'nalishni faqat "
        "diskret vaqt momentlarida (har Δt da) yangilaydi. Vaqtga asoslangan "
        "(sample-data) hodisalarning namunasi."
    ),
    notes=(
        "Yo'nalish yangilanishlari 'Hodisalar' jadvalida ko'rinadi — bu state "
        "emas, balki ichki soatga bog'liq hodisalar. Δt ni kichraytirib ta'qibni "
        "aniqlashtiring. Fazaviy portret — raketa traektoriyasi."
    ),
    variables=["raketa x", "raketa y", "vx", "vy", "τ (soat)"],
    params=[
        ParamSpec("speed", "Raketa tezligi", 1.6, 0.1, 20.0, 0.1),
        ParamSpec("period", "Yangilash davri Δt", 0.5, 0.02, 5.0, 0.01),
        ParamSpec("catch_r", "Yetish radiusi r", 0.2, 0.02, 3.0, 0.01),
        ParamSpec("tgt_x0", "Maqsad boshlang'ich x", 10.0, -30.0, 30.0, 0.5),
        ParamSpec("tgt_y0", "Maqsad boshlang'ich y", 0.0, -30.0, 30.0, 0.5),
        ParamSpec("tgt_vx", "Maqsad tezligi vx", 0.0, -5.0, 5.0, 0.1),
        ParamSpec("tgt_vy", "Maqsad tezligi vy", 1.0, -5.0, 5.0, 0.1),
        ParamSpec("rkt_x0", "Raketa boshlang'ich x", 0.0, -30.0, 30.0, 0.5),
        ParamSpec("rkt_y0", "Raketa boshlang'ich y", 0.0, -30.0, 30.0, 0.5),
    ],
    build=build,
    default_T=12.0,
    default_method="AUTO",
    two_dim=True,
    icon="🚀",
    category="Ta'qib",
)

register(PRESET)
