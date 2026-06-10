"""Mushuk va sichqon — ta'qib gibrid tizimi (dissertatsiya 1.3-rasm).

Sichqon doimiy tezlik bilan o'z uyasiga (teshik) tomon to'g'ri chiziq bo'ylab
yuguradi. Mushuk esa sichqonning joriy holatiga qarab ta'qib qiladi (ta'qib
egri chizig'i). Savol: mushuk sichqonni teshikka yetib borguncha ushlab
olaolmi?

Uzluksiz dinamika ("ta'qib" rejimi):
    mushuk:  (cx,cy)' = V_m · birlik(sichqon - mushuk)
    sichqon: (mx,my)' = V_s · birlik(teshik - sichqon)

Diskret o'tishlar (terminal hodisalar):
    ushlandi: |mushuk - sichqon| = r   → "to'xtash" rejimiga
    qochdi:   |sichqon - teshik| = r   → "to'xtash" rejimiga

Bu — ikki ishtirokchili ta'qib masalasi va hodisaga asoslangan o'tishlarning
namunasi.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.hybrid_automaton import HybridAutomaton, Mode, Transition
from .registry import ParamSpec, Preset, register


def build(params: Dict[str, float]) -> HybridAutomaton:
    vm = float(params["v_cat"])
    vs = float(params["v_mouse"])
    hx = float(params["hole_x"])
    hy = float(params["hole_y"])
    cx0, cy0 = float(params["cat_x"]), float(params["cat_y"])
    mx0, my0 = float(params["mouse_x"]), float(params["mouse_y"])
    r = float(params["catch_r"])

    def f_chase(t, s):
        cx, cy, mx, my = s
        # Mushuk sichqonga qarab harakatlanadi. Masofani ushlash radiusida
        # cheklaymiz — bu maydonni singulyar bo'lishdan saqlaydi (yaqinlashganda
        # 1/d² hadlar portlamaydi). r dan tashqarida dinamika o'zgarmaydi.
        dcm = max(np.hypot(mx - cx, my - cy), r)
        # Sichqon teshikka qarab harakatlanadi
        dmh = max(np.hypot(hx - mx, hy - my), r)
        return np.array([
            vm * (mx - cx) / dcm,
            vm * (my - cy) / dcm,
            vs * (hx - mx) / dmh,
            vs * (hy - my) / dmh,
        ])

    def f_stop(t, s):
        return np.zeros(4)

    def dist_cat_mouse(t, s):
        return np.hypot(s[0] - s[2], s[1] - s[3]) - r

    def dist_mouse_hole(t, s):
        return np.hypot(s[2] - hx, s[3] - hy) - r

    chase = Mode(name="ta'qib", f=f_chase, description="Mushuk sichqonni ta'qib qiladi")
    stop = Mode(name="to'xtash", f=f_stop, description="Harakat tugadi")

    caught = Transition(
        from_mode="ta'qib", to_mode="to'xtash",
        guard=dist_cat_mouse, direction=-1.0,
        name="Mushuk sichqonni ushladi",
    )
    escaped = Transition(
        from_mode="ta'qib", to_mode="to'xtash",
        guard=dist_mouse_hole, direction=-1.0,
        name="Sichqon teshikka yetdi (qochdi)",
    )

    return HybridAutomaton(
        modes=[chase, stop],
        transitions=[caught, escaped],
        initial_mode="ta'qib",
        initial_state=[cx0, cy0, mx0, my0],
        var_names=["mushuk x", "mushuk y", "sichqon x", "sichqon y"],
        max_events=10,
    )


PRESET = Preset(
    key="cat_mouse",
    name="Mushuk va sichqon (ta'qib)",
    description=(
        "Sichqon teshikka yuguradi, mushuk uni ta'qib qiladi. Mushuk ushlaydimi "
        "yoki sichqon qochadimi? Hodisaga asoslangan o'tishli gibrid tizim."
    ),
    notes=(
        "Dissertatsiya 1.3-rasm. Tezliklarni o'zgartirib natijani o'zgartiring: "
        "v_cat > v_mouse bo'lsa odatda ushlaydi. Fazaviy portret — mushuk yo'li."
    ),
    variables=["mushuk x", "mushuk y", "sichqon x", "sichqon y"],
    params=[
        ParamSpec("v_cat", "Mushuk tezligi V_m", 1.0, 0.1, 10.0, 0.1),
        ParamSpec("v_mouse", "Sichqon tezligi V_s", 0.78, 0.1, 10.0, 0.01),
        ParamSpec("cat_x", "Mushuk boshlang'ich x", 0.0, -20.0, 20.0, 0.5),
        ParamSpec("cat_y", "Mushuk boshlang'ich y", 0.0, -20.0, 20.0, 0.5),
        ParamSpec("mouse_x", "Sichqon boshlang'ich x", 5.0, -20.0, 20.0, 0.5),
        ParamSpec("mouse_y", "Sichqon boshlang'ich y", 4.0, -20.0, 20.0, 0.5),
        ParamSpec("hole_x", "Teshik x", 12.0, -20.0, 20.0, 0.5),
        ParamSpec("hole_y", "Teshik y", 4.0, -20.0, 20.0, 0.5),
        ParamSpec("catch_r", "Ushlash/teshik radiusi r", 0.1, 0.01, 2.0, 0.01),
    ],
    build=build,
    default_T=20.0,
    default_method="AUTO",
    two_dim=True,
    icon="🐱",
    category="Ta'qib",
)

register(PRESET)
