"""Algebraik-differensial tenglamalar (DAE) va indeksni qisqartirish.

Dissertatsiyaning 2.1-§ qismi gibrid tizimlarni algebraik-differensial
tenglamalar (DAE) ko'rinishida ifodalash va ularning indeksini kamaytirishga
bag'ishlangan. Bu modul shu g'oyani amalga oshiradi.

Yarim-oshkor (semi-explicit) DAE quyidagi ko'rinishda bo'ladi:

    dx/dt = f(t, x, y)        (differensial qism)
    0     = g(t, x, y)        (algebraik cheklov)

bu yerda x — differensial o'zgaruvchilar, y — algebraik o'zgaruvchilar.

Indeks — algebraik cheklovni necha marta differensiallaganda oshkor ODE
hosil bo'lishini bildiradi. Indeks-1 DAE da g(t, x, y) = 0 dan y ni bevosita
(har bir qadamda Nyuton usuli bilan) topish mumkin. Yuqori indeksli DAE
(masalan, mayatnik — indeks-3) avval differensiallash orqali indeks-1 ga
keltiriladi (qarang: presets/pendulum_dae.py).

Yechish strategiyasi (indeks-1 uchun):
  - Har bir f baholanishida g(t, x, y) = 0 tenglama y bo'yicha Nyuton usuli
    bilan yechiladi (oldingi y dan iliq start bilan).
  - Natijada masala x bo'yicha oddiy ODE ga keltiriladi va mavjud
    `simulate()` mexanizmi (AUTO usul almashinuvi bilan) qo'llaniladi.
  - So'ng traektoriya algebraik y qiymatlari bilan to'ldiriladi (grafik uchun).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from .hybrid_automaton import HybridAutomaton, HybridTrajectory, Mode, Piece
from .solver_manager import SimulationConfig, simulate


@dataclass
class DAESystem:
    """Yarim-oshkor indeks-1 DAE ta'rifi.

    Atributlar:
        f: differensial o'ng tomon f(t, x, y) -> dx/dt.
        g: algebraik cheklov g(t, x, y) -> qoldiq (nolga teng bo'lishi kerak).
        x0: differensial o'zgaruvchilar boshlang'ich qiymati.
        y0: algebraik o'zgaruvchilar boshlang'ich (mos) taxminiy qiymati.
        x_names: differensial o'zgaruvchilar nomlari.
        y_names: algebraik o'zgaruvchilar nomlari.
    """

    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
    g: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
    x0: np.ndarray
    y0: np.ndarray
    x_names: List[str] = field(default_factory=list)
    y_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.x0 = np.array(self.x0, dtype=float)
        self.y0 = np.array(self.y0, dtype=float)


def _alg_jacobian(
    g_of_y: Callable[[np.ndarray], np.ndarray], y: np.ndarray, r0: np.ndarray
) -> np.ndarray:
    """Algebraik qoldiqning y bo'yicha Yakobianini chekli ayirma bilan hisoblaydi."""
    m = y.size
    jac = np.empty((r0.size, m), dtype=float)
    for j in range(m):
        dy = 1e-7 * max(1.0, abs(y[j]))
        yp = y.copy()
        yp[j] += dy
        jac[:, j] = (np.asarray(g_of_y(yp), dtype=float) - r0) / dy
    return jac


def solve_algebraic(
    g_of_y: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> np.ndarray:
    """g(y) = 0 algebraik tenglamani Nyuton usuli bilan yechadi.

    y0 — iliq start (oldingi qadamdagi yechim). Sof son usuli; foydalanuvchi
    ifodalarini eval qilmaydi.
    """
    y = np.array(y0, dtype=float)
    r = np.atleast_1d(np.asarray(g_of_y(y), dtype=float))
    for _ in range(max_iter):
        if np.linalg.norm(r) < tol:
            break
        jac = _alg_jacobian(g_of_y, y, r)
        try:
            dy = np.linalg.solve(jac, -r)
        except np.linalg.LinAlgError:
            dy = np.linalg.lstsq(jac, -r, rcond=None)[0]
        y = y + dy
        r = np.atleast_1d(np.asarray(g_of_y(y), dtype=float))
    return y


def _augment_with_algebraic(traj: HybridTrajectory, dae: DAESystem) -> None:
    """Traektoriyani algebraik o'zgaruvchilar bilan to'ldiradi (grafik uchun).

    Har bir bo'lakdagi x tugunlari uchun g(t, x, y) = 0 dan y topiladi va
    holat vektoriga qo'shiladi. Zich chiqish ham [x; y] ni qaytaradigan
    qilib o'raladi.
    """
    dy = len(dae.y_names)
    new_pieces: List[Piece] = []
    for piece in traj.pieces:
        x_nodes = np.asarray(piece.x, dtype=float)
        y_vals = np.empty((x_nodes.shape[0], dy), dtype=float)
        y_guess = np.array(dae.y0, dtype=float)
        for i in range(x_nodes.shape[0]):
            ti = float(piece.t[i])
            xi = x_nodes[i]
            yi = solve_algebraic(
                lambda yy, _t=ti, _x=xi: np.asarray(dae.g(_t, _x, yy), dtype=float),
                y_guess,
            )
            y_guess = yi
            y_vals[i] = yi
        combined = np.hstack([x_nodes, y_vals])

        # Zich chiqishni [x; y] ni qaytaradigan qilib o'raymiz.
        x_dense = piece.dense

        def dense(t, _xd=x_dense, _dae=dae):
            xv = np.atleast_1d(np.asarray(_xd(t), dtype=float))
            yv = solve_algebraic(
                lambda yy, _t=float(t), _x=xv: np.asarray(
                    _dae.g(_t, _x, yy), dtype=float
                ),
                np.array(_dae.y0, dtype=float),
            )
            return np.concatenate([xv, yv])

        new_pieces.append(
            Piece(mode=piece.mode, method=piece.method, t=piece.t,
                  x=combined, dense=dense if x_dense is not None else None)
        )

    traj.pieces = new_pieces
    traj.var_names = list(dae.x_names) + list(dae.y_names)


def solve_dae(dae: DAESystem, config: SimulationConfig) -> HybridTrajectory:
    """Indeks-1 yarim-oshkor DAE ni integrallaydi.

    Algebraik cheklov har bir f baholanishida y bo'yicha yechiladi va masala
    x bo'yicha ODE ga keltirilib, mavjud `simulate()` (AUTO usul almashinuvi
    bilan) qo'llaniladi. Natija — algebraik o'zgaruvchilar bilan to'ldirilgan
    HybridTrajectory, shu sababli butun vizualizatsiya/CSV mexanizmi
    o'zgarishsiz ishlaydi.
    """
    y_cache = {"y": np.array(dae.y0, dtype=float)}

    def reduced_f(t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = solve_algebraic(
            lambda yy: np.asarray(dae.g(t, x, yy), dtype=float), y_cache["y"]
        )
        y_cache["y"] = y
        return np.asarray(dae.f(t, x, y), dtype=float)

    automaton = HybridAutomaton(
        modes=[Mode("dae", reduced_f, description="DAE (indeks-1 ga keltirilgan)")],
        transitions=[],
        initial_mode="dae",
        initial_state=dae.x0,
        var_names=list(dae.x_names),
    )
    traj = simulate(automaton, config)
    if traj.pieces:
        _augment_with_algebraic(traj, dae)
    return traj
