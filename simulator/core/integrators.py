"""Qat'iy (doimiy) qadamli oshkora sonli usullar (dissertatsiya §2.3, §3.1).

Dissertatsiyaning 2–5-jadvallari turli sonli usullarning bir xil tenglamada
TO'G'RI yoki NOTO'G'RI grafik berishini ko'rsatadi. Bu "noto'g'ri" hodisa,
asosan, qadamni avtomatik moslamaydigan (yoki qattiq qadamli) oshkora usullarda
yuzaga keladi: qat'iy (stiff) masalada qadam barqarorlik sohasidan chiqib,
yechim "portlaydi" yoki keskin cho'qqi yetarlicha tugun bo'lmagani uchun
yo'qoladi.

scipy'ning RK45/Radau/BDF/LSODA usullari ichki adaptiv qadam nazoratiga ega
bo'lgani uchun ko'pincha to'g'ri (lekin qat'iy masalada juda qimmat) natija
beradi. Shu sababli dissertatsiyaning "Noto'g'ri" hukmini halol namoyish etish
uchun bu modul DOIMIY QADAMLI usullarni qo'shadi:

  * `euler`  — oshkora (forward) Euler usuli (1-tartib). §3.1 "SIMPLE_AUTO".
  * `rk4`    — doimiy qadamli klassik 4-tartibli Runge-Kutta. §3.1.

Ikkala usul ham faqat o'ng tomon f(t, x) ni baholaydi — hech qanday `eval`
yo'q. Ular bir rejimli (transitionsiz) uzluksiz ODE test modellari uchun
mo'ljallangan (dissertatsiyaning §2.3/§3.2 test funksiyalari).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

VectorField = Callable[[float, np.ndarray], np.ndarray]


def euler(
    f: VectorField, t_span: Tuple[float, float], x0: np.ndarray, n_steps: int
) -> "Tuple[np.ndarray, np.ndarray]":
    """Oshkora Euler usuli: x_{k+1} = x_k + h·f(t_k, x_k).

    Qaytaradi: (t_nodes, x_nodes) — shakli (n_steps+1,) va (n_steps+1, dim).
    Yechim "portlasa" (inf/nan), shu nuqtagacha bo'lgan tugunlar qaytadi.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    x = np.array(x0, dtype=float)
    ts = np.empty(n_steps + 1, dtype=float)
    xs = np.empty((n_steps + 1, x.size), dtype=float)
    ts[0] = t0
    xs[0] = x
    # Beqaror usul portlashi (overflow/nan) — kutilgan holat, ogohlantirmaymiz.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(n_steps):
            t = t0 + k * h
            x = x + h * np.asarray(f(t, x), dtype=float)
            ts[k + 1] = t + h
            xs[k + 1] = x
            if not np.all(np.isfinite(x)):
                return ts[: k + 2], xs[: k + 2]
    return ts, xs


def rk4(
    f: VectorField, t_span: Tuple[float, float], x0: np.ndarray, n_steps: int
) -> "Tuple[np.ndarray, np.ndarray]":
    """Doimiy qadamli klassik 4-tartibli Runge-Kutta usuli.

    Qaytaradi: (t_nodes, x_nodes). Yechim "portlasa" shu nuqtagacha qaytadi.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    h = (t1 - t0) / n_steps
    x = np.array(x0, dtype=float)
    ts = np.empty(n_steps + 1, dtype=float)
    xs = np.empty((n_steps + 1, x.size), dtype=float)
    ts[0] = t0
    xs[0] = x
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(n_steps):
            t = t0 + k * h
            k1 = np.asarray(f(t, x), dtype=float)
            k2 = np.asarray(f(t + 0.5 * h, x + 0.5 * h * k1), dtype=float)
            k3 = np.asarray(f(t + 0.5 * h, x + 0.5 * h * k2), dtype=float)
            k4 = np.asarray(f(t + h, x + h * k3), dtype=float)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            ts[k + 1] = t + h
            xs[k + 1] = x
            if not np.all(np.isfinite(x)):
                return ts[: k + 2], xs[: k + 2]
    return ts, xs


FIXED_STEP_METHODS = {"Euler": euler, "RK4": rk4}


def integrate_on_grid(
    method: str,
    f: VectorField,
    t_span: Tuple[float, float],
    x0: np.ndarray,
    grid: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Qat'iy qadamli usulni integrallab, natijani berilgan vaqt to'riga
    chiziqli interpolyatsiya qiladi (naive grafik kabi).

    Qaytaradi: shakli (len(grid), dim). Yechim portlagan joydan keyin NaN.
    """
    fn = FIXED_STEP_METHODS[method]
    ts, xs = fn(f, t_span, np.asarray(x0, dtype=float), n_steps)
    grid = np.asarray(grid, dtype=float)
    dim = xs.shape[1]
    out = np.full((grid.size, dim), np.nan, dtype=float)
    if ts.size < 2:
        return out
    t_max = ts[-1]
    for j in range(dim):
        col = xs[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            continue
        # Faqat integrallash yetib borgan oraliqda interpolyatsiya qilamiz.
        vals = np.interp(grid, ts, col, left=col[0], right=np.nan)
        vals[grid > t_max + 1e-12] = np.nan
        out[:, j] = vals
    return out
