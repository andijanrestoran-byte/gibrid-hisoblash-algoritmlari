"""Adaptiv grafik nuqtalarini tanlash (dissertatsiyaning asosiy natijasi).

Muhim kuzatuv: ODE yechimi son jihatdan to'g'ri bo'lsa ham, agar grafik
solver qaytargan kam sonli tugunlar orasini to'g'ri chiziq bilan tutashtirib
chizilsa, keskin cho'qqilar (peak) yo'qolib ketishi mumkin. Mathematica yoki
Simulink kabi tizimlar ham ba'zan shu sababli xato grafik chizadi.

Yechim — zich chiqish (dense output) funksiyasidan foydalanib, hosila katta
bo'lgan joylarda qo'shimcha nuqtalar qo'shish (adaptiv tanlash). Bu modul:

  * `naive_series`   — faqat solver tugunlari (sodda, lekin cho'qqi yo'qolishi
                       mumkin).
  * `adaptive_series`— rekursiv ikkiga bo'lish orqali egrilik katta joylarda
                       nuqta zichlashtiriladi (cho'qqilar saqlanadi).
  * `build_plot_data`— ikkala rejim uchun Plotly'ga tayyor JSON tuzilma
                       (vaqt grafigi, fazaviy portret, hodisa chiziqlari,
                       usul jurnali).

Bo'laklar (Piece) chegarasida reset tufayli sakrash bo'lishi mumkin; shu
joylarda chiziq uzilishi uchun qatorlarga `None` (JSON'da null) qo'yiladi.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .hybrid_automaton import HybridTrajectory, Piece


# Adaptiv tanlash sozlamalari.
DEFAULT_MAX_DEPTH = 16          # rekursiya maksimal chuqurligi
DEFAULT_REL_TOL = 1.5e-3        # nisbiy bo'sag'a (y-diapazoniga nisbatan)


def _refine(
    dense: Callable[[float], np.ndarray],
    ta: float,
    tb: float,
    xa: np.ndarray,
    xb: np.ndarray,
    depth: int,
    max_depth: int,
    tol: float,
    min_dt: float,
    out_t: List[float],
    out_x: List[np.ndarray],
) -> None:
    """[ta, tb] oralig'ini rekursiv ravishda zichlashtiradi.

    O'rta nuqtaning haqiqiy qiymati (dense orqali) chiziqli interpolyatsiyadan
    `tol` dan ko'proq farq qilsa, oraliq ikkiga bo'linadi. Chap qism, so'ng
    o'rta nuqta, so'ng o'ng qism qo'shiladi — natija vaqt bo'yicha tartibli
    bo'ladi. Endpointlar bu funksiyada qo'shilmaydi (chaqiruvchi qo'shadi).
    """
    if depth >= max_depth or (tb - ta) <= min_dt:
        return
    tm = 0.5 * (ta + tb)
    xm = dense(tm)
    lin = 0.5 * (xa + xb)
    err = float(np.max(np.abs(xm - lin)))
    if err > tol:
        _refine(dense, ta, tm, xa, xm, depth + 1, max_depth, tol, min_dt, out_t, out_x)
        out_t.append(tm)
        out_x.append(xm)
        _refine(dense, tm, tb, xm, xb, depth + 1, max_depth, tol, min_dt, out_t, out_x)


def adaptive_piece_points(
    piece: Piece,
    max_depth: int = DEFAULT_MAX_DEPTH,
    rel_tol: float = DEFAULT_REL_TOL,
) -> "tuple[np.ndarray, np.ndarray]":
    """Bitta bo'lak uchun adaptiv (t, x) nuqtalarini hisoblaydi.

    Asos sifatida solver tugunlari olinadi (umumiy tuzilma yo'qolmasin),
    keyin har bir qo'shni juftlik orasi kerak bo'lsa zichlashtiriladi.
    """
    base_t = np.asarray(piece.t, dtype=float)
    base_x = np.asarray(piece.x, dtype=float)
    if base_t.size <= 1 or piece.dense is None:
        return base_t, base_x

    # y-diapazoni bo'yicha bo'sag'a (har bir komponentaning eng katta amplitudasi).
    span_y = float(np.max(np.max(base_x, axis=0) - np.min(base_x, axis=0)))
    tol = rel_tol * span_y + 1e-12
    total_dt = float(base_t[-1] - base_t[0])
    min_dt = total_dt * 1e-6

    dense = piece.dense
    out_t: List[float] = [float(base_t[0])]
    out_x: List[np.ndarray] = [base_x[0]]
    for i in range(base_t.size - 1):
        ta, tb = float(base_t[i]), float(base_t[i + 1])
        xa, xb = base_x[i], base_x[i + 1]
        _refine(dense, ta, tb, xa, xb, 0, max_depth, tol, min_dt, out_t, out_x)
        out_t.append(tb)
        out_x.append(xb)

    return np.array(out_t, dtype=float), np.array(out_x, dtype=float)


def _to_float_list(arr: np.ndarray) -> List[float]:
    return [float(v) for v in np.asarray(arr, dtype=float).ravel()]


def _series_with_breaks(
    pieces_points: List["tuple[np.ndarray, np.ndarray]"], comp: int
) -> "tuple[List[Optional[float]], List[Optional[float]]]":
    """Bo'laklarni `None` uzilishi bilan ulab, bitta (t, y) qatorini yasaydi.

    Reset sakrashi joyida chiziq uzilishi uchun bo'laklar orasiga None
    qo'yiladi. Bu Plotly'da uzluksiz qism bilan sakrashni aralashtirib
    yubormaydi.
    """
    t_out: List[Optional[float]] = []
    y_out: List[Optional[float]] = []
    for k, (tp, xp) in enumerate(pieces_points):
        if k > 0:
            t_out.append(None)
            y_out.append(None)
        t_out.extend(float(v) for v in tp)
        y_out.extend(float(v) for v in xp[:, comp])
    return t_out, y_out


def _collect_points(
    traj: HybridTrajectory, mode: str, max_depth: int, rel_tol: float
) -> List["tuple[np.ndarray, np.ndarray]"]:
    """Har bir bo'lak uchun tanlangan rejimda (t, x) nuqtalarini yig'adi."""
    pts = []
    for piece in traj.pieces:
        if mode == "adaptive":
            tp, xp = adaptive_piece_points(piece, max_depth, rel_tol)
        else:  # naive
            tp, xp = np.asarray(piece.t, float), np.asarray(piece.x, float)
        pts.append((tp, xp))
    return pts


def build_series(
    traj: HybridTrajectory,
    mode: str = "adaptive",
    max_depth: int = DEFAULT_MAX_DEPTH,
    rel_tol: float = DEFAULT_REL_TOL,
) -> Dict:
    """Berilgan rejim uchun grafik qatorlarini (JSON tuzilma) qaytaradi.

    Tuzilma:
        {
          "mode": "naive"|"adaptive",
          "variables": [nomlar],
          "series": [{"name": var, "t": [...], "y": [...]}],   # vaqt grafigi
          "phase": {"x": [...], "y": [...], "x_name":, "y_name":} | None,
          "n_points": butun,
        }
    """
    pts = _collect_points(traj, mode, max_depth, rel_tol)
    var_names = list(traj.var_names)
    dim = len(var_names)

    series = []
    for comp in range(dim):
        t_out, y_out = _series_with_breaks(pts, comp)
        series.append({"name": var_names[comp], "t": t_out, "y": y_out})

    # Fazaviy portret (faqat 2 o'zgaruvchili tizimlar uchun).
    phase = None
    if dim >= 2:
        x_out: List[Optional[float]] = []
        y_out: List[Optional[float]] = []
        for k, (tp, xp) in enumerate(pts):
            if k > 0:
                x_out.append(None)
                y_out.append(None)
            x_out.extend(float(v) for v in xp[:, 0])
            y_out.extend(float(v) for v in xp[:, 1])
        phase = {
            "x": x_out,
            "y": y_out,
            "x_name": var_names[0],
            "y_name": var_names[1],
        }

    n_points = int(sum(len(tp) for tp, _ in pts))
    return {
        "mode": mode,
        "variables": var_names,
        "series": series,
        "phase": phase,
        "n_points": n_points,
    }


def build_plot_data(
    traj: HybridTrajectory,
    max_depth: int = DEFAULT_MAX_DEPTH,
    rel_tol: float = DEFAULT_REL_TOL,
) -> Dict:
    """Frontendga to'liq grafik ma'lumotini (naive + adaptiv) JSON shaklida qaytaradi.

    Bu — view/API qatlamiga uzatiladigan asosiy funksiya. Plotly bilan
    naive va adaptiv rejimlarni yonma-yon taqqoslash uchun ikkala qatorni
    ham o'z ichiga oladi, shuningdek hodisa chiziqlari va usul jurnalini.
    """
    naive = build_series(traj, "naive", max_depth, rel_tol)
    adaptive = build_series(traj, "adaptive", max_depth, rel_tol)

    events = [
        {
            "time": float(ev.time),
            "from_mode": ev.from_mode,
            "to_mode": ev.to_mode,
            "label": ev.transition_name or f"{ev.from_mode}→{ev.to_mode}",
        }
        for ev in traj.events
    ]

    methods = [
        {
            "t_start": float(log.t_start),
            "t_end": float(log.t_end),
            "method": log.method,
            "mode": log.mode,
            "switched": bool(log.switched),
        }
        for log in traj.interval_log
    ]

    return {
        "variables": list(traj.var_names),
        "naive": naive,
        "adaptive": adaptive,
        "events": events,
        "methods": methods,
        "t_start": traj.t_start,
        "t_end": traj.t_end,
    }
