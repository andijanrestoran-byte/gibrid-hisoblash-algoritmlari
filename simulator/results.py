"""Simulyatsiya natijasini JSON tuzilmaga aylantirish (serializatsiya).

Bu modul Django'ga bog'liq emas, lekin core ni ishlatadi. U HybridTrajectory
ni frontend va baza uchun yagona, JSON ga seriyalanadigan lug'atga
o'giradi: grafik ma'lumotlari, hodisalar jadvali, usul jurnali va xulosa.
"""

from __future__ import annotations

import csv
import io
from typing import Dict, List

import numpy as np

from .core.hybrid_automaton import HybridTrajectory
from .core.plotting import build_plot_data


def _vec(arr) -> List[float]:
    return [float(v) for v in np.asarray(arr, dtype=float).ravel()]


def _json_safe(obj):
    """NaN/Inf qiymatlarni None ga almashtiradi (yaroqli JSON kafolati).

    JSON standartida NaN/Infinity yo'q; ular brauzerdagi JSON.parse ni
    buzadi. Bu funksiya natijani rekursiv tozalaydi.
    """
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def serialize_trajectory(traj: HybridTrajectory, two_dim: bool = False) -> Dict:
    """Traektoriyani to'liq JSON tuzilmaga aylantiradi."""
    events_table = [
        {
            "time": float(ev.time),
            "from_mode": ev.from_mode,
            "to_mode": ev.to_mode,
            "transition_name": ev.transition_name,
            "x_before": _vec(ev.x_before),
            "x_after": _vec(ev.x_after),
            "localized_by": ev.localized_by,
            "bisection_time": (
                None if ev.bisection_time is None else float(ev.bisection_time)
            ),
        }
        for ev in traj.events
    ]

    log_table = [
        {
            "t_start": float(log.t_start),
            "t_end": float(log.t_end),
            "mode": log.mode,
            "method": log.method,
            "n_steps": int(log.n_steps),
            "nfev": int(log.nfev),
            "stiffness": float(log.stiffness),
            "switched": bool(log.switched),
            "reason": log.reason,
        }
        for log in traj.interval_log
    ]

    result = {
        "summary": {
            "success": bool(traj.success),
            "zeno": bool(traj.zeno),
            "message": traj.message,
            "n_events": len(traj.events),
            "n_pieces": len(traj.pieces),
            "t_start": traj.t_start,
            "t_end": traj.t_end,
        },
        "var_names": list(traj.var_names),
        "two_dim": bool(two_dim and traj.dim >= 2),
        "plot": build_plot_data(traj),
        "events_table": events_table,
        "log_table": log_table,
    }
    # Yaroqli JSON kafolati: NaN/Inf -> None.
    return _json_safe(result)


def trajectory_to_csv(traj: HybridTrajectory) -> str:
    """Traektoriyani CSV matnga aylantiradi (naive tugunlar bo'yicha).

    Ustunlar: t, <o'zgaruvchilar...>, mode, method. Har bo'lakdagi solver
    tugunlari ketma-ket yoziladi.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    header = ["t"] + list(traj.var_names) + ["mode", "method"]
    writer.writerow(header)
    for piece in traj.pieces:
        for i in range(len(piece.t)):
            row = [f"{piece.t[i]:.12g}"]
            row += [f"{piece.x[i, j]:.12g}" for j in range(piece.x.shape[1])]
            row += [piece.mode, piece.method]
            writer.writerow(row)
    return buf.getvalue()
