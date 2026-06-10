"""Usullarni qiyoslash va aniqlikni baholash (dissertatsiya §2.3, §3.3).

Dissertatsiyaning amaliy o'zagi — bir xil differensial tenglamani turli sonli
usullar bilan yechib, qaysi usul TO'G'RI, qaysi biri NOTO'G'RI grafik/yechim
berishini jadval ko'rinishida solishtirish (2–5-jadvallar: Maple/Mathematica/
Matlab usullari uchun "To'g'ri/Noto'g'ri"). Bu modul shu jadvalni qayta
hosil qiladi:

  * `sample_on_grid`  — traektoriyani bir tekis vaqt to'rida namunalaydi
                        (bo'laklarning zich chiqishidan foydalanib).
  * `compare_methods` — RK45/Radau/BDF/LSODA usullarining har birini ishga
                        tushiradi, aniq yechim (yoki yuqori aniqlikdagi etalon)
                        ga nisbatan maksimal va nisbiy xatoni, nfev va qadamlar
                        sonini hisoblaydi hamda sifat hukmini (To'g'ri/Noto'g'ri)
                        chiqaradi.

Aniqlik mezoni: nisbiy maksimal xato `VERDICT_REL_TOL` dan kichik bo'lsa,
usul "To'g'ri" deb baholanadi (ya'ni grafik vizual jihatdan haqiqiy yechimga
mos keladi). Bu dissertatsiyaning sifat jihatidagi "To'g'ri/Noto'g'ri"
tasnifiga to'g'ri keladi.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional

import numpy as np

from .hybrid_automaton import HybridAutomaton, HybridTrajectory
from .integrators import FIXED_STEP_METHODS, integrate_on_grid
from .solver_manager import SimulationConfig, simulate

# Qiyoslanadigan adaptiv usullar (scipy: ochiq/yopiq, bir/ko'p qadamli).
COMPARE_METHODS = ("RK45", "Radau", "BDF", "LSODA")

# Doimiy qadamli oshkora usullar (dissertatsiya §3.1) — bir rejimli uzluksiz
# test modellarida qo'shimcha qiyoslanadi. Qat'iy masalada beqarorlashib
# "Noto'g'ri" beradi (2–5-jadvallardagi hodisa).
FIXED_STEP_NAMES = tuple(FIXED_STEP_METHODS.keys())  # ("Euler", "RK4")
FIXED_STEP_COUNT = 400  # doimiy qadamlar soni (h = (t1-t0)/N)

# Etalon (reference) yechim: aniq yechim bo'lmaganda yuqori aniqlikdagi
# yopiq usul bilan hisoblanadi.
REFERENCE_METHOD = "Radau"
REFERENCE_RTOL = 1e-11
REFERENCE_ATOL = 1e-13

# Sifat hukmi: nisbiy maksimal xato shu chegaradan kichik bo'lsa "To'g'ri".
VERDICT_REL_TOL = 2e-2

# Qiyoslash to'rining standart zichligi.
DEFAULT_GRID = 800


def sample_on_grid(traj: HybridTrajectory, t_grid: np.ndarray) -> np.ndarray:
    """Traektoriyani berilgan vaqt to'rida namunalaydi.

    Har bir `t` uchun shu vaqtni o'z ichiga olgan oxirgi bo'lak topiladi va
    uning zich chiqishi (dense output) orqali qiymat hisoblanadi. Hodisa
    (reset) chegarasida oxirgi mos bo'lak — hodisadan keyingi bo'lak —
    tanlanadi, bu uzluksizlikni saqlaydi. Zich chiqish bo'lmasa, eng yaqin
    tugun olinadi.

    Qaytaradi: shakli (len(t_grid), dim) bo'lgan massiv.
    """
    dim = traj.dim
    grid = np.asarray(t_grid, dtype=float)
    out = np.full((grid.size, dim), np.nan, dtype=float)
    pieces = traj.pieces
    if not pieces:
        return out

    for k, t in enumerate(grid):
        chosen = None
        for p in pieces:
            if p.t[0] - 1e-12 <= t <= p.t[-1] + 1e-12:
                chosen = p  # oxirgi mos bo'lak (hodisadan keyingi davomi)
        if chosen is None:
            chosen = pieces[0] if t < pieces[0].t[0] else pieces[-1]
        tt = min(max(t, float(chosen.t[0])), float(chosen.t[-1]))
        if chosen.dense is not None:
            out[k] = np.asarray(chosen.dense(tt), dtype=float)
        else:
            idx = int(np.argmin(np.abs(chosen.t - tt)))
            out[k] = chosen.x[idx]
    return out


def _exact_on_grid(
    exact: Callable[[float], np.ndarray], grid: np.ndarray, dim: int
) -> np.ndarray:
    """Aniq yechimni vaqt to'rida hisoblaydi."""
    out = np.full((grid.size, dim), np.nan, dtype=float)
    for k, t in enumerate(grid):
        out[k] = np.atleast_1d(np.asarray(exact(float(t)), dtype=float))
    return out


def _totals(traj: HybridTrajectory) -> "tuple[int, int]":
    """Traektoriya bo'yicha umumiy nfev va qadamlar sonini yig'adi."""
    nfev = sum(int(lg.nfev) for lg in traj.interval_log)
    nsteps = sum(int(lg.n_steps) for lg in traj.interval_log)
    return nfev, nsteps


def compare_methods(
    automaton: HybridAutomaton,
    base_config: SimulationConfig,
    exact: Optional[Callable[[float], np.ndarray]] = None,
    methods: "tuple[str, ...]" = COMPARE_METHODS,
    n_grid: int = DEFAULT_GRID,
) -> Dict:
    """Sonli usullarni bir xil masalada qiyoslaydi (dissertatsiya 2–5-jadval).

    Har bir usul foydalanuvchi bergan aniqlik (rtol/atol) bilan, lekin AUTO
    emas, qat'iy belgilangan holda ishga tushiriladi. Natijalar aniq yechim
    (berilgan bo'lsa) yoki yuqori aniqlikdagi etalonga nisbatan baholanadi.

    Qaytaradi:
        {
          "grid": [...],                    # vaqt to'ri
          "dim": int,
          "var_names": [...],
          "reference_label": str,           # "Aniq yechim" yoki "Etalon ..."
          "reference_series": [[...], ...], # har o'zgaruvchi uchun etalon y
          "rows": [                         # jadval satrlari (usul bo'yicha)
            {"method", "success", "max_error", "rel_error",
             "nfev", "n_steps", "verdict"}  # verdict: correct|wrong|error
          ],
          "series": [                       # grafik uchun har usul traektoriyasi
            {"method", "y": [[...], ...]}   # y[var] — to'rdagi qiymatlar
          ],
        }
    """
    t0, t_end = base_config.t_span
    grid = np.linspace(float(t0), float(t_end), int(n_grid))
    dim = automaton.dim

    # Etalon yechim.
    if exact is not None:
        ref = _exact_on_grid(exact, grid, dim)
        ref_label = "Aniq yechim"
    else:
        ref_cfg = replace(
            base_config,
            method=REFERENCE_METHOD,
            rtol=REFERENCE_RTOL,
            atol=REFERENCE_ATOL,
        )
        ref_traj = simulate(automaton, ref_cfg)
        ref = sample_on_grid(ref_traj, grid)
        ref_label = f"Etalon ({REFERENCE_METHOD}, rtol={REFERENCE_RTOL:g})"

    ref_scale = float(np.nanmax(np.abs(ref))) if np.any(np.isfinite(ref)) else 1.0
    ref_scale = max(ref_scale, 1e-12)

    rows: List[Dict] = []
    series: List[Dict] = []
    for m in methods:
        cfg = replace(base_config, method=m)
        try:
            traj = simulate(automaton, cfg)
        except Exception:
            rows.append({
                "method": m, "success": False, "max_error": None,
                "rel_error": None, "nfev": 0, "n_steps": 0, "verdict": "error",
                "fixed_step": False,
            })
            series.append({"method": m, "y": [[None] * grid.size for _ in range(dim)]})
            continue

        ys = sample_on_grid(traj, grid)
        nfev, nsteps = _totals(traj)

        if not traj.success:
            verdict = "error"
            max_err = rel = None
        else:
            diff = np.abs(ys - ref)
            max_err = float(np.nanmax(diff)) if np.any(np.isfinite(diff)) else float("inf")
            rel = max_err / ref_scale
            verdict = "correct" if rel < VERDICT_REL_TOL else "wrong"

        rows.append({
            "method": m,
            "success": bool(traj.success),
            "max_error": max_err,
            "rel_error": rel,
            "nfev": nfev,
            "n_steps": nsteps,
            "verdict": verdict,
            "fixed_step": False,
        })
        # Grafik uchun: har o'zgaruvchining to'rdagi qiymatlari (NaN -> None).
        y_by_var = [
            [None if not np.isfinite(v) else float(v) for v in ys[:, j]]
            for j in range(dim)
        ]
        series.append({"method": m, "y": y_by_var})

    # Doimiy qadamli oshkora usullar — faqat bir rejimli, transitionsiz
    # uzluksiz ODE modellari uchun (dissertatsiya §2.3/§3.2 test funksiyalari).
    if len(automaton.modes) == 1 and not automaton.transitions:
        mode_f = automaton.get_mode(automaton.initial_mode).f
        x0 = np.asarray(automaton.initial_state, dtype=float)
        for m in FIXED_STEP_NAMES:
            try:
                ys = integrate_on_grid(
                    m, mode_f, (float(t0), float(t_end)), x0, grid, FIXED_STEP_COUNT
                )
            except Exception:
                rows.append({
                    "method": m, "success": False, "max_error": None,
                    "rel_error": None, "nfev": 0, "n_steps": FIXED_STEP_COUNT,
                    "verdict": "error", "fixed_step": True,
                })
                series.append({"method": m, "y": [[None] * grid.size for _ in range(dim)]})
                continue

            diff = np.abs(ys - ref)
            diverged = not np.all(np.isfinite(ys))
            if np.any(np.isfinite(diff)):
                max_err = float(np.nanmax(diff))
            else:
                max_err = float("inf")
            finite_err = np.isfinite(max_err)
            rel = (max_err / ref_scale) if finite_err else None
            # Beqarorlashgan (portlagan) yoki nisbiy xato katta -> Noto'g'ri.
            if diverged or not finite_err or (rel is not None and rel >= VERDICT_REL_TOL):
                verdict = "wrong"
            else:
                verdict = "correct"

            rows.append({
                "method": m,
                "success": not diverged,
                "max_error": max_err if finite_err else None,
                "rel_error": rel,
                "nfev": FIXED_STEP_COUNT * (4 if m == "RK4" else 1),
                "n_steps": FIXED_STEP_COUNT,
                "verdict": verdict,
                "fixed_step": True,
            })
            y_by_var = [
                [None if not np.isfinite(v) else float(v) for v in ys[:, j]]
                for j in range(dim)
            ]
            series.append({"method": m, "y": y_by_var})

    ref_series = [
        [None if not np.isfinite(v) else float(v) for v in ref[:, j]]
        for j in range(dim)
    ]

    return {
        "grid": [float(t) for t in grid],
        "dim": dim,
        "var_names": list(automaton.var_names),
        "reference_label": ref_label,
        "reference_series": ref_series,
        "rows": rows,
        "series": series,
    }
