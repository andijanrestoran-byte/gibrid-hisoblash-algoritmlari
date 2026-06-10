"""Hodisalarni (o'tish nuqtalarini) aniqlash.

Integrallash qadamida guard funksiyasi g(t, x) ning ishorasi o'zgarsa,
shu qadam ichida g(t, x(t)) = 0 tenglamaning ildizi bor. Aynan shu nuqta —
diskret o'tish (hodisa) vaqti.

Bu modulda hodisani aniq lokalizatsiya qilishning ikki yo'li bor:

  1. `scipy.integrate.solve_ivp` ning ichki `events` mexanizmi (solver_manager
     da ishlatiladi) — yuqori darajada optimallashtirilgan.

  2. Mustaqil bisektsiya usuli (`bisection_localize`) — dissertatsiyada
     talab qilinganidek, taqqoslash uchun noldan yozilgan. U zich chiqish
     (dense output) funksiyasidan foydalanib, hodisani 1e-9 aniqlikkacha
     topadi.

Ikkala natija solver_manager da solishtirilib, EventRecord ga yoziladi.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# guard(t) -> skalyar (dense output bilan birlashtirilgan: t -> g(t, x(t)))
ScalarOfTime = Callable[[float], float]

# Bisektsiya standart aniqligi (dissertatsiya talabi).
DEFAULT_TOL = 1e-9


def sign_changes(values: np.ndarray, direction: float = 0.0) -> np.ndarray:
    """Ketma-ket qiymatlarda ishora o'zgargan indekslarni qaytaradi.

    `values[i]` va `values[i+1]` orasida nol kesib o'tilgan bo'lsa, `i`
    indeksi natijaga kiradi.

    direction:
        +1 — faqat manfiydan musbatga (g o'sib nolni kesganda),
        -1 — faqat musbatdan manfiyga (g kamayib nolni kesganda),
         0 — har qanday yo'nalish.
    """
    values = np.asarray(values, dtype=float)
    lo = values[:-1]
    hi = values[1:]
    crossed = np.sign(lo) != np.sign(hi)
    # Aynan nolga tushgan qiymatni ham hodisa deb hisoblaymiz.
    crossed |= (hi == 0.0) & (lo != 0.0)
    if direction > 0:
        crossed &= hi > lo
    elif direction < 0:
        crossed &= hi < lo
    return np.nonzero(crossed)[0]


def bisection_localize(
    g_of_t: ScalarOfTime,
    t_lo: float,
    t_hi: float,
    tol: float = DEFAULT_TOL,
    max_iter: int = 200,
) -> float:
    """Bisektsiya usuli bilan g_of_t(t) = 0 ildizini topadi.

    Bu — dissertatsiyada talab qilingan, scipy'dan mustaqil ravishda
    yozilgan ildiz lokalizatori. `g_of_t` odatda zich chiqish funksiyasi
    bilan birlashtirilgan guard bo'ladi: g_of_t(t) = guard(t, sol(t)).

    Shartlar:
        g_of_t(t_lo) va g_of_t(t_hi) qarama-qarshi ishorada bo'lishi kerak.

    Argumentlar:
        g_of_t: bir o'lchovli uzluksiz funksiya.
        t_lo, t_hi: ildizni o'rab turgan oraliq (t_lo < t_hi).
        tol: vaqt bo'yicha talab qilingan aniqlik (default 1e-9).
        max_iter: maksimal iteratsiyalar soni (xavfsizlik chegarasi).

    Qaytaradi:
        Ildiz vaqti t* (taxminan), |t_hi - t_lo| < tol bo'lguncha aniqlangan.
    """
    f_lo = g_of_t(t_lo)
    f_hi = g_of_t(t_hi)

    # Chegaralardan biri allaqachon nol bo'lsa, uni darhol qaytaramiz.
    if f_lo == 0.0:
        return t_lo
    if f_hi == 0.0:
        return t_hi
    if np.sign(f_lo) == np.sign(f_hi):
        raise ValueError(
            "bisection_localize: oraliq chegaralarida ishora o'zgarmagan "
            f"(g(t_lo)={f_lo:.3e}, g(t_hi)={f_hi:.3e})."
        )

    a, b = float(t_lo), float(t_hi)
    fa = f_lo
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = g_of_t(m)
        if fm == 0.0 or 0.5 * (b - a) < tol:
            return m
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def make_guard_of_time(
    guard: Callable[[float, np.ndarray], float],
    dense: Callable[[float], np.ndarray],
) -> ScalarOfTime:
    """guard(t, x) va zich chiqish sol(t) dan g_of_t(t) = guard(t, sol(t)) yasaydi.

    Bu yordamchi funksiya bisektsiya lokalizatoriga uzatish uchun qulay
    bir o'lchovli funksiya hosil qiladi.
    """

    def g_of_t(t: float) -> float:
        return float(guard(t, np.asarray(dense(t), dtype=float)))

    return g_of_t
