"""Usulni boshqarish va simulyatsiya yadrosi (AUTO_ODE g'oyasi).

Bu modul dissertatsiyaning markaziy g'oyalaridan birini — sonli usulni
masala "qat'iyligi" (stiffness)ga qarab avtomatik almashtirishni amalga
oshiradi:

  * Simulyatsiya boshida ochiq (explicit) RK45 usuli ishlatiladi.
  * Har bir vaqt oynasidan oldin masalaning qat'iylik ko'rsatkichi
    baholanadi (Yakobian matritsasi spektral radiusi son usulida).
  * Qat'iylik belgisi aniqlansa (rate * davomiylik chegaradan oshsa),
    avtomatik ravishda yashirin (implicit) Radau yoki BDF usuliga o'tiladi.
  * Tizim soddalashsa — yana ochiq usulga qaytiladi (gisterezis bilan).

Foydalanuvchi usulni qo'lda ham belgilashi mumkin: AUTO / RK45 / Radau /
BDF / LSODA.

`simulate()` funksiyasi gibrid avtomatni to'liq integrallaydi: har bir
uzluksiz intervalni yechadi, guard nolini kesganda hodisani aniqlaydi,
reset qo'llaydi va rejimni almashtiradi. Natija — HybridTrajectory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .hybrid_automaton import (
    EventRecord,
    HybridAutomaton,
    HybridTrajectory,
    IntervalLog,
    Piece,
    Transition,
)
from .event_detection import (
    bisection_localize,
    make_guard_of_time,
    sign_changes,
)

# Ruxsat etilgan usullar.
VALID_METHODS = ("AUTO", "RK45", "Radau", "BDF", "LSODA")
EXPLICIT_METHODS = ("RK45", "LSODA")  # LSODA o'zi ham moslashadi, lekin ochiq deb hisoblaymiz
IMPLICIT_METHODS = ("Radau", "BDF")

# AUTO rejim qat'iylik chegaralari (gisterezis uchun ikki qiymat).
# Dissertatsiyada qat'iylik = Yakobian xos qiymatlari orasidagi farq
# (o'z qiymatlari nisbati R = max|Re λ| / min|Re λ|).
# R > STIFF_RATIO_ON  -> yashirin usulga o'tish.
# R < STIFF_RATIO_OFF -> ochiq usulga qaytish.
STIFF_RATIO_ON = 50.0
STIFF_RATIO_OFF = 10.0


@dataclass
class SimulationConfig:
    """Simulyatsiya sozlamalari.

    Atributlar:
        t_span: (t0, t_end) vaqt oralig'i.
        rtol, atol: solver nisbiy va absolyut aniqliklari.
        method: 'AUTO' / 'RK45' / 'Radau' / 'BDF' / 'LSODA'.
        auto_implicit: AUTO rejimda qat'iylikda o'tiladigan yashirin usul
            ('Radau' yoki 'BDF').
        n_auto_windows: AUTO rejimda butun oraliq nechta oynaga bo'linadi
            (usul almashinuvini lokalizatsiya qilish uchun).
        max_step: solver maksimal qadami (default cheksiz).
        max_wall_seconds: hisoblash uchun maksimal real vaqt (xavfsizlik
            chegarasi — patologik parametrlar veb-so'rovni bloklamasligi uchun).
    """

    t_span: Tuple[float, float]
    rtol: float = 1e-6
    atol: float = 1e-9
    method: str = "AUTO"
    auto_implicit: str = "Radau"
    n_auto_windows: int = 60
    max_step: float = np.inf
    max_wall_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"Noma'lum usul: {self.method!r}. Ruxsat etilgan: {VALID_METHODS}."
            )
        if self.auto_implicit not in IMPLICIT_METHODS:
            raise ValueError(
                f"auto_implicit faqat {IMPLICIT_METHODS} dan biri bo'lishi mumkin."
            )
        t0, t_end = self.t_span
        if not (t_end > t0):
            raise ValueError("t_span da t_end > t0 bo'lishi shart.")


def _jacobian(
    f: Callable[[float, np.ndarray], np.ndarray], t: float, x: np.ndarray
) -> np.ndarray:
    """Yakobian J = df/dx ni oldinga chekli ayirmalar bilan hisoblaydi.

    Sof son usuli; foydalanuvchi ifodalarini hech qachon eval qilmaydi.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    f0 = np.asarray(f(t, x), dtype=float)
    jac = np.empty((n, n), dtype=float)
    for i in range(n):
        dx = 1e-6 * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += dx
        jac[:, i] = (np.asarray(f(t, xp), dtype=float) - f0) / dx
    return jac


def estimate_stiffness(
    f: Callable[[float, np.ndarray], np.ndarray], t: float, x: np.ndarray
) -> float:
    """Qat'iylik NISBATINI (xos qiymatlar farqini) son usulida baholaydi.

    Dissertatsiyaga muvofiq, qat'iylik (stiffness) — Yakobian matritsasi
    J = df/dx ning xos qiymatlari orasidagi farq bilan o'lchanadi. Tizimda
    bir vaqtda juda tez va juda sekin o'zgaruvchi komponentlar mavjud
    bo'lsa (ya'ni |Re λ| lar keskin farq qilsa), masala qat'iy hisoblanadi.

    Qat'iylik nisbati:
        R = max|Re λ| / min|Re λ|     (faqat faol, nolga teng bo'lmagan modalar)

    Agar ikkitadan kam faol moda bo'lsa (masalan, skalyar tenglama yoki sof
    integrator), qat'iylik yo'q deb hisoblanadi va R = 1 qaytariladi.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 1.0
    jac = _jacobian(f, t, x)
    re = np.abs(np.linalg.eigvals(jac).real)
    active = np.sort(re[re > 1e-12])  # nolga teng bo'lmagan (faol) modalar
    if active.size < 2:
        return 1.0
    return float(active[-1] / active[0])


class SolverManager:
    """AUTO rejimda usul tanlash va almashish mantiqini boshqaradi.

    Holatni (joriy usul) saqlaydi, shuning uchun gisterezis (almashish
    chegaralarining ikki xilligi) ishlaydi va usul juda tez-tez sakrab
    almashmaydi.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.current_method: Optional[str] = None
        self._started = False

    def select(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        x: np.ndarray,
    ) -> Tuple[str, float, bool, str]:
        """Joriy oyna uchun usulni tanlaydi.

        Qaytaradi: (usul, qat'iylik_nisbati_R, almashdimi, sabab_izohi).
        """
        cfg = self.config
        ratio = estimate_stiffness(f, t, x)

        if cfg.method != "AUTO":
            # Qo'lda tanlangan usul — qat'iylikni faqat ma'lumot uchun baholaymiz.
            return cfg.method, ratio, False, "Qo'lda tanlangan usul"

        prev = self.current_method
        if not self._started:
            # Dissertatsiya (AUTO_ODE): boshida ochiq usul (RKF45/RK45).
            chosen = "RK45"
            reason = "Boshlang'ich oraliq — ochiq usul (RK45)"
            self._started = True
        elif prev in IMPLICIT_METHODS:
            if ratio < STIFF_RATIO_OFF:
                chosen = "RK45"
                reason = (
                    f"Qat'iylik pasaydi (R={ratio:.0f}) — "
                    f"ochiq usulga (RK45) qaytildi"
                )
            else:
                chosen = prev
                reason = f"Qat'iylik saqlanmoqda (R={ratio:.0f}) — yashirin usulda"
        else:  # prev ochiq usul edi
            if ratio > STIFF_RATIO_ON:
                chosen = cfg.auto_implicit
                reason = (
                    f"Qat'iylik aniqlandi (R={ratio:.0f}) — "
                    f"yashirin usulga ({cfg.auto_implicit}) o'tildi"
                )
            else:
                chosen = "RK45"
                reason = f"Tizim silliq (R={ratio:.0f}) — ochiq usul (RK45)"

        switched = prev is not None and chosen != prev
        self.current_method = chosen
        return chosen, ratio, switched, reason


def _make_scipy_event(transition: Transition):
    """Transition guard'idan solve_ivp uchun terminal hodisa funksiyasi yasaydi."""
    guard = transition.guard

    def event(t: float, y: np.ndarray) -> float:
        return float(guard(t, np.asarray(y, dtype=float)))

    event.terminal = True
    event.direction = transition.direction
    return event


def _wrap_dense(sol) -> Callable[[float], np.ndarray]:
    """solve_ivp natijasidan zich chiqish funksiyasini ajratib oladi."""
    ode_sol = sol.sol

    def dense(t):
        return np.asarray(ode_sol(t), dtype=float)

    return dense


def _independent_bisection(
    transition: Transition, sol, te_scipy: float
) -> Optional[float]:
    """scipy'dan mustaqil ravishda hodisani bisektsiya bilan lokalizatsiya qiladi.

    Dissertatsiya talabi: taqqoslash uchun ildiz lokalizatsiyasi noldan
    ham yoziladi. Bu yerda solver bo'lagining zich chiqishi (dense output)
    bo'yicha guard ishorasi o'zgargan oraliq topiladi va `bisection_localize`
    chaqiriladi. Xatolik bo'lsa (masalan, bracket topilmasa) None qaytaradi.
    """
    try:
        dense = _wrap_dense(sol)
        g_of_t = make_guard_of_time(transition.guard, dense)
        t_nodes = sol.t
        g_vals = np.array([g_of_t(tt) for tt in t_nodes])
        idxs = sign_changes(g_vals)  # har qanday yo'nalishdagi kesishish
        if idxs.size == 0:
            return None
        i = int(idxs[-1])
        a, b = float(t_nodes[i]), float(t_nodes[i + 1])
        if np.sign(g_of_t(a)) == np.sign(g_of_t(b)):
            return None
        return float(bisection_localize(g_of_t, a, b))
    except Exception:
        return None


def simulate(
    automaton: HybridAutomaton, config: SimulationConfig
) -> HybridTrajectory:
    """Gibrid avtomatni to'liq integrallaydi.

    Algoritm (gibrid vaqt hisobi):
      1. Joriy rejimda f(t, x) ni bitta oyna davomida integrallaymiz.
      2. Shu rejimdan chiquvchi har bir o'tishning guard'i terminal hodisa
         sifatida solve_ivp ga uzatiladi.
      3. Guard nolini kessa, solve_ivp hodisani lokalizatsiya qiladi;
         biz uni mustaqil bisektsiya bilan ham tekshiramiz.
      4. Reset qo'llanadi, rejim almashadi, integrallash davom etadi.
      5. AUTO rejimda har oynadan oldin usul qat'iylikka qarab tanlanadi.
      6. Zeno himoyasi: hodisalar soni max_events dan oshsa, to'xtatiladi.

    Qaytaradi: HybridTrajectory (bo'laklar, hodisalar, usul jurnali).
    """
    t0, t_end = config.t_span
    span = float(t_end - t0)
    traj = HybridTrajectory(var_names=list(automaton.var_names))
    manager = SolverManager(config)

    t = float(t0)
    x = np.array(automaton.initial_state, dtype=float)
    mode_name = automaton.initial_mode
    n_events = 0

    # AUTO rejimda oynalarga bo'lamiz; aks holda butun qoldiqni bir zarbda.
    if config.method == "AUTO":
        window_len = span / max(1, config.n_auto_windows)
    else:
        window_len = np.inf

    # Cheksiz tsikldan himoya: progress bo'lmasa to'xtaymiz.
    stall_guard = 0
    wall_start = time.perf_counter()

    while t < t_end - 1e-12:
        # Xavfsizlik: real vaqt chegarasidan oshmaslik (patologik parametrlar).
        if time.perf_counter() - wall_start > config.max_wall_seconds:
            traj.success = False
            traj.message = (
                f"Vaqt chegarasi ({config.max_wall_seconds:g}s) — masala juda "
                f"og'ir. Parametrlarni soddalashtiring (t={t:g} gacha hisoblandi)."
            )
            break
        mode = automaton.get_mode(mode_name)
        transitions = automaton.transitions_from(mode_name)
        t_target = min(t + window_len, t_end)
        if t_target <= t + 1e-14:
            break

        method, rate, switched, reason = manager.select(mode.f, t, x)

        scipy_events = [_make_scipy_event(tr) for tr in transitions]
        sol = solve_ivp(
            mode.f,
            (t, t_target),
            x,
            method=method,
            rtol=config.rtol,
            atol=config.atol,
            dense_output=True,
            events=scipy_events if scipy_events else None,
            max_step=config.max_step,
        )

        if not sol.success:
            traj.success = False
            traj.message = f"Integrallash xatosi ({method}): {sol.message}"
            break

        dense = _wrap_dense(sol)
        traj.pieces.append(
            Piece(mode=mode_name, method=method, t=sol.t, x=sol.y.T, dense=dense)
        )
        traj.interval_log.append(
            IntervalLog(
                t_start=float(sol.t[0]),
                t_end=float(sol.t[-1]),
                mode=mode_name,
                method=method,
                n_steps=max(len(sol.t) - 1, 0),
                nfev=int(sol.nfev),
                stiffness=rate,
                switched=switched,
                reason=reason,
            )
        )

        # Hodisa ro'y berganmi?
        fired: Optional[Transition] = None
        te: Optional[float] = None
        if sol.status == 1 and scipy_events:
            best_idx = None
            best_time = None
            for i, te_arr in enumerate(sol.t_events):
                if te_arr.size > 0:
                    cand = float(te_arr[0])
                    if best_time is None or cand < best_time:
                        best_time = cand
                        best_idx = i
            if best_idx is not None:
                fired = transitions[best_idx]
                te = best_time

        if fired is not None and te is not None:
            x_event = dense(te)
            bis_time = _independent_bisection(fired, sol, te)
            x_after = fired.apply_reset(te, x_event)
            traj.events.append(
                EventRecord(
                    time=float(te),
                    from_mode=mode_name,
                    to_mode=fired.to_mode,
                    x_before=np.array(x_event, dtype=float),
                    x_after=np.array(x_after, dtype=float),
                    transition_name=fired.name,
                    localized_by="scipy",
                    bisection_time=bis_time,
                )
            )
            t = float(te)
            x = x_after
            mode_name = fired.to_mode
            n_events += 1
            if n_events >= automaton.max_events:
                traj.zeno = True
                traj.message = (
                    "Zeno himoyasi: maksimal hodisalar soniga "
                    f"({automaton.max_events}) yetildi."
                )
                break
        else:
            new_t = float(sol.t[-1])
            # Progress tekshiruvi (cheksiz tsikldan himoya).
            if new_t <= t + 1e-14:
                stall_guard += 1
                if stall_guard > 3:
                    traj.success = False
                    traj.message = "Integrallash oldinga siljimadi (turg'unlik)."
                    break
            else:
                stall_guard = 0
            t = new_t
            x = sol.y[:, -1]

    if not traj.message and traj.success:
        traj.message = (
            f"Muvaffaqiyatli: t=[{t0:g}, {t_end:g}], {len(traj.events)} ta hodisa."
        )
    return traj
