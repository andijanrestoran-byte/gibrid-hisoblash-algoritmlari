"""Gibrid avtomat modeli.

Gibrid tizim — uzluksiz dinamikani (ODE) va diskret o'tishlarni (hodisa,
sakrash, mode almashishi) birlashtirgan tizim. Bu modulda shu tizimni
ifodalovchi sof Python ma'lumot strukturalari aniqlanadi:

    Mode        — bitta uzluksiz holat (o'ng tomon f(t, x) va invariant).
    Transition  — bir mode'dan ikkinchisiga o'tish (guard + reset).
    HybridAutomaton — modelning to'liq ta'rifi.

Shuningdek, simulyatsiya natijasini saqlovchi strukturalar:

    Piece            — bitta uzluksiz integrallash bo'lagi (bitta mode,
                       bitta usul, zich chiqish bilan).
    EventRecord      — ro'y bergan har bir hodisa haqida jurnal yozuvi.
    IntervalLog      — qaysi intervalda qaysi usul ishlatilgani haqida yozuv.
    HybridTrajectory — yuqoridagilarning yig'indisi (to'liq traektoriya).

Dizayn tamoyili: bu yerda hech qanday `eval`/`exec` yo'q. f, guard va reset
funksiyalari — oddiy Python chaqiriladigan obyektlar (callable). Ular faqat
preset modullarda kod sifatida yoziladi (qarang: simulator/presets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

# Funksiya turlari uchun qisqartmalar (faqat o'qishni osonlashtirish uchun).
VectorField = Callable[[float, np.ndarray], np.ndarray]      # f(t, x) -> dx/dt
ScalarGuard = Callable[[float, np.ndarray], float]           # g(t, x) -> skalyar
ResetMap = Callable[[float, np.ndarray], np.ndarray]         # R(t, x) -> x_yangi
Invariant = Callable[[float, np.ndarray], bool]              # inv(t, x) -> bool


@dataclass
class Mode:
    """Gibrid avtomatning bitta uzluksiz holati (rejimi).

    Atributlar:
        name: rejim nomi (transition'lar shu nom orqali bog'lanadi).
        f: o'ng tomon funksiyasi, dx/dt = f(t, x). x — NumPy massivi.
        invariant: (ixtiyoriy) rejim amal qiladigan soha sharti inv(t, x).
            Faqat hujjatlash/diagnostika uchun; integrallashni to'xtatmaydi
            (to'xtatish guard'lar orqali amalga oshadi).
        description: rejimning o'zbekcha qisqa tavsifi.
    """

    name: str
    f: VectorField
    invariant: Optional[Invariant] = None
    description: str = ""


@dataclass
class Transition:
    """Bir rejimdan ikkinchisiga diskret o'tish.

    Atributlar:
        from_mode: o'tish boshlanadigan rejim nomi.
        to_mode: o'tishdan keyingi rejim nomi.
        guard: skalyar funksiya g(t, x). Hodisa g ning ishorasi o'zgargan
            (nolni kesib o'tgan) payt ro'y beradi.
        reset: holatni qayta o'rnatuvchi funksiya x_yangi = R(t, x_eski).
            None bo'lsa, holat o'zgarmaydi (faqat rejim almashadi).
        direction: nol kesib o'tish yo'nalishi (scipy konvensiyasi):
            +1 — faqat g manfiydan musbatga o'tganda,
            -1 — faqat g musbatdan manfiyga o'tganda,
             0 — har qanday yo'nalishda.
        name: o'tishning o'zbekcha tavsifi (jurnalga chiqadi).
    """

    from_mode: str
    to_mode: str
    guard: ScalarGuard
    reset: Optional[ResetMap] = None
    direction: float = 0.0
    name: str = ""

    def apply_reset(self, t: float, x: np.ndarray) -> np.ndarray:
        """Reset funksiyasini qo'llaydi (yoki yo'q bo'lsa, x ni qaytaradi)."""
        if self.reset is None:
            return np.array(x, dtype=float)
        return np.array(self.reset(t, np.array(x, dtype=float)), dtype=float)


@dataclass
class HybridAutomaton:
    """Gibrid avtomatning to'liq ta'rifi.

    Atributlar:
        modes: rejimlar ro'yxati.
        transitions: o'tishlar ro'yxati.
        initial_mode: boshlang'ich rejim nomi.
        initial_state: boshlang'ich holat vektori x(t0).
        var_names: o'zgaruvchilar nomlari (grafik/CSV uchun). Bo'sh bo'lsa
            avtomatik ravishda x0, x1, ... deb nomlanadi.
        max_events: Zeno himoyasi — ruxsat etilgan maksimal hodisalar soni.
    """

    modes: List[Mode]
    transitions: List[Transition]
    initial_mode: str
    initial_state: Sequence[float]
    var_names: List[str] = field(default_factory=list)
    max_events: int = 10000

    def __post_init__(self) -> None:
        self.initial_state = np.array(self.initial_state, dtype=float)
        self._mode_by_name = {m.name: m for m in self.modes}
        if self.initial_mode not in self._mode_by_name:
            raise ValueError(
                f"Boshlang'ich rejim '{self.initial_mode}' modellarda topilmadi."
            )
        # O'tishlardagi rejim nomlari mavjudligini tekshiramiz.
        for tr in self.transitions:
            if tr.from_mode not in self._mode_by_name:
                raise ValueError(f"Transition'da noma'lum from_mode: {tr.from_mode}")
            if tr.to_mode not in self._mode_by_name:
                raise ValueError(f"Transition'da noma'lum to_mode: {tr.to_mode}")
        # O'zgaruvchilar nomlarini to'ldiramiz.
        dim = len(self.initial_state)
        if not self.var_names:
            self.var_names = [f"x{i}" for i in range(dim)]
        elif len(self.var_names) != dim:
            raise ValueError("var_names uzunligi initial_state bilan mos kelmadi.")

    @property
    def dim(self) -> int:
        """Holat vektori o'lchami."""
        return len(self.initial_state)

    def get_mode(self, name: str) -> Mode:
        """Nom bo'yicha rejimni qaytaradi."""
        return self._mode_by_name[name]

    def transitions_from(self, mode_name: str) -> List[Transition]:
        """Berilgan rejimdan chiquvchi barcha o'tishlarni qaytaradi."""
        return [tr for tr in self.transitions if tr.from_mode == mode_name]


# --------------------------------------------------------------------------- #
#  Natija (traektoriya) strukturalari
# --------------------------------------------------------------------------- #


@dataclass
class Piece:
    """Bitta uzluksiz integrallash bo'lagi.

    Har bir bo'lak bitta rejimda, bitta sonli usul bilan integrallangan
    vaqt oralig'iga to'g'ri keladi. Hodisa yoki AUTO usul almashinuvi ro'y
    berganda yangi bo'lak boshlanadi.

    Atributlar:
        mode: shu bo'lakdagi faol rejim nomi.
        method: ishlatilgan sonli usul ('RK45', 'Radau', ...).
        t: solver qaytargan vaqt tugunlari (1D massiv).
        x: holat qiymatlari, shakli (len(t), dim).
        dense: zich chiqish funksiyasi sol(t) -> x (ixtiyoriy). Adaptiv
            grafik nuqtalarini hisoblash uchun ishlatiladi.
    """

    mode: str
    method: str
    t: np.ndarray
    x: np.ndarray
    dense: Optional[Callable[[float], np.ndarray]] = None

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.x = np.asarray(self.x, dtype=float)


@dataclass
class EventRecord:
    """Ro'y bergan bitta hodisa (diskret o'tish) haqida jurnal yozuvi.

    Atributlar:
        time: hodisa vaqti.
        from_mode / to_mode: o'tishdan oldingi va keyingi rejimlar.
        x_before: reset'dan oldingi holat.
        x_after: reset'dan keyingi holat.
        transition_name: o'tishning o'zbekcha tavsifi.
        localized_by: hodisa qaysi usul bilan lokalizatsiya qilingani
            ('scipy' yoki 'bisection').
        bisection_time: mustaqil bisektsiya bergan vaqt (taqqoslash uchun).
    """

    time: float
    from_mode: str
    to_mode: str
    x_before: np.ndarray
    x_after: np.ndarray
    transition_name: str = ""
    localized_by: str = "scipy"
    bisection_time: Optional[float] = None


@dataclass
class IntervalLog:
    """Qaysi vaqt oralig'ida qaysi usul ishlatilgani haqida yozuv.

    AUTO rejimda usul almashinuvini foydalanuvchiga ko'rsatish uchun zarur.

    Atributlar:
        t_start / t_end: interval chegaralari.
        mode: faol rejim.
        method: shu intervalda ishlatilgan usul.
        n_steps: solver qabul qilgan qadamlar soni.
        nfev: o'ng tomon f baholanishlari soni.
        stiffness: qat'iylik ko'rsatkichi (Yakobian spektral radiusi).
        switched: bu intervalda usul oldingisiga nisbatan almashganmi.
        reason: almashish (yoki tanlash) sababi (o'zbekcha izoh).
    """

    t_start: float
    t_end: float
    mode: str
    method: str
    n_steps: int
    nfev: int
    stiffness: float
    switched: bool = False
    reason: str = ""


@dataclass
class HybridTrajectory:
    """Simulyatsiyaning to'liq natijasi.

    Atributlar:
        pieces: uzluksiz integrallash bo'laklari (vaqt bo'yicha tartibda).
        events: ro'y bergan hodisalar jurnali.
        interval_log: usul tanlash/almashish jurnali.
        var_names: o'zgaruvchilar nomlari.
        success: simulyatsiya muvaffaqiyatli tugadimi.
        message: yakuniy holat haqida xabar (masalan, Zeno chegarasi).
        zeno: Zeno (max_events) chegarasiga yetilganmi.
    """

    pieces: List[Piece] = field(default_factory=list)
    events: List[EventRecord] = field(default_factory=list)
    interval_log: List[IntervalLog] = field(default_factory=list)
    var_names: List[str] = field(default_factory=list)
    success: bool = True
    message: str = ""
    zeno: bool = False

    @property
    def dim(self) -> int:
        """Holat vektori o'lchami."""
        return len(self.var_names)

    def concat_t(self) -> np.ndarray:
        """Barcha bo'laklarning vaqt tugunlarini ketma-ket ulaydi (naive).

        Bo'laklar chegarasidagi takroriy nuqtalar saqlanadi — bu sakrash
        (reset) joylarini grafikda aniq ko'rsatishga yordam beradi.
        """
        if not self.pieces:
            return np.array([])
        return np.concatenate([p.t for p in self.pieces])

    def concat_x(self) -> np.ndarray:
        """Barcha bo'laklarning holat qiymatlarini ketma-ket ulaydi."""
        if not self.pieces:
            return np.zeros((0, self.dim))
        return np.concatenate([p.x for p in self.pieces], axis=0)

    @property
    def t_start(self) -> float:
        return float(self.pieces[0].t[0]) if self.pieces else 0.0

    @property
    def t_end(self) -> float:
        return float(self.pieces[-1].t[-1]) if self.pieces else 0.0

    def event_times(self) -> List[float]:
        """Hodisa vaqtlari ro'yxati (grafikdagi vertikal chiziqlar uchun)."""
        return [ev.time for ev in self.events]
