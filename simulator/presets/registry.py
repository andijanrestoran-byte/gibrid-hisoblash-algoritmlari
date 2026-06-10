"""Preset ro'yxati va umumiy tuzilmalar.

`ParamSpec` — forma maydonining metama'lumoti (nom, yorliq, default, chegaralar).
`Preset`    — bitta misolning to'liq ta'rifi: avtomat quruvchi funksiya,
              parametrlar va default sozlamalar.

Bu modulning oxirida barcha misollar `PRESETS` lug'atiga yig'iladi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..core.hybrid_automaton import HybridAutomaton


@dataclass
class ParamSpec:
    """Parametr formasi uchun metama'lumot.

    Atributlar:
        name: parametrning ichki nomi (build() ga uzatiladigan kalit).
        label: foydalanuvchiga ko'rinadigan o'zbekcha yorliq.
        default: standart qiymat.
        min_value, max_value: validatsiya chegaralari (ixtiyoriy).
        step: forma maydoni qadami.
        help_text: qisqa izoh.
        integer: qiymat butun songa yaxlitlanishi kerakmi.
    """

    name: str
    label: str
    default: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    help_text: str = ""
    integer: bool = False


@dataclass
class Preset:
    """Bitta tayyor misolning to'liq ta'rifi.

    Atributlar:
        key: URL va kod uchun qisqa kalit (masalan, 'bouncing_ball').
        name: o'zbekcha nomi.
        description: o'zbekcha qisqa tavsif.
        notes: dissertatsiya bilan bog'liqlik haqida izoh.
        variables: holat o'zgaruvchilari nomlari.
        params: misolga xos parametrlar (ParamSpec ro'yxati).
        build: parametrlar lug'atidan HybridAutomaton quruvchi funksiya.
        default_T: standart yakuniy vaqt.
        default_method: standart sonli usul.
        default_rtol, default_atol: standart aniqliklar.
        compare_default: naive/adaptiv taqqoslash standart yoniqligi.
        two_dim: fazaviy portret mavjudligini bildiradi.
    """

    key: str
    name: str
    description: str
    variables: List[str]
    params: List[ParamSpec]
    build: Callable[[Dict[str, float]], HybridAutomaton]
    default_T: float = 5.0
    default_method: str = "AUTO"
    default_rtol: float = 1e-6
    default_atol: float = 1e-9
    compare_default: bool = False
    two_dim: bool = False
    notes: str = ""
    icon: str = "🧮"      # kartochka uchun belgi
    category: str = ""    # qisqa toifa yorlig'i (masalan, "Mexanika")
    kind: str = "ode"     # "ode" (gibrid avtomat) yoki "dae" (algebraik-differensial)
    structure: Optional[Dict] = None  # strukturaviy tahlil uchun tenglama tuzilmasi
    comparable: bool = True  # "Usullarni qiyoslash" tabini ko'rsatish (faqat ODE)
    # Aniq (analitik) yechim quruvchi: params -> (t -> x(t) massiv). Berilgan
    # bo'lsa, usullarni qiyoslashda etalon sifatida ishlatiladi (dissertatsiya
    # §2.3 test funksiyalari: sharp_peak, smooth_linear, stiff_demo).
    exact: Optional[Callable[[Dict[str, float]], Callable[[float], "object"]]] = None

    def merged_params(self, overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Default parametrlarni foydalanuvchi qiymatlari bilan birlashtiradi."""
        values = {p.name: p.default for p in self.params}
        if overrides:
            for k, v in overrides.items():
                if k in values:
                    values[k] = v
        return values


# Ro'yxatni modul yuklanganda to'ldiramiz (pastdagi import'lar tsiklni
# oldini olish uchun shu yerda turadi).
PRESETS: "Dict[str, Preset]" = {}


def register(preset: Preset) -> None:
    """Misolni global ro'yxatga qo'shadi."""
    PRESETS[preset.key] = preset


def get_preset(key: str) -> Preset:
    """Kalit bo'yicha misolni qaytaradi (topilmasa KeyError)."""
    return PRESETS[key]


def all_presets() -> List[Preset]:
    """Barcha misollar ro'yxatini (qo'shilgan tartibda) qaytaradi."""
    return list(PRESETS.values())


# Misollarni ro'yxatga olish. Import qilishning o'zi register() ni chaqiradi.
from . import bouncing_ball  # noqa: E402,F401
from . import thermostat  # noqa: E402,F401
from . import three_state  # noqa: E402,F401
from . import sharp_peak  # noqa: E402,F401
from . import smooth_linear  # noqa: E402,F401
from . import stiff_demo  # noqa: E402,F401
from . import van_der_pol  # noqa: E402,F401
from . import cat_mouse  # noqa: E402,F401
from . import rocket_pursuit  # noqa: E402,F401
from . import pendulum_dae  # noqa: E402,F401
