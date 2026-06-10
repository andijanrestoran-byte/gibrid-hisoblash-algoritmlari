"""GibridSim yadrosi — Django'ga bog'liq bo'lmagan sof Python kutubxonasi.

Bu paket gibrid avtomatlarni (uzluksiz ODE + diskret o'tishlar) sonli
modellashtirish uchun barcha matematik mantiqni o'z ichiga oladi. Hech bir
modul Django'ni import qilmaydi, shuning uchun uni mustaqil ravishda
(masalan, pytest yoki oddiy skript orqali) ishlatish va sinash mumkin.
"""

from .hybrid_automaton import (
    Mode,
    Transition,
    HybridAutomaton,
    Piece,
    EventRecord,
    IntervalLog,
    HybridTrajectory,
)
from .solver_manager import simulate, SolverManager, SimulationConfig
from .dae import DAESystem, solve_dae, solve_algebraic

__all__ = [
    "Mode",
    "Transition",
    "HybridAutomaton",
    "Piece",
    "EventRecord",
    "IntervalLog",
    "HybridTrajectory",
    "simulate",
    "SolverManager",
    "SimulationConfig",
    "DAESystem",
    "solve_dae",
    "solve_algebraic",
]
