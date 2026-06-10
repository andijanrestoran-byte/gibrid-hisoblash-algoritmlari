"""Tayyor misollar (presets) paketi.

Har bir misol gibrid tizimning bitta klassik namunasini ifodalaydi va
dissertatsiyaning muayyan natijasini ko'rsatadi. Misollar `registry`
orqali ro'yxatga olinadi.

Misollardagi tenglamalar faqat shu Python modullarda kod sifatida turadi —
foydalanuvchi kiritgan matnli ifodalar hech qachon eval/exec qilinmaydi.
"""

from .registry import PRESETS, get_preset, all_presets, ParamSpec, Preset

__all__ = ["PRESETS", "get_preset", "all_presets", "ParamSpec", "Preset"]
