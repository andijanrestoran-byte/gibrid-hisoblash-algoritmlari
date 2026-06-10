"""Simulyatsiya parametrlari formasi.

Forma misolga qarab dinamik tuziladi: umumiy solver maydonlari (T, rtol,
atol, usul) + shu misolga xos parametrlar (preset.params asosida). Forma
ham HTML render qilish (chap paneldagi parametrlar), ham validatsiya uchun
ishlatiladi.

Xavfsizlik: barcha qiymatlar faqat son sifatida qabul qilinadi va
chegaralar bo'yicha tekshiriladi. Hech qanday matnli ifoda eval qilinmaydi.
"""

from __future__ import annotations

from django import forms

from .core.solver_manager import VALID_METHODS
from .presets.registry import Preset

METHOD_CHOICES = [
    ("AUTO", "AUTO (avtomat almashinuv)"),
    ("RK45", "RK45 (ochiq)"),
    ("Radau", "Radau (yashirin)"),
    ("BDF", "BDF (yashirin)"),
    ("LSODA", "LSODA (moslashuvchan)"),
]

IMPLICIT_CHOICES = [
    ("Radau", "Radau"),
    ("BDF", "BDF"),
]

_INPUT_ATTRS = {"class": "form-control form-control-sm"}
_SELECT_ATTRS = {"class": "form-select form-select-sm"}


class SimulationForm(forms.Form):
    """Misolga moslashtirilgan parametrlar formasi."""

    T = forms.FloatField(
        label="Yakuniy vaqt T",
        min_value=1e-6,
        max_value=1000.0,
        widget=forms.NumberInput(attrs={**_INPUT_ATTRS, "step": "any"}),
    )
    rtol = forms.FloatField(
        label="Nisbiy aniqlik rtol",
        min_value=1e-13,
        max_value=1e-1,
        initial=1e-6,
        widget=forms.NumberInput(attrs={**_INPUT_ATTRS, "step": "any"}),
    )
    atol = forms.FloatField(
        label="Absolyut aniqlik atol",
        min_value=1e-15,
        max_value=1e-1,
        initial=1e-9,
        widget=forms.NumberInput(attrs={**_INPUT_ATTRS, "step": "any"}),
    )
    method = forms.ChoiceField(
        label="Sonli usul",
        choices=METHOD_CHOICES,
        widget=forms.Select(attrs=_SELECT_ATTRS),
    )
    auto_implicit = forms.ChoiceField(
        label="AUTO yashirin usuli",
        choices=IMPLICIT_CHOICES,
        initial="Radau",
        required=False,
        widget=forms.Select(attrs=_SELECT_ATTRS),
    )

    def __init__(self, preset: Preset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preset = preset
        # Umumiy maydonlarning boshlang'ich qiymatlari misoldan olinadi.
        self.fields["T"].initial = preset.default_T
        self.fields["rtol"].initial = preset.default_rtol
        self.fields["atol"].initial = preset.default_atol
        self.fields["method"].initial = preset.default_method

        # Misolga xos parametrlar uchun maydonlarni dinamik qo'shamiz.
        self._param_field_names = []
        for spec in preset.params:
            field = forms.FloatField(
                label=spec.label,
                initial=spec.default,
                min_value=spec.min_value,
                max_value=spec.max_value,
                help_text=spec.help_text,
                widget=forms.NumberInput(
                    attrs={**_INPUT_ATTRS, "step": str(spec.step or "any")}
                ),
            )
            name = f"p_{spec.name}"
            self.fields[name] = field
            self._param_field_names.append((name, spec))

    def clean_method(self):
        method = self.cleaned_data["method"]
        if method not in VALID_METHODS:
            raise forms.ValidationError("Noma'lum sonli usul.")
        return method

    def preset_params(self) -> dict:
        """Validatsiyadan o'tgan misol parametrlarini (p_ prefiksisiz) qaytaradi."""
        out = {}
        for name, spec in self._param_field_names:
            value = self.cleaned_data[name]
            if spec.integer:
                value = int(round(value))
            out[spec.name] = value
        return out

    def param_fields(self):
        """Shablon uchun: faqat misolga xos maydonlar ro'yxati."""
        return [self[name] for name, _ in self._param_field_names]
