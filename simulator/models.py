"""Ma'lumotlar bazasi modellari."""

from django.db import models


class SimulationRun(models.Model):
    """Saqlangan bitta simulyatsiya natijasi.

    Parametrlar va natija JSON ko'rinishida saqlanadi, shuning uchun har
    qanday misol uchun yagona model yetarli. Bu /history/ sahifasida
    ko'rsatiladi va natijani qayta ko'rish/CSV yuklab olish imkonini beradi.
    """

    preset_key = models.CharField("Misol kaliti", max_length=64)
    preset_name = models.CharField("Misol nomi", max_length=128)
    created_at = models.DateTimeField("Yaratilgan vaqt", auto_now_add=True)

    # Kirish parametrlari: solver sozlamalari + misolga xos parametrlar.
    params = models.JSONField("Parametrlar", default=dict)
    # To'liq natija: grafik ma'lumotlari, hodisalar, usul jurnali, xulosa.
    result = models.JSONField("Natija", default=dict)

    method = models.CharField("Usul", max_length=16, default="AUTO")
    n_events = models.IntegerField("Hodisalar soni", default=0)
    success = models.BooleanField("Muvaffaqiyatli", default=True)
    zeno = models.BooleanField("Zeno chegarasi", default=False)
    message = models.TextField("Xabar", blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Simulyatsiya"
        verbose_name_plural = "Simulyatsiyalar"

    def __str__(self):
        return f"{self.preset_name} ({self.created_at:%Y-%m-%d %H:%M})"
