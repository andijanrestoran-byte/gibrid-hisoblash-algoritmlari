"""Admin paneli sozlamalari."""

from django.contrib import admin

from .models import SimulationRun


@admin.register(SimulationRun)
class SimulationRunAdmin(admin.ModelAdmin):
    list_display = ("preset_name", "method", "n_events", "success", "zeno",
                    "created_at")
    list_filter = ("preset_key", "method", "success", "zeno")
    readonly_fields = ("created_at",)
    search_fields = ("preset_name", "preset_key")
