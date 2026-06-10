"""Simulator ilovasi URL'lari."""

from django.urls import path

from . import views

app_name = "simulator"

urlpatterns = [
    path("", views.index, name="index"),
    path("simulate/<str:preset_key>/", views.simulate_page, name="simulate"),
    path("api/simulate/<str:preset_key>/", views.api_simulate, name="api_simulate"),
    path("api/compare/<str:preset_key>/", views.api_compare, name="api_compare"),
    path("api/run/<int:pk>/", views.api_run_detail, name="api_run_detail"),
    path("history/", views.history, name="history"),
    path("run/<int:pk>/csv/", views.run_csv, name="run_csv"),
]
