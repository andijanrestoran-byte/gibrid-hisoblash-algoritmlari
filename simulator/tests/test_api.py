"""/api/simulate/ va boshqa view'lar testlari (Django bilan)."""

import json

import pytest
from django.urls import reverse

from simulator.models import SimulationRun
from simulator.presets import get_preset


def _valid_payload(preset_key, overrides=None):
    """Misol uchun to'liq, validatsiyadan o'tadigan JSON yuk yasaydi."""
    preset = get_preset(preset_key)
    payload = {
        "T": preset.default_T,
        "rtol": preset.default_rtol,
        "atol": preset.default_atol,
        "method": "RK45",  # testlar tez bo'lishi uchun
        "auto_implicit": "Radau",
    }
    for spec in preset.params:
        payload[f"p_{spec.name}"] = spec.default
    if overrides:
        payload.update(overrides)
    return payload


def test_index_page(client):
    resp = client.get(reverse("simulator:index"))
    assert resp.status_code == 200
    assert b"GibridSim" in resp.content


def test_simulate_page_renders(client):
    resp = client.get(reverse("simulator:simulate", args=["bouncing_ball"]))
    assert resp.status_code == 200
    assert b"sim-form" in resp.content


def test_simulate_page_unknown_preset(client):
    resp = client.get(reverse("simulator:simulate", args=["yoq_bunaqa"]))
    assert resp.status_code == 404


@pytest.mark.django_db
def test_api_simulate_success(client):
    url = reverse("simulator:api_simulate", args=["bouncing_ball"])
    payload = _valid_payload("bouncing_ball", {"T": 2.0})
    resp = client.post(url, data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["summary"]["n_events"] > 0
    # Natija bazaga saqlanishi kerak.
    assert SimulationRun.objects.filter(id=data["run_id"]).exists()


@pytest.mark.django_db
def test_api_simulate_validation_error(client):
    url = reverse("simulator:api_simulate", args=["bouncing_ball"])
    # T manfiy — validatsiya xatosi kutilmoqda.
    payload = _valid_payload("bouncing_ball", {"T": -5.0})
    resp = client.post(url, data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 400
    data = resp.json()
    assert data["ok"] is False
    assert "errors" in data


@pytest.mark.django_db
def test_api_simulate_bad_json(client):
    url = reverse("simulator:api_simulate", args=["bouncing_ball"])
    resp = client.post(url, data="{buzuq", content_type="application/json")
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


@pytest.mark.django_db
def test_api_simulate_stiff_switches(client):
    """AUTO rejimda qat'iy sistema natijasida usul almashinuvi jurnalga tushishi."""
    url = reverse("simulator:api_simulate", args=["stiff_demo"])
    payload = _valid_payload("stiff_demo", {"method": "AUTO", "p_n": 8})
    resp = client.post(url, data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    log = resp.json()["result"]["log_table"]
    methods = {row["method"] for row in log}
    assert "RK45" in methods and "Radau" in methods


@pytest.mark.django_db
def test_api_simulate_dae_pendulum(client):
    """DAE misoli (mayatnik) API orqali hisoblanib, 5 o'zgaruvchi qaytarishi."""
    url = reverse("simulator:api_simulate", args=["pendulum_dae"])
    payload = _valid_payload("pendulum_dae", {"T": 3.0})
    resp = client.post(url, data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert len(data["result"]["var_names"]) == 5  # x, y, vx, vy, λ
    assert data["result"]["plot"]["adaptive"]["phase"] is not None


@pytest.mark.django_db
def test_run_csv_download(client):
    url = reverse("simulator:api_simulate", args=["thermostat"])
    payload = _valid_payload("thermostat", {"T": 10.0})
    resp = client.post(url, data=json.dumps(payload), content_type="application/json")
    run_id = resp.json()["run_id"]

    csv_resp = client.get(reverse("simulator:run_csv", args=[run_id]))
    assert csv_resp.status_code == 200
    assert csv_resp["Content-Type"].startswith("text/csv")
    body = csv_resp.content.decode("utf-8")
    assert body.splitlines()[0].startswith("t,")
