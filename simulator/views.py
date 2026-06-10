"""Ko'rinishlar (views): sahifalar va JSON API.

Simulyatsiya foydalanuvchi so'rovida sinxron tarzda (view ichida)
bajariladi — Celery/Redis ishlatilmaydi, chunki bu misollar tez hisoblanadi.
"""

from __future__ import annotations

import json

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST

from .core.solver_manager import SimulationConfig, simulate
from .core.accuracy import compare_methods
from .core.dae import solve_dae
from .forms import SimulationForm
from .models import SimulationRun
from .presets import all_presets, get_preset
from .presets.registry import PRESETS
from .results import serialize_trajectory, trajectory_to_csv


def _run_preset(preset, preset_params, config):
    """Misolni turiga qarab (ODE yoki DAE) hisoblaydi va traektoriya qaytaradi."""
    model = preset.build(preset_params)
    if preset.kind == "dae":
        return solve_dae(model, config)
    return simulate(model, config)


def index(request):
    """Bosh sahifa: loyiha tavsifi va misollar kartochkalari."""
    return render(request, "simulator/index.html", {"presets": all_presets()})


def simulate_page(request, preset_key):
    """Bitta misol uchun simulyatsiya sahifasi (forma + tablar)."""
    if preset_key not in PRESETS:
        return render(request, "simulator/not_found.html",
                      {"preset_key": preset_key}, status=404)
    preset = get_preset(preset_key)
    form = SimulationForm(preset)
    # DAE misollari uchun strukturaviy tahlil (insidentlik + indeks).
    structure = None
    if preset.structure:
        from .core.structural import analyze
        structure = analyze(preset.structure)
    can_compare = preset.kind == "ode" and preset.comparable
    context = {
        "preset": preset,
        "form": form,
        "api_url": reverse("simulator:api_simulate", args=[preset_key]),
        "compare_url": (
            reverse("simulator:api_compare", args=[preset_key]) if can_compare else ""
        ),
        "can_compare": can_compare,
        "compare_default": preset.compare_default,
        "two_dim": preset.two_dim,
        "structure": structure,
        # Qayta ko'rish uchun: ?run=<id> bo'lsa, shu natijani yuklash.
        "run_id": request.GET.get("run", ""),
    }
    return render(request, "simulator/simulate.html", context)


@require_POST
def api_simulate(request, preset_key):
    """JSON API: parametrlarni qabul qiladi, simulyatsiya qiladi, natija qaytaradi.

    So'rov tanasi — JSON: {T, rtol, atol, method, auto_implicit, p_<param>...}.
    Javob — {ok, run_id, result} yoki validatsiya xatosi (400).
    """
    if preset_key not in PRESETS:
        return JsonResponse({"ok": False, "error": "Misol topilmadi."}, status=404)
    preset = get_preset(preset_key)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"ok": False, "error": "Noto'g'ri JSON."}, status=400)

    form = SimulationForm(preset, data=payload)
    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)

    cd = form.cleaned_data
    T = min(float(cd["T"]), settings.GIBRIDSIM_MAX_T)
    preset_params = form.preset_params()

    try:
        config = SimulationConfig(
            t_span=(0.0, T),
            rtol=float(cd["rtol"]),
            atol=float(cd["atol"]),
            method=cd["method"],
            auto_implicit=cd.get("auto_implicit") or "Radau",
        )
        traj = _run_preset(preset, preset_params, config)
        result = serialize_trajectory(traj, two_dim=preset.two_dim)
    except Exception as exc:  # son usuli xatosi — foydalanuvchiga xabar beramiz
        return JsonResponse(
            {"ok": False, "error": f"Hisoblash xatosi: {exc}"}, status=400
        )

    stored_params = {
        "T": T,
        "rtol": float(cd["rtol"]),
        "atol": float(cd["atol"]),
        "method": cd["method"],
        "auto_implicit": cd.get("auto_implicit") or "Radau",
        "preset_params": preset_params,
    }
    run = SimulationRun.objects.create(
        preset_key=preset.key,
        preset_name=preset.name,
        params=stored_params,
        result=result,
        method=cd["method"],
        n_events=result["summary"]["n_events"],
        success=result["summary"]["success"],
        zeno=result["summary"]["zeno"],
        message=result["summary"]["message"],
    )

    return JsonResponse({"ok": True, "run_id": run.id, "result": result})


@require_POST
def api_compare(request, preset_key):
    """JSON API: bir masalani turli sonli usullar bilan yechib qiyoslaydi.

    Dissertatsiya §2.3/§3.3: RK45/Radau/BDF/LSODA usullarining aniqligini
    aniq yechim (yoki yuqori aniqlikdagi etalon) ga nisbatan baholaydi va
    "To'g'ri/Noto'g'ri" jadvalini qaytaradi (2–5-jadvallarning analogi).

    So'rov tanasi — `api_simulate` bilan bir xil; `method` e'tiborga olinmaydi
    (har bir usul navbat bilan sinaladi). Javob — {ok, compare}.
    """
    if preset_key not in PRESETS:
        return JsonResponse({"ok": False, "error": "Misol topilmadi."}, status=404)
    preset = get_preset(preset_key)
    if preset.kind != "ode" or not preset.comparable:
        return JsonResponse(
            {"ok": False, "error": "Bu misol uchun usullarni qiyoslash mavjud emas."},
            status=400,
        )

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"ok": False, "error": "Noto'g'ri JSON."}, status=400)

    form = SimulationForm(preset, data=payload)
    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)

    cd = form.cleaned_data
    T = min(float(cd["T"]), settings.GIBRIDSIM_MAX_T)
    preset_params = form.preset_params()

    try:
        base_config = SimulationConfig(
            t_span=(0.0, T),
            rtol=float(cd["rtol"]),
            atol=float(cd["atol"]),
            method="RK45",  # qiyoslashda har usul alohida o'rnatiladi
        )
        automaton = preset.build(preset_params)
        exact = preset.exact(preset_params) if preset.exact else None
        compare = compare_methods(automaton, base_config, exact=exact)
    except Exception as exc:
        return JsonResponse(
            {"ok": False, "error": f"Qiyoslash xatosi: {exc}"}, status=400
        )

    return JsonResponse({"ok": True, "compare": compare})


@require_GET
def api_run_detail(request, pk):
    """Saqlangan natijani qaytaradi (qayta ko'rish uchun)."""
    run = get_object_or_404(SimulationRun, pk=pk)
    return JsonResponse(
        {
            "ok": True,
            "run_id": run.id,
            "preset_key": run.preset_key,
            "params": run.params,
            "result": run.result,
        }
    )


def history(request):
    """Saqlangan simulyatsiyalar ro'yxati."""
    runs = SimulationRun.objects.all()[:200]
    return render(request, "simulator/history.html", {"runs": runs})


@require_GET
def run_csv(request, pk):
    """Saqlangan simulyatsiyani parametrlardan qayta hisoblab, CSV qaytaradi."""
    run = get_object_or_404(SimulationRun, pk=pk)
    if run.preset_key not in PRESETS:
        return HttpResponse("Misol topilmadi.", status=404)
    preset = get_preset(run.preset_key)
    p = run.params
    config = SimulationConfig(
        t_span=(0.0, float(p.get("T", preset.default_T))),
        rtol=float(p.get("rtol", preset.default_rtol)),
        atol=float(p.get("atol", preset.default_atol)),
        method=p.get("method", preset.default_method),
        auto_implicit=p.get("auto_implicit", "Radau"),
    )
    traj = _run_preset(preset, p.get("preset_params", {}), config)
    csv_text = trajectory_to_csv(traj)

    response = HttpResponse(csv_text, content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = (
        f'attachment; filename="gibridsim_{run.preset_key}_{run.id}.csv"'
    )
    return response
