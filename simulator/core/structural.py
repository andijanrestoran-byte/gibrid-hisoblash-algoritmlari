"""Strukturaviy tahlil va indeksni qisqartirish (Pantelides algoritmi, §2.1).

Dissertatsiya gibrid/DAE tizimlarini INSIDENTLIK MATRITSASI orqali strukturaviy
tahlil qilishni va indeksni avtomatik kamaytirishni tasvirlaydi (Pantelides,
Gear). Bu modul shu g'oyani amalga oshiradi:

  * Insidentlik matritsasi: satrlar — tenglamalar, ustunlar — o'zgaruvchilar;
    element 1, agar o'zgaruvchi shu tenglamada qatnashsa.
  * Transversal: har bir tenglamani aniqlanishi kerak bo'lgan (eng yuqori
    tartibli hosila yoki algebraik) o'zgaruvchiga moslashtirish.
  * Agar to'liq moslik topilmasa, tizim strukturaviy degeneratsiyalangan —
    ya'ni indeksi > 1. Pantelides algoritmi mos KONSTRUKSIYANI topguncha
    ayrim tenglamalarni (cheklovlarni) differensiallaydi. Differensiallashlar
    soni strukturaviy indeksni beradi.

Tenglamalar birinchi tartibli ko'rinishda beriladi: o'zgaruvchi belgilari
"x" (tartib 0), "x'" (1-hosila), "x''" (2-hosila) ko'rinishida yoziladi.
Misol uchun mayatnik (pendulum_dae) — strukturaviy indeksi 3.
"""

from __future__ import annotations

from typing import Dict, List, Set


def _parse(tok: str):
    """\"x''\" belgisini (\"x\", 2) ko'rinishiga ajratadi."""
    base = tok.rstrip("'")
    return base, len(tok) - len(base)


def _build(equations: List[Dict]):
    """Tenglamalardan o'zgaruvchi tugunlari va insidentlikni quradi."""
    toks = [[_parse(t) for t in eq["vars"]] for eq in equations]
    maxord: Dict[str, int] = {}
    for row in toks:
        for b, o in row:
            maxord[b] = max(maxord.get(b, 0), o)

    nid: Dict = {}
    base: List[str] = []
    order: List[int] = []
    vname: List[str] = []
    A: List[int] = []  # A[v] = d(v)/dt tuguni id (yoki -1)

    def new_var(b: str, o: int) -> int:
        nid[(b, o)] = len(vname)
        base.append(b)
        order.append(o)
        vname.append(b + "'" * o)
        A.append(-1)
        return nid[(b, o)]

    for b in sorted(maxord):
        mo = maxord[b]
        for o in range(mo + 1):
            new_var(b, o)
        for o in range(mo):
            A[nid[(b, o)]] = nid[(b, o + 1)]

    incidence: List[Set[int]] = []
    for row in toks:
        incidence.append({nid[(b, o)] for b, o in row})

    return {
        "nid": nid, "base": base, "order": order, "vname": vname,
        "A": A, "incidence": incidence,
    }


def incidence_matrix(equations: List[Dict]):
    """Insidentlik matritsasini (0/1) va o'zgaruvchi/tenglama nomlarini qaytaradi."""
    g = _build(equations)
    var_names = g["vname"]
    eq_names = [eq.get("name", f"E{i+1}") for i, eq in enumerate(equations)]
    matrix = []
    for inc in g["incidence"]:
        matrix.append([1 if j in inc else 0 for j in range(len(var_names))])
    return eq_names, var_names, matrix


def pantelides(equations: List[Dict]) -> Dict:
    """Pantelides algoritmi: strukturaviy indeks va differensiallash qadamlari.

    Qaytaradi:
        structural_index — strukturaviy indeks,
        diff_per_eq      — har bir boshlang'ich tenglama necha marta
                           differensiallangani,
        steps            — qisqartirishning o'zbekcha tavsifi.
    """
    g = _build(equations)
    base, order, vname, A = g["base"], g["order"], g["vname"], list(g["A"])
    incidence = [set(s) for s in g["incidence"]]
    eorig = list(range(len(equations)))
    edl = [0] * len(equations)
    B = [-1] * len(equations)
    assign: Dict[int, int] = {}

    nid = g["nid"]

    def new_var(b: str, o: int) -> int:
        nid[(b, o)] = len(vname)
        base.append(b)
        order.append(o)
        vname.append(b + "'" * o)
        A.append(-1)
        return nid[(b, o)]

    def der_var(v: int) -> int:
        if A[v] == -1:
            A[v] = new_var(base[v], order[v] + 1)
        return A[v]

    def augment(i: int, ce: Set[int], cv: Set[int]) -> bool:
        ce.add(i)
        for v in incidence[i]:
            if A[v] == -1 and v not in assign:
                assign[v] = i
                return True
        for v in incidence[i]:
            if A[v] == -1 and v not in cv:
                cv.add(v)
                if augment(assign[v], ce, cv):
                    assign[v] = i
                    return True
        return False

    diff_per_eq = [0] * len(equations)
    n0 = len(equations)
    for k in range(n0):
        i = k
        for _ in range(1000):  # xavfsizlik chegarasi
            ce: Set[int] = set()
            cv: Set[int] = set()
            if augment(i, ce, cv):
                break
            for v in list(cv):
                der_var(v)
            for l in list(ce):
                new_inc = {der_var(v) for v in incidence[l]}
                incidence.append(new_inc)
                eorig.append(eorig[l])
                edl.append(edl[l] + 1)
                B.append(-1)
                new_idx = len(incidence) - 1
                B[l] = new_idx
                diff_per_eq[eorig[l]] = max(diff_per_eq[eorig[l]], edl[l] + 1)
                for v in list(incidence[l]):
                    if assign.get(v) == l:
                        assign[A[v]] = new_idx
            i = B[i]

    constraint_diffs = [diff_per_eq[k] for k, eq in enumerate(equations)
                        if eq.get("constraint")]
    if constraint_diffs:
        structural_index = max(constraint_diffs) + 1
    else:
        structural_index = 1 if any(eq.get("constraint") for eq in equations) else 0

    steps = []
    for k, eq in enumerate(equations):
        if eq.get("constraint") and diff_per_eq[k] > 0:
            steps.append(
                f"Cheklov '{eq.get('name', 'E'+str(k+1))}' {diff_per_eq[k]} marta "
                f"differensiallandi."
            )
    if structural_index >= 1:
        steps.append(f"Strukturaviy indeks = {structural_index}; tizim indeks-1 ga "
                     f"keltirildi.")

    return {
        "structural_index": structural_index,
        "diff_per_eq": diff_per_eq,
        "steps": steps,
    }


def analyze(structure: Dict) -> Dict:
    """Insidentlik + Pantelidesni birlashtirib, ko'rsatish uchun JSON qaytaradi."""
    equations = structure["equations"]
    eq_names, var_names, matrix = incidence_matrix(equations)
    pan = pantelides(equations)
    rows = [{"name": eq_names[i], "cells": matrix[i]} for i in range(len(eq_names))]
    return {
        "title": structure.get("title", "Strukturaviy tahlil"),
        "eq_names": eq_names,
        "var_names": var_names,
        "matrix": matrix,
        "rows": rows,
        "structural_index": pan["structural_index"],
        "steps": pan["steps"],
        "constraints": [eq.get("name", f"E{i+1}")
                        for i, eq in enumerate(equations) if eq.get("constraint")],
    }
