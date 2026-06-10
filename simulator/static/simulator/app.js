/* GibridSim frontend mantiqi.
 *
 * Forma yuborilganda sahifa qayta yuklanmaydi: fetch orqali /api/simulate/
 * chaqiriladi va Plotly grafiklari hamda jadvallar JavaScriptda yangilanadi.
 * Grafiklar yorug'/qorong'i rejimga moslashadi.
 */
(function () {
  "use strict";

  const cfg = JSON.parse(document.getElementById("sim-config").textContent);
  const form = document.getElementById("sim-form");
  const statusEl = document.getElementById("status");
  const runBtn = document.getElementById("run-btn");
  const runLabel = document.getElementById("run-label");
  const csvBtn = document.getElementById("csv-btn");
  const compareToggle = document.getElementById("compare-toggle");
  const loadingTime = document.getElementById("loading-time");

  let lastResult = null;

  const PLOTLY_CFG = { responsive: true, displaylogo: false, displayModeBar: "hover" };
  // O'zgaruvchi traektoriyalari ranglari.
  const COLORS = ["#4f46e5", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"];
  // Usul ranglari (CSS dagi m-badge bilan mos).
  const METHOD_COLOR = { RK45: "#3b82f6", Radau: "#f59e0b", BDF: "#8b5cf6", LSODA: "#14b8a6", AUTO: "#4f46e5", Euler: "#ec4899", RK4: "#06b6d4" };

  function isDark() {
    return document.documentElement.getAttribute("data-bs-theme") === "dark";
  }

  /* ---- Mavzuga mos Plotly layout asosi ---- */
  function themedLayout(extra) {
    const dark = isDark();
    const font = dark ? "#cbd2e0" : "#374151";
    const grid = dark ? "rgba(255,255,255,0.08)" : "rgba(17,24,39,0.07)";
    const zero = dark ? "rgba(255,255,255,0.18)" : "rgba(17,24,39,0.18)";
    const base = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter, system-ui, sans-serif", color: font, size: 12 },
      margin: { t: 26, r: 12, b: 42, l: 54 },
      xaxis: { gridcolor: grid, zerolinecolor: zero, linecolor: grid },
      yaxis: { gridcolor: grid, zerolinecolor: zero, linecolor: grid },
      hovermode: "closest",
    };
    return Object.assign(base, extra || {});
  }

  function csrfToken() {
    const el = form.querySelector("[name=csrfmiddlewaretoken]");
    return el ? el.value : "";
  }

  function fmt(x, digits) {
    if (x === null || x === undefined) return "";
    const d = digits === undefined ? 6 : digits;
    if (typeof x !== "number") return String(x);
    if (Math.abs(x) !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e5)) {
      return x.toExponential(d - 1);
    }
    return x.toFixed(d);
  }
  function fmtVec(arr) { return "[" + arr.map((v) => fmt(v, 4)).join(", ") + "]"; }

  function gatherPayload() {
    const payload = {};
    form.querySelectorAll("input, select").forEach((el) => {
      if (!el.name || el.name === "csrfmiddlewaretoken") return;
      if (el.type === "checkbox") return;
      payload[el.name] = el.value;
    });
    return payload;
  }

  function eventDecor(events) {
    const dark = isDark();
    const col = dark ? "rgba(200,205,220,0.45)" : "rgba(80,80,90,0.5)";
    const shapes = [];
    const annotations = [];
    events.forEach((ev) => {
      shapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: ev.time, x1: ev.time, y0: 0, y1: 1,
        line: { color: col, width: 1, dash: "dot" },
      });
      annotations.push({
        x: ev.time, yref: "paper", y: 1.0, text: "▼", showarrow: false,
        font: { color: col, size: 9 }, yanchor: "bottom",
      });
    });
    return { shapes, annotations };
  }

  /* AUTO_ODE usul almashinuvini grafik fonida rangli zonalar bilan ko'rsatadi.
   * Har bir interval o'z usuli rangida (past shaffoflikda) bo'yaladi; usul
   * almashgan chegarada vertikal marker chiziladi. Dissertatsiya §3.2. */
  function methodDecor(methods) {
    const shapes = [];
    const annotations = [];
    if (!methods || methods.length < 1) return { shapes, annotations };
    // Bir nechta usul ishlatilganda fonni bo'yaymiz (bitta usul bo'lsa shart emas).
    const distinct = new Set(methods.map((m) => m.method));
    if (distinct.size > 1) {
      methods.forEach((m) => {
        shapes.push({
          type: "rect", xref: "x", yref: "paper", layer: "below",
          x0: m.t_start, x1: m.t_end, y0: 0, y1: 1,
          fillcolor: METHOD_COLOR[m.method] || "#888",
          opacity: 0.07, line: { width: 0 },
        });
        if (m.switched) {
          shapes.push({
            type: "line", xref: "x", yref: "paper",
            x0: m.t_start, x1: m.t_start, y0: 0, y1: 1,
            line: { color: METHOD_COLOR[m.method] || "#888", width: 1.5, dash: "dash" },
          });
          annotations.push({
            x: m.t_start, yref: "paper", y: 0, yanchor: "bottom",
            text: "⇄ " + m.method, showarrow: false,
            font: { color: METHOD_COLOR[m.method] || "#888", size: 9 },
            bgcolor: "rgba(127,127,127,0.12)",
          });
        }
      });
    }
    return { shapes, annotations };
  }

  function timeTraces(modeData) {
    return modeData.series.map((s, i) => ({
      x: s.t, y: s.y, name: s.name, mode: "lines", type: "scatter",
      line: { color: COLORS[i % COLORS.length], width: 2 },
      connectgaps: false,
    }));
  }

  function renderTimeSingle(result) {
    const data = result.plot.adaptive;
    const dec = eventDecor(result.plot.events);
    const mdec = methodDecor(result.plot.methods);
    Plotly.react("plot-single", timeTraces(data), themedLayout({
      xaxis: Object.assign(themedLayout().xaxis, { title: "t" }),
      yaxis: Object.assign(themedLayout().yaxis, { title: "qiymat" }),
      legend: { orientation: "h", y: 1.12 },
      shapes: dec.shapes.concat(mdec.shapes),
      annotations: dec.annotations.concat(mdec.annotations),
      title: { text: "Adaptiv · " + data.n_points + " nuqta", font: { size: 12 } },
    }), PLOTLY_CFG);
  }

  function renderTimeCompare(result) {
    ["naive", "adaptive"].forEach((modeKey) => {
      const data = result.plot[modeKey];
      const dec = eventDecor(result.plot.events);
      Plotly.react(modeKey === "naive" ? "plot-naive" : "plot-adaptive",
        timeTraces(data), themedLayout({
          xaxis: Object.assign(themedLayout().xaxis, { title: "t" }),
          yaxis: Object.assign(themedLayout().yaxis, { title: "qiymat" }),
          showlegend: false,
          shapes: dec.shapes, annotations: dec.annotations,
          title: { text: data.n_points + " nuqta", font: { size: 11 } },
        }), PLOTLY_CFG);
    });
  }

  function renderTime(result) {
    const compare = compareToggle && compareToggle.checked;
    document.getElementById("plot-single").style.display = compare ? "none" : "block";
    document.getElementById("plot-compare").style.display = compare ? "flex" : "none";
    if (compare) renderTimeCompare(result); else renderTimeSingle(result);
  }

  function renderPhase(result) {
    if (!cfg.twoDim) return;
    const ph = result.plot.adaptive.phase;
    if (!ph) return;
    Plotly.react("plot-phase", [{
      x: ph.x, y: ph.y, mode: "lines", type: "scatter",
      line: { color: "#4f46e5", width: 1.8 }, connectgaps: false,
    }], themedLayout({
      xaxis: Object.assign(themedLayout().xaxis, { title: ph.x_name }),
      yaxis: Object.assign(themedLayout().yaxis, { title: ph.y_name }),
      title: { text: "Fazaviy portret", font: { size: 12 } },
    }), PLOTLY_CFG);
  }

  function renderEvents(result) {
    const body = document.getElementById("events-body");
    const rows = result.events_table;
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="6" class="text-muted">Hodisa yo\'q.</td></tr>';
      return;
    }
    body.innerHTML = rows.map((ev, i) => {
      const bis = ev.bisection_time === null ? "—" : fmt(ev.bisection_time, 9);
      return "<tr><td class='text-muted'>" + (i + 1) + "</td>" +
        "<td>" + fmt(ev.time, 9) + "</td>" +
        "<td>" + bis + "</td>" +
        "<td>" + (ev.transition_name || ev.from_mode + "→" + ev.to_mode) + "</td>" +
        "<td>" + fmtVec(ev.x_before) + "</td>" +
        "<td>" + fmtVec(ev.x_after) + "</td></tr>";
    }).join("");
  }

  function renderLog(result) {
    const body = document.getElementById("log-body");
    const rows = result.log_table;
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="7" class="text-muted">Jurnal bo\'sh.</td></tr>';
      return;
    }
    body.innerHTML = rows.map((lg) => {
      const cls = "m-badge m-" + lg.method + (lg.switched ? " switched" : "");
      const badge = '<span class="' + cls + '">' + lg.method + "</span>" +
        (lg.switched ? ' <span class="small" style="color:var(--gs-accent)">⇄</span>' : "");
      return "<tr><td>[" + fmt(lg.t_start, 4) + ", " + fmt(lg.t_end, 4) + "]</td>" +
        "<td>" + lg.mode + "</td>" +
        "<td>" + badge + "</td>" +
        "<td>" + lg.n_steps + "</td>" +
        "<td>" + lg.nfev + "</td>" +
        "<td>" + fmt(lg.stiffness, 3) + "</td>" +
        '<td class="text-muted">' + lg.reason + "</td></tr>";
    }).join("");
  }

  function setStatus(html, kind) {
    const cls = kind === "error" ? "status-err" : kind === "ok" ? "status-ok"
      : kind === "warn" ? "status-warn" : "";
    statusEl.innerHTML = '<div class="status-chip ' + cls + '"><span class="status-dot"></span><div>' + html + "</div></div>";
  }

  function renderAll(result, runId) {
    lastResult = result;
    if (cfg.canCompare) invalidateCompare();
    const s = result.summary;
    renderTime(result);
    renderPhase(result);
    renderEvents(result);
    renderLog(result);

    if (runId) {
      csvBtn.classList.remove("disabled");
      csvBtn.href = cfg.csvBase + runId + "/csv/";
    }
    const kind = s.zeno ? "warn" : s.success ? "ok" : "error";
    const label = s.zeno ? "Zeno chegarasi" : s.success ? "Muvaffaqiyatli" : "Xato";
    setStatus("<strong>" + label + "</strong> — " + s.n_events + " hodisa, " + s.n_pieces +
      " bo'lak<br><span class='small' style='color:var(--gs-muted)'>" + (s.message || "") + "</span>", kind);
  }

  function setBusy(busy) {
    runBtn.disabled = busy;
    runLabel.innerHTML = busy
      ? '<span class="spinner-mini"></span> Hisoblanmoqda…'
      : "▶ Hisoblash";
    if (loadingTime) loadingTime.classList.toggle("show", busy);
  }

  async function runSimulation(evt) {
    if (evt) evt.preventDefault();
    setBusy(true);
    setStatus("Hisoblanmoqda…");
    try {
      const resp = await fetch(cfg.apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken() },
        body: JSON.stringify(gatherPayload()),
      });
      const data = await resp.json();
      if (!data.ok) {
        let msg = data.error || "Validatsiya xatosi.";
        if (data.errors) {
          msg = Object.entries(data.errors).map(([k, v]) => k + ": " + v.join(", ")).join("<br>");
        }
        setStatus(msg, "error");
        return;
      }
      renderAll(data.result, data.run_id);
    } catch (err) {
      setStatus("Tarmoq xatosi: " + err, "error");
    } finally {
      setBusy(false);
    }
  }

  async function loadRun(runId) {
    setBusy(true);
    setStatus("Saqlangan natija yuklanmoqda…");
    try {
      const resp = await fetch(cfg.runDetailBase + runId + "/");
      const data = await resp.json();
      if (data.ok) renderAll(data.result, data.run_id);
      else setStatus("Natija topilmadi.", "error");
    } catch (err) {
      setStatus("Yuklash xatosi: " + err, "error");
    } finally {
      setBusy(false);
    }
  }

  /* ===================== Usullarni qiyoslash ===================== */
  const VERDICT = {
    correct: { label: "✓ To'g'ri", cls: "m-correct" },
    wrong: { label: "✗ Noto'g'ri", cls: "m-wrong" },
    error: { label: "⚠ Xato", cls: "m-errverdict" },
  };
  let compareLoaded = false;

  function renderCompareTable(cmp) {
    const body = document.getElementById("compare-body");
    body.innerHTML = cmp.rows.map((r) => {
      const v = VERDICT[r.verdict] || VERDICT.error;
      // Beqaror (portlagan) yoki juda katta xato -> "beqaror" deb ko'rsatamiz.
      const unstable = r.rel_error === null || r.rel_error > 10;
      const me = r.max_error === null || unstable ? "—" : fmt(r.max_error, 3);
      const re = unstable ? "beqaror ∞"
        : (r.rel_error * 100 < 1e-2 ? r.rel_error.toExponential(2)
          : (r.rel_error * 100).toFixed(3) + "%");
      const mc = METHOD_COLOR[r.method] || "#888";
      const tag = r.fixed_step
        ? " <span class='small' style='color:var(--gs-muted)'>(doimiy qadam)</span>" : "";
      return "<tr><td><span class='m-badge' style='background:" + mc +
        "22;color:" + mc + "'>" + r.method + "</span>" + tag + "</td>" +
        "<td><span class='m-badge " + v.cls + "'>" + v.label + "</span></td>" +
        "<td>" + me + "</td><td>" + re + "</td>" +
        "<td>" + r.nfev + "</td><td>" + r.n_steps + "</td></tr>";
    }).join("");
  }

  function renderComparePlot(cmp) {
    // Faqat birinchi o'zgaruvchini ko'rsatamiz (eng aniq taqqoslash uchun).
    const traces = [];
    traces.push({
      x: cmp.grid, y: cmp.reference_series[0], name: cmp.reference_label,
      mode: "lines", type: "scatter",
      line: { color: isDark() ? "#e5e7eb" : "#111827", width: 3 },
      connectgaps: false,
    });
    cmp.series.forEach((s) => {
      traces.push({
        x: cmp.grid, y: s.y[0], name: s.method, mode: "lines", type: "scatter",
        line: { color: METHOD_COLOR[s.method] || "#888", width: 1.4, dash: "dot" },
        connectgaps: false,
      });
    });
    // Y-diapazonni etalon bo'yicha cheklaymiz — beqaror (portlagan) usullar
    // o'qni buzmasligi uchun (ular ekrandan chiqib ketadi, kichik xato ko'rinadi).
    const ref0 = cmp.reference_series[0].filter((v) => v !== null && isFinite(v));
    let yrange = null;
    if (ref0.length) {
      const lo = Math.min.apply(null, ref0), hi = Math.max.apply(null, ref0);
      const pad = Math.max((hi - lo) * 0.25, 0.5);
      yrange = [lo - pad, hi + pad];
    }
    const yax = Object.assign(themedLayout().yaxis, { title: cmp.var_names[0] || "qiymat" });
    if (yrange) yax.range = yrange;
    Plotly.react("plot-compare-methods", traces, themedLayout({
      xaxis: Object.assign(themedLayout().xaxis, { title: "t" }),
      yaxis: yax,
      legend: { orientation: "h", y: 1.14 },
      title: { text: "Usullar va " + cmp.reference_label + " (1-o'zgaruvchi)", font: { size: 12 } },
    }), PLOTLY_CFG);
  }

  function setCompareStatus(html, kind) {
    const el = document.getElementById("compare-status");
    if (!el) return;
    const cls = kind === "error" ? "status-err" : kind === "ok" ? "status-ok" : "";
    el.innerHTML = html
      ? '<div class="status-chip ' + cls + '"><span class="status-dot"></span><div>' + html + "</div></div>"
      : "";
  }

  async function runCompare() {
    if (!cfg.canCompare || !cfg.compareUrl) return;
    setCompareStatus("Usullar qiyoslanmoqda…");
    try {
      const resp = await fetch(cfg.compareUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken() },
        body: JSON.stringify(gatherPayload()),
      });
      const data = await resp.json();
      if (!data.ok) {
        setCompareStatus(data.error || "Qiyoslash xatosi.", "error");
        return;
      }
      renderCompareTable(data.compare);
      renderComparePlot(data.compare);
      const nOk = data.compare.rows.filter((r) => r.verdict === "correct").length;
      setCompareStatus("<strong>" + nOk + "/" + data.compare.rows.length +
        "</strong> usul to'g'ri natija berdi (etalon: " + data.compare.reference_label + ").", "ok");
      compareLoaded = true;
    } catch (err) {
      setCompareStatus("Tarmoq xatosi: " + err, "error");
    }
  }

  const compareRunBtn = document.getElementById("compare-run-btn");
  if (compareRunBtn) compareRunBtn.addEventListener("click", runCompare);
  // Yangi simulyatsiya bo'lsa, eski qiyoslashni eskirgan deb belgilaymiz.
  function invalidateCompare() {
    compareLoaded = false;
    setCompareStatus("Parametrlar o'zgardi — qayta qiyoslang.", "");
  }

  form.addEventListener("submit", runSimulation);
  if (compareToggle) {
    compareToggle.addEventListener("change", function () {
      if (lastResult) renderTime(lastResult);
    });
  }
  // Mavzu almashganda grafiklarni qayta chizamiz.
  document.addEventListener("gs-theme-change", function () {
    if (lastResult) { renderTime(lastResult); renderPhase(lastResult); }
  });

  // Tab ochilganda undagi Plotly grafiklarini o'lchamga moslaymiz.
  // (Yashirin tabda grafik 0 kenglikda chizilib qoladi — bu fazaviy
  //  portretni "ishlamayotgandek" ko'rsatardi.)
  document.querySelectorAll('[data-bs-toggle="tab"]').forEach(function (btn) {
    btn.addEventListener("shown.bs.tab", function (e) {
      const target = document.querySelector(e.target.getAttribute("data-bs-target"));
      if (!target) return;
      // Qiyoslash tabi birinchi marta ochilganda avtomatik hisoblaymiz.
      if (e.target.id === "compare-tab-btn" && !compareLoaded) runCompare();
      target.querySelectorAll(".js-plotly-plot").forEach(function (p) {
        Plotly.Plots.resize(p);
      });
    });
  });

  if (cfg.runId) loadRun(cfg.runId);
  else runSimulation(null);
})();
