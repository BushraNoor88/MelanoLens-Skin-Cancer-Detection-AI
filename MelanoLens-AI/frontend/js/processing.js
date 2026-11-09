/* Edge Booster Workspace: build params from the controls, validate, call /api/process, show gallery. */
(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("controls");
  const controls = [...form.querySelectorAll("[data-param]")];
  const DETECTORS = ["canny", "sobel", "prewitt", "laplacian", "log"];
  const ODD = { kernel_size: 1, booster_kernel: 1, canny_blur: 1, close_kernel: 1 };
  let defaults = {}, params = {}, rid = ML.getResultId(), viewer = null;
  /* presentation only: status pill + loading overlay */
  function setState(kind, text) {
    const pill = $("ws-state"); pill.className = `pill ${kind}`; $("ws-state-text").textContent = text;
    $("results-card").classList.toggle("is-loading", kind === "busy");
  }

  /* ---------- reading / writing controls ---------- */
  const byParam = (name) => controls.filter((c) => c.dataset.param === name);
  function readControl(name) {
    const [c] = byParam(name);
    if (c.dataset.type === "bool") return c.checked;
    if (c.dataset.type === "radio") return byParam(name).find((r) => r.checked)?.value;
    if (c.dataset.type === "number") return Number(c.value);
    return c.value;
  }
  function writeControl(name, value) {
    for (const c of byParam(name)) {
      if (c.dataset.type === "bool") c.checked = !!value;
      else if (c.dataset.type === "radio") c.checked = c.value === value;
      else c.value = value;
    }
  }
  const collect = () => Object.fromEntries([...new Set(controls.map((c) => c.dataset.param))].map((n) => [n, readControl(n)]));
  const populate = (p) => { for (const n of Object.keys(p)) if (byParam(n).length) writeControl(n, p[n]); refreshUI(); };

  /* ---------- UI that depends on other controls ---------- */
  function refreshUI() {
    const p = collect();
    for (const c of controls) { const out = c.closest(".field")?.querySelector("output"); if (out && c.type === "range") out.textContent = c.value; }
    $("bilateral-group").hidden = p.filter !== "bilateral";
    $("alpha-warn").hidden = !(p.booster_alpha > 3);
    const k = $("p-booster_kernel"), field = k.closest(".field");
    const range = p.booster_method === "morphological" ? [3, 21] : [1, 7];
    k.min = range[0]; k.max = range[1];
    if (Number(k.value) < range[0]) k.value = range[0]; if (Number(k.value) > range[1]) k.value = range[1];
    k.disabled = p.booster_method === "multiscale"; field.style.opacity = k.disabled ? .5 : 1;
    field.querySelector("output").textContent = k.value;
    for (const d of DETECTORS) $(`grp-${d}`).classList.toggle("group-active", d === p.detector);
    for (const n of ["canny_low", "canny_high"]) { const c = $(`p-${n}`); c.disabled = p.canny_auto; c.closest(".field").style.opacity = p.canny_auto ? .45 : 1; }
    $("p-canny_percentile").disabled = !p.canny_auto; $("p-canny_percentile").closest(".field").style.opacity = p.canny_auto ? 1 : .45;
  }

  /* ---------- validation (the server validates again) ---------- */
  function validate(p) {
    const errs = {};
    for (const n of Object.keys(ODD)) if (p[n] % 2 === 0) errs[n] = "Must be an odd number.";
    if (p.open_kernel !== 0 && (p.open_kernel < 3 || p.open_kernel % 2 === 0)) errs.open_kernel = "Use 0 (off) or an odd value from 3 to 15.";
    if ((p.filter === "median" || p.filter === "bilateral") && p.kernel_size < 3) errs.kernel_size = "Median and Bilateral need a kernel of at least 3.";
    if (!p.canny_auto && p.canny_low >= p.canny_high) errs.canny_low = "Lower threshold must be smaller than the upper threshold.";
    if (p.min_area_pct >= p.max_area_pct) errs.min_area_pct = "Minimum area must be smaller than maximum area.";
    if (p.booster_method === "morphological" && p.booster_kernel < 3) errs.booster_kernel = "Morphological kernel must be at least 3.";
    return errs;
  }
  function showErrors(errs) {
    for (const c of controls) { const f = c.closest(".field"); if (f) f.classList.remove("invalid"); }
    document.querySelectorAll(".err").forEach((e) => (e.textContent = ""));
    for (const [n, msg] of Object.entries(errs)) {
      byParam(n)[0].closest(".field").classList.add("invalid");
      const e = $(`e-${n}`); if (e) e.textContent = msg;
    }
    $("form-errors").replaceChildren(...(Object.keys(errs).length ? [ML.alertBox("error", "Please fix the highlighted settings before processing.")] : []));
    return Object.keys(errs).length === 0;
  }

  /* ---------- results gallery ---------- */
  function thumb(name, urls, onOpen) {
    return ML.el("button", { class: "thumb", type: "button", onclick: () => onOpen(name) }, [
      ML.el("img", { src: urls[name], alt: labelFor(name), loading: "lazy" }), ML.el("span", { text: labelFor(name) })]);
  }
  function openModal(name) {
    const meta = window.__meta;
    $("modal-title").textContent = labelFor(name);
    $("modal-download").href = meta.download_urls[name];
    $("modal").hidden = false;
    $("modal-viewer").replaceChildren(); $("modal-toolbar").replaceChildren();
    const host = ML.el("div"); $("modal-viewer").append(host);
    const tb = viewerToolbar(() => viewer);
    $("modal-toolbar").append(tb.bar);
    viewer = createViewer(host, { onZoom: tb.setLabel });
    viewer.setImages(meta.image_urls[name]);
    $("modal-close").focus();
  }
  const closeModal = () => { $("modal").hidden = true; };
  $("modal-close").addEventListener("click", closeModal);
  $("modal").addEventListener("click", (e) => { if (e.target === $("modal")) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("modal").hidden) closeModal(); });

  function renderResults(meta) {
    window.__meta = meta;
    const has = (n) => meta.images.includes(n);
    $("results-empty").hidden = true; $("results").hidden = false;
    $("gallery-main").replaceChildren(...["original", "preprocessed", "enhanced", "gradient", "edges", "contour_overlay", "mask", "reference_mask", "difference"]
      .filter(has).map((n) => thumb(n, meta.image_urls, openModal)));
    const rows = [];
    for (const d of DETECTORS) {
      rows.push(ML.el("div", { class: "row-label", text: DETECTOR_NAMES[d] }));
      for (const m of ["conv", "enh"]) {
        const n = `edges_${m}_${d}`; rows.push(has(n) ? thumb(n, meta.image_urls, openModal) : ML.el("span", { class: "muted", text: "n/a" }));
      }
    }
    $("gallery-pairs").replaceChildren(ML.el("span"), ML.el("b", { text: "Conventional" }), ML.el("b", { text: "Enhanced (Edge Booster)" }), ...rows);
    const msgs = [...(meta.warnings || [])];
    $("warnings").replaceChildren(...msgs.map((w) => ML.alertBox("warn", w)));
    const t = meta.canny_thresholds_used;
    $("canny-used").textContent = t ? `Canny thresholds used (low / high): conventional ${t.conventional.low} / ${t.conventional.high}, enhanced ${t.enhanced.low} / ${t.enhanced.high}${t.conventional.auto ? " (auto)" : " (manual)"}.` : "";
  }

  /* ---------- actions ---------- */
  form.addEventListener("input", () => { refreshUI(); params = collect(); ML.setParams(params); showErrors(validate(params)); });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    params = collect();
    if (!showErrors(validate(params))) return;
    const btn = $("process-btn"); btn.disabled = true; ML.setStatus($("status"), "Processing…", true); setState("busy", "Processing…");
    try {
      const meta = await ML.postJSON("/api/process", { result_id: rid, params: { ...params, candidate_index: null } });
      ML.setParams(params); renderResults(meta); ML.setStatus($("status"), "Done."); setState("ok", "Results ready");
    } catch (err) { ML.setStatus($("status"), ""); setState("", "Processing failed"); $("form-errors").replaceChildren(ML.alertBox("error", err.message)); }
    btn.disabled = false;
  });
  $("reset-btn").addEventListener("click", () => { params = { ...defaults }; ML.setParams({}); populate(params); showErrors({}); ML.toast("Settings reset to defaults."); });
  const PRESET = { canny_auto: true, canny_percentile: 85, kernel_size: 9, canny_blur: 9, close_kernel: 11, selection: "closest_to_center", ignore_border_px: 5 };
  $("preset-btn").addEventListener("click", () => {
    params = { ...collect(), ...PRESET }; ML.setParams(params); populate(params); showErrors(validate(params));
    ML.toast("Dermoscopy preset loaded. Press Process.");
  });
  $("delete-btn").addEventListener("click", async () => {
    if (!confirm("Delete the uploaded image and all generated results from this computer?")) return;
    try { await ML.api(`/api/results/${rid}`, { method: "DELETE" }); ML.clearResultId(); window.location.href = "upload.html"; }
    catch (err) { ML.toast(err.message, "error"); }
  });

  /* ---------- start ---------- */
  (async () => {
    if (!rid) { $("no-result").hidden = false; return; }
    try {
      const d = await ML.api("/api/defaults");
      defaults = d.defaults; params = { ...defaults, ...ML.getParams() };
      $("workspace").hidden = false; populate(params); setState("", "Ready to process");
      const meta = await ML.api(`/api/results/${rid}`);
      if (meta.processed) { renderResults(meta); setState("ok", "Results ready"); }
    } catch (err) {
      if (err.status === 404) { ML.clearResultId(); $("no-result").hidden = false; }
      else { $("workspace").hidden = false; $("form-errors").replaceChildren(ML.alertBox("error", err.message)); }
    }
  })();
})();
