/* CNN section of the Boundary Analysis page: runs /api/cnn/segment and /api/cnn/classify for the current
   result and shows the stored outputs next to the classical pipeline. Nothing here invents numbers. */
(() => {
  const $ = (id) => document.getElementById(id);
  const rid = ML.getResultId();
  if (!rid || !$("cnn")) return;
  let status = null;
  const pct = (v) => (v == null ? "n/a" : `${(v * 100).toFixed(1)}%`);
  const code = (t) => ML.el("code", { text: t });

  const emit = (name, detail) => document.dispatchEvent(new CustomEvent(name, { detail }));
  const unavailable = (k) => (k.available ? null : k.reason || "Not available.");

  function renderStatus() {
    const msgs = [];
    for (const [key, label, cmd] of [["segmentation", "Segmentation model", status.train_commands.segmentation], ["classification", "Classification model", status.train_commands.classification]]) {
      const r = unavailable(status[key]);
      if (r) msgs.push(ML.el("div", { class: "alert warn" }, [ML.el("span", {}, [`${label}: ${r} `, status[key].tensorflow_installed ? ["Train it locally with ", code(cmd), "."] : ""].flat())]));
    }
    $("cnn-status").replaceChildren(...msgs);
    $("cnn-seg-btn").disabled = !status.segmentation.available;
    $("cnn-cls-btn").disabled = !status.classification.available;
  }

  const fig = (name, caption, url) => ML.el("figure", { class: "figure-frame" }, [
    ML.el("img", { src: url, alt: caption, loading: "lazy" }),
    ML.el("figcaption", {}, [caption, " ", ML.el("button", { class: "linkbtn", type: "button", text: "View", "aria-label": `View ${caption} in the viewer`, onclick: () => emit("melanolens:view", name) })]),
  ]);

  function metricRow(label, m) {
    return ML.el("tr", {}, [ML.el("td", { text: label }), ...["dice", "iou", "precision", "recall", "pixel_accuracy"].map((k) => ML.el("td", { class: "num", text: ML.fmt(m[k]) }))]);
  }

  function renderSeg(meta) {
    const s = meta.cnn && meta.cnn.segmentation;
    if (!s) return;
    const u = meta.image_urls;
    $("cnn-seg-result").hidden = false;
    const figs = [["original", "Original image"], ["cnn_mask", "CNN predicted mask"], ["cnn_region", "CNN lesion region"], ["cnn_overlay", "CNN boundary overlay"],
      ["contour_overlay", "Classical contour overlay"], ["cnn_compare", "Classical (cyan) vs CNN (magenta)"], ["cnn_probability", "CNN probability map"]].filter(([n]) => u[n]);
    $("cnn-figs").replaceChildren(...figs.map(([n, c]) => fig(n, c, u[n])));
    $("cnn-seg-stats").replaceChildren(
      ...[["CNN lesion area", pct(s.stats.lesion_area_fraction)], ["CNN perimeter", `${s.stats.perimeter_px.toLocaleString()} px`], ["Inference time", `${s.inference_ms} ms`]]
        .map(([l, v]) => ML.el("div", { class: "stat" }, [ML.el("small", { text: l }), ML.el("b", { text: v })])));
    $("cnn-seg-warnings").replaceChildren(...s.warnings.map((w) => ML.alertBox("warn", w)));

    const c = s.comparison, parts = [];
    if (c.classical_available) {
      const a = c.agreement_cnn_vs_classical;
      parts.push(ML.el("h4", { text: "Agreement between the two methods" }),
        ML.el("div", { class: "stat-grid" }, [["Dice (CNN vs classical)", a.dice], ["IoU (CNN vs classical)", a.iou]].map(([l, v]) => ML.el("div", { class: "stat" }, [ML.el("small", { text: l }), ML.el("b", { text: ML.fmt(v) })]))));
    } else if (c.classical_unavailable_reason) parts.push(ML.alertBox("info", `No classical-vs-CNN comparison: ${c.classical_unavailable_reason}`));
    if (c.vs_reference) {
      parts.push(ML.el("h4", { text: "Scores against the reference mask you uploaded" }),
        ML.el("div", { class: "table-wrap" }, ML.el("table", {}, [
          ML.el("thead", {}, ML.el("tr", {}, ["Method", "Dice", "IoU", "Precision", "Recall", "Pixel acc."].map((h) => ML.el("th", { text: h })))),
          ML.el("tbody", {}, [metricRow("CNN (U-Net)", c.vs_reference.cnn), ...(c.vs_reference.classical ? [metricRow("Classical (selected contour)", c.vs_reference.classical)] : [])])])));
    } else parts.push(ML.el("p", { class: "muted", text: "No reference mask is attached to this image, so no accuracy scores are shown." }));
    parts.push(ML.el("p", { class: "muted", text: c.note }));
    $("cnn-compare").replaceChildren(...parts);
    $("cnn-seg-model").textContent = `Model: ${s.model.name}, input ${s.model.input_size}×${s.model.input_size}, threshold ${s.model.threshold}, post-processing: ${s.model.postprocessing}.`
      + (s.model.trained_at ? ` Trained ${s.model.trained_at.replace("T", " ")}.` : "");
  }

  function renderCls(meta) {
    const r = meta.cnn && meta.cnn.classification;
    if (!r) return;
    const p = r.prediction, numeric = r.model.classes.every((c) => /^\d+$/.test(c));
    const bars = Object.entries(p.probabilities).map(([name, v]) => ML.el("div", { class: "bar-row" }, [
      ML.el("span", { class: "lbl", text: numeric ? `Class ${name}` : name }),
      ML.el("div", { class: "bar" }, [ML.el("div", { class: "track" }, ML.el("i", { class: "fill enh", "data-w": v * 100 })), ML.el("span", { class: "v", text: pct(v) })])]));
    const box = $("cnn-cls-result"); box.hidden = false;
    box.replaceChildren(
      ML.el("div", { class: "pred" }, [ML.el("div", {}, [ML.el("small", { class: "muted", text: "Model prediction" }), ML.el("div", { class: "pred-class", text: numeric ? `Class ${p.predicted_class}` : p.predicted_class })]),
        ML.el("div", {}, [ML.el("small", { class: "muted", text: "Confidence" }), ML.el("div", { class: "pred-conf", text: pct(p.confidence) })])]),
      ML.el("div", { class: "bars", style: "margin:1rem 0" }, bars),
      ...(p.decision && p.decision.threshold != null ? [ML.el("p", { class: "muted", text: `Decision rule: "${p.decision.positive_class}" is predicted when its probability is at least ${p.decision.threshold} (a threshold tuned on validation data to catch more ${p.decision.positive_class} cases, at the cost of more false alarms). This image's ${p.decision.positive_class} probability: ${pct(p.decision.positive_probability)}. The confidence shown is the probability of the predicted class.` })] : []),
      ML.alertBox("warn", `${r.kind_note} ${p.disclaimer}`),
      ML.el("p", { class: "muted", text: `Model: ${r.model.name}, input ${r.model.input_size}×${r.model.input_size}` + (r.model.pretrained_weights ? `, ${r.model.pretrained_weights} pre-trained weights` : ", no pre-trained weights")
        + `, classes: ${r.model.classes.join(", ")}` + (numeric ? " (numeric label codes exactly as written in the dataset CSV)" : "") + (r.model.trained_at ? `. Trained ${r.model.trained_at.replace("T", " ")}.` : ".") + " Confidence is the model's softmax output, not a measured probability of being correct." }));
    requestAnimationFrame(() => requestAnimationFrame(() => box.querySelectorAll(".fill").forEach((f) => { f.style.width = `${f.dataset.w}%`; })));
  }

  async function refresh() {
    const meta = await ML.api(`/api/results/${rid}`);
    emit("melanolens:meta", meta);
    renderSeg(meta); renderCls(meta);
    return meta;
  }

  async function run(kind, btn, statusEl) {
    btn.disabled = true;
    ML.setStatus(statusEl, kind === "segment" ? "Running U-Net segmentation…" : "Running classification…", true);
    try {
      await ML.postJSON(`/api/cnn/${kind}`, { result_id: rid });
      await refresh();
      ML.setStatus(statusEl, "Done.");
    } catch (e) {
      ML.setStatus(statusEl, ""); statusEl.append(ML.alertBox("error", e.message));
    } finally { btn.disabled = false; }
  }
  $("cnn-seg-btn").addEventListener("click", () => run("segment", $("cnn-seg-btn"), $("cnn-seg-status")));
  $("cnn-cls-btn").addEventListener("click", () => run("classify", $("cnn-cls-btn"), $("cnn-cls-status")));

  (async () => {
    try {
      status = await ML.api("/api/cnn/status"); renderStatus();
      const meta = await ML.api(`/api/results/${rid}`);
      renderSeg(meta); renderCls(meta);
      if (location.hash === "#cnn") setTimeout(() => $("cnn").scrollIntoView({ behavior: "smooth" }), 400);
    } catch (e) { $("cnn-status").replaceChildren(ML.alertBox("error", e.message)); }
  })();
})();
