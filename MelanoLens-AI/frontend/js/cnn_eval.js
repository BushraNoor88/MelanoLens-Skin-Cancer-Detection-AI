/* Evaluation page: shows the saved HELD-OUT test-set results of both CNNs (from /api/cnn/evaluation). */
(() => {
  const body = document.getElementById("cnne-body");
  if (!body) return;
  const f = (v, d = 3) => (v == null ? "n/a" : Number(v).toFixed(d));
  const card = (title, kids) => ML.el("div", { class: "card" }, [ML.el("h3", { text: title }), ...kids]);
  const stat = (label, value, sub) => ML.el("div", { class: "stat" }, [ML.el("small", { text: label }), ML.el("b", { text: value }), ...(sub ? [ML.el("small", { style: "text-transform:none;letter-spacing:0", text: sub })] : [])]);
  const svg = (tag, attrs = {}, kids = []) => { const n = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); n.append(...[].concat(kids)); return n; };

  /* tiny SVG line chart: series = [{name, color, xs, ys}] */
  function lineChart(series, { xLabel, yLabel, yMin = 0, yMax = null, diagonal = false, title }) {
    const W = 460, H = 280, L = 46, R = 14, T = 16, B = 38;
    const allY = series.flatMap((s) => s.ys), allX = series.flatMap((s) => s.xs);
    const y1 = yMax ?? Math.max(...allY, 1e-6) * 1.05, x1 = Math.max(...allX, 1), x0 = Math.min(...allX, 0);
    const X = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R), Y = (v) => H - B - ((v - yMin) / (y1 - yMin || 1)) * (H - B - T);
    const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": title });
    for (let i = 0; i <= 4; i++) {
      const v = yMin + ((y1 - yMin) * i) / 4;
      g.append(svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "#e2e8f0" }), svg("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, v.toFixed(2)));
    }
    g.append(svg("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: "#94a3b8" }), svg("line", { x1: L, x2: L, y1: T, y2: H - B, stroke: "#94a3b8" }),
      svg("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle" }, xLabel), svg("text", { x: 12, y: (T + H - B) / 2, transform: `rotate(-90 12 ${(T + H - B) / 2})`, "text-anchor": "middle" }, yLabel));
    if (diagonal) g.append(svg("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: "#94a3b8", "stroke-dasharray": "4 4" }));
    series.forEach((s, i) => {
      g.append(svg("polyline", { fill: "none", stroke: s.color, "stroke-width": 2.2, "stroke-linejoin": "round", points: s.xs.map((x, k) => `${X(x).toFixed(1)},${Y(s.ys[k]).toFixed(1)}`).join(" ") }),
        svg("rect", { x: L + 10 + i * 130, y: T + 2, width: 10, height: 10, rx: 2, fill: s.color }), svg("text", { x: L + 25 + i * 130, y: T + 11 }, s.name));
    });
    return g;
  }

  function historyChart(h, keys, title) {
    const colors = ["#0f766e", "#8b5cf6", "#f59e0b", "#64748b"];
    const series = keys.filter((k) => h && h[k]).map((k, i) => ({ name: k.replace("_", " "), color: colors[i], xs: h[k].map((_, e) => e + 1), ys: h[k] }));
    return series.length ? lineChart(series, { xLabel: "epoch", yLabel: title, title: `Training history: ${title}` }) : ML.el("p", { class: "muted", text: "No history saved." });
  }

  function confusion(cm) {
    const max = Math.max(...cm.matrix.flat(), 1);
    return ML.el("div", { class: "table-wrap", style: "border:0" }, ML.el("table", { class: "cm" }, [
      ML.el("thead", {}, ML.el("tr", {}, [ML.el("th", { text: "true ↓ / predicted →" }), ...cm.labels.map((l) => ML.el("th", { text: l }))])),
      ML.el("tbody", {}, cm.matrix.map((row, i) => ML.el("tr", {}, [ML.el("th", { text: cm.labels[i] }),
        ...row.map((v, j) => ML.el("td", { class: "cell", style: `background:rgba(15,118,110,${(0.08 + 0.85 * v / max).toFixed(2)});color:${v / max > 0.5 ? "#fff" : "#101828"};outline:${i === j ? "2px solid #5eead4" : "none"}`, text: v }))])))]));
  }

  /* scatter: each dot = one test image; above the diagonal = U-Net scored higher */
  function scatter(rows) {
    const W = 460, H = 320, L = 46, R = 14, T = 14, B = 40, X = (v) => L + v * (W - L - R), Y = (v) => H - B - v * (H - B - T);
    const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": "Per-image Dice: classical on x, U-Net on y" });
    for (let i = 0; i <= 4; i++) { const v = i / 4;
      g.append(svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "#e2e8f0" }), svg("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, v.toFixed(2)),
        svg("text", { x: X(v), y: H - B + 15, "text-anchor": "middle" }, v.toFixed(2))); }
    g.append(svg("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: "#94a3b8", "stroke-dasharray": "4 4" }),
      svg("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }, "classical Dice (this image)"),
      svg("text", { x: 12, y: (T + H - B) / 2, transform: `rotate(-90 12 ${(T + H - B) / 2})`, "text-anchor": "middle" }, "U-Net Dice (this image)"));
    rows.forEach((r) => { const c = svg("circle", { cx: X(r.classical_dice), cy: Y(r.cnn_dice), r: 3.4, fill: r.cnn_dice - r.classical_dice > 0.01 ? "#0f766e" : (r.classical_dice - r.cnn_dice > 0.01 ? "#8b5cf6" : "#94a3b8"), "fill-opacity": 0.7 });
      c.append(svg("title", {}, `${r.id}: classical ${f(r.classical_dice)}, U-Net ${f(r.cnn_dice)}`)); g.append(c); });
    return g;
  }

  function compareCard(c) {
    if (!c) return card("Classical vs CNN segmentation (test set)", [ML.el("div", { class: "alert info" }, ML.el("span", {}, ["Not computed yet. Run ", ML.el("code", { text: "cd backend && python -m cnn.segmentation.compare" }), " on your computer, then reload this page. It scores the classical pipeline and the U-Net on the same held-out test images."]))]);
    const h = c.headline, P = c.paired.dice, ci = (m) => `${f(m.mean)} [${f(m.ci95[0])}, ${f(m.ci95[1])}]`;
    const row = (name, m) => ML.el("tr", {}, [ML.el("td", { text: name }), ML.el("td", { class: "num", text: ci(m.dice) }), ML.el("td", { class: "num", text: ci(m.iou) }),
      ...["precision", "recall", "pixel_accuracy"].map((k) => ML.el("td", { class: "num", text: m[k] ? f(m[k].mean) : "n/a" }))]);
    const sign = (v) => (v >= 0 ? "+" : "") + v.toFixed(3), excl = P.ci95[0] > 0 || P.ci95[1] < 0;
    return card("Classical vs CNN segmentation (test set)", [
      ML.el("p", { class: "muted", text: `Both methods were scored on the same ${c.n_test} held-out test images, at reference-mask resolution, with the same metric code. The classical method shown is ${c.classical.headline_method.replace(":", " / ")}, chosen on the validation split (${c.classical.selection.n_validation} images), never on the test split. Classical parameters: ${typeof c.classical.parameters === "string" ? c.classical.parameters : "custom"}.` }),
      ML.el("div", { class: "table-wrap" }, ML.el("table", {}, [ML.el("thead", {}, ML.el("tr", {}, ["Method", "Dice mean [95% CI]", "IoU mean [95% CI]", "Precision", "Recall", "Pixel acc."].map((x) => ML.el("th", { text: x })))),
        ML.el("tbody", {}, [row("Classical (Edge Booster pipeline)", h.classical), row("U-Net (CNN)", h.cnn)])])),
      ML.el("div", { class: "stat-grid", style: "margin-top:1rem" }, [stat("Dice difference (U-Net − classical)", sign(P.mean_difference), `95% CI [${f(P.ci95[0])}, ${f(P.ci95[1])}] ${excl ? "excludes" : "includes"} 0`),
        stat("Images where U-Net scored higher", `${P.cnn_better} / ${P.n}`, `ties ${P.ties}, classical higher ${P.classical_better} (tie = within 0.01)`),
        stat("Wilcoxon signed-rank p", P.wilcoxon_p == null ? "n/a" : String(P.wilcoxon_p), "paired, on per-image Dice"),
        stat("No contour found", `${c.classical.n_no_mask} / ${c.n_test}`, `classical; U-Net empty: ${c.cnn.n_empty_prediction}`)]),
      ML.el("div", { class: "eval-two", style: "margin-top:1rem" }, [ML.el("div", {}, [ML.el("h4", { text: "Every test image" }), scatter(c.per_image),
        ML.el("div", { class: "legend", style: "margin-top:.5rem" }, [ML.el("span", {}, [ML.el("i", { style: "background:#0f766e" }), "U-Net higher"]), ML.el("span", {}, [ML.el("i", { style: "background:#8b5cf6" }), "classical higher"]), ML.el("span", {}, [ML.el("i", { style: "background:#94a3b8" }), "tie"])])]),
        ML.el("div", {}, [ML.el("h4", { text: "All 10 classical variants (test set)" }),
          ML.el("div", { class: "table-wrap" }, ML.el("table", {}, [ML.el("thead", {}, ML.el("tr", {}, ["Detector / mode", "Dice mean [95% CI]", "No mask"].map((x) => ML.el("th", { text: x })))),
            ML.el("tbody", {}, c.classical_all_methods.map((m) => ML.el("tr", { class: m.method === c.classical.headline_method ? "selected-row" : "" }, [ML.el("td", { text: m.method.replace(":", " / ") }), ML.el("td", { class: "num", text: ci(m.dice) }), ML.el("td", { class: "num", text: m.n_no_mask })])))])),
          ML.el("p", { class: "muted", text: "Highlighted = the variant chosen on the validation split." })])]),
      ML.el("ul", { class: "muted", style: "margin-top:.8rem" }, c.protocol.map((t) => ML.el("li", { text: t })))]);
  }

  function segCard(d) {
    if (!d.available) return card("Segmentation (U-Net)", [unavailable(d, "segmentation")]);
    const r = d.results, m = r.metrics, ms = (k) => (m[k] ? `${f(m[k].mean)} ± ${f(m[k].std)}` : "n/a");
    return card("Segmentation (U-Net)", [
      ML.el("p", { class: "muted", text: `Evaluated on ${r.n_test} held-out test images at original resolution (train ${r.split_sizes.train}, validation ${r.split_sizes.val}, test ${r.split_sizes.test}). Values are mean ± std over images.` }),
      ML.el("div", { class: "stat-grid" }, [stat("Dice", ms("dice"), `median ${f(m.dice && m.dice.median)}`), stat("IoU", ms("iou")), stat("Precision", ms("precision")), stat("Recall", ms("recall")), stat("Pixel accuracy", ms("pixel_accuracy"))]),
      ML.el("h4", { text: "Training history (not test results)" }), historyChart(d.history, ["loss", "val_loss"], "loss"),
      ML.el("p", { class: "muted", text: "The classical pipeline has no dataset-wide test score; compare it with the CNN image by image on the Boundary Analysis page." })]);
  }

  /* Default (argmax) vs tuned decision threshold, both measured on the held-out TEST split. */
  function thresholdBlock(r, thr) {
    if (!r.decision_threshold) return [];
    const dt = r.decision_threshold, pos = dt.positive_class, b = r.at_default_threshold;
    const row = (label, m) => { const c = m.per_class.find((x) => x.class === pos);
      return ML.el("tr", {}, [ML.el("td", { text: label }), ML.el("td", { class: "num", text: f(m.accuracy) }), ML.el("td", { class: "num", text: f(c.recall) }), ML.el("td", { class: "num", text: f(c.precision) }), ML.el("td", { class: "num", text: f(m.macro_f1) })]); };
    const kids = [ML.el("h4", { text: `Decision threshold for "${pos}"` }),
      ML.el("p", { class: "muted", text: `Chosen on the validation split only (${dt.tuned_on}), criterion: ${dt.criterion}${dt.target_recall ? ` (target recall ${dt.target_recall})` : ""}. Test results below were measured once, afterwards. A lower threshold catches more ${pos} cases (higher recall) but raises more false alarms (lower precision).` }),
      ML.el("div", { class: "table-wrap" }, ML.el("table", {}, [
        ML.el("thead", {}, ML.el("tr", {}, ["Rule (test set)", "Accuracy", `${pos} recall`, `${pos} precision`, "Macro F1"].map((h) => ML.el("th", { text: h })))),
        ML.el("tbody", {}, [row("Default (highest probability)", b), row(`Tuned (P ≥ ${dt.value})`, r)])]))];
    if (thr && thr.sweep) kids.push(ML.el("details", {}, [ML.el("summary", { text: "Validation sweep (threshold vs recall/precision)" }),
      ML.el("div", { class: "table-wrap" }, ML.el("table", {}, [ML.el("thead", {}, ML.el("tr", {}, ["Threshold", "Recall", "Precision", "Flagged"].map((h) => ML.el("th", { text: h })))),
        ML.el("tbody", {}, thr.sweep.map((x) => ML.el("tr", { class: x.threshold === dt.value ? "selected-row" : "" }, [ML.el("td", { class: "num", text: x.threshold.toFixed(2) }), ML.el("td", { class: "num", text: f(x.recall) }), ML.el("td", { class: "num", text: f(x.precision) }), ML.el("td", { class: "num", text: x.flagged })])))]))]));
    return kids;
  }

  function clsCard(d) {
    if (!d.available) return card("Classification (MobileNetV2)", [unavailable(d, "classification")]);
    const r = d.results, kids = [
      ML.el("p", { class: "muted", text: `Evaluated on ${r.n_test} held-out test images. Test class counts: ${Object.entries(r.test_class_counts).map(([k, v]) => `${k} ${v}`).join(", ")}.` }),
      ML.el("div", { class: "stat-grid" }, [stat("Accuracy", f(r.accuracy)), stat("Macro precision", f(r.macro_precision)), stat("Macro recall", f(r.macro_recall)), stat("Macro F1", f(r.macro_f1)), stat("ROC-AUC", f(r.roc_auc), r.roc_auc_note)]),
      ML.el("p", { class: "muted", text: r.macro_note }),
      ...thresholdBlock(r, d.threshold),
      ML.el("h4", { text: "Confusion matrix (test set)" }), confusion(r.confusion_matrix), ML.el("p", { class: "muted", text: r.confusion_matrix.note }),
      ML.el("h4", { text: "Per-class results" }),
      ML.el("div", { class: "table-wrap" }, ML.el("table", {}, [ML.el("thead", {}, ML.el("tr", {}, ["Class", "Precision", "Recall", "F1", "Support"].map((h) => ML.el("th", { text: h })))),
        ML.el("tbody", {}, r.per_class.map((c) => ML.el("tr", {}, [ML.el("td", { text: c.class }), ...[c.precision, c.recall, c.f1].map((v) => ML.el("td", { class: "num", text: f(v) })), ML.el("td", { class: "num", text: c.support })])))])),
    ];
    if (r.roc_curve) kids.push(ML.el("h4", { text: "ROC curve (test set)" }), lineChart([{ name: `AUC ${f(r.roc_auc)}`, color: "#0f766e", xs: r.roc_curve.fpr, ys: r.roc_curve.tpr }], { xLabel: "false positive rate", yLabel: "true positive rate", yMax: 1, diagonal: true, title: "ROC curve" }));
    kids.push(ML.el("h4", { text: "Training history (not test results)" }), historyChart(d.history, ["loss", "val_loss", "accuracy", "val_accuracy"], "loss / accuracy"));
    return card("Classification (MobileNetV2)", kids);
  }

  function unavailable(d, kind) {
    const cmd = kind === "segmentation" ? "python -m cnn.segmentation.train" : "python -m cnn.classification.train";
    return ML.el("div", { class: "alert info" }, ML.el("span", {}, [d.message + " Run ", ML.el("code", { text: `cd backend && ${cmd}` }), " on your computer, then reload this page."]));
  }

  (async () => {
    try {
      const d = await ML.api("/api/cnn/evaluation");
      body.replaceChildren(ML.el("div", { class: "eval-two" }, [segCard(d.segmentation), clsCard(d.classification)]),
        ...(d.segmentation.available ? [ML.el("div", { style: "margin-top:1rem" }, compareCard(d.segmentation.comparison))] : []));
    } catch (e) { body.replaceChildren(ML.alertBox("error", e.message)); }
  })();
})();
