/* Boundary Analysis page: viewer, before/after slider, measurements, candidate switching, reference comparison. */
(() => {
  const $ = (id) => document.getElementById(id);
  const rid = ML.getResultId();
  let meta = null, mainViewer = null, cmpViewer = null;

  const fillSelect = (sel, names, value) => {
    sel.replaceChildren(...names.map((n) => ML.el("option", { value: n, text: labelFor(n) })));
    if (value && names.includes(value)) sel.value = value;
  };
  const stat = (label, value) => ML.el("div", { class: "stat" }, [ML.el("small", { text: label }), ML.el("b", { text: value })]);

  function showEmpty(html) { const e = $("empty"); e.hidden = false; e.replaceChildren(...html); $("content").hidden = true; }

  function setupViewers() {
    const tb1 = viewerToolbar(() => mainViewer); $("viewer-toolbar").replaceChildren(tb1.bar);
    $("viewer").replaceChildren(); mainViewer = createViewer($("viewer"), { onZoom: tb1.setLabel });
    const tb2 = viewerToolbar(() => cmpViewer); $("cmp-toolbar").replaceChildren(tb2.bar);
    $("cmp-viewer").replaceChildren(); cmpViewer = createViewer($("cmp-viewer"), { onZoom: tb2.setLabel });
  }

  function showMain() {
    const n = $("view-select").value;
    mainViewer.setImages(meta.image_urls[n]);
    $("dl-current").href = meta.download_urls[n];
  }
  function showCompare() {
    cmpViewer.setImages(meta.image_urls[$("cmp-left").value], meta.image_urls[$("cmp-right").value]);
    cmpViewer.setSplit(Number($("cmp-slider").value));
  }

  function render() {
    $("empty").hidden = true; $("content").hidden = false;
    const names = meta.images;
    const first = names.includes("contour_overlay") ? "contour_overlay" : "original";
    fillSelect($("view-select"), names, first);
    fillSelect($("cmp-left"), names, "original"); fillSelect($("cmp-right"), names, names.includes("enhanced") ? "enhanced" : names[0]);
    $("dl-overlay").href = meta.download_urls.contour_overlay || "#";
    const maskLink = $("dl-mask");
    if (meta.download_urls.mask) { maskLink.href = meta.download_urls.mask; maskLink.removeAttribute("aria-disabled"); maskLink.style.opacity = 1; }
    else { maskLink.removeAttribute("href"); maskLink.setAttribute("aria-disabled", "true"); maskLink.style.opacity = .5; }
    showMain(); showCompare();

    const b = meta.boundary;
    $("messages").replaceChildren(...(meta.warnings || []).map((w) => ML.alertBox("warn", w)));
    $("meas-context").textContent = `${labelFor(`edges_${b.mode === "enhanced" ? "enh" : "conv"}_${b.detector}`)} · ${b.candidate_count} candidate(s)`
      + (b.detector === "canny" && meta.canny_thresholds_used ? ` · Canny ${meta.canny_thresholds_used[b.mode].low} / ${meta.canny_thresholds_used[b.mode].high}${meta.canny_thresholds_used[b.mode].auto ? " (auto)" : ""}` : "");
    const m = b.measurements;
    $("stats").replaceChildren(...(m ? [
      stat("Area", `${m.area_px.toLocaleString()} px`), stat("Perimeter", `${m.perimeter_px.toLocaleString()} px`),
      stat("Circularity", ML.fmt(m.circularity)),
      stat("Bounding box", `${m.bbox.width} × ${m.bbox.height} px`), stat("Box origin (x, y)", `${m.bbox.x}, ${m.bbox.y}`),
      stat("Centroid (x, y)", `${m.centroid.x}, ${m.centroid.y}`), stat("Touches image border", m.touches_border ? "yes" : "no"),
    ] : [ML.alertBox("warn", b.message || "No reliable candidate contour found.")]));

    $("selection").value = b.selection;
    $("cand-body").replaceChildren(...(b.candidates.length ? b.candidates.map((c) => ML.el("tr", { class: c.index === b.selected_index ? "selected-row" : "" }, [
      ML.el("td", { text: c.index === b.selected_index ? `${c.index} (selected)` : c.index }),
      ML.el("td", { class: "num", text: c.area_px.toLocaleString() }), ML.el("td", { class: "num", text: c.perimeter_px.toLocaleString() }),
      ML.el("td", { class: "num", text: ML.fmt(c.circularity) }),
      ML.el("td", {}, c.index === b.selected_index ? [] : [ML.el("button", { class: "btn ghost small", type: "button", text: "Use this", onclick: () => reprocess({ candidate_index: c.index }) })]),
    ])) : [ML.el("tr", {}, ML.el("td", { colspan: 5, class: "muted", text: "No candidates found." }))]));

    const ref = meta.reference.available;
    $("ref-none").hidden = ref; $("ref-body").hidden = !ref;
    if (ref) {
      $("ref-img").src = meta.image_urls.reference_mask;
      for (const [id, key, ok, no] of [["pred-img", "mask", "Candidate mask", "No candidate mask"], ["diff-img", "difference", "Difference image", "No difference image"]]) {
        const img = $(id), url = meta.image_urls[key];
        if (url) img.src = url; else img.removeAttribute("src");
        img.alt = url ? ok : no; img.hidden = !url;            // avoid a broken-image icon when the backend has no such image
      }
      const row = meta.evaluation.rows.find((r) => r.detector === b.detector && r.mode === b.mode);
      $("ref-stats").replaceChildren(...(row && row.mask_found ? [["Dice", row.dice], ["IoU", row.iou], ["Precision", row.precision], ["Recall", row.recall], ["F1", row.f1], ["Boundary F1", row.boundary_f1]]
        .map(([l, v]) => stat(l, ML.fmt(v))) : [ML.alertBox("warn", "No candidate mask for this detector and mode, so no mask metrics can be computed.")]));
      $("ref-note").textContent = `Metrics compare the filled candidate mask with the reference mask for the selected candidate. Boundary tolerance: ${meta.evaluation.tolerance_px} px.` +
        (meta.reference.resized ? " The reference mask was resized with nearest-neighbour interpolation to match the processed image." : "");
    }
    $("notes").replaceChildren(...meta.notes.map((t) => ML.el("li", { text: t })));
  }

  async function reprocess(change) {
    try {
      const p = { ...meta.params, ...change };
      meta = await ML.postJSON("/api/process", { result_id: rid, params: p });
      ML.setParams({ ...ML.getParams(), ...Object.fromEntries(Object.entries(p).filter(([k]) => k !== "candidate_index")) });
      render(); ML.toast("Updated.");
    } catch (e) { ML.toast(e.message, "error"); }
  }

  /* The CNN section (cnn.js) adds cnn_* images: refresh the pickers without losing the user's choices. */
  document.addEventListener("melanolens:meta", (e) => {
    if (!meta) return;
    meta = e.detail;
    for (const id of ["view-select", "cmp-left", "cmp-right"]) { const v = $(id).value; fillSelect($(id), meta.images, v); }
  });
  document.addEventListener("melanolens:view", (e) => {         // "View" buttons on CNN figures
    if (!meta.images.includes(e.detail)) return;
    $("view-select").value = e.detail; showMain(); $("viewer-h").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  $("view-select").addEventListener("change", showMain);
  $("cmp-left").addEventListener("change", showCompare); $("cmp-right").addEventListener("change", showCompare);
  $("cmp-slider").addEventListener("input", (e) => cmpViewer.setSplit(Number(e.target.value)));
  $("selection").addEventListener("change", (e) => reprocess({ selection: e.target.value, candidate_index: null }));

  (async () => {
    if (!rid) return showEmpty(["No image loaded yet. ", ML.el("a", { href: "upload.html", text: "Upload an image" }), " first."]);
    try {
      meta = await ML.api(`/api/results/${rid}`);
      if (!meta.processed) return showEmpty(["This image has not been processed yet. ", ML.el("a", { href: "workspace.html", text: "Open the workspace" }), " and press Process."]);
      setupViewers(); render();
    } catch (e) {
      if (e.status === 404) { ML.clearResultId(); showEmpty(["That result no longer exists. ", ML.el("a", { href: "upload.html", text: "Upload an image" }), "."]); }
      else showEmpty([e.message]);
    }
  })();
})();
