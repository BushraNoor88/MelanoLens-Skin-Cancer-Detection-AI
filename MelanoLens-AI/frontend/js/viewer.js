/* MelanoLens AI - zoom/pan viewer with optional before/after slider. Vanilla JS, no libraries. */
function createViewer(container, { onZoom } = {}) {
  container.classList.add("viewer");
  const stage = ML.el("div", { class: "stage" });
  const base = ML.el("img", { alt: "", draggable: "false" });
  const over = ML.el("img", { class: "over", alt: "", draggable: "false", hidden: true });
  const divider = ML.el("div", { class: "divider", hidden: true });
  stage.append(base, over, divider);
  container.append(stage);
  container.tabIndex = 0;
  container.setAttribute("role", "img");

  let scale = 1, tx = 0, ty = 0, fit = 1, split = 50;
  const MIN = 0.1, MAX = 40;

  const apply = () => {
    stage.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    container.classList.toggle("pixelated", scale / fit > 2);
    onZoom && onZoom(Math.round(100 * scale / fit));
  };
  function reset() {
    const w = base.naturalWidth || 1, h = base.naturalHeight || 1;
    fit = Math.min(container.clientWidth / w, container.clientHeight / h, 1e9);
    scale = fit; tx = (container.clientWidth - w * scale) / 2; ty = (container.clientHeight - h * scale) / 2;
    apply();
  }
  function zoomAt(factor, cx, cy) {
    const next = Math.min(MAX * fit, Math.max(MIN * fit, scale * factor));
    const k = next / scale;
    tx = cx - (cx - tx) * k; ty = cy - (cy - ty) * k; scale = next; apply();
  }
  const center = () => [container.clientWidth / 2, container.clientHeight / 2];
  function setSplit(pct) {
    split = pct;
    over.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;           // left part shows the "over" (left) image
    divider.style.left = `${pct}%`;
  }

  /** Show one image, or two images (left = before, right = after) with a slider split. */
  function setImages(leftUrl, rightUrl = null) {
    const compare = !!rightUrl;
    over.hidden = divider.hidden = !compare;
    let loaded = 0;
    const done = () => { if (++loaded === (compare ? 2 : 1)) { reset(); if (compare) setSplit(split); } };
    base.onload = done; over.onload = done;
    if (compare) { base.src = rightUrl; over.src = leftUrl; } else { base.src = leftUrl; over.removeAttribute("src"); }
    container.setAttribute("aria-label", compare ? "Before and after comparison" : "Image viewer");
  }

  container.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = container.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  let drag = null;
  container.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, tx, ty }; container.setPointerCapture(e.pointerId); container.classList.add("dragging"); });
  container.addEventListener("pointermove", (e) => { if (drag) { tx = drag.tx + e.clientX - drag.x; ty = drag.ty + e.clientY - drag.y; apply(); } });
  const end = () => { drag = null; container.classList.remove("dragging"); };
  container.addEventListener("pointerup", end); container.addEventListener("pointercancel", end);
  container.addEventListener("keydown", (e) => {
    if (e.key === "+" || e.key === "=") zoomAt(1.25, ...center());
    else if (e.key === "-") zoomAt(1 / 1.25, ...center());
    else if (e.key === "0") reset();
  });
  window.addEventListener("resize", () => base.naturalWidth && reset());

  return { setImages, setSplit, reset, zoomIn: () => zoomAt(1.25, ...center()), zoomOut: () => zoomAt(1 / 1.25, ...center()) };
}

/** Standard toolbar (zoom in/out/reset + label) wired to a viewer. Returns the toolbar element and a label updater. */
function viewerToolbar(getViewer, extras = []) {
  const label = ML.el("span", { class: "zoom-label", "aria-live": "polite", text: "100%" });
  const bar = ML.el("div", { class: "viewer-bar" }, [
    ML.el("button", { class: "btn ghost small", type: "button", "aria-label": "Zoom out", text: "−", onclick: () => getViewer().zoomOut() }),
    ML.el("button", { class: "btn ghost small", type: "button", "aria-label": "Zoom in", text: "+", onclick: () => getViewer().zoomIn() }),
    ML.el("button", { class: "btn ghost small", type: "button", text: "Reset view", onclick: () => getViewer().reset() }),
    label, ...extras,
  ]);
  return { bar, setLabel: (pct) => { label.textContent = `${pct}%`; } };
}
