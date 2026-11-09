/* Upload page: client-side checks (the server validates again), upload, then run the default pipeline. */
(() => {
  const $ = (id) => document.getElementById(id);
  const ALLOWED = ["jpg", "jpeg", "png"];
  let maxBytes = 10 * 1024 * 1024;
  let image = null, mask = null;

  ML.api("/api/health").then((h) => { maxBytes = h.max_upload_mb * 1024 * 1024; }).catch((e) => showMessage("error", e.message));

  /* Upload with real byte progress (fetch cannot report upload progress). Errors use the same shape as ML.api. */
  function uploadWithProgress(form, onProgress) {
    return new Promise((resolve, reject) => {
      const x = new XMLHttpRequest();
      x.open("POST", "/api/upload"); x.responseType = "json";
      x.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
      x.onload = () => {
        const b = x.response;
        if (x.status >= 200 && x.status < 300) return resolve(b);
        const err = new Error(b?.error?.message || `Request failed (${x.status}).`); err.code = b?.error?.code; err.status = x.status; reject(err);
      };
      x.onerror = () => reject(new Error("Cannot reach the MelanoLens server. Is `python app.py` running?"));
      x.send(form);
    });
  }
  function showUploadProgress(frac) {
    const pct = Math.round(frac * 100);
    $("status").replaceChildren(ML.el("span", { class: "spinner", "aria-hidden": "true" }), `Uploading and validating… ${pct}%`,
      ML.el("div", { class: "progress", "aria-hidden": "true", style: `--p:${pct}%` }, ML.el("i")));
  }

  const ext = (f) => f.name.split(".").pop().toLowerCase();
  function validate(file, label) {
    if (!ALLOWED.includes(ext(file))) return `${label}: invalid file type ".${ext(file)}". Use JPG, JPEG or PNG.`;
    if (file.size === 0) return `${label}: the file is empty.`;
    if (file.size > maxBytes) return `${label}: ${ML.fmtBytes(file.size)} exceeds the ${ML.fmtBytes(maxBytes)} limit.`;
    return null;
  }
  function showMessage(kind, text) { $("messages").replaceChildren(ML.alertBox(kind, text)); }
  const clearMessage = () => $("messages").replaceChildren();

  function setImage(file) {
    clearMessage();
    const problem = validate(file, "Image");
    if (problem) { image = null; showMessage("error", problem); render(); return; }
    image = file; render();
  }

  function render() {
    $("process-btn").disabled = !image; $("clear-btn").disabled = !image && !mask;
    const info = $("info"), preview = $("preview");
    info.replaceChildren();
    if (!image) { preview.replaceChildren(ML.el("span", { class: "muted", text: "No image selected" })); return; }
    const url = URL.createObjectURL(image);
    const img = ML.el("img", { alt: "Preview of the selected image", src: url });
    preview.replaceChildren(img);
    img.onload = () => {                                   // dimensions are only known after decoding
      info.replaceChildren(...[["Name", image.name], ["Type", ext(image).toUpperCase()], ["Size", ML.fmtBytes(image.size)],
        ["Dimensions", `${img.naturalWidth} × ${img.naturalHeight} px`], ["Reference mask", mask ? mask.name : "none"]]
        .flatMap(([k, v]) => [ML.el("dt", { text: k }), ML.el("dd", { text: v })]));
    };
    img.onerror = () => { showMessage("error", "The browser could not read this image. It may be corrupted."); image = null; render(); };
  }

  const dz = $("dropzone");
  dz.addEventListener("click", () => $("file-input").click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); } });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); if (e.dataTransfer.files[0]) setImage(e.dataTransfer.files[0]); });
  $("file-input").addEventListener("change", (e) => { if (e.target.files[0]) setImage(e.target.files[0]); });
  $("mask-input").addEventListener("change", (e) => {
    const f = e.target.files[0]; mask = null;
    if (f) { const p = validate(f, "Mask"); if (p) { showMessage("error", p); e.target.value = ""; } else { mask = f; clearMessage(); } }
    render();
  });
  $("clear-btn").addEventListener("click", () => { image = mask = null; $("file-input").value = ""; $("mask-input").value = ""; clearMessage(); ML.setStatus($("status"), ""); render(); });

  $("process-btn").addEventListener("click", async () => {
    const btn = $("process-btn"); btn.disabled = true; clearMessage();
    try {
      showUploadProgress(0);
      const form = new FormData(); form.append("image", image); if (mask) form.append("mask", mask);
      const up = await uploadWithProgress(form, showUploadProgress);
      ML.setResultId(up.result_id); ML.setParams({});
      ML.setStatus($("status"), "Running the pipeline with default settings…", true);
      await ML.postJSON("/api/process", { result_id: up.result_id, params: {} });
      window.location.href = "workspace.html";
    } catch (e) {
      ML.setStatus($("status"), ""); showMessage("error", e.message); btn.disabled = false;
    }
  });
  render();
})();
