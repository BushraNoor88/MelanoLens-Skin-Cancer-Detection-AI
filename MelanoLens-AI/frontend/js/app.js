/* MelanoLens AI - shared helpers: API calls, page chrome, state, toasts. */
document.documentElement.classList.add("js");
const ML = (() => {
  const KEY = "melanolens.resultId";
  const PARAMS_KEY = "melanolens.params";
  const DISCLAIMER = "MelanoLens AI is an educational computer vision project for experimental skin lesion boundary enhancement and image analysis. It is not a medical diagnostic tool. Its outputs must not be used to diagnose skin cancer or make treatment decisions.";
  const LOGO = `<svg viewBox="0 0 64 64" aria-hidden="true"><circle cx="26" cy="26" r="20" fill="#0f2740" stroke="#22d3ee" stroke-width="4"/><path d="M15 24c2-7 9-11 15-9 6 1 11 6 10 12-1 7-8 11-14 10-4 4-12 0-12-6-1-3 0-5 1-7z" fill="#14b8a6" fill-opacity=".35" stroke="#e6fffb" stroke-width="2" stroke-linejoin="round"/><line x1="41" y1="41" x2="58" y2="58" stroke="#22d3ee" stroke-width="6" stroke-linecap="round"/></svg>`;
  const PAGES = [
    ["index.html", "Dashboard"], ["upload.html", "Upload"], ["workspace.html", "Edge Booster"],
    ["analysis.html", "Boundary Analysis"], ["evaluation.html", "Evaluation"], ["about.html", "About"],
  ];

  /** Create an element: el("div", {class:"x", onclick: fn}, "text" | node | [children]) */
  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === false || v == null) continue;
      if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of [].concat(children)) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return node;
  }

  /** fetch wrapper: returns parsed JSON, throws Error(message) using the API error schema. */
  async function api(path, options = {}) {
    let res;
    try { res = await fetch(path, options); }
    catch { throw new Error("Cannot reach the MelanoLens server. Is `python app.py` running?"); }
    const type = res.headers.get("content-type") || "";
    const body = type.includes("json") ? await res.json() : null;
    if (!res.ok) {
      const err = new Error(body?.error?.message || `Request failed (${res.status}).`);
      err.code = body?.error?.code; err.status = res.status; throw err;
    }
    return body;
  }
  const postJSON = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

  const getResultId = () => sessionStorage.getItem(KEY);
  const setResultId = (id) => sessionStorage.setItem(KEY, id);
  const clearResultId = () => { sessionStorage.removeItem(KEY); sessionStorage.removeItem(PARAMS_KEY); };
  const getParams = () => { try { return JSON.parse(sessionStorage.getItem(PARAMS_KEY)) || {}; } catch { return {}; } };
  const setParams = (p) => sessionStorage.setItem(PARAMS_KEY, JSON.stringify(p));

  const imageUrl = (rid, type) => `/api/results/${rid}/image/${type}`;
  const downloadUrl = (rid, type) => `/api/results/${rid}/download/${type}`;

  function toast(message, kind = "info") {
    let wrap = document.querySelector(".toast-wrap");
    if (!wrap) { wrap = el("div", { class: "toast-wrap", role: "status", "aria-live": "polite" }); document.body.append(wrap); }
    const t = el("div", { class: `toast ${kind === "error" ? "error" : ""}`, text: message });
    wrap.append(t); setTimeout(() => t.remove(), 4500);
  }

  function alertBox(kind, text) { return el("div", { class: `alert ${kind}`, role: kind === "error" ? "alert" : "status", text }); }
  function setStatus(node, text, busy = false) {
    node.replaceChildren(...(busy ? [el("span", { class: "spinner", "aria-hidden": "true" })] : []), text,
      ...(busy ? [el("div", { class: "progress indeterminate", "aria-hidden": "true" }, el("i"))] : []));
  }
  const fmtBytes = (n) => n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(2)} MB`;
  const fmt = (v, d = 3) => (v === null || v === undefined) ? "n/a" : (typeof v === "number" ? v.toFixed(d) : String(v));

  function chrome() {
    const page = document.body.dataset.page;
    const toggle = el("button", { class: "nav-toggle", type: "button", "aria-expanded": "false", "aria-controls": "site-nav", "aria-label": "Toggle navigation" }, el("span"));
    const wrap = el("div", { class: "nav-wrap", id: "site-nav" }, [
      el("nav", { class: "nav", "aria-label": "Main" }, PAGES.map(([href, label]) =>
        el("a", { href, "aria-current": href === page ? "page" : false, text: label }))),
      el("a", { class: "header-cta", href: "upload.html", text: "Analyze Lesion →" }),
    ]);
    const header = el("header", { class: "site-header" }, el("div", { class: "container" }, [
      el("a", { class: "brand", href: "index.html", "aria-label": "MelanoLens AI home" }, []), toggle, wrap,
    ]));
    header.querySelector(".brand").innerHTML = `${LOGO}<span>Melano<em>Lens</em> AI</span>`;
    toggle.addEventListener("click", () => { const open = wrap.classList.toggle("open"); toggle.setAttribute("aria-expanded", String(open)); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && wrap.classList.contains("open")) { wrap.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); toggle.focus(); } });
    const onScroll = () => header.classList.toggle("scrolled", window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true }); onScroll();

    const footer = el("footer", { class: "site-footer" }, el("div", { class: "container" }, [
      el("div", { class: "foot-grid" }, [
        el("div", {}, [el("a", { class: "brand", href: "index.html" }), el("p", { text: "See the Edges. Understand the Boundaries. An educational computer vision project for experimental skin lesion boundary enhancement." })]),
        el("div", {}, [el("h4", { text: "Platform" }), ...PAGES.slice(1, 5).map(([href, label]) => el("a", { href, text: label }))]),
        el("div", {}, [el("h4", { text: "Project" }), el("a", { href: "about.html", text: "About" }), el("a", { href: "index.html#pipeline", text: "How it works" }),
          el("a", { href: "https://github.com/BushraNoor88/MelanoLens-AI", target: "_blank", rel: "noopener", text: "Source on GitHub" })]),
      ]),
      el("p", { class: "disclaimer" }, [el("strong", { text: "Medical disclaimer: " }), DISCLAIMER]),
      el("div", { class: "foot-bottom", text: "Runs locally; images are not sent to third-party services." }),
    ]));
    footer.querySelector(".brand").innerHTML = `${LOGO}<span>Melano<em>Lens</em> AI</span>`;
    document.body.prepend(el("a", { class: "skip-link", href: "#main", text: "Skip to content" }), header);
    document.body.append(footer);
    const main = document.getElementById("main"); if (main) main.classList.add("page-enter");
  }

  /** Download an image by URL (works for same-origin attachment endpoints). */
  function download(url) { const a = el("a", { href: url }); document.body.append(a); a.click(); a.remove(); }

  document.addEventListener("DOMContentLoaded", chrome);
  return { el, api, postJSON, getResultId, setResultId, clearResultId, getParams, setParams, imageUrl, downloadUrl,
           toast, alertBox, setStatus, fmtBytes, fmt, download, DISCLAIMER };
})();
