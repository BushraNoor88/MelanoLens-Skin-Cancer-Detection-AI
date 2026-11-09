/* MelanoLens AI - presentation-only behaviour: scroll reveal, button glow, hero before/after slider.
   Nothing here touches the API or processing logic. */
(() => {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  document.addEventListener("DOMContentLoaded", () => {
    // scroll reveal
    const items = document.querySelectorAll(".reveal");
    if (reduce || !("IntersectionObserver" in window)) items.forEach((n) => n.classList.add("in"));
    else {
      const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
      items.forEach((n) => io.observe(n));
    }

    // pointer-follow glow for button press
    document.addEventListener("pointerdown", (e) => {
      const b = e.target.closest(".btn"); if (!b) return;
      const r = b.getBoundingClientRect(); b.style.setProperty("--rx", `${e.clientX - r.left}px`); b.style.setProperty("--ry", `${e.clientY - r.top}px`);
    });

    // hero before/after slider (real pipeline output; keyboard + pointer accessible via range input)
    const lens = document.querySelector("[data-lens]");
    if (lens) {
      const range = lens.querySelector(".lens-range");
      const set = (v) => lens.style.setProperty("--split", `${v}%`);
      range.addEventListener("input", () => { set(range.value); stop(); });
      let t = null, dir = 1, v = 50;
      const stop = () => { if (t) { cancelAnimationFrame(t); t = null; } };
      if (!reduce) {                       // gentle auto-sweep until the visitor interacts
        const tick = () => { v += dir * 0.18; if (v > 78 || v < 22) dir *= -1; range.value = v; set(v); t = requestAnimationFrame(tick); };
        t = requestAnimationFrame(tick);
      }
      set(50);
    }
  });
})();
