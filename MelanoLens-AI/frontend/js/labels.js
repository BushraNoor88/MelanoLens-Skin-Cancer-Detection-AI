/* Human-readable names for result image types returned by the API. */
const DETECTOR_NAMES = { canny: "Canny", sobel: "Sobel", prewitt: "Prewitt", laplacian: "Laplacian", log: "LoG" };
const BASE_LABELS = {
  original: "Original", preprocessed: "Preprocessed", enhanced: "Edge Booster output", gradient: "Gradient map",
  edges: "Selected edge map", contour_overlay: "Contour overlay", mask: "Candidate mask",
  reference_mask: "Reference mask", difference: "Difference (TP / FP / FN)",
  cnn_mask: "CNN predicted mask", cnn_region: "CNN lesion region", cnn_overlay: "CNN boundary overlay",
  cnn_probability: "CNN probability map", cnn_compare: "Classical (cyan) vs CNN (magenta) boundary",
};
function labelFor(name) {
  if (BASE_LABELS[name]) return BASE_LABELS[name];
  const m = name.match(/^(edges|mask)_(conv|enh)_([a-z]+)$/);
  if (m) return `${DETECTOR_NAMES[m[3]] || m[3]} ${m[1] === "edges" ? "edges" : "mask"} (${m[2] === "conv" ? "conventional" : "enhanced"})`;
  return name;
}
