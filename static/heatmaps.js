/** Pure canvas renderers for the point inspector — per-head q/k/v heatmaps
 *  and the ‖q‖-per-head bar strip. No app state: give them data, they draw. */

const HEAT_NEG = [57, 135, 229];    // --attn blue
const HEAT_POS = [217, 89, 38];     // --mlp orange
const HEAT_MID = [13, 13, 13];      // page background

/** Per-head heatmap: one row per head, one column per head_dim coordinate,
 *  symmetric normalization by the tensor's max |x| (blue − · orange +). */
export function drawHeatmap(canvas, heads) {
  const rows = heads.length, cols = heads[0].length;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  let amax = 1e-6;
  for (const row of heads) for (const v of row) amax = Math.max(amax, Math.abs(v));
  let o = 0;
  for (const row of heads) {
    for (const v of row) {
      const t = Math.max(-1, Math.min(1, v / amax));
      const c = t < 0 ? HEAT_NEG : HEAT_POS;
      const a = Math.abs(t);
      img.data[o++] = HEAT_MID[0] + (c[0] - HEAT_MID[0]) * a;
      img.data[o++] = HEAT_MID[1] + (c[1] - HEAT_MID[1]) * a;
      img.data[o++] = HEAT_MID[2] + (c[2] - HEAT_MID[2]) * a;
      img.data[o++] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** One thin bar per q head, height ∝ ‖q_h‖ — which heads are loud here. */
export function drawHeadNorms(el, heads) {
  el.innerHTML = '';
  const norms = heads.map(row => Math.hypot(...row));
  const max = Math.max(...norms, 1e-6);
  norms.forEach((n, i) => {
    const bar = document.createElement('div');
    bar.className = 'headNormBar';
    bar.style.height = `${Math.max((n / max) * 100, 4)}%`;
    bar.title = `head ${i} · ‖q‖ = ${n.toFixed(1)}`;
    el.appendChild(bar);
  });
}
