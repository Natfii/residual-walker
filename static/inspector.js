/** The point-inspector panel: per-head q/k/v for the selected attention stop.
 *
 *  The lens answers "what would fire here"; this panel answers "what is on
 *  the k/v board here": heatmaps fetched on demand from /api/qkv (one
 *  forward's snapshot covers every position), plus the exact attention
 *  weights that already ride in each path packet.
 *
 *  The host supplies displayToken, tokenAtPos, and a view() accessor read on
 *  every update: view() → { steps, promptLen, headsInfo, deferFetch }.
 *  deferFetch is true while the walk animates unpaused — fetching then is
 *  churn, so the panel waits and the host re-updates on pause. */

import { drawHeatmap, drawHeadNorms } from './heatmaps.js';

const $ = id => document.getElementById(id);

const HINTS = {
  none: 'click a point on the path (or scrub with ←/→) to inspect it',
  embed: "embedding point — no attention has happened yet; pick a blue point",
  mlp: 'MLP point — no q/k/v here; pick a blue attention point',
};

export function initInspector({ displayToken, tokenAtPos, view }) {
  let target = null;               // "pos:layer" the panel currently wants
  let fetchTimer = null;           // debounce timer for /api/qkv
  const cache = new Map();         // "pos:layer" → /api/qkv payload

  function reset() {
    cache.clear();
    target = null;
    clearTimeout(fetchTimer);
    $('inspHint').textContent = HINTS.none;
    $('inspHint').style.display = 'block';
    $('inspBody').style.display = 'none';
  }

  function setStatus(text) {
    $('inspStatus').textContent = text;
    $('inspStatus').style.display = text ? 'block' : 'none';
  }

  function renderQkv(data) {
    setStatus('');
    drawHeatmap($('qCanvas'), data.q);
    drawHeatmap($('kCanvas'), data.k);
    drawHeatmap($('vCanvas'), data.v);
    drawHeadNorms($('headNorms'), data.q);
  }

  /** Refresh the panel for point k of path p (a TokenPath or null). */
  function update(k, p) {
    const { steps, promptLen, headsInfo, deferFetch } = view();
    const kind = steps[k]?.kind;
    if (!p || kind !== 'attn') {
      $('inspHint').textContent = HINTS[kind === 'embed' || kind === 'mlp' ? kind : 'none'];
      $('inspHint').style.display = 'block';
      $('inspBody').style.display = 'none';
      return;
    }
    const layer = steps[k].layer;
    const pos = promptLen - 1 + p.index;
    $('inspHint').style.display = 'none';
    $('inspBody').style.display = 'block';
    $('inspWhere').textContent =
      `layer ${layer} · query from “${displayToken(tokenAtPos(pos))}” (pos ${pos})`;
    if (headsInfo) {
      const group = headsInfo.n_heads / headsInfo.n_kv_heads;
      $('inspGqa').textContent =
        `${headsInfo.n_heads} q heads → ${headsInfo.n_kv_heads} kv heads`
        + (group > 1 ? ` · groups of ${group} share k/v` : '');
    }

    /* "where this query looked" — rides in the packet, no fetch needed */
    const listEl = $('attnList');
    listEl.innerHTML = '';
    const sources = p.attn?.[k] || [];
    if (!sources.length) {
      const note = document.createElement('div');
      note.className = 'inspNote';
      note.textContent = p.attn
        ? 'all attention mass on the sink / itself at this layer'
        : 'no attention data from this server';
      listEl.appendChild(note);
    }
    const wmax = sources.length ? sources[0][1] || 1 : 1;
    for (const [j, w] of sources) {
      const row = document.createElement('div');
      row.className = 'lensRow';
      const tok = document.createElement('span');
      tok.className = 'tok';
      tok.textContent = displayToken(tokenAtPos(j));
      tok.title = `position ${j} · weight ${w} (max over heads)`;
      const track = document.createElement('div'); track.className = 'track';
      const fill = document.createElement('div'); fill.className = 'fill';
      fill.style.width = `${Math.max((w / wmax) * 100, 2)}%`;
      track.appendChild(fill);
      const pct = document.createElement('span');
      pct.className = 'pct'; pct.textContent = w.toFixed(2);
      row.append(tok, track, pct);
      listEl.appendChild(row);
    }

    /* q/k/v heatmaps — on-demand, cached, debounced; pointless mid-animation */
    const key = `${pos}:${layer}`;
    target = key;
    const cached = cache.get(key);
    if (cached) { renderQkv(cached); return; }
    if (deferFetch) { setStatus('pause (or click a point) to fetch q/k/v'); return; }
    setStatus('fetching q/k/v…');
    clearTimeout(fetchTimer);
    fetchTimer = setTimeout(async () => {
      try {
        const resp = await fetch(`/api/qkv?pos=${pos}&layer=${layer}`);
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        cache.set(key, data);
        if (target === key) renderQkv(data);
      } catch (err) {
        if (target === key) setStatus(`q/k/v unavailable: ${err.message}`);
      }
    }, 120);
  }

  return { update, reset };
}
