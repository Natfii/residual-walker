/** The recording overlay card: prompt + story line, step readout, logit-lens
 *  bars, patch echo, and shadow honesty — stamped onto every composited MP4
 *  frame so exports tell the whole story on their own.
 *
 *  Pure drawing: everything it shows arrives in the `view` argument —
 *  { promptText, genText, lensState, stepsTotal, patchInfo, honesty,
 *    touring, displayToken }. `honesty` is null when no walk has started. */

function trimLeft(ctx, text, maxW) {
  if (ctx.measureText(text).width <= maxW) return text;
  // jump-cut near the target length first — the char-by-char loop alone costs
  // hundreds of measureText calls per frame once generations get long
  let t = text;
  const avg = ctx.measureText(t).width / t.length;
  const keep = Math.min(t.length, Math.ceil((maxW / avg) * 1.4) + 2);
  t = t.slice(t.length - keep);
  while (t.length > 1 && ctx.measureText('…' + t).width > maxW) t = t.slice(1);
  return '…' + t;
}

export function drawRecordingOverlay(ctx, w, h, view) {
  const { promptText, genText, lensState, stepsTotal,
          patchInfo, honesty, touring, displayToken } = view;
  const s = h / 950;
  const pad = 14 * s, margin = 16 * s;
  const cardW = Math.min(480 * s, w - 2 * margin);
  const rowH = 24 * s;
  const lensRows = lensState ? lensState.rows.length : 0;
  const honestyH = honesty != null ? 18 * s : 0;
  const patchH = patchInfo?.active ? 18 * s : 0;
  const cardH = pad * 2 + 20 * s + (lensState ? 26 * s + lensRows * rowH : 0) + honestyH + patchH;
  const x = margin, y = h - margin - cardH;
  const KIND_COLORS = { embed: '#ffffff', attn: '#3987e5', mlp: '#d95926' };

  ctx.save();
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = '#1a1a19';
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, cardW, cardH, 10 * s);
  ctx.fill();
  ctx.stroke();
  ctx.globalAlpha = 1;

  const textX = x + pad, maxW = cardW - pad * 2;
  let ty = y + pad + 12 * s;

  /* story line: prompt in secondary ink, generated tokens bold white; the
     generation claims its width first, the prompt left-trims into what's left,
     so the newest words always stay on film */
  ctx.font = `600 ${13.5 * s}px system-ui, sans-serif`;
  const gen = trimLeft(ctx, genText, maxW);
  const genW = ctx.measureText(gen).width;
  ctx.font = `${13.5 * s}px system-ui, sans-serif`;
  const prRoom = maxW - genW - 4 * s;
  const pr = prRoom > 12 * s ? trimLeft(ctx, promptText, prRoom) : '';
  ctx.fillStyle = '#c3c2b7';
  ctx.fillText(pr, textX, ty);
  const prW = ctx.measureText(pr).width;
  ctx.fillStyle = '#ffffff';
  ctx.font = `600 ${13.5 * s}px system-ui, sans-serif`;
  ctx.fillText(gen, textX + prW, ty);

  if (lensState) {
    ty += 24 * s;
    const info = lensState.info;
    ctx.fillStyle = KIND_COLORS[info.kind];
    ctx.beginPath();
    ctx.arc(textX + 5 * s, ty - 4 * s, 5 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#c3c2b7';
    ctx.font = `${12 * s}px system-ui, sans-serif`;
    const where = info.kind === 'embed' ? 'embedding — the path begins'
      : `${info.kind === 'attn' ? 'attention' : 'MLP'} add · layer ${info.layer}`;
    ctx.fillText(`step ${lensState.k}/${stepsTotal} · ${where}` +
      (lensState.usingJ ? ' · J-lens' : ''), textX + 16 * s, ty);

    const chipW = 96 * s, pctW = 46 * s;
    const trackX = textX + chipW + 8 * s;
    const trackW = maxW - chipW - pctW - 16 * s;
    lensState.rows.forEach((r, i) => {
      const ry = ty + 10 * s + i * rowH;
      ctx.font = `${i === 0 ? '700 ' : ''}${11.5 * s}px ui-monospace, Consolas, monospace`;
      ctx.fillStyle = i === 0 ? '#ffffff' : '#c3c2b7';
      ctx.fillText(trimLeft(ctx, displayToken(r.t), chipW), textX, ry + 12 * s);
      ctx.fillStyle = '#0d0d0d';
      ctx.beginPath();
      ctx.roundRect(trackX, ry + 3 * s, trackW, 10 * s, 3 * s);
      ctx.fill();
      ctx.fillStyle = '#199e70';
      ctx.beginPath();
      ctx.roundRect(trackX, ry + 3 * s, Math.max(trackW * r.p, 2), 10 * s, 3 * s);
      ctx.fill();
      ctx.fillStyle = i === 0 ? '#c3c2b7' : '#898781';
      ctx.font = `${11 * s}px system-ui, sans-serif`;
      const pct = `${(r.p * 100).toFixed(1)}%`;
      ctx.fillText(pct, textX + maxW - ctx.measureText(pct).width, ry + 12 * s);
    });
    ty += 10 * s + lensRows * rowH;
  }

  if (patchInfo?.active) {
    ctx.fillStyle = '#9085e9';
    ctx.font = `${11 * s}px system-ui, sans-serif`;
    const parts = [];
    const shortLabel = t => t.length > 26 ? t.slice(0, 25) + '…' : t;
    if (patchInfo.add) parts.push(`+${shortLabel(patchInfo.add.trim())}`);
    if (patchInfo.remove) parts.push(`−${shortLabel(patchInfo.remove.trim())}`);
    if (patchInfo.source === 'phrase') parts.push('(phrase vibes)');
    const verb = patchInfo.mode === 'swap' ? '🔁 J-swap' : '💉 nudge';
    const range = patchInfo.sticky ? `${patchInfo.layer}→${patchInfo.layer_end}` : `${patchInfo.layer}`;
    ctx.fillText(`${verb} ${parts.join(' ')} @ layer ${range} ×${patchInfo.alpha}`, textX, ty + 14 * s);
    ty += 18 * s;
  }
  if (honesty != null) {
    ctx.fillStyle = '#898781';
    ctx.font = `${11 * s}px system-ui, sans-serif`;
    const mode = touring ? ' · touring the hidden dimensions' : ' · best angle';
    ctx.fillText(`shadow honesty ${(honesty * 100).toFixed(1)}%${mode}`, textX, ty + 14 * s);
  }
  ctx.restore();
}
