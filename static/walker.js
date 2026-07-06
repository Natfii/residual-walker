/** Residual Walker frontend — scene, walk state machine, lenses, tour,
 *  patching UI, pause/scrub, and MP4 recording. Served with no-cache:
 *  this module and the server's WS protocol version together. */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

const COLORS = { attn: 0x3987e5, mlp: 0xd95926, embed: 0xffffff, ghost: 0x898781 };

/* ---------- scene setup ---------- */
const stage = document.getElementById('stage');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d0d);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 4000);
camera.position.set(70, 48, 95);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
stage.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotateSpeed = 0.6;

scene.add(new THREE.HemisphereLight(0x8899bb, 0x0d0d0d, 1.0));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.3);
keyLight.position.set(60, 90, 40);
scene.add(keyLight);

const grid = new THREE.GridHelper(280, 28, 0x2c2c2a, 0x1f1f1e);
grid.position.y = -58;
scene.add(grid);

/* shared line materials — step-kind colors ride in the vertex-color buffer */
const lineMats = {
  path:  new LineMaterial({ vertexColors: true, linewidth: 3 }),
  ghost: new LineMaterial({ color: COLORS.ghost, linewidth: 1.6, transparent: true, opacity: 0.28 }),
};
const sphereGeo = new THREE.SphereGeometry(0.62, 20, 14);
const embedGeo = new THREE.SphereGeometry(0.95, 22, 16);
const sphereMats = {
  attn:  new THREE.MeshStandardMaterial({ color: COLORS.attn, emissive: COLORS.attn, emissiveIntensity: 0.4, roughness: 0.35 }),
  mlp:   new THREE.MeshStandardMaterial({ color: COLORS.mlp, emissive: COLORS.mlp, emissiveIntensity: 0.4, roughness: 0.35 }),
  embed: new THREE.MeshStandardMaterial({ color: COLORS.embed, emissive: 0x888888, emissiveIntensity: 0.35, roughness: 0.3 }),
  ghost: new THREE.MeshStandardMaterial({ color: COLORS.ghost, transparent: true, opacity: 0.4, roughness: 0.6 }),
};
const patchGeo = new THREE.OctahedronGeometry(1.15);
const patchMat = new THREE.MeshStandardMaterial({
  color: 0x9085e9, emissive: 0x9085e9, emissiveIntensity: 0.5, roughness: 0.3,
});
const headMarker = new THREE.Mesh(
  new THREE.SphereGeometry(1.05, 24, 18),
  new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.65, roughness: 0.2 })
);
headMarker.visible = false;
scene.add(headMarker);

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  Object.values(lineMats).forEach(m => m.resolution.set(w, h));
}
new ResizeObserver(resize).observe(stage);
resize();

/* ---------- DOM handles ---------- */
const $ = id => document.getElementById(id);
const guessEl = $('guess'), statusEl = $('status'), storyEl = $('story');
const lensEl = $('lens'), stepReadout = $('stepReadout');
$('temp').oninput = e => $('tempVal').textContent = Number(e.target.value).toFixed(2);
$('speed').oninput = e => $('speedVal').textContent = `${e.target.value} ms`;
/* strength: slider for coarse sweeps, number box for fine-tuning — the
   number box is the source of truth (type 0.055 if the window demands it) */
$('patchAlpha').oninput = e => { $('patchAlphaNum').value = e.target.value; };
$('patchAlphaNum').oninput = e => { $('patchAlpha').value = e.target.value; };
$('orbit').onchange = e => controls.autoRotate = e.target.checked;
controls.autoRotate = $('orbit').checked;

/* ---------- lens mode (logit vs Jacobian) ---------- */
let lensMode = 'logit';
let jlensAvailable = false;
function setLensMode(m) {
  lensMode = m;
  $('lensModeLogit').classList.toggle('on', m === 'logit');
  $('lensModeJ').classList.toggle('on', m === 'jlens');
  $('lensTitle').textContent = m === 'jlens'
    ? 'J-lens — “what it’s disposed to say”'
    : 'Logit lens — “if we fired right now”';
  if (current && lensState) updateLens(lensState.k);   // re-render the live step
}
$('lensModeLogit').onclick = () => setLensMode('logit');
$('lensModeJ').onclick = () => setLensMode('jlens');
function defaultPatchAlpha() {
  // sticky compounds across every layer in its range, so it wants a feather
  // (the steering window sits near 0.05-0.1); an exact paper swap is alpha = 1;
  // the one-shot nudge pushes harder
  const alpha = $('patchSticky').checked ? '0.1'
    : $('patchMode').value === 'swap' ? '1.0' : '2.5';
  $('patchAlpha').value = alpha;
  $('patchAlphaNum').value = alpha;
}
$('patchMode').onchange = defaultPatchAlpha;
$('patchSticky').onchange = () => {
  $('patchEndField').style.display = $('patchSticky').checked ? 'block' : 'none';
  defaultPatchAlpha();
};

/* configure the UI from the server before any walk */
fetch('/api/info').then(r => r.json()).then(info => {
  $('modelChip').textContent =
    `${info.model} · ${info.n_layers} layers · ${info.hidden}-dim stream · ${info.device}`;
  jlensAvailable = !!(info.jlens && info.jlens.available);
  if (jlensAvailable) {
    $('lensSwitch').style.display = 'flex';
    $('patchSwapOpt').disabled = false;
  }
  // sticky default range ends before the motor zone (final quarter of layers)
  $('patchLayerEnd').value = Math.max(0, info.n_layers - 1 - Math.floor(info.n_layers / 4));
}).catch(() => {});

/** Make control characters in a token visible without breaking layout. */
function displayToken(t) {
  return t.replace(/\n/g, '⏎').replace(/^ /, '␣');
}

/* ---------- 2048-D → 3D projection (the hand-shadow machinery) ----------
 * The server sends each path point as its top-12 PCA coordinates. The "best
 * angle" shadow is simply coordinates 0-2. The grand tour spins an exactly
 * orthonormal 3-frame through the 12-D space via Givens rotations whose speeds
 * have irrational ratios, so the frame never repeats or degenerates. */
let tourInfo = null;            // {dims, var_ratios} from the walk's meta packet
let tourT = 0;                  // tour clock (0 = home / best angle)
let tourLambda = 0;             // blend: 0 = home shadow, 1 = touring frame
let basis = null;               // current 3×dims frame, rebuilt when touring
const TOUR_PLANES = [[0,3],[1,4],[2,5],[0,6],[1,7],[2,8],[0,9],[1,10],[2,11]];
const TOUR_SPEEDS = [1, 1/Math.sqrt(2), 1/Math.sqrt(3), 1/Math.sqrt(5),
                     1/Math.sqrt(7), 1/Math.sqrt(11), 1/Math.sqrt(13),
                     1/Math.sqrt(17), 1/Math.sqrt(19)];

function computeBasis(t, dims) {
  const B = [new Float64Array(dims), new Float64Array(dims), new Float64Array(dims)];
  B[0][0] = 1; B[1][1] = 1; B[2][2] = 1;
  TOUR_PLANES.forEach(([p, q], k) => {
    if (q >= dims) return;
    const a = t * 0.3 * TOUR_SPEEDS[k];
    const c = Math.cos(a), s = Math.sin(a);
    for (const v of B) {
      const x = v[p], y = v[q];
      v[p] = c * x - s * y;
      v[q] = s * x + c * y;
    }
  });
  return B;
}

/** Project one 12-D coordinate row into the scene, blending home ↔ tour frame. */
function projectPoint(c, out) {
  let x = 0, y = 0, z = 0;
  if (tourLambda > 0 && basis) {
    for (let i = 0; i < c.length; i++) {
      x += basis[0][i] * c[i];
      y += basis[1][i] * c[i];
      z += basis[2][i] * c[i];
    }
  }
  const l = tourLambda;
  return out.set(c[0] * (1 - l) + x * l, c[1] * (1 - l) + y * l, c[2] * (1 - l) + z * l);
}

/** Fraction of the full 2048-D variance visible through the current 3-frame.
 *  PCA components are uncorrelated, so variance along a unit direction u in
 *  component space is Σ u_i² · var_i — the ratios sum directly. */
function shadowHonesty() {
  if (!tourInfo) return 0;
  const r = tourInfo.var_ratios;
  const home = r[0] + r[1] + r[2];
  if (tourLambda <= 0 || !basis) return home;
  let tour = 0;
  for (const row of basis) {
    for (let i = 0; i < r.length; i++) tour += row[i] * row[i] * r[i];
  }
  return home * (1 - tourLambda) + tour * tourLambda;
}

/** Billboard text label used for ghost path endpoints. */
function makeLabelSprite(text) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.font = '600 34px system-ui, sans-serif';
  const w = Math.ceil(ctx.measureText(text).width) + 24;
  canvas.width = w; canvas.height = 52;
  const ctx2 = canvas.getContext('2d');
  ctx2.font = '600 34px system-ui, sans-serif';
  ctx2.fillStyle = '#c3c2b7';
  ctx2.fillText(text, 12, 38);
  const tex = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, opacity: 0.85 }));
  sprite.scale.set(w * 0.055, 52 * 0.055, 1);
  return sprite;
}

/* ---------- one token's path ----------
 * The whole path is ONE Line2: reveal advances geometry.instanceCount, the
 * grand tour rewrites the existing position buffer in place, and step-kind
 * colors are baked once into the vertex-color buffer. (A per-segment Line2
 * design froze the tab: touring re-allocated GPU buffers for every segment
 * of every path on every frame.) */
class TokenPath {
  constructor(packet, steps) {
    this.coords = packet.coords;   // rows of top-12 PCA coordinates
    this.positions = this.coords.map(() => new THREE.Vector3());
    this.lens = packet.lens;
    this.jlens = packet.jlens || null;   // per-point rows, null where no transport
    this.chosen = packet.chosen;
    this.chosenProb = packet.chosen_prob;
    this.steps = steps;
    this.revealed = 0;             // number of points currently visible
    this.group = new THREE.Group();
    this.spheres = [];
    this.endSphere = null;
    this.labelSprite = null;

    const n = this.coords.length;
    const geo = new LineGeometry();
    geo.setPositions(new Array(n * 3).fill(0));
    geo.setColors(new Array(n * 3).fill(1));
    const colorArr = geo.attributes.instanceColorStart.data.array;
    const c = new THREE.Color();
    for (let seg = 0; seg < n - 1; seg++) {
      c.set(COLORS[this.steps[seg + 1].kind]);  // solid, colored by destination step
      colorArr.set([c.r, c.g, c.b, c.r, c.g, c.b], seg * 6);
    }
    geo.instanceCount = 0;
    this.line = new Line2(geo, lineMats.path);
    this.line.frustumCulled = false;
    this.group.add(this.line);

    this.refreshPositions();
    scene.add(this.group);
  }

  /** Re-project every point through the current frame and move all geometry.
   *  Called on reveal and every frame the grand tour turns — so it must not
   *  allocate: it rewrites the line's existing interleaved buffer. */
  refreshPositions() {
    const n = this.coords.length;
    for (let i = 0; i < n; i++) {
      projectPoint(this.coords[i], this.positions[i]);
    }
    const buf = this.line.geometry.attributes.instanceStart.data;
    const arr = buf.array;
    for (let seg = 0; seg < n - 1; seg++) {
      const a = this.positions[seg], b = this.positions[seg + 1];
      const o = seg * 6;
      arr[o] = a.x; arr[o + 1] = a.y; arr[o + 2] = a.z;
      arr[o + 3] = b.x; arr[o + 4] = b.y; arr[o + 5] = b.z;
    }
    buf.needsUpdate = true;
    for (const s of this.spheres) s.position.copy(this.positions[s.userData.idx]);
    if (this.endSphere) {
      const end = this.positions[this.positions.length - 1];
      this.endSphere.position.copy(end);
      this.labelSprite.position.copy(end).add(new THREE.Vector3(0, 2.4, 0));
    }
  }

  revealNext() {
    const k = this.revealed;
    if (k >= this.positions.length) return false;
    const kind = this.steps[k].kind;
    if (k === 0) {
      const start = new THREE.Mesh(embedGeo, sphereMats.embed);
      start.userData.idx = 0;
      start.position.copy(this.positions[0]);
      this.group.add(start);
      this.spheres.push(start);
    } else {
      this.line.geometry.instanceCount = k;   // k revealed segments
      const isNudge = patchInfo?.active && (k === patchInfo.step ||
        (patchInfo.sticky && k > patchInfo.step && k <= 2 * patchInfo.layer_end + 2 && kind === 'mlp'));
      const dot = isNudge
        ? new THREE.Mesh(patchGeo, patchMat)
        : new THREE.Mesh(sphereGeo, sphereMats[kind]);
      dot.userData.idx = k;
      dot.position.copy(this.positions[k]);
      this.group.add(dot);
      this.spheres.push(dot);
      if (isNudge && k === patchInfo.step) fireRing(this.positions[k], 0x9085e9, 6);
    }
    this.revealed++;
    return true;
  }

  get head() { return this.positions[Math.max(0, this.revealed - 1)]; }
  get done() { return this.revealed >= this.positions.length; }

  /** Collapse a finished path into a faint gray trail with a token label. */
  ghost() {
    if (this.endSphere) return;
    this.line.material = lineMats.ghost;
    for (const s of this.spheres) { this.group.remove(s); }
    this.spheres.length = 0;
    const end = new THREE.Mesh(sphereGeo, sphereMats.ghost);
    end.position.copy(this.positions[this.positions.length - 1]);
    this.group.add(end);
    this.endSphere = end;
    const label = makeLabelSprite(displayToken(this.chosen));
    label.position.copy(end.position).add(new THREE.Vector3(0, 2.4, 0));
    this.group.add(label);
    this.labelSprite = label;
  }

  dispose() {
    scene.remove(this.group);
    this.group.traverse(o => {
      if (o.geometry && o.geometry !== sphereGeo && o.geometry !== embedGeo && o.geometry !== patchGeo) {
        o.geometry.dispose();
      }
    });
  }
}

/* ---------- fire effect ---------- */
const rings = [];
function fireRing(position, color = 0xffffff, growth = 14) {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.9, 1.18, 48),
    new THREE.MeshBasicMaterial({ color, transparent: true, side: THREE.DoubleSide })
  );
  ring.position.copy(position);
  scene.add(ring);
  rings.push({ mesh: ring, t: 0, growth });
}

/* ---------- walk state machine ---------- */
let ws = null;
let steps = [];                    // step kinds from meta
let queue = [];                    // path packets waiting to animate
let current = null;                // TokenPath being drawn
let paths = [];                    // all TokenPaths this walk
let doneInfo = null;               // set when server sends "done"
let walking = false;
let lastStepAt = 0;
let firePauseUntil = 0;
let paused = false;
let viewStep = -1;                 // lens/scrub cursor within the current path
let promptText = '';               // mirrored for the recording overlay
let genText = '';
let lensState = null;              // {k, info, rows} at the current step
let patchInfo = null;              // activation-patch echo from meta

function setStatus(t) { statusEl.textContent = t; }

/* ---------- MP4 recording ----------
 * The film is composited on a hidden canvas: each 3D frame is copied over and
 * an overlay card (prompt + story, step readout, logit-lens bars, shadow
 * honesty) is drawn on top, so exports carry the full narrative. */
let recorder = null;
let recChunks = [];
let recCanvas = null;
let recCtx = null;

function startRecording() {
  const src = renderer.domElement;
  recCanvas = document.createElement('canvas');
  recCanvas.width = src.width;
  recCanvas.height = src.height;
  recCtx = recCanvas.getContext('2d');
  const stream = recCanvas.captureStream(60);
  const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
    ? 'video/webm;codecs=vp9' : 'video/webm';
  recChunks = [];
  recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 });
  recorder.ondataavailable = e => { if (e.data.size) recChunks.push(e.data); };
  recorder.start();
}

/** Copy the freshly-rendered 3D frame and stamp the overlay card onto it.
 *  Must run in the same rAF tick as renderer.render() so the WebGL buffer
 *  is still valid to read. */
function compositeFrame() {
  if (!recCtx || !recorder || recorder.state !== 'recording') return;
  recCtx.drawImage(renderer.domElement, 0, 0, recCanvas.width, recCanvas.height);
  drawRecordingOverlay(recCtx, recCanvas.width, recCanvas.height);
}

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

function drawRecordingOverlay(ctx, w, h) {
  const s = h / 950;
  const pad = 14 * s, margin = 16 * s;
  const cardW = Math.min(480 * s, w - 2 * margin);
  const rowH = 24 * s;
  const lensRows = lensState ? lensState.rows.length : 0;
  const honestyH = tourInfo ? 18 * s : 0;
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
    ctx.fillText(`step ${lensState.k}/${steps.length - 1} · ${where}` +
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
  if (tourInfo) {
    ctx.fillStyle = '#898781';
    ctx.font = `${11 * s}px system-ui, sans-serif`;
    const mode = tourLambda > 0.02 ? ' · touring the hidden dimensions' : ' · best angle';
    ctx.fillText(`shadow honesty ${(shadowHonesty() * 100).toFixed(1)}%${mode}`, textX, ty + 14 * s);
  }
  ctx.restore();
}

async function finishRecording() {
  if (!recorder) return;
  const rec = recorder;
  recorder = null;
  const stopped = new Promise(res => { rec.onstop = res; });
  rec.stop();
  await stopped;
  const exportEl = document.getElementById('export');
  exportEl.textContent = 'encoding MP4 on the GPU…';
  try {
    const resp = await fetch('/export', {
      method: 'POST',
      body: new Blob(recChunks, { type: 'video/webm' }),
    });
    const info = await resp.json();
    if (info.url) {
      exportEl.innerHTML = '';
      const a = document.createElement('a');
      a.href = info.url;
      a.download = info.name;
      a.textContent = `⬇ ${info.name}`;
      exportEl.appendChild(a);
    } else {
      exportEl.textContent = `export failed: ${info.error}`;
    }
  } catch (err) {
    exportEl.textContent = `export failed: ${err.message}`;
  }
}

function resetWalk() {
  for (const p of paths) p.dispose();
  paths = []; queue = []; current = null; doneInfo = null;
  for (const r of rings) scene.remove(r.mesh);
  rings.length = 0;
  headMarker.visible = false;
  guessEl.style.display = 'none';
  guessEl.classList.remove('fired');
  storyEl.innerHTML = '';
  $('promptTokens').textContent = '';
  $('promptDetails').style.display = 'none';
  $('promptDetails').open = false;
  lensEl.innerHTML = '';
  stepReadout.innerHTML = '';
  $('export').textContent = '';
  promptText = '';
  genText = '';
  lensState = null;
  patchInfo = null;
}

function startWalk() {
  if ($('patchOn').checked && $('patchMode').value === 'swap'
      && (!$('patchAdd').value.trim() || !$('patchRemove').value.trim())) {
    setStatus('J-swap trades two concepts — fill in both add and remove');
    return;
  }
  resetWalk();
  if ($('record').checked) startRecording();
  const prompt = $('prompt').value;
  promptText = prompt;
  const promptSpan = document.createElement('span');
  promptSpan.className = 'prompt';
  promptSpan.textContent = prompt;
  storyEl.appendChild(promptSpan);

  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    ws.send(JSON.stringify({
      prompt,
      temperature: Number($('temp').value),
      max_new_tokens: Number($('maxTokens').value),
      patch: $('patchOn').checked ? {
        add: $('patchAdd').value.trim() || null,
        remove: $('patchRemove').value.trim() || null,
        mode: $('patchMode').value,
        sticky: $('patchSticky').checked,
        layer: Number($('patchLayer').value),
        layer_end: Number($('patchLayerEnd').value),
        alpha: Number($('patchAlphaNum').value) || 0,
      } : null,
    }));
    setStatus('running prompt through the model…');
  };
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'meta') {
      steps = msg.steps;
      tourInfo = msg.tour;
      patchInfo = msg.patch?.active ? msg.patch : null;
      document.getElementById('dimsLabel').textContent = msg.hidden;
      document.getElementById('shadowMeter').style.display = 'block';
      $('modelChip').textContent = `${msg.model} · ${msg.n_layers} layers · ${msg.hidden}-dim stream · ${msg.device}`;
      const pt = $('promptTokens');
      pt.innerHTML = '';
      for (const t of msg.prompt_tokens) {
        const c = document.createElement('code');
        c.textContent = displayToken(t);
        pt.appendChild(c);
      }
      $('promptSummary').textContent = `prompt tokens (${msg.prompt_tokens.length})`;
      $('promptDetails').style.display = 'block';
    } else if (msg.type === 'path') {
      queue.push(msg);
    } else if (msg.type === 'done') {
      doneInfo = msg;
    } else if (msg.type === 'error') {
      setStatus(`error: ${msg.message}`);
      stopWalk(false);
    }
  };
  ws.onclose = () => { ws = null; };
  walking = true;
  setPaused(false);
  viewStep = -1;
  $('walkBtn').disabled = true;
  $('stopBtn').disabled = false;
  $('pauseBtn').disabled = false;
}

function stopWalk(flush = true) {
  if (ws) { ws.close(); ws = null; }
  if (flush && current) {
    while (current.revealNext()) {}   // snap the in-flight path to complete
    finishCurrent(false);
  }
  queue = [];
  walking = false;
  finishRecording();
  setPaused(false);
  $('walkBtn').disabled = false;
  $('stopBtn').disabled = true;
  $('pauseBtn').disabled = true;
  $('stepBackBtn').disabled = true;
  $('stepFwdBtn').disabled = true;
}

$('walkBtn').onclick = startWalk;
$('stopBtn').onclick = () => { stopWalk(); setStatus('stopped'); };

/* ---------- pause & step-by-step scrubbing ----------
 * Pausing freezes only the animation: the server keeps generating and the
 * queue keeps caching finished paths, and every step's lens data already
 * rides in its packet — so scrubbing is instant, nothing is recomputed. */
function setPaused(p) {
  paused = p;
  $('pauseBtn').textContent = p ? '▶' : '⏸';
  $('stepBackBtn').disabled = !p;
  $('stepFwdBtn').disabled = !p;
  if (!p) lastStepAt = performance.now();   // resume without a catch-up burst
}

/** One TokenPath begins: ghost the others, reset the scrub cursor. */
function startNextPath() {
  if (!queue.length) return false;
  for (const p of paths) p.ghost();
  current = new TokenPath(queue.shift(), steps);
  setStatus(`walking token ${paths.length + 1}…`);
  lastStepAt = 0;
  viewStep = -1;
  return true;
}

function stepForward() {
  if (!current) {
    if (startNextPath() && current.revealNext()) { viewStep = 0; updateLens(0); }
    return;
  }
  if (viewStep < current.revealed - 1) {          // walk the cursor back up first
    viewStep++;
    updateLens(viewStep);
  } else if (!current.done) {                     // then reveal fresh steps
    current.revealNext();
    viewStep = current.revealed - 1;
    updateLens(viewStep);
  } else {                                        // then fire and begin the next
    finishCurrent(true);
    if (startNextPath() && current.revealNext()) { viewStep = 0; updateLens(0); }
  }
}

function stepBack() {
  if (!current || viewStep <= 0) return;
  viewStep--;
  updateLens(viewStep);
}

$('pauseBtn').onclick = () => setPaused(!paused);
$('stepFwdBtn').onclick = stepForward;
$('stepBackBtn').onclick = stepBack;
window.addEventListener('keydown', e => {
  if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
  if (e.code === 'Space' && !$('pauseBtn').disabled) { e.preventDefault(); setPaused(!paused); }
  else if (paused && e.key === 'ArrowRight') { e.preventDefault(); stepForward(); }
  else if (paused && e.key === 'ArrowLeft') { e.preventDefault(); stepBack(); }
});

/* ---------- per-step UI ---------- */
function updateLens(k) {
  const info = steps[k];
  const dotColor = info.kind === 'embed' ? 'var(--embed)' : info.kind === 'attn' ? 'var(--attn)' : 'var(--mlp)';
  let where = info.kind === 'embed' ? 'embedding — the path begins' : `${info.kind === 'attn' ? 'attention' : 'MLP'} add · layer ${info.layer}`;
  if (patchInfo?.active && k === patchInfo.step) {
    where += patchInfo.mode === 'swap' ? ' · 🔁 swapped!' : ' · 💉 nudged!';
  } else if (patchInfo?.active && patchInfo.sticky && k > patchInfo.step
             && k <= 2 * patchInfo.layer_end + 2 && info.kind === 'mlp') {
    where += ' · 💉';
  }

  let rows = current.lens[k];
  let usingJ = false;
  if (lensMode === 'jlens' && current.jlens && current.jlens[k]) {
    rows = current.jlens[k];
    usingJ = true;
  }
  if (lensMode === 'jlens' && !usingJ) where += ' · no J transport — logit shown';
  else if (usingJ) where += ' · J-lens';

  stepReadout.innerHTML = '';
  const dot = document.createElement('span');
  dot.className = 'dot'; dot.style.background = dotColor;
  const txt = document.createElement('span');
  txt.textContent = `step ${k}/${steps.length - 1} · ${where}`;
  stepReadout.append(dot, txt);

  lensState = { k, info, rows, usingJ };
  lensEl.innerHTML = '';
  rows.forEach((r, i) => {
    const row = document.createElement('div');
    row.className = 'lensRow' + (i === 0 ? ' top' : '');
    const tok = document.createElement('span');
    tok.className = 'tok'; tok.textContent = displayToken(r.t);
    const codepoints = [...r.t].map(ch =>
      'U+' + ch.codePointAt(0).toString(16).toUpperCase().padStart(4, '0')).join(' ');
    tok.title = `token ${r.i ?? '?'} · ${codepoints} · ${(r.p * 100).toFixed(2)}%`;
    const track = document.createElement('div'); track.className = 'track';
    const fill = document.createElement('div'); fill.className = 'fill';
    track.appendChild(fill);
    const pct = document.createElement('span');
    pct.className = 'pct'; pct.textContent = `${(r.p * 100).toFixed(1)}%`;
    row.append(tok, track, pct);
    lensEl.appendChild(row);
    requestAnimationFrame(() => { fill.style.width = `${Math.max(r.p * 100, 1.5)}%`; });
  });

  guessEl.style.display = 'block';
  guessEl.classList.remove('fired');
  guessEl.innerHTML = '';
  guessEl.append((usingJ ? 'J· ' : '') + displayToken(rows[0].t));
  const prob = document.createElement('span');
  prob.className = 'prob'; prob.textContent = `${(rows[0].p * 100).toFixed(0)}%`;
  guessEl.appendChild(prob);
}

function finishCurrent(withEffects = true) {
  if (!current) return;
  const endPoint = current.positions[current.positions.length - 1];
  if (withEffects) {
    fireRing(endPoint);
    guessEl.classList.add('fired');
    guessEl.innerHTML = '';
    guessEl.append(displayToken(current.chosen));
    const prob = document.createElement('span');
    prob.className = 'prob';
    prob.textContent = `fired · ${(current.chosenProb * 100).toFixed(0)}%`;
    guessEl.appendChild(prob);
  }
  const span = document.createElement('span');
  span.className = 'gen flash';
  span.textContent = current.chosen;
  storyEl.appendChild(span);
  storyEl.scrollTop = storyEl.scrollHeight;   // keep the newest words in view
  genText += current.chosen;
  paths.push(current);
  current = null;
  firePauseUntil = performance.now() + (withEffects ? 700 : 0);
}

/* ---------- camera auto-framing ---------- */
const camTarget = new THREE.Vector3();
let userAdjusted = false;
controls.addEventListener('start', () => { userAdjusted = true; });

function updateCameraFrame() {
  const box = new THREE.Box3();
  let any = false;
  if (current && current.revealed > 0) {
    for (let i = 0; i < current.revealed; i++) box.expandByPoint(current.positions[i]);
    any = true;
  }
  for (const p of paths) { box.expandByPoint(p.positions[p.positions.length - 1]); any = true; }
  if (!any) return;
  box.getCenter(camTarget);
  controls.target.lerp(camTarget, 0.04);
  if (!userAdjusted) {
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 25);
    const wanted = radius * 2.1 + 25;
    const dir = camera.position.clone().sub(controls.target);
    const dist = dir.length();
    camera.position.copy(controls.target).add(dir.multiplyScalar(1 + ((wanted - dist) / dist) * 0.04));
  }
}

/* ---------- grand tour driver ---------- */
let lastFrameAt = 0;
let prevLambda = 0;

function updateTour(now) {
  const dt = Math.min((now - lastFrameAt) / 1000 || 0, 0.1);
  lastFrameAt = now;
  if (!tourInfo) return;

  if ($('tour').checked) {
    tourT += dt;
    tourLambda = Math.min(1, tourLambda + dt * 1.2);
  } else {
    tourLambda = Math.max(0, tourLambda - dt * 0.8);
    if (tourLambda === 0) tourT = 0;   // next tour departs fresh from home
  }

  const moving = tourLambda > 0 || prevLambda > 0;
  if (moving) {
    basis = computeBasis(tourT, tourInfo.dims);
    if (current) current.refreshPositions();
    for (const p of paths) p.refreshPositions();
  }
  prevLambda = tourLambda;

  const h = shadowHonesty();
  document.getElementById('honestyVal').textContent = `${(h * 100).toFixed(1)}%`;
  document.getElementById('honestyFill').style.width = `${h * 100}%`;
  document.getElementById('honestyMode').textContent =
    tourLambda <= 0.02 ? 'best angle (top-3 PCA)'
    : $('tour').checked ? 'touring the hidden dimensions…'
    : 'returning to the best angle…';
}

/* ---------- main loop ---------- */
const clock = new THREE.Clock();
function animate(now) {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();
  updateTour(now);

  if (walking || current || queue.length) {
    if (!paused && !current && queue.length && now >= firePauseUntil) startNextPath();
    if (!paused && current) {
      const stepMs = Number($('speed').value);
      if (now - lastStepAt >= stepMs) {
        lastStepAt = now;
        if (current.revealNext()) {
          viewStep = current.revealed - 1;
          updateLens(viewStep);
        }
        if (current.done) finishCurrent(true);
      }
    }
    if (!current && !queue.length && doneInfo && walking) {
      walking = false;
      setStatus(`done — ${paths.length} tokens (full text in the story box)`);
      setTimeout(finishRecording, 900);   // let the last fire-ring land on film
      $('walkBtn').disabled = false;
      $('stopBtn').disabled = true;
      $('pauseBtn').disabled = true;
      $('stepBackBtn').disabled = true;
      $('stepFwdBtn').disabled = true;
      if (ws) { ws.close(); ws = null; }
    }
  }

  /* head marker + floating guess label follow the newest point */
  const active = current;
  if (active && active.revealed > 0) {
    headMarker.visible = true;
    const cursor = Math.max(0, Math.min(viewStep, active.revealed - 1));
    headMarker.position.copy(active.positions[cursor]);
    const pulse = 1 + 0.16 * Math.sin(t * 6);
    headMarker.scale.setScalar(pulse);
  }
  if (headMarker.visible) {
    const screen = headMarker.position.clone().project(camera);
    if (screen.z < 1) {
      guessEl.style.left = `${(screen.x * 0.5 + 0.5) * stage.clientWidth}px`;
      guessEl.style.top = `${(-screen.y * 0.5 + 0.5) * stage.clientHeight}px`;
    }
  }

  /* expanding fire rings */
  for (let i = rings.length - 1; i >= 0; i--) {
    const r = rings[i];
    r.t += 0.024;
    r.mesh.scale.setScalar(1 + r.t * r.growth);
    r.mesh.material.opacity = Math.max(0, 1 - r.t * 1.4);
    r.mesh.quaternion.copy(camera.quaternion);
    if (r.mesh.material.opacity <= 0) { scene.remove(r.mesh); rings.splice(i, 1); }
  }

  updateCameraFrame();
  controls.update();
  renderer.render(scene, camera);
  compositeFrame();
}
requestAnimationFrame(animate);

/* ?autowalk=1 starts a walk on page load — useful for shareable/demo links */
if (new URLSearchParams(location.search).has('autowalk')) startWalk();
