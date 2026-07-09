/** The k/v stream layer: prompt-token ghost anchor paths plus the
 *  attention-weighted arcs connecting every token's walk to the tokens it
 *  read from at each layer.
 *
 *  Owns all of its scene objects. The host supplies the scene, two line
 *  materials (kept in the host's resize-managed material set), the live
 *  projection function, sprite/label helpers, and a `view()` accessor that
 *  reports the walk state each time arcs are rebuilt:
 *    view() → { path, step, enabled, paths, promptLen, promptTokens }
 *
 *  ALL arcs live in ONE preallocated LineSegments2 — same lesson as the
 *  host's TokenPath: per-arc objects re-allocate GPU buffers every tour
 *  frame. Weight rides in vertex-color brightness (LineMaterial has no
 *  per-vertex alpha). */

import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';

const MAX_ARCS = 512;
const ARC_SEGS = 12;
const ARC_COLOR = new THREE.Color(0xc168e0);
const PULSE_SPEED = 0.9;   // pulses per second traveling source → destination
const PULSE_SHARP = 5;     // bump tightness (higher = tighter bright blob)

export function initKvStream({ scene, arcMaterial, ghostMaterial,
                               projectPoint, makeLabelSprite, displayToken, view }) {
  let arcs = [];            // {src, srcIdx, dst, dstIdx, w, wmax, bright}
  let promptGhosts = [];    // PromptPath per prompt position 1..P-2
  let pulseT = 0;           // clock the traveling pulse rides on

  const arcGeo = new LineSegmentsGeometry();
  arcGeo.setPositions(new Float32Array(MAX_ARCS * ARC_SEGS * 6));
  arcGeo.setColors(new Float32Array(MAX_ARCS * ARC_SEGS * 6));
  arcGeo.instanceCount = 0;
  const arcLines = new LineSegments2(arcGeo, arcMaterial);
  arcLines.frustumCulled = false;
  scene.add(arcLines);

  /* Anchors for the k/v stream: arcs leaving a prompt token need a point to
   * leave from, so the server ships the prompt positions' full paths and
   * they render as extra-faint trails. Never counted by camera framing. */
  class PromptPath {
    constructor(coords, label, visible) {
      this.coords = coords;
      this.positions = coords.map(() => new THREE.Vector3());
      this.group = new THREE.Group();
      this.group.visible = visible;

      const n = coords.length;
      const geo = new LineGeometry();
      geo.setPositions(new Array(n * 3).fill(0));
      this.line = new Line2(geo, ghostMaterial);
      this.line.frustumCulled = false;
      this.group.add(this.line);

      this.labelSprite = makeLabelSprite(displayToken(label));
      this.labelSprite.material.opacity = 0.45;
      this.group.add(this.labelSprite);

      this.refreshPositions();
      scene.add(this.group);
    }

    /** Same in-place buffer rewrite as TokenPath — no allocation during tours. */
    refreshPositions() {
      const n = this.coords.length;
      for (let i = 0; i < n; i++) projectPoint(this.coords[i], this.positions[i]);
      const buf = this.line.geometry.attributes.instanceStart.data;
      const arr = buf.array;
      for (let seg = 0; seg < n - 1; seg++) {
        const a = this.positions[seg], b = this.positions[seg + 1];
        const o = seg * 6;
        arr[o] = a.x; arr[o + 1] = a.y; arr[o + 2] = a.z;
        arr[o + 3] = b.x; arr[o + 4] = b.y; arr[o + 5] = b.z;
      }
      buf.needsUpdate = true;
      this.labelSprite.position.copy(this.positions[n - 1]).add(new THREE.Vector3(0, 2.4, 0));
    }

    dispose() {
      scene.remove(this.group);
      this.line.geometry.dispose();
      this.labelSprite.material.map.dispose();
      this.labelSprite.material.dispose();
    }
  }

  /** Which drawn path owns absolute sequence position j (null: sink / absent). */
  function geometryForPos(j) {
    const { paths, promptLen } = view();
    if (j <= 0) return null;
    if (j <= promptLen - 2) return promptGhosts[j - 1] || null;
    return paths[j - (promptLen - 1)] || null;
  }

  /** Token text sitting at absolute position j. */
  function tokenAtPos(j) {
    const { paths, promptLen, promptTokens } = view();
    if (j <= 0) return '⟨sink⟩';
    if (j <= promptLen - 1) return promptTokens[j - 1] ?? '?';
    return paths[j - promptLen]?.chosen ?? '?';
  }

  /** Decide which arcs exist for the viewed path + cursor, then write buffers.
   *  Dim arcs for every attention point at or before the cursor; the most
   *  recent one is bright (so it stays lit through the following mlp step).
   *  Arc geometry: source = the source path's point 2ℓ (the state its k/v
   *  were computed from), destination = the viewed path's attn add 2ℓ+1. */
  function rebuildArcSet() {
    arcs = [];
    const { path: p, step, enabled } = view();
    if (enabled && p && p.attn && step >= 0) {
      const last = Math.min(step, p.attn.length - 1);
      let brightK = -1;
      for (let k = last; k >= 1; k--) {
        if (p.attn[k]?.length) { brightK = k; break; }
      }
      outer:
      for (let k = 1; k <= last; k++) {
        const sources = p.attn[k];
        if (!sources?.length) continue;
        const wmax = sources[0][1] || 1;   // server ships sources topk-sorted
        for (const [j, w] of sources) {
          const src = geometryForPos(j);
          if (!src) continue;
          if (arcs.length >= MAX_ARCS) break outer;
          arcs.push({ src, srcIdx: k - 1, dst: p, dstIdx: k, w, wmax, bright: k === brightK });
        }
      }
    }
    writeArcBuffers();
  }

  const _arcA = new THREE.Vector3(), _arcB = new THREE.Vector3(), _arcMid = new THREE.Vector3();
  const _arcLift = new THREE.Vector3(), _arcPrev = new THREE.Vector3(), _arcPt = new THREE.Vector3();
  const _arcUp = new THREE.Vector3(0, 1, 0);

  /** Sample every arc's bezier into the preallocated buffers, with a
   *  brightness pulse traveling source → destination (the k/v flowing into
   *  the reader). Runs on every arc-set change and every animation frame —
   *  must not allocate. */
  function writeArcBuffers() {
    if (!arcs.length) { arcGeo.instanceCount = 0; return; }
    const pos = arcGeo.attributes.instanceStart.data;
    const col = arcGeo.attributes.instanceColorStart.data;
    const pa = pos.array, ca = col.array;
    for (let i = 0; i < arcs.length; i++) {
      const arc = arcs[i];
      const A = _arcA.copy(arc.src.positions[arc.srcIdx]);
      const B = _arcB.copy(arc.dst.positions[arc.dstIdx]);
      // deterministic sideways+up lift: camera-independent, stable under orbit
      _arcMid.addVectors(A, B).multiplyScalar(0.5);
      const len = _arcLift.subVectors(B, A).length();
      _arcLift.cross(_arcUp);
      if (_arcLift.lengthSq() < 1e-6) _arcLift.set(1, 0, 0);
      _arcLift.normalize().multiplyScalar(len * 0.18);
      _arcMid.add(_arcLift);
      _arcMid.y += 1.5;

      const t01 = arc.w / arc.wmax;
      const glow = arc.bright ? 0.35 + 0.65 * t01 : 0.12 + 0.2 * t01;
      const phase = pulseT * PULSE_SPEED + i * 0.13;   // desynced per arc

      _arcPrev.copy(A);
      for (let s = 1; s <= ARC_SEGS; s++) {
        const t = s / ARC_SEGS, u = 1 - t;
        _arcPt.set(
          u * u * A.x + 2 * u * t * _arcMid.x + t * t * B.x,
          u * u * A.y + 2 * u * t * _arcMid.y + t * t * B.y,
          u * u * A.z + 2 * u * t * _arcMid.z + t * t * B.z,
        );
        // periodic bump peaked where the pulse currently sits on the arc
        const along = (s - 0.5) / ARC_SEGS;
        const wave = Math.pow(0.5 + 0.5 * Math.cos(2 * Math.PI * (along - phase)), PULSE_SHARP);
        const flow = arc.bright ? 0.6 + 0.8 * wave : 0.85 + 0.3 * wave;
        const r = ARC_COLOR.r * glow * flow;
        const g = ARC_COLOR.g * glow * flow;
        const b = ARC_COLOR.b * glow * flow;
        const o = (i * ARC_SEGS + s - 1) * 6;
        pa[o] = _arcPrev.x; pa[o + 1] = _arcPrev.y; pa[o + 2] = _arcPrev.z;
        pa[o + 3] = _arcPt.x; pa[o + 4] = _arcPt.y; pa[o + 5] = _arcPt.z;
        ca[o] = r; ca[o + 1] = g; ca[o + 2] = b;
        ca[o + 3] = r; ca[o + 4] = g; ca[o + 5] = b;
        _arcPrev.copy(_arcPt);
      }
    }
    arcGeo.instanceCount = arcs.length * ARC_SEGS;
    pos.needsUpdate = true;
    col.needsUpdate = true;
  }

  /** Per-frame driver: advances the traveling pulse (and, because positions
   *  are rewritten too, keeps arcs glued to their endpoints during tours).
   *  Called after the host has refreshed path positions for the frame. */
  function animate(timeSec) {
    pulseT = timeSec;
    if (arcs.length) writeArcBuffers();
  }

  /** Replace the prompt ghost set (called on each walk's meta packet). */
  function setPromptGhosts(coordsList, tokens, visible) {
    for (const g of promptGhosts) g.dispose();
    promptGhosts = (coordsList || []).map((c, i) => new PromptPath(c, tokens[i] ?? '?', visible));
  }

  function setGhostVisibility(v) { for (const g of promptGhosts) g.group.visible = v; }
  function refreshGhosts() { for (const g of promptGhosts) g.refreshPositions(); }

  function reset() {
    setPromptGhosts([], [], false);
    arcs = [];
    arcGeo.instanceCount = 0;
  }

  return { setPromptGhosts, setGhostVisibility, refreshGhosts,
           rebuildArcSet, animate, tokenAtPos, reset };
}
