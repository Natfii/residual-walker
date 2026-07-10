/** Legend visibility toggles for the 3D scene.
 *
 *  Every `data-legend` entry in the legend hides its scene objects, marked
 *  with a red strike:
 *    embed / attn / mlp / patch — two states: shown, or hidden (⧸). The
 *      embed entry also hides the walking token's trail line;
 *    ghosts — four states cycling normal → ⧸ end orbs hidden → ✕ orbs +
 *      trails + labels hidden → ⧹ trails hidden, orbs back → normal;
 *    arcs — mirrors the "k/v stream" checkbox both ways.
 *
 *  The host calls applyToPath() whenever a path gains geometry (reveal /
 *  ghosting); state changes re-apply to every path via the view() accessor. */

export function initLegend({ view, kvArcsBox }) {
  const state = { embed: true, attn: true, mlp: true, patch: true, ghosts: 0 };

  const items = {};
  for (const el of document.querySelectorAll('#legend .item[data-legend]')) {
    items[el.dataset.legend] = el;
    el.onclick = () => toggle(el.dataset.legend);
  }
  kvArcsBox.addEventListener('change', refreshMarks);

  function toggle(key) {
    if (key === 'arcs') {
      kvArcsBox.checked = !kvArcsBox.checked;
      kvArcsBox.dispatchEvent(new Event('change'));   // host handler redraws arcs
    } else if (key === 'ghosts') {
      state.ghosts = (state.ghosts + 1) % 4;
    } else {
      state[key] = !state[key];
    }
    const { paths, current } = view();
    for (const p of paths) applyToPath(p);
    if (current) applyToPath(current);
    refreshMarks();
  }

  function refreshMarks() {
    for (const [key, el] of Object.entries(items)) {
      const ghosts = key === 'ghosts' ? state.ghosts : 0;
      el.classList.toggle('slash', ghosts === 1 || (key === 'arcs' ? !kvArcsBox.checked : state[key] === false));
      el.classList.toggle('cross', ghosts === 2);
      el.classList.toggle('backslash', ghosts === 3);
    }
  }

  /** Push the current legend state onto one TokenPath's scene objects. */
  function applyToPath(p) {
    if (p.endSphere) {           // ghosted: faint trail + end orb + label
      const g = state.ghosts;    // 0 all · 1 orbs off · 2 all off · 3 trails off
      p.line.visible = g !== 2 && g !== 3;
      p.endSphere.visible = g !== 1 && g !== 2;
      p.labelSprite.visible = g !== 2;
    } else {                     // live: colored trail + one orb per step
      p.line.visible = state.embed;
      for (const s of p.spheres) s.visible = state[s.userData.kind] !== false;
    }
  }

  return { applyToPath, visible: key => state[key] !== false };
}
