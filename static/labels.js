/** Billboard token labels: the sprite factory shared by ghost path endpoints
 *  and the k/v stream's prompt anchors, plus the per-frame size clamp.
 *
 *  Labels are world-sized sprites, so after a long generation zooming in
 *  fills the screen with giant token text. clampLabelScales() bounds every
 *  label's projected height to a readable pixel band each frame; sprites
 *  whose group has left the scene deregister themselves there. */

import * as THREE from 'three';

const LABEL_MAX_PX = 22;
const LABEL_MIN_PX = 9;

export function initLabels({ scene, camera, stage }) {
  const labelSprites = new Set();   // every live label, for per-frame clamping

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
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: tex, transparent: true, depthTest: false, opacity: 0.85,
    }));
    sprite.scale.set(w * 0.055, 52 * 0.055, 1);
    sprite.userData.naturalScale = sprite.scale.clone();
    labelSprites.add(sprite);
    return sprite;
  }

  function clampLabelScales() {
    const pxPerWorld = stage.clientHeight /
      (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
    for (const sprite of labelSprites) {
      let root = sprite;
      while (root.parent) root = root.parent;
      if (root !== scene) { labelSprites.delete(sprite); continue; }
      const natural = sprite.userData.naturalScale;
      const dist = Math.max(camera.position.distanceTo(sprite.position), 1e-3);
      const px = (natural.y / dist) * pxPerWorld;
      const f = px > LABEL_MAX_PX ? LABEL_MAX_PX / px
              : px < LABEL_MIN_PX ? LABEL_MIN_PX / px : 1;
      sprite.scale.set(natural.x * f, natural.y * f, 1);
    }
  }

  return { makeLabelSprite, clampLabelScales };
}
