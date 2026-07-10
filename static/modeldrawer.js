/** Model switcher: chip → drawer → confirm card → in-place server swap.
 *
 *  The model chip opens a drawer listing preset + custom models from
 *  GET /api/models. Picking one raises a confirm card over the 3D stage;
 *  confirming stops any walk, POSTs /api/model, and hands the fresh info
 *  payload to the host to reset the scene. Custom HF ids are added with
 *  POST /api/models and removed with DELETE /api/models?id=…
 *
 *  A first-time switch downloads the model, which can outlive the POST
 *  connection — on a dropped request the drawer falls back to polling
 *  /api/models until the server settles on a model again. */

const POLL_MS = 2000;
const POLL_MAX = 900;   // give a first-time download up to 30 minutes

export function initModelDrawer({ applyServerInfo, resetForModelSwitch, isWalking, stopWalk }) {
  const $ = id => document.getElementById(id);
  const chip = $('modelChip'), drawer = $('modelDrawer'), backdrop = $('drawerBackdrop');
  const confirmBox = $('modelConfirm'), statusEl = $('drawerStatus');
  let currentId = null;
  let pendingId = null;    // model waiting in the confirm card
  let switching = false;

  chip.onclick = () => { if (!switching) open(); };
  backdrop.onclick = close;
  $('modelDrawerClose').onclick = close;
  window.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (confirmBox.classList.contains('open')) hideConfirm();
    else close();
  });

  async function open() {
    drawer.classList.add('open');
    backdrop.classList.add('open');
    statusEl.textContent = '';
    try {
      render(await (await fetch('/api/models')).json());
    } catch { statusEl.textContent = 'could not reach the server'; }
  }

  function close() {
    drawer.classList.remove('open');
    backdrop.classList.remove('open');
    hideConfirm();
  }

  function render(payload) {
    currentId = payload.current;
    const list = $('modelList');
    list.innerHTML = '';
    for (const m of payload.models) {
      const entry = document.createElement('button');
      entry.className = 'modelEntry' + (m.id === currentId ? ' current' : '');
      const idEl = document.createElement('span');
      idEl.className = 'id';
      idEl.textContent = m.id;
      entry.appendChild(idEl);
      const blurb = document.createElement('span');
      blurb.className = 'blurb';
      blurb.textContent = m.custom ? 'custom model' : m.blurb;
      entry.appendChild(blurb);
      if (m.custom) {
        const rm = document.createElement('span');
        rm.className = 'rm';
        rm.textContent = '✕';
        rm.title = 'remove from the list';
        rm.onclick = async e => {
          e.stopPropagation();
          const resp = await fetch(`/api/models?id=${encodeURIComponent(m.id)}`, { method: 'DELETE' });
          render(await resp.json());
        };
        entry.appendChild(rm);
      }
      entry.onclick = () => { if (m.id !== currentId && !switching) showConfirm(m.id); };
      list.appendChild(entry);
    }
  }

  $('modelAddBtn').onclick = addModel;
  $('modelAddInput').addEventListener('keydown', e => { if (e.key === 'Enter') addModel(); });

  async function addModel() {
    const input = $('modelAddInput');
    const id = input.value.trim();
    if (!id) return;
    statusEl.textContent = '';
    try {
      const resp = await fetch('/api/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      });
      const payload = await resp.json();
      if (!resp.ok) { statusEl.textContent = payload.error; return; }
      input.value = '';
      render(payload);
    } catch (err) { statusEl.textContent = err.message; }
  }

  function showConfirm(id) {
    pendingId = id;
    $('modelConfirmId').textContent = id;
    confirmBox.classList.add('open');
  }

  function hideConfirm() {
    pendingId = null;
    confirmBox.classList.remove('open');
  }

  $('modelConfirmNo').onclick = hideConfirm;
  confirmBox.onclick = e => { if (e.target === confirmBox) hideConfirm(); };
  $('modelConfirmYes').onclick = () => {
    const id = pendingId;
    hideConfirm();
    if (id) switchTo(id);
  };

  async function switchTo(id) {
    close();
    if (isWalking()) stopWalk();
    switching = true;
    chip.classList.add('switching');
    chip.textContent = `switching to ${id} — downloading/loading…`;
    $('walkBtn').disabled = true;
    try {
      let info;
      try {
        let resp;
        for (let attempt = 0; ; attempt++) {
          resp = await fetch('/api/model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id }),
          });
          // a just-stopped walk holds the server lock until its in-flight
          // token drains — give it a few seconds before taking 409 as final
          if (resp.status !== 409 || attempt >= 20) break;
          await new Promise(r => setTimeout(r, 250));
        }
        info = await resp.json();
        if (!resp.ok) throw new Error(info.error || `HTTP ${resp.status}`);
      } catch (err) {
        info = await pollUntilSettled(id, err);
      }
      resetForModelSwitch();
      applyServerInfo(info);
    } catch (err) {
      chip.textContent = `model switch failed: ${err.message}`;
      // the server rolled back to the old model — re-sync the chip after
      // the failure message has had its moment
      setTimeout(() => fetch('/api/info').then(r => r.json())
        .then(applyServerInfo).catch(() => {}), 4000);
    } finally {
      switching = false;
      chip.classList.remove('switching');
      $('walkBtn').disabled = false;
    }
  }

  /** Wait out a switch whose POST died (long download / dropped connection).
   *  Resolves with the new /api/info once the server lands on `id`; rethrows
   *  the original error if it settles on anything else. */
  async function pollUntilSettled(id, originalErr) {
    for (let i = 0; i < POLL_MAX; i++) {
      await new Promise(r => setTimeout(r, POLL_MS));
      let payload;
      try { payload = await (await fetch('/api/models')).json(); }
      catch { continue; }              // server busy loading — keep waiting
      if (payload.loading) continue;
      if (payload.current === id) return await (await fetch('/api/info')).json();
      throw originalErr;
    }
    throw new Error('timed out waiting for the model switch');
  }
}
