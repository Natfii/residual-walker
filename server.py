"""Residual Walker — watch a transformer draw its residual-stream path in 3D.

Runs a small Llama model locally, captures the residual stream at every
sub-layer boundary (embedding, then each attention add and each MLP add),
applies the logit lens at every point, projects the high-dimensional states
to 3D with PCA, and streams one path per generated token to the browser
over a WebSocket.

For supported models a pre-fitted Jacobian lens (J-lens) is downloaded and
applied alongside the logit lens: instead of asking "what if the head fired
right now", the J-lens transports the state through the model's *average*
remaining flow (J_l = E[dh_final/dh_l], one matrix per layer) before decoding
— "what is this state disposed to make the model say later". Method and
fitted lenses: Anthropic's "Verbalizable Representations Form a Global
Workspace in Language Models" + github.com/anthropics/jacobian-lens; lens
files fitted and hosted by Neuronpedia (neuronpedia/jacobian-lens).

Run:  python server.py   →   http://127.0.0.1:8471
Overrides:  RESIDUAL_WALKER_MODEL=<hf-repo-id>, RESIDUAL_WALKER_PORT=<port>,
            RESIDUAL_WALKER_JLENS=auto|off|<path-to-lens.pt>.
"""

import asyncio
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("RESIDUAL_WALKER_MODEL", "unsloth/Llama-3.2-1B")
PORT = int(os.environ.get("RESIDUAL_WALKER_PORT", "8471"))
JLENS_SPEC = os.environ.get("RESIDUAL_WALKER_JLENS", "auto")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SCENE_RADIUS = 42.0  # world units the prompt's projected states are scaled into
TOUR_DIMS = 12       # PCA components sent to the browser for the grand tour
LENS_TOP_K = 5
MAX_NEW_TOKENS_CAP = 200
SAMPLE_TOP_K = 50

# Pre-fitted Jacobian lenses (Neuronpedia, fitted with Anthropic's jlens on
# wikitext). Keyed by lowercased repo basename so mirrors match too; values
# are the file paths inside JLENS_REPO. Only models whose blocks the recorder
# reconstructs exactly (plain pre-norm attn/mlp) are listed — qwen3.5/3.6
# lenses exist but those are hybrid architectures the recorder can't walk.
JLENS_REPO = "neuronpedia/jacobian-lens"
# Pin the exact revision so a mutable Hub repo can't swap lens contents under
# us (torch.load(weights_only=True) blocks code execution, not content drift).
# Bump deliberately after inspecting upstream changes.
JLENS_REVISION = "5003e6ecd11bb085e2129d7411800b95074e4682"  # 2026-07-02


def _jlens_file(folder, base):
    return f"{folder}/jlens/Salesforce-wikitext/{base}_jacobian_lens.pt"


JLENS_MODELS = {
    "llama-3.1-8b": _jlens_file("llama3.1-8b", "Llama-3.1-8B"),
    "meta-llama-3.1-8b": _jlens_file("llama3.1-8b", "Llama-3.1-8B"),
    "llama-3.1-8b-instruct": _jlens_file("llama3.1-8b-it", "Llama-3.1-8B-Instruct"),
    "meta-llama-3.1-8b-instruct": _jlens_file("llama3.1-8b-it", "Llama-3.1-8B-Instruct"),
    "qwen2.5-7b-instruct": _jlens_file("qwen2.5-7b-it", "Qwen2.5-7B-Instruct"),
    "qwen3-1.7b": _jlens_file("qwen3-1.7b", "Qwen3-1.7B"),
    "qwen3-4b": _jlens_file("qwen3-4b", "Qwen3-4B"),
    "qwen3-8b": _jlens_file("qwen3-8b", "Qwen3-8B"),
    "qwen3-14b": _jlens_file("qwen3-14b", "Qwen3-14B"),
    "qwen3-32b": _jlens_file("qwen3-32b", "Qwen3-32B"),
}


class ResidualRecorder:
    """Forward hooks that capture the embedding output and every attention/MLP
    delta, so the residual stream can be reconstructed by cumulative addition.

    Reconstruction is exact for pre-norm blocks (Llama):
        h = h + attn(norm(h));  h = h + mlp(norm(h))
    so the stream at any sub-layer boundary is embedding + sum of deltas so far.
    """

    def __init__(self, model):
        self.embed_out = None
        self.attn_deltas = []
        self.mlp_deltas = []
        self.patch_vecs = {}        # {layer: steering vector applied this forward}
        model.model.embed_tokens.register_forward_hook(self._grab_embed)
        for layer in model.model.layers:
            layer.self_attn.register_forward_hook(self._grab_attn)
            layer.mlp.register_forward_hook(self._grab_mlp)

    def _grab_embed(self, module, inputs, output):
        self.embed_out = output.detach()

    def _grab_attn(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        self.attn_deltas.append(out.detach())

    def _grab_mlp(self, module, inputs, output):
        self.mlp_deltas.append(output.detach())

    def reset(self):
        self.embed_out = None
        self.attn_deltas.clear()
        self.mlp_deltas.clear()
        self.patch_vecs.clear()

    def stream_states(self):
        """Rebuild the residual stream at every sub-layer boundary.

        Returns a tensor of shape [n_points, seq, hidden] where
        n_points = 1 + 2 * n_layers (embedding, then attn add, mlp add, ...).
        If a steering patch was applied this forward, it is folded in at its
        layer boundary so the reconstructed path matches what the model saw.
        """
        s = self.embed_out[0]  # [seq, hidden]
        points = [s]
        for i, (attn, mlp) in enumerate(zip(self.attn_deltas, self.mlp_deltas)):
            s = s + attn[0]
            points.append(s)
            s = s + mlp[0]
            if i in self.patch_vecs:
                pad = torch.zeros_like(s)
                pad[-1] = self.patch_vecs[i]
                s = s + pad
            points.append(s)
        return torch.stack(points)


class Walker:
    """Owns the model and turns prompts into per-token path packets."""

    def __init__(self):
        print(f"[residual-walker] loading {MODEL_ID} on {DEVICE} ({DTYPE}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE)
        inner = getattr(self.model, "model", None)
        if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "embed_tokens"):
            raise SystemExit(
                f"[residual-walker] {MODEL_ID} is not a supported architecture. "
                "The recorder needs Llama-style modules (model.model.layers[i].self_attn/.mlp) — "
                "Llama, Qwen2/2.5, and Mistral family models work."
            )
        self.model.to(DEVICE).eval()
        self.recorder = ResidualRecorder(self.model)
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden = self.model.config.hidden_size
        self.jlens = self._load_jlens()
        print(f"[residual-walker] ready: {self.n_layers} layers, hidden={self.hidden}")

    def _load_jlens(self):
        """Load the Jacobian lens for MODEL_ID: {layer: J_l tensor} or None.

        J_l maps the residual at layer l's block output into the final-layer
        basis (fitted as the corpus-average Jacobian E[dh_final/dh_l]). Any
        failure downgrades to logit-lens-only rather than blocking the walk.
        """
        if JLENS_SPEC.lower() in ("off", "0", "none", ""):
            return None
        try:
            if JLENS_SPEC != "auto":
                path, source = JLENS_SPEC, JLENS_SPEC
            else:
                fname = JLENS_MODELS.get(MODEL_ID.split("/")[-1].lower())
                if fname is None:
                    print(f"[residual-walker] no pre-fitted J-lens known for {MODEL_ID} "
                          "(logit lens only; set RESIDUAL_WALKER_JLENS=<lens.pt> to supply one)")
                    return None
                from huggingface_hub import hf_hub_download
                print(f"[residual-walker] fetching J-lens {JLENS_REPO}/{fname} "
                      f"@ {JLENS_REVISION[:12]} ...")
                path = hf_hub_download(JLENS_REPO, fname, revision=JLENS_REVISION)
                source = f"{JLENS_REPO}:{fname.split('/')[0]}@{JLENS_REVISION[:12]}"
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            if ckpt.get("d_model") != self.hidden:
                print(f"[residual-walker] J-lens d_model={ckpt.get('d_model')} != "
                      f"model hidden={self.hidden}; skipping")
                return None
            jacobians = {int(l): j.to(DEVICE, DTYPE) for l, j in ckpt["J"].items()}
            print(f"[residual-walker] J-lens ready: {len(jacobians)} layers from {source}")
            return {"J": jacobians, "source": source}
        except Exception as err:
            print(f"[residual-walker] J-lens unavailable ({type(err).__name__}: {err}); "
                  "continuing with logit lens only")
            return None

    @torch.inference_mode()
    def forward_states(self, input_ids, patch=None):
        """Full forward pass; returns residual stream states [n_points, seq, hidden].

        patch steers the stream right after each block in patch["layers"] —
        a single layer normally, a layer→end range in sticky mode (re-applying
        every block is how steering outruns the network's self-repair). Modes:
          {"mode": "nudge", "layers", "alpha", "units"} — the spider→ant push at
              the last position only: h += alpha * ||h|| * units[layer]
              (units are per-layer: unembedding-row for single tokens, the
              phrase's own mean residual states for multi-token concepts).
          {"mode": "swap", "layers", "alpha", "bases"} — paper-style J-space
              swap: permute the state's coordinates in the two-J-lens-vector
              frame (bases[layer] is [hidden, 2]) at EVERY position; alpha=1 is
              exact. All positions, because the concept usually lives in the
              context tokens — attention re-imports it if they're left alone.
        """
        self.recorder.reset()
        handles = []
        if patch is not None:
            recorder, alpha = self.recorder, patch["alpha"]

            def make_hook(layer_idx):
                def steer_hook(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if patch["mode"] == "swap":
                        V = patch["bases"].get(layer_idx)
                        if V is None:
                            return output
                        h = h.clone()
                        x = h[0].float()                              # [seq, hidden]
                        gram = V.T @ V + 1e-6 * torch.eye(2, device=V.device)
                        coords = torch.linalg.solve(gram, V.T @ x.T)  # [2, seq]
                        delta = (alpha * (V @ (coords.flip(0) - coords)).T).to(h.dtype)
                        h[0] += delta
                        vec = delta[-1]
                    else:
                        unit = patch["units"].get(layer_idx)
                        if unit is None:
                            return output
                        h = h.clone()
                        vec = alpha * h[0, -1, :].norm() * unit
                        h[0, -1, :] += vec
                    recorder.patch_vecs[layer_idx] = vec.detach()
                    if isinstance(output, tuple):
                        return (h,) + tuple(output[1:])
                    return h
                return steer_hook

            handles = [
                self.model.model.layers[i].register_forward_hook(make_hook(i))
                for i in patch["layers"]
            ]
        try:
            self.model(input_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return self.recorder.stream_states()

    @torch.inference_mode()
    def phrase_layer_means(self, text):
        """Forward a phrase and return its mean residual state per block output.

        The mean is over real tokens (BOS excluded) — this is the model's own
        representation of the phrase at each depth, the ActAdd/CAA-style
        source for steering directions when a concept is more than one token.
        """
        ids = self.tokenizer(" " + text.strip(), return_tensors="pt").input_ids.to(DEVICE)
        states = self.forward_states(ids)
        bos = self.tokenizer.bos_token_id
        start = 1 if (bos is not None and ids.shape[1] > 1 and ids[0, 0].item() == bos) else 0
        return {i: states[2 * i + 2, start:, :].float().mean(dim=0) for i in range(self.n_layers)}

    def steer_directions(self, add_text, remove_text):
        """Per-layer unit steering directions {layer: vec} plus echo info.

        Single-token concepts: one unembedding-row direction, shared by every
        layer (the calibrated classic). Any multi-token concept switches both
        sides to phrase mode: activation-based directions from the model's own
        mean residual states, layer-matched.

        Returns (units, add_label, remove_label, source) — units is None when
        nothing resolves; source is "token" or "phrase".
        """
        def ids_of(text):
            return self.tokenizer.encode(" " + text.strip(), add_special_tokens=False) if text else []

        add_ids, remove_ids = ids_of(add_text), ids_of(remove_text)
        if not add_ids and not remove_ids:
            return None, None, None, None

        if len(add_ids) <= 1 and len(remove_ids) <= 1:
            direction = torch.zeros(self.hidden, device=DEVICE)
            add_tok = remove_tok = None
            if add_ids:
                row = self.model.lm_head.weight[add_ids[0]].float()
                direction = direction + row / row.norm()
                add_tok = self.tokenizer.decode([add_ids[0]])
            if remove_ids:
                row = self.model.lm_head.weight[remove_ids[0]].float()
                direction = direction - row / row.norm()
                remove_tok = self.tokenizer.decode([remove_ids[0]])
            norm = direction.norm()
            if norm < 1e-6:
                return None, add_tok, remove_tok, "token"
            unit = (direction / norm).to(DTYPE)
            return {i: unit for i in range(self.n_layers)}, add_tok, remove_tok, "token"

        add_means = self.phrase_layer_means(add_text) if add_ids else None
        remove_means = self.phrase_layer_means(remove_text) if remove_ids else None
        units = {}
        for i in range(self.n_layers):
            d = torch.zeros(self.hidden, device=DEVICE)
            if add_means is not None:
                d = d + add_means[i] / add_means[i].norm()
            if remove_means is not None:
                d = d - remove_means[i] / remove_means[i].norm()
            norm = d.norm()
            if norm > 1e-6:
                units[i] = (d / norm).to(DTYPE)
        if not units:
            return None, add_text, remove_text, "phrase"
        return units, add_text.strip() if add_text else None, \
            remove_text.strip() if remove_text else None, "phrase"

    @torch.inference_mode()
    def lens_logits(self, path_states):
        """Logit lens: final RMSNorm + unembedding applied to every path point.

        path_states: [n_points, hidden] → logits [n_points, vocab] (float32).
        """
        normed = self.model.model.norm(path_states)
        return self.model.lm_head(normed).float()

    @torch.inference_mode()
    def jlens_logits(self, path_states):
        """J-lens: transport each point into the final-layer basis with its
        block's J_l, then the same norm + unembed as the logit lens.

        Points map to the enclosing block: attn-add and mlp-add of layer l both
        use J_l (the attn midpoint has no fitted transport of its own; J_l is
        the nearest, off by that layer's MLP). The embedding point precedes
        block 0 and gets no readout. Fitted lenses omit the final block, whose
        transport is the identity by construction — there the J-lens IS the
        logit lens. Returns a list aligned with path points: [vocab] float32
        tensors, or None where no transport exists.
        """
        out = [None] * path_states.shape[0]
        if self.jlens is None:
            return out
        for k in range(1, path_states.shape[0]):
            layer = (k - 1) // 2
            J = self.jlens["J"].get(layer)
            if J is not None:
                transported = (path_states[k].float() @ J.T.float()).to(DTYPE)
            elif layer == self.n_layers - 1:
                transported = path_states[k]   # end of the flow: J = identity
            else:
                continue
            out[k] = self.model.lm_head(self.model.model.norm(transported)).float()
        return out

    def swap_basis(self, add_text, remove_text, layer):
        """Basis for a paper-style J-space swap at `layer`: the two J-lens
        vectors v_w = J_l^T u_w (u_w = unembedding row), stacked [hidden, 2].

        The swap permutes the state's coordinates in this frame and leaves the
        orthogonal complement untouched — concept A becomes concept B and vice
        versa, rather than just shoving the state along a direction.
        Returns (V, add_tok, remove_tok); V is None if the lens, the layer, or
        either token is unavailable.
        """
        if self.jlens is None or layer not in self.jlens["J"]:
            return None, None, None

        def first_id(text):
            ids = self.tokenizer.encode(" " + (text or "").strip(), add_special_tokens=False)
            return (ids[0], self.tokenizer.decode([ids[0]])) if ids else (None, None)

        a_id, a_tok = first_id(add_text)
        b_id, b_tok = first_id(remove_text)
        if a_id is None or b_id is None:
            return None, a_tok, b_tok
        J = self.jlens["J"][layer].float()
        u = self.model.lm_head.weight
        v_a = J.T @ u[a_id].float()
        v_b = J.T @ u[b_id].float()
        return torch.stack([v_a, v_b], dim=1), a_tok, b_tok

    def fit_projection(self, states):
        """Fit a PCA basis + scene scale on the prompt's states so every path in
        one generation shares the same projection (ghost trails stay comparable).

        The top TOUR_DIMS components are kept: the browser renders components
        0-2 as the "best angle" shadow and grand-tours through all of them.
        Position 0 (BOS) is excluded from the fit when possible — its residual
        norms are attention-sink outliers that would dominate the basis.
        """
        if states.shape[1] > 1:
            fit_states = states[:, 1:, :]
        else:
            fit_states = states
        flat = fit_states.transpose(0, 1).reshape(-1, self.hidden).float().cpu().numpy()
        pca = PCA(n_components=min(TOUR_DIMS, flat.shape[0])).fit(flat)
        spread = np.abs(pca.transform(flat)[:, :3]).max()
        scale = SCENE_RADIUS / max(spread, 1e-6)
        return pca, scale

    def sample(self, logits, temperature):
        """Sample the next token id from final-point logits (greedy if temp≈0)."""
        if temperature <= 0.01:
            return int(logits.argmax().item())
        topk = torch.topk(logits / temperature, SAMPLE_TOP_K)
        probs = torch.softmax(topk.values, dim=-1)
        pick = torch.multinomial(probs, 1).item()
        return int(topk.indices[pick].item())

    def step_kinds(self):
        """Ordered description of every path point: embed, then attn/mlp per layer."""
        steps = [{"kind": "embed", "layer": -1}]
        for i in range(self.n_layers):
            steps.append({"kind": "attn", "layer": i})
            steps.append({"kind": "mlp", "layer": i})
        return steps


walker = None
generation_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    global walker
    walker = await asyncio.to_thread(Walker)
    yield


app = FastAPI(lifespan=lifespan)
STATIC_DIR = Path(__file__).parent / "static"
EXPORTS_DIR = Path(__file__).parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)
app.mount("/exports", StaticFiles(directory=EXPORTS_DIR), name="exports")


# no-cache on all frontend files: they version together with the WS protocol,
# so a stale cached copy must never talk to a newer server
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/styles.css")
async def styles():
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css",
                        headers={"Cache-Control": "no-cache"})


@app.get("/walker.js")
async def walker_js():
    return FileResponse(STATIC_DIR / "walker.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/info")
async def info():
    """Model + lens availability, so the UI can configure itself before a walk."""
    return {
        "model": MODEL_ID,
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
        "n_layers": walker.n_layers,
        "hidden": walker.hidden,
        "jlens": {"available": walker.jlens is not None,
                  "source": walker.jlens["source"] if walker.jlens else None},
    }


@app.post("/export")
async def export_walk(request: Request):
    """Transcode a browser-recorded walk (WebM) to MP4 — NVENC first, CPU fallback."""
    webm = await request.body()
    if not webm:
        return JSONResponse({"error": "empty recording"}, status_code=400)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    src = EXPORTS_DIR / f"walk-{stamp}.webm"
    dst = EXPORTS_DIR / f"walk-{stamp}.mp4"
    src.write_bytes(webm)
    encoder_args = [
        ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"],
        ["-c:v", "libx264", "-crf", "20"],
    ]
    proc = None
    for enc in encoder_args:
        cmd = [
            "ffmpeg", "-y", "-i", str(src), *enc,
            # even dimensions + yuv420p keep the mp4 playable everywhere
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
        ]
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
        if proc.returncode == 0:
            src.unlink()
            return {"url": f"/exports/{dst.name}", "name": dst.name}
    detail = proc.stderr.decode(errors="replace")[-400:] if proc else "ffmpeg not found"
    return JSONResponse({"error": detail}, status_code=500)


def token_text(tokenizer, token_id):
    return tokenizer.decode([token_id])


def _top_rows(tokenizer, logits):
    """[vocab] logits → top-K [{t, p, i}] rows for the lens panel."""
    probs = torch.softmax(logits, dim=-1)
    top = probs.topk(LENS_TOP_K, dim=-1)
    return [
        {"t": token_text(tokenizer, int(i)), "p": round(float(p), 4), "i": int(i)}
        for i, p in zip(top.indices, top.values)
    ]


def build_packet(walker, path_states, pca, scale, temperature, token_index):
    """Assemble one token's path packet: 3D points, both lenses, sampled token."""
    coords = pca.transform(path_states.float().cpu().numpy()) * scale
    logits = walker.lens_logits(path_states)
    probs = torch.softmax(logits, dim=-1)
    top = probs.topk(LENS_TOP_K, dim=-1)
    lens = [
        [
            {"t": token_text(walker.tokenizer, int(i)), "p": round(float(p), 4), "i": int(i)}
            for i, p in zip(row_ids, row_ps)
        ]
        for row_ids, row_ps in zip(top.indices, top.values)
    ]
    jlens = None
    if walker.jlens is not None:
        jlens = [
            _top_rows(walker.tokenizer, row) if row is not None else None
            for row in walker.jlens_logits(path_states)
        ]
    next_id = walker.sample(logits[-1], temperature)
    return {
        "type": "path",
        "index": token_index,
        "coords": [[round(float(c), 3) for c in row] for row in coords],
        "lens": lens,
        "jlens": jlens,
        "chosen": token_text(walker.tokenizer, next_id),
        "chosen_prob": round(float(probs[-1, next_id]), 4),
    }, next_id


@app.websocket("/ws")
async def walk(ws: WebSocket):
    await ws.accept()
    if generation_lock.locked():
        await ws.send_json({"type": "error", "message": "A walk is already running."})
        await ws.close()
        return

    async with generation_lock:
        try:
            req = await ws.receive_json()
            prompt = str(req.get("prompt", "")).strip() or "The capital of France is"
            temperature = float(req.get("temperature", 0.7))
            max_new = min(int(req.get("max_new_tokens", 10)), MAX_NEW_TOKENS_CAP)

            patch = None
            patch_echo = {"active": False}
            patch_req = req.get("patch") or {}
            if patch_req.get("add") or patch_req.get("remove"):
                layer = max(0, min(walker.n_layers - 1, int(patch_req.get("layer", walker.n_layers // 2))))
                alpha = max(0.0, min(6.0, float(patch_req.get("alpha", 1.5))))
                mode = str(patch_req.get("mode", "nudge"))
                sticky = bool(patch_req.get("sticky"))
                if sticky:
                    # default range stops before the final quarter — re-injecting
                    # in the motor zone just parrots the token instead of steering
                    default_end = walker.n_layers - 1 - walker.n_layers // 4
                    layer_end = int(patch_req.get("layer_end", default_end))
                    layer_end = max(layer, min(walker.n_layers - 1, layer_end))
                    layers = list(range(layer, layer_end + 1))
                else:
                    layer_end = layer
                    layers = [layer]
                if mode == "swap":
                    bases, add_tok, remove_tok = {}, None, None
                    for l in layers:
                        basis, add_tok, remove_tok = walker.swap_basis(
                            patch_req.get("add"), patch_req.get("remove"), l
                        )
                        if basis is not None:
                            bases[l] = basis
                    if bases and alpha > 0:
                        patch = {"mode": "swap", "layers": layers, "alpha": alpha, "bases": bases}
                else:
                    units, add_tok, remove_tok, source = walker.steer_directions(
                        patch_req.get("add"), patch_req.get("remove")
                    )
                    if units is not None and alpha > 0:
                        patch = {"mode": "nudge", "layers": layers, "alpha": alpha, "units": units}
                if patch is not None:
                    patch_echo = {
                        "active": True, "mode": mode, "add": add_tok, "remove": remove_tok,
                        "layer": layer, "layer_end": layer_end, "alpha": alpha,
                        "sticky": sticky, "step": 2 * layer + 2,
                        "source": source if mode != "swap" else "token",
                    }

            ids = walker.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            # PCA basis is always fit on the unpatched forward, so nudged and
            # clean walks of the same prompt share a projection and compare 1:1.
            states = await asyncio.to_thread(walker.forward_states, ids)
            pca, scale = await asyncio.to_thread(walker.fit_projection, states)
            if patch is not None:
                states = await asyncio.to_thread(walker.forward_states, ids, patch)

            await ws.send_json({
                "type": "meta",
                "model": MODEL_ID,
                "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
                "n_layers": walker.n_layers,
                "hidden": walker.hidden,
                "jlens": {"available": walker.jlens is not None,
                          "source": walker.jlens["source"] if walker.jlens else None},
                "steps": walker.step_kinds(),
                "tour": {
                    "dims": int(pca.n_components_),
                    "var_ratios": [round(float(r), 5) for r in pca.explained_variance_ratio_],
                },
                "patch": patch_echo,
                "prompt_tokens": [
                    token_text(walker.tokenizer, int(t)) for t in ids[0][1:]
                ],
            })

            generated = []
            eos = walker.tokenizer.eos_token_id
            for i in range(max_new):
                if i > 0:
                    states = await asyncio.to_thread(walker.forward_states, ids, patch)
                packet, next_id = await asyncio.to_thread(
                    build_packet, walker, states[:, -1, :], pca, scale, temperature, i
                )
                await ws.send_json(packet)
                if next_id == eos:
                    break
                generated.append(next_id)
                ids = torch.cat(
                    [ids, torch.tensor([[next_id]], device=DEVICE)], dim=1
                )

            await ws.send_json({
                "type": "done",
                "text": walker.tokenizer.decode(generated),
            })
        except WebSocketDisconnect:
            pass  # user hit Stop or closed the tab mid-walk
        except Exception as err:  # surface real failures to the UI, not just the console
            try:
                await ws.send_json({"type": "error", "message": f"{type(err).__name__}: {err}"})
            except Exception:
                pass
            raise


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
