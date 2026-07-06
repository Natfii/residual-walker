"""Residual Walker — watch a transformer draw its residual-stream path in 3D.

Runs a small Llama model locally, captures the residual stream at every
sub-layer boundary (embedding, then each attention add and each MLP add),
applies the logit lens at every point, projects the high-dimensional states
to 3D with PCA, and streams one path per generated token to the browser
over a WebSocket.

Run:  python server.py   →   http://127.0.0.1:8471
Overrides:  RESIDUAL_WALKER_MODEL=<hf-repo-id>, RESIDUAL_WALKER_PORT=<port>.
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SCENE_RADIUS = 42.0  # world units the prompt's projected states are scaled into
TOUR_DIMS = 12       # PCA components sent to the browser for the grand tour
LENS_TOP_K = 5
MAX_NEW_TOKENS_CAP = 48
SAMPLE_TOP_K = 50


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

    def stream_states(self):
        """Rebuild the residual stream at every sub-layer boundary.

        Returns a tensor of shape [n_points, seq, hidden] where
        n_points = 1 + 2 * n_layers (embedding, then attn add, mlp add, ...).
        """
        s = self.embed_out[0]  # [seq, hidden]
        points = [s]
        for attn, mlp in zip(self.attn_deltas, self.mlp_deltas):
            s = s + attn[0]
            points.append(s)
            s = s + mlp[0]
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
        print(f"[residual-walker] ready: {self.n_layers} layers, hidden={self.hidden}")

    @torch.inference_mode()
    def forward_states(self, input_ids):
        """Full forward pass; returns residual stream states [n_points, seq, hidden]."""
        self.recorder.reset()
        self.model(input_ids, use_cache=False)
        return self.recorder.stream_states()

    @torch.inference_mode()
    def lens_logits(self, path_states):
        """Logit lens: final RMSNorm + unembedding applied to every path point.

        path_states: [n_points, hidden] → logits [n_points, vocab] (float32).
        """
        normed = self.model.model.norm(path_states)
        return self.model.lm_head(normed).float()

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


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


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


def build_packet(walker, path_states, pca, scale, temperature, token_index):
    """Assemble one token's path packet: 3D points, logit lens, sampled token."""
    coords = pca.transform(path_states.float().cpu().numpy()) * scale
    logits = walker.lens_logits(path_states)
    probs = torch.softmax(logits, dim=-1)
    top = probs.topk(LENS_TOP_K, dim=-1)
    lens = [
        [
            {"t": token_text(walker.tokenizer, int(i)), "p": round(float(p), 4)}
            for i, p in zip(row_ids, row_ps)
        ]
        for row_ids, row_ps in zip(top.indices, top.values)
    ]
    next_id = walker.sample(logits[-1], temperature)
    return {
        "type": "path",
        "index": token_index,
        "coords": [[round(float(c), 3) for c in row] for row in coords],
        "lens": lens,
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

            ids = walker.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            states = await asyncio.to_thread(walker.forward_states, ids)
            pca, scale = await asyncio.to_thread(walker.fit_projection, states)

            await ws.send_json({
                "type": "meta",
                "model": MODEL_ID,
                "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
                "n_layers": walker.n_layers,
                "hidden": walker.hidden,
                "steps": walker.step_kinds(),
                "tour": {
                    "dims": int(pca.n_components_),
                    "var_ratios": [round(float(r), 5) for r in pca.explained_variance_ratio_],
                },
                "prompt_tokens": [
                    token_text(walker.tokenizer, int(t)) for t in ids[0][1:]
                ],
            })

            generated = []
            eos = walker.tokenizer.eos_token_id
            for i in range(max_new):
                if i > 0:
                    states = await asyncio.to_thread(walker.forward_states, ids)
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
