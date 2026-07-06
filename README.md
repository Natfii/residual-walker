# Residual Walker

Watch a real transformer draw its residual-stream path through 3D space, one
Euler step at a time, until it fires a token.

Every layer in a transformer updates the hidden state by **addition**:
`h = h + attention(norm(h))`, then `h = h + mlp(norm(h))` — each `+` is one
forward-Euler step. Residual Walker runs a small LLM locally, captures the
hidden state at every sub-layer boundary (embedding + each attention add +
each MLP add), projects the states down to 3D with PCA, and animates the
path growing in your browser. At every point it applies the **logit lens**
(final norm + unembedding) so you can watch which token the model *would*
pick if the path stopped there.

## Quick start

**Windows, no Python needed** — grab `ResidualWalker.exe` (from the repo's
Releases page or build it yourself, below), drop it in this folder,
double-click. First run bootstraps everything: a private Python environment,
the right PyTorch build (CUDA if you have an NVIDIA GPU, CPU otherwise), and
a model of your choice from a picker. Later runs go straight to launch.

**With Python 3.10+ installed** (Windows/Linux/macOS):

```bash
python launcher.py
```

Same bootstrap, same picker. The browser opens automatically when the
server is ready.

**Fully manual** (if you'd rather own the environment):

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python server.py     # then open http://127.0.0.1:8471
```

(Use `/whl/cpu` instead of `/whl/cu128` on machines without an NVIDIA GPU;
on Linux/macOS the venv paths are `.venv/bin/...`.)

## What you're looking at

- **White sphere** — the token's embedding: where the path starts.
- **Blue segments/spheres** — attention adds (context pulled from other tokens).
- **Orange segments/spheres** — MLP adds (facts/associations recalled).
- **Floating label** — the logit-lens top guess *at that point on the path*;
  the side panel shows the top-5 with probabilities.
- **Ring flash** — the head fires: the finished vector is compared against
  every vocabulary direction and the winner becomes the next token.
- **Gray ghosts** — earlier tokens' paths. Every generated token is a
  complete new trip through all the layers.

Try `The capital of France is` at temperature 0 and watch "Paris" take over
the lens partway through the layers — that's the retrieval moment.

## Grand tour & shadow honesty

The 3D view is a *shadow* of the model's full hidden space (2048 dimensions
for Llama 3.2 1B) — like a hand shadow on a wall, cast from the single most
informative angle (top-3 PCA). Tick **🖐 grand tour** to slowly turn the
hand: the projection frame rotates through the top-12 principal directions
and hidden structure swings into view while the paths morph. Untick and it
eases back to the best angle.

The **shadow honesty** meter (top-left) shows what fraction of the paths'
true spread is visible through the current angle — watch it drop as the
tour turns away from the best angle and recover as it swings back.

## MP4 export

Tick **⏺ record walk → MP4** before hitting Walk. The scene is recorded
live (including your own camera moves) with an overlay card carrying the
prompt, the story so far, the step readout, and the logit-lens bars — so
exports tell the whole story on their own. When the walk ends the server
transcodes with NVENC on the GPU (CPU x264 fallback) and a download link
appears; files land in `exports/`. Requires `ffmpeg` on your PATH.

## Models

Any ungated HuggingFace causal LM with Llama-style module structure works —
Llama, Qwen 2/2.5, and Mistral families. The launcher's picker offers a few
presets; paste any other id at the prompt. Notes:

- Meta's official `meta-llama/*` repos are approval-gated; the `unsloth/*`
  mirrors are the same weights, ungated.
- Change your saved choice with `python launcher.py --reset-model`
  (or delete the `.model` file).
- Bigger models mean more layers per path and slower walks; 1B-class models
  are the sweet spot for watching individual steps.

## Configuration

| Env var | Meaning | Default |
|---|---|---|
| `RESIDUAL_WALKER_MODEL` | HuggingFace model id (overrides the saved pick) | `unsloth/Llama-3.2-1B` |
| `RESIDUAL_WALKER_PORT` | Server port (the launcher auto-bumps if busy) | `8471` |
| `RESIDUAL_WALKER_CPU` | Set to `1` to force CPU PyTorch at setup | unset |

URL parameter: `?autowalk=1` starts a walk on page load.

## Building the portable exe

```bash
build_exe.bat        # → dist/ResidualWalker.exe (~10 MB)
```

The exe is only the launcher — pure standard library, compiled with
PyInstaller. All heavy dependencies download at first run, which is what
keeps it portable and the repo tiny.

## Honest caveats

- The 3D view keeps the three directions with the most variance and drops
  the rest — distances are suggestive, not exact. The grand tour exists to
  make that limitation *visible*.
- Mid-layer logit-lens guesses are often garbage in small models; that's
  real behavior, not a bug. The signal cleans up in the late layers.
- Residual-stream norms genuinely grow across layers — the path
  accelerating outward near the head is real geometry, not an artifact.
- The PCA basis is fit per walk, so paths within one walk are comparable;
  paths across different walks are not.
