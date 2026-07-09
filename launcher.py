"""Residual Walker launcher — one entry point that bootstraps everything.

First run: downloads the `uv` package manager, provisions a private Python
environment (downloading Python itself if the machine has none), installs
the right PyTorch build (CUDA if an NVIDIA GPU is present, CPU otherwise),
lets you pick a model, pre-downloads it, then starts the server and opens
your browser. Later runs skip straight to launch.

Pure standard library, so it compiles to a small portable .exe with
PyInstaller (see build_exe.bat). Run it from the app folder — it needs
server.py / static / requirements.txt beside it.

Flags:  --no-browser   don't open a browser tab (CI / tests)
        --port N       preferred port (auto-bumps if busy)
        --reset-model  forget the saved model choice and ask again
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
TOOLS_DIR = APP_DIR / ".tools"
VENV_DIR = APP_DIR / ".venv"
DEPS_MARKER = VENV_DIR / ".deps-ok"
MODEL_FILE = APP_DIR / ".model"

IS_WINDOWS = sys.platform == "win32"
UV_URL = (
    "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    if IS_WINDOWS
    else "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
)
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

MODEL_PRESETS = [
    ("unsloth/Llama-3.2-1B", "Llama 3.2 1B — recommended first walk: 16 big steps, "
                             "dramatic firework paths (the demo gif)"),
    ("Qwen/Qwen3-1.7B", "Qwen3 1.7B — 28 gentler layers, pre-fitted J-lens ✓"),
    ("Qwen/Qwen3-4B", "Qwen3 4B — richer paths, pre-fitted J-lens ✓"),
    ("unsloth/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct — chat-tuned paths"),
    ("Qwen/Qwen2.5-1.5B", "Qwen 2.5 1.5B — a different model family to compare"),
]


def log(msg):
    print(f"[launcher] {msg}", flush=True)


def venv_python():
    return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def ensure_uv():
    """Return a usable uv binary, downloading a private copy if none exists."""
    on_path = shutil.which("uv")
    if on_path:
        return Path(on_path)
    local = TOOLS_DIR / ("uv.exe" if IS_WINDOWS else "uv")
    if local.exists():
        return local
    TOOLS_DIR.mkdir(exist_ok=True)
    archive = TOOLS_DIR / UV_URL.rsplit("/", 1)[-1]
    log(f"downloading uv ({UV_URL.rsplit('/', 1)[-1]}) ...")
    urllib.request.urlretrieve(UV_URL, archive)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(TOOLS_DIR)
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                if member.name.endswith("/uv"):
                    member.name = "uv"
                    t.extract(member, TOOLS_DIR)
    archive.unlink()
    if not local.exists():
        raise SystemExit("uv download failed — install uv manually (https://docs.astral.sh/uv/) and rerun.")
    if not IS_WINDOWS:
        local.chmod(0o755)
    return local


def has_nvidia_gpu():
    return os.environ.get("RESIDUAL_WALKER_CPU") != "1" and shutil.which("nvidia-smi") is not None


def ensure_env():
    """Create the virtualenv and install dependencies once."""
    if DEPS_MARKER.exists():
        return
    uv = ensure_uv()
    gpu = has_nvidia_gpu()
    log(f"provisioning Python environment ({'CUDA' if gpu else 'CPU'} PyTorch) — first run only, a few GB ...")
    run = lambda *cmd: subprocess.run([str(c) for c in cmd], cwd=APP_DIR, check=True)
    run(uv, "venv", VENV_DIR, "--python", "3.12", "--allow-existing")
    run(
        uv, "pip", "install", "-p", venv_python(),
        "-r", APP_DIR / "requirements.txt",
        "--extra-index-url", CUDA_INDEX if gpu else CPU_INDEX,
    )
    DEPS_MARKER.touch()
    log("environment ready")


def choose_model(reset=False):
    """Model id from env > saved choice > interactive menu > default preset."""
    env_model = os.environ.get("RESIDUAL_WALKER_MODEL")
    if env_model:
        return env_model
    if reset and MODEL_FILE.exists():
        MODEL_FILE.unlink()
    if MODEL_FILE.exists():
        return MODEL_FILE.read_text().strip()

    default_id = MODEL_PRESETS[0][0]
    if sys.stdin is None or not sys.stdin.isatty():
        return default_id
    print("\nWhich model should Residual Walker use? (downloads on first use)")
    for i, (model_id, blurb) in enumerate(MODEL_PRESETS, 1):
        print(f"  {i}. {model_id:<34} {blurb}")
    print("  or paste any HuggingFace id (Llama/Qwen/Mistral-family, ungated)")
    answer = input(f"choice [1]: ").strip() or "1"
    if answer.isdigit() and 1 <= int(answer) <= len(MODEL_PRESETS):
        model_id = MODEL_PRESETS[int(answer) - 1][0]
    else:
        model_id = answer
    MODEL_FILE.write_text(model_id)
    return model_id


def predownload(model_id):
    log(f"fetching {model_id} (cached after the first time) ...")
    code = f"from huggingface_hub import snapshot_download; snapshot_download({model_id!r})"
    subprocess.run([str(venv_python()), "-c", code], check=True)


def free_port(preferred):
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("no free port found")


def wait_for_server(port, proc, tries=90):
    import time
    url = f"http://127.0.0.1:{port}/"
    for _ in range(tries):
        if proc.poll() is not None:
            raise SystemExit("server exited during startup — see the log above")
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except OSError:
            time.sleep(1)
    raise SystemExit("server did not come up in time")


def main():
    parser = argparse.ArgumentParser(description="Residual Walker launcher")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8471)
    parser.add_argument("--reset-model", action="store_true")
    args = parser.parse_args()

    if not (APP_DIR / "server.py").exists():
        raise SystemExit("server.py not found — run the launcher from the Residual Walker folder.")

    ensure_env()
    model_id = choose_model(reset=args.reset_model)
    predownload(model_id)

    port = free_port(args.port)
    env = {**os.environ, "RESIDUAL_WALKER_MODEL": model_id, "RESIDUAL_WALKER_PORT": str(port)}
    log(f"starting server for {model_id} on port {port} ...")
    proc = subprocess.Popen([str(venv_python()), str(APP_DIR / "server.py")], cwd=APP_DIR, env=env)
    try:
        wait_for_server(port, proc)
        url = f"http://127.0.0.1:{port}/"
        log(f"ready — {url}")
        if not args.no_browser:
            import webbrowser
            webbrowser.open(url)
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
