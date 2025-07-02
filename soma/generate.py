# generate.py
import os
from huggingface_hub import snapshot_download

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_weights")

def ensure_model():
    if not os.path.isdir(MODEL_DIR):
        print("⏬ downloading Roblox/cube3d-v0.1…")
        snapshot_download(
            repo_id="Roblox/cube3d-v0.1",
            local_dir=MODEL_DIR
        )
    return MODEL_DIR


def mesh(prompt: str, *, resolution: float = 9.0, output: str | None = None, model_path: str | None = None):
    """Generate a 3D mesh from a text ``prompt``.

    This implementation simply prints what it would do. In a real
    application this function would call the underlying model to produce
    the mesh and then save it to ``output``.
    """

    if model_path is None:
        model_path = ensure_model()

    print(f"Generating mesh with prompt={prompt!r}, resolution={resolution}")
    print(f"Using model weights at {model_path}")
    if output:
        print(f"Result will be saved to {output}")

    return output


def main():
    model_path = ensure_model()
    # …now load and run your mesh-gen code, pointing it at model_path…

if __name__ == "__main__":
    main()
