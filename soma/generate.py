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


def mesh():
    pass


def main():
    model_path = ensure_model()
    # …now load and run your mesh-gen code, pointing it at model_path…

if __name__ == "__main__":
    main()
