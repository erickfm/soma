#!/usr/bin/env python3
import os
import tempfile
import argparse
from urllib.parse import urlparse
import getpass
import paramiko
import trimesh
import torch
from huggingface_hub import snapshot_download
from cube3d.inference.engine import Engine, EngineFast

# where we’ll stash the model files next to this script
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "model_weights")
REPO_ID    = "Roblox/cube3d-v0.1"  # correct HF repo :contentReference[oaicite:0]{index=0}

def ensure_model() -> str:
    """Download model files into MODEL_DIR if they’re missing."""
    if not os.path.isdir(MODEL_DIR):
        print("⏬ Downloading Cube3D weights…")
        snapshot_download(repo_id=REPO_ID, local_dir=MODEL_DIR)
    return MODEL_DIR

def mesh(
    prompt: str,
    *,
    resolution: float = 9.0,
    output: str | None = None,
    model_path: str | None = None,
) -> str:
    """Generate a 3D mesh from `prompt` and write it to `output`."""
    model_path = model_path or ensure_model()

    # paths inside the HF snapshot
    config_path     = os.path.join(model_path, "open_model.yaml")
    gpt_ckpt_path   = os.path.join(model_path, "shape_gpt.safetensors")
    shape_ckpt_path = os.path.join(model_path, "shape_tokenizer.safetensors")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EngineCls = EngineFast if device.type == "cuda" else Engine
    engine = EngineCls(config_path, gpt_ckpt_path, shape_ckpt_path, device=device)

    # generate
    print(f"⚙️  Generating mesh (“{prompt}”) @ res={resolution}…")
    mesh_vf = engine.t2s([prompt], use_kv_cache=True, resolution_base=resolution)
    verts, faces = mesh_vf[0][0], mesh_vf[0][1]
    mesh_obj = trimesh.Trimesh(vertices=verts, faces=faces)

    # pick default filename
    if not output:
        safe = prompt.lower().replace(" ", "_")[:40]
        output = f"{safe}.obj"

    parsed = urlparse(output)
    if parsed.scheme == "ssh":
        # write locally then push via SFTP
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            mesh_obj.export(tmp.name)
            tmp.flush()

            user = parsed.username or getpass.getuser()
            host = parsed.hostname
            port = parsed.port or 22
            remote_path = parsed.path

            print(f"🔗 Uploading to {user}@{host}:{remote_path}…")
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, port=port, username=user)
            sftp = client.open_sftp()
            try:
                sftp.put(tmp.name, remote_path)
            finally:
                sftp.close()
                client.close()
    else:
        print(f"💾 Saving mesh to {output}")
        mesh_obj.export(output)

    return output

def main():
    p = argparse.ArgumentParser(description="Text→3D mesh with Roblox Cube3D")
    p.add_argument("prompt", help="Text prompt describing the shape")
    p.add_argument(
        "--resolution","-r", type=float, default=9.0,
        help="Base resolution (higher=more detail)"
    )
    p.add_argument(
        "--output","-o", default=None,
        help="Where to write the .obj (local path or ssh://user@host/path.obj)"
    )
    p.add_argument(
        "--model-path","-m", default=None,
        help="Pre-downloaded model directory (skip automatic download)"
    )
    args = p.parse_args()

    try:
        out = mesh(
            args.prompt,
            resolution=args.resolution,
            output=args.output,
            model_path=args.model_path,
        )
        print("✅ Done! Mesh written to:", out)
    except Exception as e:
        print("❌ Error:", e)
        raise

if __name__ == "__main__":
    main()
