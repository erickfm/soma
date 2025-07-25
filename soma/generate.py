"""
Library module for Cube3D mesh generation.

Provides ensure_model() and mesh() functions for programmatic use.
"""
import os
import tempfile
from urllib.parse import urlparse
import getpass
import paramiko
import trimesh
import torch
from huggingface_hub import snapshot_download
from cube3d.inference.engine import Engine, EngineFast

# Directory where model weights will be stored (next to this file)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_weights")
# Correct Hugging Face repo for Cube3D
REPO_ID = "Roblox/cube3d-v0.1"
# Clear memory cache
torch.cuda.empty_cache()

def ensure_model() -> str:
    """
    Download model weights if MODEL_DIR is missing.
    Returns the path to the model directory.
    """
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
    """
    Generate a 3D mesh from text prompt.
    Saves to `output` (local path or ssh://) and returns the output path.
    """
    # ensure we have model files
    model_path = model_path or ensure_model()

    # prepare checkpoint and config locations
    config_path     = os.path.join("cube3d/configs/", "open_model.yaml")
    gpt_ckpt_path   = os.path.join(model_path, "shape_gpt.safetensors")
    shape_ckpt_path = os.path.join(model_path, "shape_tokenizer.safetensors")

    # pick CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EngineCls = EngineFast if device.type == "cuda" else Engine
    engine = EngineCls(config_path, gpt_ckpt_path, shape_ckpt_path, device=device)

    # run text-to-mesh
    with torch.inference_mode():
        verts, faces = engine.t2s([prompt], use_kv_cache=True, resolution_base=resolution)[0][:2]
    mesh_obj = trimesh.Trimesh(vertices=verts, faces=faces)

    # fallback filename if none provided
    if output is None:
        safe = prompt.lower().replace(" ", "_")[:40]
        output = f"{safe}.obj"

    parsed = urlparse(output)
    if parsed.scheme == "ssh":
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            mesh_obj.export(tmp.name)
            tmp.flush()

            user = 'erick'
            _pass = "''''"
            host = parsed.hostname
            port = parsed.port or 22
            path = parsed.path

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, port=port, username=user, password=_pass)
            sftp = client.open_sftp()
            try:
                def sftp_mkdirs(sftp, remote_path):
                    dirs = os.path.dirname(remote_path).split('/')
                    path_so_far = ''
                    for d in dirs:
                        if not d:
                            continue
                        path_so_far += '/' + d
                        try:
                            sftp.stat(path_so_far)
                        except FileNotFoundError:
                            sftp.mkdir(path_so_far)

                sftp_mkdirs(sftp, path)
                sftp.put(tmp.name, path)
            finally:
                sftp.close()
                client.close()
    else:
        mesh_obj.export(output)

    return output
