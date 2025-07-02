# generate.py
import os
import tempfile
from urllib.parse import urlparse

import getpass
import paramiko
import trimesh
from huggingface_hub import snapshot_download
import torch
from cube3d.inference.engine import Engine, EngineFast

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_weights")

def ensure_model():
    if not os.path.isdir(MODEL_DIR):
        print("⏬ downloading Roblox/cube…")
        snapshot_download(
            repo_id="Roblox/cube",
            local_dir=MODEL_DIR
        )
    return MODEL_DIR


def mesh(prompt: str, *, resolution: float = 9.0, output: str | None = None, model_path: str | None = None) -> str:
    """Generate a 3D mesh from a text ``prompt`` and save it to ``output``.

    This function downloads model weights if needed and then creates a
    very simple mesh based on keywords in the ``prompt``.  The mesh is
    saved locally or via ``ssh`` using :mod:`paramiko`.
    """

    if model_path is None:
        model_path = ensure_model()

    # Use Roblox Cube engine to generate the mesh
    config_path = os.path.join(model_path, "open_model.yaml")
    gpt_ckpt_path = os.path.join(model_path, "shape_gpt.safetensors")
    shape_ckpt_path = os.path.join(model_path, "shape_tokenizer.safetensors")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EngineCls = EngineFast if device.type == "cuda" else Engine
    engine = EngineCls(config_path, gpt_ckpt_path, shape_ckpt_path, device=device)

    mesh_v_f = engine.t2s([prompt], use_kv_cache=True, resolution_base=resolution)
    vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
    mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)

    if output is None:
        output = "mesh.obj"

    parsed = urlparse(output)
    if parsed.scheme == "ssh":
        # Save to a temporary file then copy via SFTP
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            mesh_obj.export(tmp.name)
            tmp.flush()

            host = parsed.hostname
            path = parsed.path
            username = parsed.username or getpass.getuser()
            port = parsed.port or 22

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, port=port, username=username)
            sftp = client.open_sftp()
            try:
                sftp.put(tmp.name, path)
            finally:
                sftp.close()
                client.close()
    else:
        mesh_obj.export(output)

    return output


def main():
    model_path = ensure_model()
    # …now load and run your mesh-gen code, pointing it at model_path…

if __name__ == "__main__":
    main()
