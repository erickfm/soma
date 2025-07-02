# generate.py
import os
import tempfile
from urllib.parse import urlparse

import getpass
import paramiko
import trimesh
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


def mesh(prompt: str, *, resolution: float = 9.0, output: str | None = None, model_path: str | None = None) -> str:
    """Generate a 3D mesh from a text ``prompt`` and save it to ``output``.

    This function downloads model weights if needed and then creates a
    very simple mesh based on keywords in the ``prompt``.  The mesh is
    saved locally or via ``ssh`` using :mod:`paramiko`.
    """

    if model_path is None:
        model_path = ensure_model()

    # Map prompt keywords to simple geometric primitives
    prompt_lower = prompt.lower()
    if "sphere" in prompt_lower:
        mesh_obj = trimesh.creation.icosphere(subdivisions=3, radius=resolution)
    elif "cylinder" in prompt_lower:
        mesh_obj = trimesh.creation.cylinder(radius=resolution / 2, height=resolution)
    elif "cone" in prompt_lower:
        mesh_obj = trimesh.creation.cone(radius=resolution / 2, height=resolution)
    else:
        mesh_obj = trimesh.creation.box(extents=(resolution, resolution, resolution))

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
