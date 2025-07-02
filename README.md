# soma
Generate 3D meshes from text prompts or input images via a tiny CLI.

Example usage:

```bash
python mesh.py --prompt "flying red dragon"
```

By default the mesh is saved via SSH to `192.168.1.140` under
`~/soma/output/<uuid>.obj`. Provide the `--output` argument to override
the destination.
