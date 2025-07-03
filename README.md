# soma
Generate 3D meshes from text prompts or input images via a tiny CLI.
Meshes are generated using an adapted form of the open source [Roblox Cube](https://github.com/Roblox/cube) engine.

Example usage:

```bash
python mesh.py --prompt "flying red dragon"
```

By default the mesh is saved via SSH under
`~/soma/output/<uuid>.obj`. Provide the `--output` argument to override
the destination.

The `--resolution` flag controls the Cube decoder's resolution base and
defaults to `9.0`.
