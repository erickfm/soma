import argparse
import uuid

from soma.generate import ensure_model, mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a 3D mesh from a text prompt")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate the mesh from")
    parser.add_argument("--resolution", type=float, default=9.0,
                        help="Base resolution of the mesh (default: 9.0)")
    parser.add_argument("--output", help="Output path for the mesh")
    return parser.parse_args()


def main():
    args = parse_args()

    output = args.output
    if output is None:
        output = f"ssh://192.168.1.140/~/soma/output/{uuid.uuid4().hex}.obj"

    model_path = ensure_model()
    mesh(args.prompt, resolution=args.resolution, output=output, model_path=model_path)


if __name__ == "__main__":
    main()
