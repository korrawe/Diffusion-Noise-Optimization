import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Patch old numpy
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.unicode_
np.str = np.str_


def convert(input_path: Path, output_path: Path):
    with input_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    np.savez(output_path, **data)
    return data


SMPL_PATH = Path(__file__).parents[1] / "body_models" / "smpl"
INPUT_PATH = SMPL_PATH / "SMPL_NEUTRAL.pkl"
OUTPUT_PATH = SMPL_PATH / "SMPL_NEUTRAL.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("input", nargs="*", default=[INPUT_PATH])

    args = parser.parse_args()

    input_paths = args.input
    if len(input_paths) != 1:
        print("Please provide a single path as input")
        sys.exit(1)
    input_path = Path(input_paths[0])
    if not input_path.exists():
        print("File not found:", input_path, file=sys.stderr)
        sys.exit(1)
    convert(input_path, args.out)

    data = np.load(args.out, allow_pickle=True)
    print("Converted data:")
    for k, v in data.items():
        print(
            f"\t{f'[{k}]':20s} => {f'<{v.dtype}>':>10s} array of shape {str(v.shape):16s}"
        )


if __name__ == "__main__":
    main()
    convert(INPUT_PATH, OUTPUT_PATH)
