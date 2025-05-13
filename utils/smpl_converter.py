import argparse
import inspect
import pickle
import sys
from pathlib import Path
from typing import Any

import torch

# Patch for chumpy with Python > 3.10
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import chumpy
import numpy as np
import scipy
import scipy.sparse


def convert_types(key: str, value: Any):
    if isinstance(value, (scipy.sparse.csc_matrix, scipy.sparse.csc.csc_matrix)):
        return value.toarray()
    if isinstance(value, chumpy.Ch):
        return value.r  # Get raw numpy array
    return value


def convert(input_path: Path, output_path: Path):
    with input_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    def convert_types_wrapper(k, v):
        value = convert_types(k, v)
        if type(v) is not type(value):
            print("Converted type:", type(v), "->", type(value))
        return value

    data = {k: convert_types_wrapper(k, v) for k, v in data.items()}
    np.savez(output_path, **data)
    return data


def compare(path1: Path, path2: Path):
    data1 = np.load(path1, allow_pickle=True)
    data2 = np.load(path2, allow_pickle=True)

    equal = True
    numpy_comparable = True

    if not (keys1 := set(data1.keys())) == (keys2 := set(data2.keys())):
        print("Keys differ")
        print(keys1.difference(keys2), "are in file 1 but not in file 2")
        print(keys2.difference(keys1), "are in file 2 but not in file 1")
        equal = False

    for k in keys1.intersection(keys2):
        v1, v2 = data1[k], data2[k]
        if (t1 := type(v1)) is not (t2 := type(v2)):
            print(f"{f'[{k}]':>20s}: Types differ:  {t1} in file 1, but {t2} in file 2")
            equal = False
            continue
        if hasattr(v1, "shape") and hasattr(v2, "shape") and v1.shape != v2.shape:
            print(f"{f'[{k}]':>20s}: Shapes differ: {v1.shape} in file 1, but {v2.shape} in file 2")
            equal = False
            numpy_comparable = False
        if hasattr(v1, "dtype") and hasattr(v2, "dtype") and v1.dtype != v2.dtype:
            print(f"{f'[{k}]':>20s}: Dtypes differ: {v1.dtype} in file 1, but {v2.dtype} in file 2")
            equal = False
            numpy_comparable = False
        is_numpy_like = isinstance(v1, (np.ndarray, chumpy.Ch, torch.Tensor, scipy.sparse.csc_matrix))
        if is_numpy_like and numpy_comparable:
            if np.issubdtype(v1.dtype, np.number):
                if not np.allclose(v1, v2):
                    print(f"{f'[{k}]':>20s}: Arrays differ: Mean difference of {(v1 - v2).abs().mean()}")
                    equal = False
            else:
                if not np.equal(v1, v2):
                    print(f"{f'[{k}]':>20s}: Arrays differ")
                    equal = False
        if not is_numpy_like and v1 != v2:
            # Use strict equality (problem?)
            print(f"{f'[{k}]':>20s}: Values differ: {v1} in file 1, but {v2} in file 2")
            equal = False

    return equal


SMPL_PATH = Path(__file__).parents[1] / "body_models" / "smpl"
INPUT_PATH = SMPL_PATH / "SMPL_NEUTRAL.pkl"
OUTPUT_PATH = SMPL_PATH / "SMPL_NEUTRAL.npz"


def convert_main(args: argparse.Namespace):
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
        print(f"\t{f'[{k}]':20s} => {f'<{v.dtype}>':>10s} array of shape {str(v.shape):16s}")


def compare_main(args: argparse.Namespace):
    equal = compare(args.files[0], args.files[1])
    if equal:
        print("File contents are identical!")
    else:
        print("File contents differ.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser("convert", help="Convert .pkl to .npz and convert all to numpy arrays")
    convert_parser.add_argument("-o", "--out", type=Path, default=OUTPUT_PATH)
    convert_parser.add_argument("input", nargs="*", default=[INPUT_PATH])

    compare_parser = subparsers.add_parser("compare", help="Compare two .npz files for numerical equality")
    compare_parser.add_argument(
        "files",
        nargs=2,
        type=Path,
        help="Input files",
    )

    args = parser.parse_args()
    if args.command == "compare":
        compare_main(args)
    if args.command == "convert":
        convert_main(args)
