#!/usr/bin/env python3
"""
Converts a Flax model checkpoint (usually the output of the run_t5_mlm_flax.py
script) to PyTorch format.

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from n2c2_ss.utils import path_normalizer
import argparse
import transformers


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("flax_dir", type=path_normalizer, help="Input flax checkpoint dir")
    parser.add_argument("pytorch_dir", type=path_normalizer, help="Output pytorch checkpoint dir")
    args = parser.parse_args()

    if not args.flax_dir.is_dir():
        raise IOError("Not a directory: {args.flax_dir}")

    args.pytorch_dir.mkdir(parents=True, exist_ok=True)

    pt_model = transformers.T5ForConditionalGeneration.from_pretrained(args.flax_dir, from_flax=True)
    pt_model.save_pretrained(args.pytorch_dir)


if __name__ == "__main__":
    main()
