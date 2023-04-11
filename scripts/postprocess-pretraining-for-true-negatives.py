#!/usr/bin/env python3
"""
Writes the output of `extract-pretraining.py` to a text file so a researcher can
modify it.  The intention is for the resarcher to cut out SDoH content so what's
left are true negatives, wrt to the SDoH task.  A separate script will convert
this to the format output by `extract-examples.py`.

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from n2c2_ss.utils import path_normalizer, postprocess_t5_decoded
import argparse
import logging
import textwrap


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--num-examples", type=int, help=textwrap.dedent("""\
        Number of examples to read from the top of the pretraining file"""))

    parser.add_argument("--verbose", action="store_true", help="Print info-level log messages")
    parser.add_argument("--debug", action="store_true", help="Print debugging-level log messages")

    parser.add_argument("input", type=path_normalizer, help=textwrap.dedent("""\
        Path to the extracted pretraining .txt file"""))

    parser.add_argument("output_filename", type=path_normalizer, help=textwrap.dedent("""\
        Unnormalized examples will be written to this .txt file, with examples separated
        by a delimiter"""))

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING)

    args.output_filename.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # lock down output directory permissions so other users can't see protected data

    document_delimiter = "€" * 79

    with open(args.input, "r", encoding="utf-8") as f, open(args.output_filename, "w", encoding="utf-8") as fw:
        for line_idx, line in enumerate(f):
            if line_idx >= args.num_examples:
                break
            fw.write(f"\n{postprocess_t5_decoded(line)}\n")
            fw.write(f"\n{document_delimiter}\n")


if __name__ == "__main__":
    main()
