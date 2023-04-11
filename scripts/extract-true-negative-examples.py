#!/usr/bin/env python3
"""
Extracts examples from the manually modified output of `postprocess-pretraining-
for-true-negatives.py`.

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from n2c2_ss.utils import path_normalizer, normalize_for_t5_tokenizer
import argparse
import json
import logging
import re
import textwrap


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--verbose", action="store_true", help="Print info-level log messages")
    parser.add_argument("--debug", action="store_true", help="Print debugging-level log messages")

    parser.add_argument("input", type=path_normalizer, help=textwrap.dedent("""\
        Path to the input file"""))

    parser.add_argument("output_filename", type=path_normalizer, help=textwrap.dedent("""\
        Examples file to write"""))

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING)

    args.output_filename.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # lock down output directory permissions so other users can't see protected data

    document_delimiter = "€" * 79

    with open(args.input, "r", encoding="utf-8") as f, open(args.output_filename, "w", encoding="utf-8") as fw:
        document = []
        example_idx = 0
        for line in f:
            if line.strip() != document_delimiter:
                document.append(line)
            else:
                document = "".join(document).strip()
                if document:
                    example_data = {
                        "translation": {
                            "n": 0,
                            "text_bound": "loose",
                            "document_id": f"tn{example_idx:05d}",
                            "en": normalize_for_t5_tokenizer(document),
                            "n2c2": "",
                        }
                    }
                    print(json.dumps(example_data), file=fw)

                    paragraphs = re.split(r"\n\n", document)
                    for paragraph in paragraphs[1:]:
                        paragraph = paragraph.strip()
                        if paragraph:
                            example_data = {
                                "translation": {
                                    "n": 0,
                                    "text_bound": "loose",
                                    "document_id": f"tn{example_idx:05d}",
                                    "en": normalize_for_t5_tokenizer(paragraph),
                                    "n2c2": "",
                                }
                            }
                            print(json.dumps(example_data), file=fw)
                document = []
                example_idx += 1


if __name__ == "__main__":
    main()
