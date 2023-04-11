#!/usr/bin/env python3
"""
Extracts text from the MIMIC III corpus or from text files for use with the
Huggingface T5 pretraining script `run_t5_mlm_flax.py`.

The document data is processed in a similar way as `extract-examples.py` which
notably currently collapses whitespace and converts newlines to "\\n" strings to
keep them explicit.

Not completely sure about this but I think if we give `run_t5_mlm_flax.py` text
that is longer than --max-length it will truncate the remainder of the text?  So
this script will chunk documents longer than --max-length.

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from n2c2_ss.utils import path_normalizer, normalize_for_t5_tokenizer
import argparse
import csv
import gzip
import logging
import re
import textwrap


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model", default="google/t5-v1_1-base", help=textwrap.dedent("""\
        Huggingface model or checkpoint from which to load tokenizer that is used to
        calculate document chunk sizes.  E.g., 'google/t5-v1_1-base'. """))

    parser.add_argument("--max-length", default=512, type=int, help=textwrap.dedent("""\
        Maximum length of document text to include in each example"""))

    parser.add_argument("--stride", default=256, type=int, help=textwrap.dedent("""\
        Number of tokens that overlap in sliding window of size --max-length over
        document text."""))

    parser.add_argument("--split", required=False, type=str, help=textwrap.dedent("""\
        Which range of lines to read and process into output.  Format is  '--split 2/10'
        which means to create the second split out of ten total splits. The output file
        will have a suffix appended to it like ".split02_10"."""))

    parser.add_argument("--split-num-docs", required=False, type=int, help=textwrap.dedent("""\
        Form splits with respect to this number of documents.  Allows skipping one pass
        through the data to count the documents when the number of documents is already
        known."""))

    parser.add_argument("--verbose", action="store_true", help="Print info-level log messages")
    parser.add_argument("--debug", action="store_true", help="Print debugging-level log messages")

    parser.add_argument("input", type=path_normalizer, help=textwrap.dedent("""\
        Path to the MIMIC III NOTEEVENTS.csv.gz file or a directory containing .txt
        files"""))

    parser.add_argument("output_filename", type=path_normalizer, help=textwrap.dedent("""\
        Examples will be written to this .txt file, with one example per line"""))

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING)

    if args.max_length < 1:
        raise ValueError(f"--max-length must be a positive integer")

    if args.stride < 1:
        raise ValueError(f"--stride must be a positive integer")

    mimic_csv_gz = None
    text_dir = None
    if args.input:
        if args.input.is_file():
            mimic_csv_gz = args.input
        elif args.input.is_dir():
            text_dir = args.input
        else:
            raise ValueError(f"input is not a directory nor a file")

    args.output_filename.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # lock down output directory permissions so other users can't see protected data

    skip_begin_inclusive, skip_end_exclusive = None, None
    if args.split:
        split_num, num_splits = parse_split_string(args.split)
        split_suffix = get_split_suffix(split_num, num_splits)
        args.output_filename = args.output_filename.with_suffix(f"{args.output_filename.suffix}.{split_suffix}")

        logger.debug("Counting number of documents for splits")
        if mimic_csv_gz:
            documents_gen = range(args.split_num_docs) if args.split_num_docs else gen_mimic_documents(mimic_csv_gz)
        else:
            documents_gen = gen_text_filenames(text_dir)  # don't actually need the document content for this, just how many items we have to deal with
        num_documents, skip_begin_inclusive, skip_end_exclusive = get_split_offsets(documents_gen, split_num, num_splits)
        logger.debug("Split %s/%s for %s documents contains rows %s up to %s", split_num, num_splits, num_documents, skip_begin_inclusive, skip_end_exclusive)

    # Delay import because importing pytorch takes time.  This makes running --help
    # cheaper and also makes it easier for the user to debug certain problems with
    # their command-line usage.
    from transformers import T5Tokenizer  # pylint: disable=import-outside-toplevel
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    unk_token_replacement = "\\"  # T5 tokenizer doesn't have the backslash, so it'll map back to "<unk>" when we retokenize these chunks of text.
    if unk_token_replacement in tokenizer.get_vocab():
        raise ValueError("Backslash is in the tokenizer vocabulary, need to pick another token that the tokenizer maps to <unk>.")

    characters = set()
    num_junk_documents = 0
    num_processed_documents = 0
    num_chunks = 0
    with open(args.output_filename, "w", encoding="utf-8") as fw:
        logger.info("Writing output to '%s'", args.output_filename)
        for text_idx, text in enumerate(gen_mimic_documents(mimic_csv_gz) if mimic_csv_gz else gen_text_documents(text_dir)):
            if skip_begin_inclusive is not None and (text_idx < skip_begin_inclusive or text_idx >= skip_end_exclusive):
                continue
            if is_junk_mimic_document(text):
                num_junk_documents += 1
                continue
            logger.debug("Processing document %s", text_idx)
            text = normalize_for_t5_tokenizer(text)
            characters.update(text)
            num_processed_documents += 1
            for chunk in gen_document_chunks(tokenizer, args.max_length, args.stride, text, unk_token_replacement=unk_token_replacement):
                num_chunks += 1
                fw.write(chunk)
                fw.write("\n")

    logger.info("processed %s documents into %s chunks; ignored %s junk documents", num_processed_documents, num_chunks, num_junk_documents)

    for character in sorted(characters):
        ids = tokenizer(character).input_ids
        if len(ids) > 1 and ids[1] == tokenizer.unk_token_id:  # check length because whitespace gets stripped completely
            converted = character.encode(encoding="unicode_escape").decode("ascii")
            character = character if character == converted else converted
            logger.warning("Tokenizer does not know '%s'", character)


def is_junk_mimic_document(text):
    """
    Apparently the MIMIC III NOTEEVENTS have what looks like some binary format data
    in the "text" field.  Looking for the following keywords seems to be fairly
    reliable.
    """
    return any(x in text for x in ("bjbj", "DDefault", "VTable", "RTable"))


def parse_split_string(split_string):
    """
    Parses a string of the form "2/10" and returns the two components as `int`s.
    Checks that the split information is at least somewhat valid and raises
    exceptions if not.
    """
    err_msg = "Invalid split value '{}'; should be a value like '2/10' which would designate the second split out of a total of ten splits".format(split_string)
    try:
        m = re.match(r"^([0-9]+)/([0-9]+)$", split_string)
    except Exception as e:
        raise ValueError(err_msg) from e

    if not m:
        raise ValueError(err_msg)
    split_num, num_splits = map(int, (m.group(1), m.group(2)))
    if num_splits < 1 or split_num < 1 or split_num > num_splits:
        raise ValueError(err_msg)
    return split_num, num_splits


def get_split_suffix(split_num, num_splits):
    """
    Returns a string intended to be used as a filename suffix for split files, like
    "split02_10".
    """
    max_split_digits = len(str(num_splits))
    return f"split{split_num:0{max_split_digits}d}_{num_splits:0{max_split_digits}d}"


def get_split_offsets(gen, split_num, num_splits):
    """
    Determine the zero-based begin (inclusive) and end (exclusive) offsets of the
    split `split_num` out of the total number of splits `num_splits` for the items
    contained in the generator `gen`.  Exhausts the generator.
    """
    input_len = 0
    for _ in gen:
        input_len += 1
    chunk_size = input_len // num_splits
    skip_begin_inclusive = chunk_size * (split_num - 1)  # split 1 will start reading from input zero
    skip_end_exclusive = input_len if split_num == num_splits else chunk_size * split_num  # last split will take up the slack of extra examples
    return input_len, skip_begin_inclusive, skip_end_exclusive


def gen_mimic_documents(filename):
    """
    Generates the "TEXT" cell from each row in the `NOTEEVENTS.csv.gz` file that
    MIMIC III provides.
    """
    with gzip.open(filename, "rt") as f:
        csvreader = csv.reader(f)
        for row_num, row in enumerate(csvreader, start=1):
            if row_num == 1:
                try:
                    text_column_idx = row.index("TEXT")
                except:
                    raise IndexError(f"Could not find the header 'TEXT' in the NOTEEVENTS.csv.gz file '{filename}'")  # pylint: disable=raise-missing-from
            else:
                yield row[text_column_idx].strip()


def gen_text_filenames(dirname):
    """
    Generates the text filenames of each file in the `text_dir`.
    """
    yield from dirname.glob("*.txt")


def gen_text_documents(dirname):
    """
    Generates the text concent from each file in the `text_dir`.
    """
    for filename in gen_text_filenames(dirname):
        with open(filename, "r", encoding="utf-8") as f:
            yield f.read()


def gen_document_chunks(tokenizer, max_length, stride, text, unk_token_replacement="\\"):
    """
    Yields spans of text of approximately `max_length` tokens by running a sliding
    window over the document text.

    There's an argument for trying to start and end on line breaks, on newlines, or
    on tokens.  OTOH, there's also an argument for just chopping the stream of
    subwords wherever and letting the model sort it out.  Since the latter is much
    easier to do, we'll try that first.
    """
    def decode_keeping_unks(ids):
        decoded_strs = []
        subset = []
        for k, i in enumerate(ids):
            if i == tokenizer.unk_token_id:
                decoded_strs.append(tokenizer.decode(subset))
                if subset and tokenizer.convert_ids_to_tokens(subset[-1]).endswith("\u2581"):
                    decoded_strs.append(" ")  # adds an explicit space before the upcoming unk character
                decoded_strs.append(unk_token_replacement)
                if (k + 1) < len(ids) and tokenizer.convert_ids_to_tokens(ids[k + 1]).startswith("\u2581"):
                    decoded_strs.append(" ")  # adds an explicit space after the unk character
                subset = []
            elif i in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id):
                pass  # don't include these special tokens
            else:
                subset.append(i)
        if subset:
            decoded_strs.append(tokenizer.decode(subset))
        chunk = "".join(decoded_strs)
        return chunk

    # Originally used `tokenizer.truncate_sequences` to implement a kind of sliding
    # window a bit more efficiently but ran into a problem where one token ID would
    # sometimes round-trip into more than one token ID, making it hard to exactly fill
    # out each chunk to the max_length.

    tentative_max_length = max_length
    while True:
        t = tokenizer(text, truncation="longest_first", max_length=tentative_max_length, stride=stride, return_overflowing_tokens=True)
        decoded_chunk = decode_keeping_unks(t.input_ids)
        decoded_chunk_tokenized_len = len(tokenizer(decoded_chunk).input_ids)
        if max_length < decoded_chunk_tokenized_len:
            logger.warning("tokenized decoded chunk length %s exceeds max_length %s, reducing tentative_max_length %s and trying again", decoded_chunk_tokenized_len, max_length, tentative_max_length)
            tentative_max_length -= (decoded_chunk_tokenized_len - max_length)  # might overshoot but it's quite slow when the delta is large
        else:
            yield decoded_chunk
            if not t.overflowing_tokens:
                break
            text = decode_keeping_unks(t.overflowing_tokens)


if __name__ == "__main__":
    main()
