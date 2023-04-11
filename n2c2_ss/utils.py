"""
Misc helper functions.

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
import pathlib
import re


def path_normalizer(filename: str) -> pathlib.Path:
    """
    Resolves user dir and symlinks to an absolute file path.  Meant to be used as
    the value for argparse's `type` parameter.
    """
    return pathlib.Path(filename).expanduser().resolve()


# T5 doesn't have a few common characters in its vocabulary, so we'll add them.
#
# It's easy to add them as whole tokens, but we lose information about whitespace,
# which complicates mapping model output back to the original string input.  For
# example, since ">" is not in the T5 vocabulary, ">25 years" and "> 25 years"
# would be tokenized in an identical way and would appear as "> 25 years" in the
# model output.
#
# It doesn't appear as easy to modify the sentencepiece portion of the tokenizer,
# which is what would allow us to recover spacing on either side of the
# characters.  Instead, we'll introduce a special non-space token "®" that will
# denote the lack of a space.  The formerly ambiguous cases will appear like "> ®
# 25 years" and "> 25 years" in the model output.  On the flip side,
# postprocess_t5_decoded() will change these to ">25 years" and "> 25 years".
#
# Another way to do this is to create four special tokens ">", ">®", "®>", and
# "®>®".  This has the upside of reducing the sequence length.  However, it also
# divides up knowledge between these four cases, which may or may not be a problem
# depending on how many training examples we have and how they are distributed.
_NOT_A_SPACE = "®"
_ORIGINAL_TO_ESCAPED_STRINGS_DICT = {
    # "{}" has a special meaning when used in the masking-objective pretraining
    # scripts so we'll replace these with a special symbol.
    "{": "€C",
    "}": "€c",

    # These characters are fine to use as-is, including them here just makes it easier
    # to add the _NOT_A_SPACE and to add them to the tokenizer.
    "<": "€>",
    ">": ">",  # this is actually in the vocabulary, it's just "<" that isn't in the vocab for some odd reason
    "~": "€t",

    # Whitespace is currently being stripped from the beginning and ending of each
    # line.  This is questionable as it messes up formatting / indentation semantics.
    # Nevertheless, it means that we don't need to consider adding a _NOT_A_SPACE on
    # either side of the escaped newline character.
    "\n": "€n",
}


def get_additional_t5_tokens():
    """
    Returns a list of special strings that should be added as tokens to the T5
    tokenizer.
    """
    return list(_ORIGINAL_TO_ESCAPED_STRINGS_DICT.values()) + [_NOT_A_SPACE]


def strip_lines(text):
    """
    Strips whitespace from the beginning and ending of each line in `text`. Also
    converts Windows line-endings to Linux line-endings.
    """
    return "\n".join(line.strip() for line in text.strip().splitlines())


def normalize_for_t5_tokenizer(text):
    """
    Minimal text preprocessing to deal with whitespace and characters that aren't in
    the T5 vocabulary.

    This is lossy due to the removal of leading and trailing whitespace from each
    line, and from collapsing runs of whitespace.

    This is meant to process the raw text before running the T5 tokenizer.  The T5
    tokenizer should have the output of `get_additional_t5_tokens` added.  After
    having obtained T5 model output and run it through the tokenizer decoding, the
    complementary function `postprocess_t5_decoded` should be used to recover the
    original characters and their surrounding whitespace, to the extent that this is
    possible.
    """
    text = strip_lines(text)
    text = re.sub(r"\n\n\n+", "\n\n", text)  # collapse multiple empty lines to just one empty line

    # Newlines aren't associated with _NOT_A_SPACE since we know that each line has
    # its leading and trailing whitespace stripped.  Handle these specially so that
    # unnecessary _NOT_A_SPACE are not inserted.
    newline_replacement = _ORIGINAL_TO_ESCAPED_STRINGS_DICT.get("\n")
    if newline_replacement is not None:
        text = text.replace("\n", f" {newline_replacement} ")

    # Add _NOT_A_SPACE on the left-side of all special tokens.  Adding only on the
    # left side makes it a bit easier to use a regex implementation without having
    # problems with runs of special tokens.
    for original in _ORIGINAL_TO_ESCAPED_STRINGS_DICT:
        if original == "\n":
            continue  # already handled
        escaped = re.escape(original)
        text = re.sub(rf"(\S)(?={escaped})", rf"\1 {_NOT_A_SPACE} ", text, re.UNICODE)

    # Add _NOT_A_SPACE on the right-side of all special tokens.
    for original in _ORIGINAL_TO_ESCAPED_STRINGS_DICT:
        if original == "\n":
            continue  # already handled
        escaped = re.escape(original)
        text = re.sub(rf"({escaped})(?=\S)", rf"\1 {_NOT_A_SPACE} ", text, re.UNICODE)

    # Do the replacements
    for original, replacement in _ORIGINAL_TO_ESCAPED_STRINGS_DICT.items():
        if original == "\n":
            continue  # already handled
        text = text.replace(original, replacement)

    # And finally, clean up any runs of whitespace that we may have introduced
    text = re.sub(r"\s+", " ", text, re.UNICODE).strip()
    return text


def postprocess_t5_decoded(text):
    """
    Replaces special symbols added by `normalize_for_t5_tokenizer` with their original symbols.
    Deals with the space vs. not-a-space between special symbols and surrounding
    text.  Won't recover the original text exactly, since it can't recover any
    leading, trailing, or internal runs of whitespace.
    """
    text = re.sub(rf"\s*{re.escape(_NOT_A_SPACE)}\s*", "", text, re.UNICODE)
    for original, replacement in _ORIGINAL_TO_ESCAPED_STRINGS_DICT.items():
        text = text.replace(replacement, original)

    text = strip_lines(text)
    text = re.sub(r"\n\n\n+", "\n\n", text)  # collapse multiple empty lines to just one empty line
    return text
