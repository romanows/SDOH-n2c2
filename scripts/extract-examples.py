#!/usr/bin/env python3
"""
Converts the n2c2 track 2 social determinants of health BRAT-annotated data into
a structured representation that can be predicted by a seq2seq model.

For input/source text like:

    "Patient lives with husband and is a nonsmoker"

The corresponding structured sequence that this script produces would be
something like:

    <LivingStatus> lives <StatusTime> [current] lives <TypeLiving> [with_family] with her husband
    {AND}
    <StatusTime> [none] nonsmoker [Tobacco] nonsmoker

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from collections import deque
from functools import reduce
from n2c2_ss import brat
from n2c2_ss.structured_representation import (
    Argument,
    DEFAULT_CONJUNCTION_TOKEN,
    DEFAULT_SUBTYPE_MARKUP,
    DEFAULT_TYPE_MARKUP,
    Event,
    parse_event_sequence,
    serialize_event_sequence,
    strip_spans,
)
from n2c2_ss.utils import path_normalizer, normalize_for_t5_tokenizer
import argparse
import json
import logging
import os.path
import re
import textwrap


logger = logging.getLogger(__name__)


def extract(txt_filename, ann_filename):
    """
    Extracts text and individual events from the given pairs of .txt .ann files.
    Returns a sequence of `Event` objects, which are themselves essentially sequences
    of `Argument` objects.  Events and event arguments are ordered left-to-right
    according to their begin_offset_inclusive, with ties broken by their
    end_offset_inclusive.
    """
    with open(txt_filename, "r", encoding="utf-8") as f:
        document_text = f.read()

    with open(ann_filename, "r", encoding="utf-8") as f:
        try:
            id_to_annotation_dict = brat.parse(f)
        except Exception as e:
            raise Exception(f"when parsing '{ann_filename}'") from e

    if not brat.validate_references(id_to_annotation_dict):
        raise Exception(f"Invalid annotation file '{ann_filename}'")

    target_id_to_attribute_dict = {v.target_id: v for k, v in id_to_annotation_dict.items() if k.upper().startswith("A")}

    # TAnnotations are mentions/entities/argument-types.  E.g., "Tobacco".  They are
    # associated with begin_inclusive, end_exclusive spans.
    #
    # EAnnotations define the events, what we're calling "events" due to the
    # similarity with an in-house system's approach.  They are comprised of
    # TAnnotations, where the first TAnnotation is the "trigger".  All edges in an
    # event go from the trigger to the subsequent TAnnotations.
    #
    # AAnnotations are subtypes on the TAnnotations.  These are concepts that capture
    # things like whether the event happened in the past, the present, or is being
    # asserted to never have happened at all.

    # Convert the extracted data into a format that's closer to the structured
    # sequence with which we're going to train the model.
    events = []
    event_annotations = [v for k, v in id_to_annotation_dict.items() if k.upper().startswith("E")]
    for event_annotation in event_annotations:
        arguments = []
        for textbound_id in [event_annotation.trigger_id, *(x.id_ for x in event_annotation.arguments)]:

            # The event argument `type_` seems to be a shortened version of the full
            # textbound.type_ label; we see "Status" instead of "StatusTime" or
            # "StatusEmploy"; also there is a type_ = "Status2" for some reason?  These thus
            # seem to be redundant and we won't use them.

            textbound = id_to_annotation_dict.get(textbound_id)
            spans = [(span.begin_inclusive, span.end_exclusive) for span in textbound.spans if span.begin_inclusive != span.end_exclusive]  # discard zero-width spans
            argument_span = reduce(lambda x, y: (min(x[0], y[0]), max(x[1], y[1])), spans)

            attribute = target_id_to_attribute_dict.get(textbound_id)
            attribute_value = None if attribute is None else attribute.value  # there is a .type_ but I think it's always completely determined by the mention/entity/trigger?

            token_str = normalize_for_t5_tokenizer(" ".join(document_text[begin_inclusive:end_exclusive] for begin_inclusive, end_exclusive in spans))

            # TODO: probably replace PHI-tagged spans with a special token to reduce sequence length?
            # TODO: replace spans of tokens with the first and last token to reduce sequence length?
            # TODO: often there is redundant info, where the LivingStatus and LivingType are the same, maybe we can fuse those into one LivingStatusType to save sequence length?

            arguments.append(Argument(argument_span, textbound.type_, attribute_value, sorted(spans), token_str))
            arguments.sort()

        event_span = reduce(lambda x, y: (min(x[0], y.span[0]), max(x[1], y.span[1])), arguments, (float("inf"), float("-inf")))
        events.append(Event(event_span, arguments))

    events.sort()
    return document_text, events


def get_n_event_examples(text, events, n, is_loose_only=False):
    """
    Gets examples that cover all sequences of `n` events that don't overlap any other
    event from `events`.  Idea is to explode one "document" into several examples from
    the presumably easy example containing one event all the way up to the presumably
    harder containing all possible events.

    When possible, returns a "tight" text string and a "loose" text string.  The
    tight text string is the span of text covered from the first event arguments
    through the last.  The loose text string has additional tokens to the left
    and/or to the right such that these extra characters are not covered by any
    other event arguments from `events`.  This is also meant to provide a bit more
    helpful variation in the training data.
    """
    # TODO: this is going to need a unittest to make sure we've got it correct.

    if not events:
        return [("loose", text, [])]

    examples = []
    event_deque = deque()
    for event in events:
        event_deque.append(event)
        if len(event_deque) == n:
            example = list(event_deque)
            example_begin_inclusive = example[0].span[0]
            example_end_exclusive = example[-1].span[1]
            is_overlap = False  # not all overlaps are automatically disqualifying TODO: deal with false positives.  "X Y Z" with events "X Y" and "X Z", only "X Z" should be banned.  Same with "X Y Z" and events "X Y" and "Y Z", it's fine for those events to share "Y" since the corresponding text spans will cover the full event content without including the corresponding held-out event evidence.
            loose_begin_inclusive = 0
            loose_end_exclusive = len(text)
            for other_event in events:
                if other_event not in example:
                    if other_event.span[-1] < example_begin_inclusive:
                        loose_begin_inclusive = max(loose_begin_inclusive, other_event.span[-1])
                    if example_end_exclusive < other_event.span[0]:
                        loose_end_exclusive = min(loose_end_exclusive, other_event.span[0])
                    if other_event.span[-1] <= example_begin_inclusive or example_end_exclusive < other_event.span[0]:
                        continue  # clear separation
                    logger.debug("span overlap with '%s': %s", other_event, example)
                    is_overlap = True
            if not is_overlap:
                tight_example_text = text[example_begin_inclusive:example_end_exclusive].strip()

                # TODO: if there is no event to the left or right, we shouldn't remove junk from
                # the left or right respectively.
                loose_example_text = re.sub(r"([,.:;?!'\"]|\s)*$", "", re.sub(r"^([,.:;?!'\"]|\s)*", "", text[loose_begin_inclusive:loose_end_exclusive]))

                if not is_loose_only:
                    examples.append(("tight", tight_example_text, example))

                if is_loose_only or tight_example_text != loose_example_text and loose_example_text not in tight_example_text:  # the latter condition if we deleted punctuation that would otherwise be in the tight version
                    examples.append(("loose", loose_example_text, example))
            event_deque.popleft()
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--explode-document", action="store_true", help=textwrap.dedent("""\
        Creates examples out of all sub-sequences of non-overlapping events in a
        document, with varying amounts of associated text as the input/source text. This
        creates many (redundant) examples out of one document that may "ease" a machine
        learning model's adaptation to a full-document-length sequence of events. In
        general, this is meant to be set when creating training datasets and should not
        be set when creating test datasets."""))

    parser.add_argument("--type-prefix", default=DEFAULT_TYPE_MARKUP[0], help=textwrap.dedent("""\
        String used to prefix the event "type" in the serialized, structured output"""))

    parser.add_argument("--type-suffix", default=DEFAULT_TYPE_MARKUP[1], help=textwrap.dedent("""\
        String used to suffix the event "type" in the serialized, structured output"""))

    parser.add_argument("--subtype-prefix", default=DEFAULT_SUBTYPE_MARKUP[0], help=textwrap.dedent("""\
        String used to prefix the event "subtype" in the serialized, structured
        output"""))

    parser.add_argument("--subtype-suffix", default=DEFAULT_SUBTYPE_MARKUP[1], help=textwrap.dedent("""\
        String used to suffix the event "subtype" in the serialized, structured
        output"""))

    parser.add_argument("--conjunction", default=DEFAULT_CONJUNCTION_TOKEN, help=textwrap.dedent("""\
        String placed between two events"""))

    parser.add_argument("annotation_dir", type=path_normalizer, help=textwrap.dedent("""\
        Directory path that holds .txt and .ann annotation files"""))

    parser.add_argument("output_filename", type=path_normalizer, help=textwrap.dedent("""\
        Main output will be written to this filename.  Additional output may be written
        to variations of this filename."""))

    parser.add_argument("--verbose", action="store_true", help="Print info-level log messages")
    parser.add_argument("--debug", action="store_true", help="Print debugging-level log messages")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING)

    if not args.annotation_dir.is_dir():
        raise ValueError(f"Given annotation dir is not a directory: {args.annotation_dir}")

    args.output_filename.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # lock down output directory permissions so other users can't see protected data
    special_tokens_filename = args.output_filename.parent / f"{args.output_filename.stem}.special-tokens.txt"

    type_markup = (args.type_prefix, args.type_suffix)
    subtype_markup = (args.subtype_prefix, args.subtype_suffix)

    special_tokens = set()
    with open(args.output_filename, "w", encoding="utf-8") as fw:
        logger.info("writing examples to '%s'", args.output_filename)
        for ann_filename in sorted(args.annotation_dir.glob("*.ann")):
            txt_filename = ann_filename.with_suffix(".txt")
            if not txt_filename.is_file():
                logger.error("no text document is available at '%s' for standoff annotation file at '%s', skipping", txt_filename, ann_filename)
                continue
            logger.info("processing document %s", txt_filename)
            text, events = extract(txt_filename, ann_filename)
            for n in range(min(1, len(events)) if args.explode_document else len(events), len(events) + 1):
                for example_text_bound, example_text, example_events in get_n_event_examples(text, events, n, is_loose_only=not args.explode_document):
                    events_str = serialize_event_sequence(example_events, special_tokens, args.conjunction, type_markup=type_markup, subtype_markup=subtype_markup)
                    example_text = normalize_for_t5_tokenizer(example_text)

                    stripped_example_events = strip_spans(example_events)
                    round_trip_example_events, is_error = parse_event_sequence(events_str)
                    if is_error:
                        logger.error("Round-trip parsing error reported when parsing document %s", txt_filename)
                    if stripped_example_events != round_trip_example_events:
                        raise ValueError(f"Round-trip event sequence serialization and re-parsing failed:\nORIGINAL: {stripped_example_events}\nREPARSED: {round_trip_example_events}")

                    example_data = {
                        "translation": {
                            "n": n,
                            "text_bound": example_text_bound,
                            "document_id": os.path.splitext(os.path.basename(txt_filename))[0],
                            "en": example_text,
                            "n2c2": events_str,
                        }
                    }
                    print(json.dumps(example_data), file=fw)

    with open(special_tokens_filename, "w", encoding="utf-8") as fw:
        logger.info("writing special tokens to '%s'", special_tokens_filename)
        for special_token in sorted(special_tokens):
            print(special_token, file=fw)


if __name__ == "__main__":
    main()
