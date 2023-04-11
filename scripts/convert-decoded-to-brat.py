#!/usr/bin/env python3
"""
Converts the structured sequence model output to BRAT annotations that can be
compared to the reference BRAT annotations with brat_scoring.

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from collections import Counter
from n2c2_ss.brat import AnnotationIdBuilder, TAnnotation, Span, AAnnotation, EAnnotation, EArgument, serialize
from n2c2_ss.structured_representation import parse_event_sequence, recover_spans, NoFeasibleSpans
from n2c2_ss.utils import path_normalizer
import argparse
import json
import logging
import re
import textwrap


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("annotation_dir", type=path_normalizer, help=textwrap.dedent("""\
        Directory path that holds the reference .txt and .ann annotation files.  Needed
        so we can load the original text documents."""))

    parser.add_argument("decoded_filename", type=path_normalizer, help=textwrap.dedent("""\
        Filename that holds decoded output"""))

    parser.add_argument("output_dir", type=path_normalizer, help=textwrap.dedent("""\
        Directory to which BRAT documents created from the decoded output will be
        written"""))

    parser.add_argument("--verbose", action="store_true", help="Print info-level log messages")
    parser.add_argument("--debug", action="store_true", help="Print debugging-level log messages")

    args = parser.parse_args()

    if not args.annotation_dir.is_dir():
        raise ValueError(f"Given annotation dir is not a directory: {args.annotation_dir}")

    if not args.decoded_filename.is_file():
        raise ValueError(f"Given decoded filename is not a file: {args.decoded_filename}")

    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)  # lock down output directory permissions so other users can't see protected data

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING)

    with open(args.decoded_filename, "r", encoding="utf-8") as fr:
        for line in fr:
            d = json.loads(line)
            document_id = d["translation"]["document_id"]

            logger.info("processing document %s", document_id)

            preprocessed_text =  d["translation"]["en"]  # not the original text, can't be used to determine offsets since, at a minimum, whitespace may have been altered
            events, is_error = parse_event_sequence(d["translation"]["hyp"])

            original_text_path = args.annotation_dir / f"{document_id}.txt"
            logger.debug("loading original document text from: %s", original_text_path)
            with open(original_text_path, "r", encoding="utf-8") as frtxt:
                original_text = frtxt.read()

            if is_error:
                logger.warning("Parsing error reported when parsing document %s", document_id)

            log_hallucinated_tokens(preprocessed_text, events)

            recover_span_events = None
            try:
                recover_span_events = recover_spans(original_text, events)
            except NoFeasibleSpans:
                logger.exception("Could not recover spans in document %s", document_id)
            except:
                logger.exception("Could not recover spans in document %s", document_id)  # other errors

            text_tokens_spans = [(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r"\S+", original_text)]

            annotations = []
            try:
                if recover_span_events and recover_span_events[0] and recover_span_events[0][0]:
                    annotations = convert_events_to_brat(document_id, recover_span_events if recover_span_events else events, mock_span=text_tokens_spans[0][1])
            except Exception as e:
                logger.exception("error converting events in document '%s'", document_id)
                raise e

            # Copy .txt file to destination
            with open(args.output_dir / f"{document_id}.txt", "w", encoding="utf-8") as fwtxt:
                fwtxt.write(original_text)

            # Write .ann file
            id_to_annotation_dict = {x.id_: x for x in annotations}
            serialize(args.output_dir / f"{document_id}.ann", id_to_annotation_dict)


def log_hallucinated_tokens(preprocessed_text, events):
    """
    Logs hallucinated tokens, tokens that are in the model output but which are not
    exact matches for subtrings of the source text.  Informative, a risk with this
    style of model, but trivially remediated by restricting the decoder output, if
    it becomes necessary.
    """
    hallucinated_tokens = set()
    for event in events:
        for ea in event.arguments:
            for token in ea.token_str.split():
                if token not in preprocessed_text:
                    hallucinated_tokens.add(token)
    if hallucinated_tokens:
        logger.warning("Hallucinated tokens '%s' in text:\n%s\n", ", ".join(sorted(hallucinated_tokens)), preprocessed_text)


def convert_events_to_brat(document_id, events, mock_span=None):
    """
    Converts the event sequence representation back to n2c2-style BRAT annotations.
    Highly dependent on the `extract-examples.py` conventions that convert the
    original BRAT annotations to the event sequence.
    """
    aidb = AnnotationIdBuilder()
    annotations = []
    for event in events:

        e_label_counter = Counter()
        event_type = None
        event_t_id = None
        event_arguments = []
        for ea in event.arguments:
            this_t_id = aidb("T")

            if ea.token_spans:
                token_spans = [Span(p, q) for p, q in ea.token_spans]
            else:
                logger.warning("Used mock span for %s", ea)
                token_spans = [Span(0, 0) if mock_span is None else Span(*mock_span)]

            try:
                annotations.append(TAnnotation(this_t_id, ea.type, token_spans, ea.token_str))  # TODO: this copy uses the span information
            except:
                logger.exception("Failed to create TAnnotation, skipping")
                continue

            if ea.subtype:
                annotations.append(AAnnotation(aidb("A"), f"{ea.type}Val", this_t_id, ea.subtype))

            trigger_types = set(["Alcohol", "Drug", "Employment", "LivingStatus", "Tobacco"])
            if ea.type in trigger_types:
                if event_type:
                    logger.warning("More than one event trigger found in document '%s' event: %s", document_id, event)
                event_type = ea.type
                event_t_id = this_t_id
            else:
                type_to_e_label_dict = {"StatusEmploy": "Status", "StatusTime": "Status", "TypeLiving": "Type"}
                e_label = type_to_e_label_dict.get(ea.type, ea.type)
                e_label_counter[e_label] += 1
                e_label_count = e_label_counter[e_label]  # will let us add "Status2"-type argument labels
                e_label = f"{e_label}{e_label_count}" if e_label_count > 1 else e_label
                event_arguments.append(EArgument(e_label, this_t_id))

        if event_type:
            annotations.append(EAnnotation(aidb("E"), event_type, event_t_id, event_arguments))
        else:
            # Maybe fail gracefully by making one of the arguments a trigger?  Could we get
            # some credit for the non-trigger arguments or is it better to drop the event,
            # treating it as noise?  Not sure what the best strategy is here.
            #
            # Maybe we should attempt to coerce a trigger?  One case it'd be hard with the
            # word "illicits" but in another case, there are annotations about "beers".  Maybe
            # we could get this to go away a bit more by augmenting the training data.
            logger.warning("Event in document '%s' has no trigger annotation: %s", document_id, event)
            x = event_arguments.pop(0)
            annotations.append(EAnnotation(aidb("E"), x.role, x.id_, event_arguments))

    return annotations


if __name__ == "__main__":
    main()
