"""
Helpers for the structured sequence representation of social determinants of
health events.

BRAT annotations are converted to this structured sequence representation,
models are trained on it, and then the representation is converted back to BRAT.

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from collections import defaultdict, namedtuple
from n2c2_ss.utils import postprocess_t5_decoded
from ortools.sat.python import cp_model
from typing import Sequence, Set
import logging
import re


logger = logging.getLogger(__name__)


# Spans are used when extracting information from annotations so that events and
# arguments can be ordered left-to-right.  This field is `None` in event
# sequences that are parsed from model output rather than manual annotation.
# Similarly, token_spans are None in Argument objects created when parsing model
# output.
Event = namedtuple("Event", ["span", "arguments"])
Argument = namedtuple("Argument", ["span", "type", "subtype", "token_spans", "token_str"])


DEFAULT_TYPE_MARKUP = ("«", "")
DEFAULT_SUBTYPE_MARKUP = ("»", "")
DEFAULT_CONJUNCTION_TOKEN = "€AND"


def serialize_event_sequence(events: Sequence[Event], special_tokens: Set[str] = None, conjunction_token=DEFAULT_CONJUNCTION_TOKEN, type_markup=DEFAULT_TYPE_MARKUP, subtype_markup=DEFAULT_SUBTYPE_MARKUP) -> str:
    """
    Serializes a sequence of `Event` objects to our special string representation.

    `special_tokens` is a `set` to which the serialization of special tokens like
    TAnnotations/textbounds, AAnnotations/attributes, and event conjunctions are
    added as they are encountered.
    """
    special_tokens.add(conjunction_token)
    return f" {conjunction_token} ".join(serialize_event(event, special_tokens, type_markup=type_markup, subtype_markup=subtype_markup) for event in events)


def serialize_event(event: Event, special_tokens: Set[str] = None, type_markup=DEFAULT_TYPE_MARKUP, subtype_markup=DEFAULT_SUBTYPE_MARKUP) -> str:
    """
    Serializes a `Event` object to our special string representation.

    `special_tokens` is a `set` to which the serialization of special tokens like
    TAnnotations/textbounds, AAnnotations/attributes, and event conjunctions are
    added as they are encountered.
    """
    event_str = []

    type_prefix, type_suffix = type_markup
    subtype_prefix, subtype_suffix = subtype_markup

    for argument in event.arguments:
        type_ = f"{type_prefix}{argument.type}{type_suffix}"
        subtype = f"{subtype_prefix}{argument.subtype}{subtype_suffix}"
        token_text = argument.token_str

        if special_tokens is not None:
            special_tokens.add(type_)
            if argument.subtype is not None:
                special_tokens.add(subtype)

        if argument.subtype is not None:
            event_str.append(f"{type_} {subtype} {token_text}")
        else:
            event_str.append(f"{type_} {token_text}")  # many of the arguments don't have subtypes, let's try reducing sequence length by allowing it to be optional

    return " ".join(event_str)


def strip_spans(events):
    """
    Replaces the `span` field values in `Event` and `Argument` with `None`.  Makes
    it easier to compare event sequences created from BRAT (with span information)
    to event sequences parsed from model output (with no span information).
    """
    stripped_events = []
    for event in events:
        stripped_arguments = []
        for argument in event.arguments:
            stripped_arguments.append(Argument(None, argument.type, argument.subtype, None, argument.token_str))
        stripped_events.append(Event(None, stripped_arguments))
    return stripped_events


def _build_model_assignments(text_tokens_spans, events):
    """
    Creates variables for each event argument token that correspond to starting
    offsets in the original text.  Creates a model with hard left-to-right ordering
    constraints at the event, argument, and token level.
    """
    assignments = defaultdict(lambda: defaultdict(dict))
    model = cp_model.CpModel()

    eat_to_event_token_dict = {}  # may go away, this is a way we've hacked in support for substrings but it could stand to have a more principled/robust implementation

    # TODO: support graceful failure when not all token assignments can be satisfied?

    # Create the variables
    for e, event in enumerate(events):
        for a, argument in enumerate(event.arguments):
            for t, event_token in enumerate(postprocess_t5_decoded(argument.token_str).split()):  # TODO: note this iffy call to postprocess_t5_decoded, may want to handle this better or perform it outside of this function
                # TODO: has the effect of ignoring event tokens that don't exactly match a text
                # token; we should handle this because otherwise left-to-right constraints won't
                # work transitively through to other event arguments and events

                # TODO:, because we have annotated tokens like "(-)" but whitespace-separated
                # tokens like "(-)," that don't quite match, we're going to try changing the exact
                # match to a substring match in the lookup below.
                admissable_token_indexes = [o for o, (text_token, _) in enumerate(text_tokens_spans) if event_token in text_token]  # tokens in event arguments must map to exact matches in the original text
                if not admissable_token_indexes:
                    # Even when constrained decoding is active, this can happen when other special
                    # symbols (like our newline symbols) are only partially output by the system or
                    # when the system output gets cutoff due to output sequence length restrictions
                    # midway through producing a special token.
                    logger.warning("Token %s not found in text; not currently supported, skipping token", event_token)
                    assignments[e][a][t] = None
                    eat_to_event_token_dict[(e, a, t)] = None
                else:
                    token_var = model.NewIntVarFromDomain(cp_model.Domain.FromValues(admissable_token_indexes), f"e{e}a{a}t{t}")
                    assignments[e][a][t] = token_var
                    eat_to_event_token_dict[(e, a, t)] = event_token

    # Note: once the possibility of "no mapping" was allowed for a token (and thus a
    # argument and thus an event) it because complex to track the first and "latest"
    # token in a token sequence, event argument, and event.  Splitting it up is
    # slightly less efficient but we're paying that cost to keep the code more
    # understandable.

    # Add constraints for all tokens
    for ed in assignments.values():  # minimize distance between first and last token within an event argument
        for ad in ed.values():
            last_t = None
            for t in ad.values():
                if t is not None:
                    if last_t is not None:
                        model.Add(last_t < t)  # tokens within a event argument must be strictly left-to-right-ordered
                    last_t = t

    # Constrain the relationship between event argument in the same event
    for ed in assignments.values():  # minimize distance between first and last token within a event argument
        last_t = None
        for ad in ed.values():
            t = next((t for t in ad.values() if t is not None), None)  # get first token variable in the event argument
            if t is not None:
                if last_t is not None:
                    model.Add(last_t <= t)  # first token of every event argument in the same event are ordered left-to-right
                last_t = t

    # Constrain the relationship between events
    last_t = None
    for ed in assignments.values():  # minimize distance between first and last token within a event argument
        f = next((t for ad in ed.values() for t in ad.values() if t is not None), None)
        if f is not None:
            if last_t is not None:
                model.Add(last_t <= t)  # first token of the first event argument in different events are ordered left-to-right
            last_t = t

    return assignments, model, eat_to_event_token_dict


def _build_objective(assignments):
    # TODO: should/could we force tokens to "clump" in the case "W X Y Y Y Z" such
    # that we'd prefer "W X Y ... Z" over "W X ... Y ... Z" or even "W X ... Y Z"?
    # Does taking the product of all pairwise distances with skips do this: "W X" "X Y" ... "W Y" ...

    objective = 0
    for ed in assignments.values():  # minimize distance between first and last token within an event argument
        for ad in ed.values():
            if len(ad) > 1:
                cpvars = list(ad.values())
                objective += cpvars[-1] - cpvars[0]

    for ed in assignments.values():  # minimize distance between the first token in the first event argument of an event and the last token of the last argument
        if len(ed) > 1:
            arguments = list(ed.values())
            objective += list(arguments[-1].values())[-1] - list(arguments[0].values())[0]
    return objective


def _recover_token_span(text_tokens_spans, eat_to_event_token_dict, e, a, t, token_idx):
    original_token, (begin_inclusive, _) = text_tokens_spans[token_idx]
    event_token = eat_to_event_token_dict[(e, a, t)]
    i = begin_inclusive + original_token.index(event_token)  # dancing around here because we allow our event tokens to be substrings of the original sentence's whitespace-separated tokens
    token_span = (i, i + len(event_token))
    return token_span


def _postprocess_for_brat_scoring(text, token_spans):
    if not text or not token_spans:
        return token_spans, []

    # Combine adjacent spans that don't cross a newline boundary
    reversed_token_spans = list(reversed(token_spans))

    fixed_token_spans = []
    fixed_token_spans.append(reversed_token_spans.pop())

    while reversed_token_spans:
        prev_token_span = fixed_token_spans[-1]
        next_token_span = reversed_token_spans.pop()

        prev_next_between_text = text[prev_token_span[1]:next_token_span[0]]
        if re.match(r"^[ ]*\Z", prev_next_between_text):
            logger.debug("combining adjacent spans for '%s'", text[prev_token_span[0]:next_token_span[1]])
            fixed_token_spans[-1] = (prev_token_span[0], next_token_span[1])
        else:
            fixed_token_spans.append(next_token_span)

    token_spans = fixed_token_spans

    # Still need to combine token spans that don't cross a newline boundary (this is a
    # brat_scoring requirement).  Break spans that cross a newline.  Include an empty
    # span and an extra " " in the token_str for extra newlines, also a brat_scoring
    # quirk.
#    fixed_token_spans = []
#    token_strs = []
#    reversed_token_spans = list(reversed(token_spans))
#    fixed_token_spans.append(reversed_token_spans.pop())
#    while len(reversed_token_spans):
#        x = reversed_token_spans.pop()
#        begin_inclusive = fixed_token_spans[-1][0]
#        text_subset = text[begin_inclusive:x[1]]
#        if "\n" not in text_subset:
#            fixed_token_spans[-1] = (begin_inclusive, x[1])
#        elif text_subset.startswith("\n"):
## TODO: blargh, this is complicated
#            fixed_token_spans.append((begin_inclusive, begin_inclusive))  # append the empty span for what should be a subsequent newline, brat_scoring quirk
#            reversed_token_spans.append((begin_inclusive + 1, )
#        else:
#            fixed_token_spans[-1] = (begin_inclusive, begin_inclusive + text_subset.index("\n"))
#            if x[1] - fixed_token_spans[-1][1] > 0:
#                reversed_token_spans.append()
#
#        token_strs.append(text[token_span[0]:token_span[1]])

    token_strs = [text[begin_inclusive:end_exclusive] for begin_inclusive, end_exclusive in token_spans]

    return token_spans, token_strs


class NoFeasibleSpans(Exception):
    """
    When `recover_spans` fails to find a feasible mapping from event tokens to spans.
    """


def recover_spans(text, events):
    """
    Recovers the `span` fields in `Event` and `Argument`.  Not guaranteed to recover
    the original spans when there are multiple copies of a token.

    Depends upon the `events` following conventions:
        * Events appear in left-to-right order with respect to the `text` according
            to their first event argument's first token
        * Event arguments in the same event appear in left-to-right order according to
            their first token
        * Tokens in the same event argument appear in left-to-right order
        * All tokens appear in `text`
    """
    # TODO: IDK if we need to allow for mid-token offsets.  Seems to become more
    # onerous for single character "tokens".  If supported, we would probably want to
    # include a bias towards left, right, and bilateral whitespace.
    text_tokens_spans = [(m.group(0), (m.start(), m.end())) for m in re.finditer(r"\S+", text)]

    assignments, model, eat_to_event_token_dict = _build_model_assignments(text_tokens_spans, events)

    objective = _build_objective(assignments)

    model.Minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.linearization_level = 0
    solver.parameters.enumerate_all_solutions = True

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise NoFeasibleSpans()

    # TODO: print(f"is optimal = {status == cp_model.OPTIMAL} and is feasible = {status == cp_model.FEASIBLE} with cost = {solver.ObjectiveValue()}")

    recovered_span_events = []
    for (e, ed), event in zip(assignments.items(), events):
        recovered_span_event_arguments = []
        for (a, ad), argument in zip(ed.items(), event.arguments):
            token_spans = []
            for t, cpvar in ad.items():
                if cpvar is None:
                    continue  # can happen in rare cases when the system output token was not found in the text
                token_idx = solver.Value(cpvar)
                try:
                    token_span = _recover_token_span(text_tokens_spans, eat_to_event_token_dict, e, a, t, token_idx)
                except Exception as exception:
                    raise IndexError(f"Bad token index {token_idx} for ({e}, {a}, {t})") from exception
                token_spans.append(token_span)

            token_spans, token_strs = _postprocess_for_brat_scoring(text, token_spans)
            if token_spans:
                recovered_span_event_arguments.append(Argument((token_spans[0][0], token_spans[-1][1]), argument.type, argument.subtype, token_spans, " ".join(token_strs)))

        event_begin_inclusive, event_end_exclusive = None, None
        if event_begin_inclusive is None:
            event_begin_inclusive = token_spans[0][0]
        event_end_exclusive = token_spans[-1][1]

        if recovered_span_event_arguments:
            recovered_span_events.append(Event((event_begin_inclusive, event_end_exclusive), recovered_span_event_arguments))
    return recovered_span_events


def parse_event_sequence(events_str: str, conjunction_token=DEFAULT_CONJUNCTION_TOKEN, type_markup=DEFAULT_TYPE_MARKUP, subtype_markup=DEFAULT_SUBTYPE_MARKUP):
    """
    Parses a serialized sequence of events back into a sequence of `Event` objects,
    which themselves contain a sequence of `Argument` objects.  Fails gracefully
    when ill-formed or illegal strings are encountered and returns a flag to
    indicate that some parsing problem occurred.

    Currently does not support anything but the default markup for event argument
    types and subtypes, nor anything but the default conjunction.
    """
    events, is_error = [], False
    events_str = re.sub(r"\s+", " ", events_str).strip()
    event_tokens = []

    def process_event_tokens():
        nonlocal events, is_error, event_tokens
        event_str = " ".join(event_tokens)  # recreate the event string to keep the function definitions more consistent and easier to test
        event, is_event_error = parse_event(event_str, type_markup=type_markup, subtype_markup=subtype_markup)
        if event is not None:
            events.append(event)
        is_error |= is_event_error
        event_tokens = []

    for events_token in events_str.split():
        if events_token != conjunction_token:
            event_tokens.append(events_token)
        else:
            process_event_tokens()
    if event_tokens:
        process_event_tokens()

    return events, is_error


def parse_event(event_str: str, type_markup=DEFAULT_TYPE_MARKUP, subtype_markup=DEFAULT_SUBTYPE_MARKUP):
    """
    Parses an event, which is a sequence of one or more event arguments.  Event
    arguments start with a type, optionally contain one subtype, and contain at
    least one token.

    This attempts to fail gracefully by ignoring tokens before the first event
    argument name.  It ignores repeated arguments.  It returns an event with no
    tokens if event argument names occur before tokens are observed in the input. If
    an argument is encountered when collecting event argument tokens, the collection
    is ended and all content through the next event argument name is ignored.

    Note that this doesn't check to make sure types or subtypes are valid or can
    appear in the combinations they appear in.  It doesn't check that tokens are in
    the text and appear in the left-to-right order in which we expect.  All that
    must come from subsequent validation and alignment steps.
    """
    is_error = False
    error_explanations = []  # may be developed in the future, may be refactored to include more identifying data optionally injected into these functions to explain things
    arguments = []
    ea_type, ea_subtype, ea_tokens = None, None, []

    type_prefix_re, type_suffix_re = map(re.escape, type_markup)
    subtype_prefix_re, subtype_suffix_re = map(re.escape, subtype_markup)

    for event_token in event_str.split():

        type_match = re.match(rf"^{type_prefix_re}(.+){type_suffix_re}$", event_token)
        subtype_match = re.match(rf"^{subtype_prefix_re}(.+){subtype_suffix_re}$", event_token) if type_match is None else None

        if not ea_type:
            # Looking for a "type"-type token to start an event argument
            if not type_match:
                is_error = True
                error_explanations.append("Non-name content encountered when expecting a name, ignoring")
            else:
                ea_type = type_match.group(1)
            continue

        # At this point, we're building the argument and tokens for an event argument and
        # looking for the next event argument to start.

        if type_match:
            # Seeing another name means that we end the previous event argument and start a
            # new one.
            if not ea_tokens:
                is_error = True
                error_explanations.append("Event argument missing token string, continuing")
            arguments.append(Argument(None, ea_type, ea_subtype, None, " ".join(ea_tokens)))
            ea_type, ea_subtype, ea_tokens = type_match.group(1), None, []
            continue

        if subtype_match:
            if ea_subtype and not ea_tokens:
                is_error = True
                error_explanations.append("Event argument has more than one subtype in a row, ignoring")
            elif ea_subtype and ea_tokens:
                is_error = True
                error_explanations.append("Event argument encountered an unexpected subtype when collecting tokens, stopping collection and starting looking for an argument type")
                arguments.append(Argument(None, ea_type, ea_subtype, None, " ".join(ea_tokens)))
                ea_type, ea_subtype, ea_tokens = None, None, []
            else:
                ea_subtype = subtype_match.group(1)
            continue

        # If the event_token isn't a type nor a subtype, it's a regular token.
        ea_tokens.append(event_token)

    if ea_type:
        # add the event argument we were working on when the string ended
        arguments.append(Argument(None, ea_type, ea_subtype, None, " ".join(ea_tokens)))

    return Event(None, arguments), is_error
