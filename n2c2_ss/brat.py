"""
Parse and serialize the BRAT annotation format.  Lightweight, only tackling the
n2c2 subset of BRAT.

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Sequence, Optional
import contextlib
import re


class BratParseError(Exception):
    """
    Error while parsing a line from a BRAT .ann file.
    """


@dataclass(frozen=True, order=True)
class Span:
    """
    Represents a span of text as begin, end offsets.
    """
    begin_inclusive: int
    end_exclusive: int

    def __post_init__(self):
        if self.begin_inclusive < 0:
            raise ValueError(f"invalid begin_inclusive value '{self.begin_inclusive}'")
        if self.end_exclusive < 0:
            raise ValueError(f"invalid end_exclusive value '{self.end_exclusive}'")
        if self.end_exclusive < self.begin_inclusive:
            raise ValueError(f"end_exclusive '{self.end_exclusive}' cannot precede begin_inclusive '{self.begin_inclusive}'")

    def as_tuple(self):
        return (self.begin_inclusive, self.end_exclusive)

    def __repr__(self):
        return f"({self.begin_inclusive}, {self.end_exclusive})"


def _validate_id(type_, id_):
    if type_ is None:
        return bool(re.match(rf"^[TREAMN#][0-9]+$", id_))  # checks whether its a well-formed BRAT ID of any type
    return bool(re.match(rf"^{type_}[0-9]+$", id_))  # checks whether it's a well-formed BRAT ID of the expected type


def _contains_whitespace(s):
    return bool(re.match(r".*\s", s))


@dataclass
class TAnnotation:
    """
    Textbound annotation.
    """
    id_: str
    type_: str
    spans: Sequence[Span]
    text: str

    def __post_init__(self):
        if not _validate_id("T", self.id_):
            raise ValueError(f"Invalid TAnnotation identifier '{self.id_}'")
        if _contains_whitespace(self.type_):
            raise ValueError(f"TAnnotation type '{self.type_}' cannot contain whitespace")
        if len(self.text.splitlines()) > 1:
            raise ValueError(f"TAnnotation text cannot contain newlines")  # don't print text due to privacy concerns

        spans = [x.as_tuple() for x in self.spans]
        if spans != sorted(spans):
            raise ValueError(f"TAnnotation spans not sorted: {self.spans}")

        p, q = None, None
        for begin_inclusive, end_exclusive in spans:
            if p is None:
                p, q = begin_inclusive, end_exclusive
            else:
                if begin_inclusive != p and begin_inclusive < q or begin_inclusive == p and end_exclusive != p:
                    raise ValueError(f"TAnnotation spans '{spans}' cannot overlap")

        # n2c2 brat_scoring code requires that all spans before the last span in a multi-
        # span annotation ends on a space character.
        if len(spans) > 1:
            for span in spans[:-1]:
                text_span_end_exclusive = span[1] - spans[0][0]  # end-exclusive relative to the start of this annotation's covered text
                if text_span_end_exclusive > len(self.text):
                    raise ValueError(f"span {span} in spans {spans} too long for text with length {len(self.text)}: '{self.text}'")  # TODO: not sure this is correct; what if the original text has a lot of whitespace that the span should cover.  That can get compressed in BRAT's text column?
                if self.text[text_span_end_exclusive] != " ":
                    raise ValueError(f"span {span} in spans {spans} ends in a non-space character (this is not allowed by brat_scoring) for text: '{self.text}'")

    @staticmethod
    def parse(line):
        """
        Parses one row taken from an .ann file as a TAnnotation.  Raises BratParseError
        if there is an error parsing the string `line`.  Does minimal, local valiation.
        """
        line = line.strip(" \n\r")  # not removing tabs as perhaps some of our functions rely on the tab separator to correctly parse a line?
        m = re.match(r"^(?P<id>T[0-9]+)\t(?P<type>\S+) (?P<spans>[0-9 ;]+)\t(?P<text>.*)$", line)
        if not m:
            raise BratParseError("Error parsing TAnnotation")
        spans_str = m.group("spans")
        spans = []
        for span_str in spans_str.split(";"):
            n = re.match(r"^([0-9]+) ([0-9]+)$", span_str)
            if not n:
                raise BratParseError(f"Error parsing span string '{spans_str}' in TAnnotation")
            spans.append(Span(int(n.group(1)), int(n.group(2))))

        return TAnnotation(m.group("id"), m.group("type"), spans, m.group("text"))

    def serialize(self):
        span_str = ";".join(f"{x.begin_inclusive} {x.end_exclusive}" for x in self.spans)
        return f"{self.id_}\t{self.type_} {span_str}\t{self.text}"


@dataclass
class AAnnotation:
    """
    Attribute annotation.
    """
    id_: str
    name: str
    target_id: str
    value: Optional[str]

    def __post_init__(self):
        if not _validate_id("A", self.id_):
            raise ValueError(f"Invalid AAnnotation identifier '{self.id_}'")
        if _contains_whitespace(self.name):
            raise ValueError(f"AAnnotation name '{self.name}' cannot contain whitespace")
        if not _validate_id(None, self.target_id):  # we'll accept all possible identifier types here, other parts of processing can fail on annotation types we don't support
            raise ValueError(f"Invalid AAnnotation target identifier '{self.target_id}'")

    @staticmethod
    def parse(line):
        """
        Parses one row taken from an .ann file as a AAnnotation.  Raises BratParseError
        if there is an error parsing the string `line`.  Does minimal, local valiation.
        """
        line = line.strip(" \n\r")  # not removing tabs as perhaps some of our functions rely on the tab separator to correctly parse a line?
        m = re.match(r"^(?P<id>A[0-9]+)\t(?P<name>\S+) (?P<target_id>[^ ]+)(?: (?P<value>.*))?$", line)
        if not m:
            raise BratParseError("Error parsing AAnnotation")
        return AAnnotation(m.group("id"), m.group("name"), m.group("target_id"), m.group("value"))

    def serialize(self):
        if self.value is None:
            return f"{self.id_}\t{self.name} {self.target_id}"
        return f"{self.id_}\t{self.name} {self.target_id} {self.value}"


@dataclass(frozen=True)
class EArgument:
    """
    Event arguments (the triggering annotation is associated with these argument
    annotations).
    """
    role: str
    id_: str

    def __post_init__(self):
        if _contains_whitespace(self.role):
            raise ValueError(f"EArgument role '{self.role}' cannot contain whitespace")
        if not _validate_id(None, self.id_):
            raise ValueError(f"Invalid EArgument role identifier '{self.id_}'")

    def __repr__(self):
        return f"{self.role}:{self.id_}"


@dataclass
class EAnnotation:
    """
    Event annotation.
    """
    id_: str
    type_: str
    trigger_id: str
    arguments: Sequence[EArgument]  # will be an empty sequence if there are no arguments

    def __post_init__(self):
        if not _validate_id("E", self.id_):
            raise ValueError(f"Invalid EAnnotation identifier '{self.id_}'")
        if _contains_whitespace(self.type_):
            raise ValueError(f"EAnnotation type '{self.type_}' cannot contain whitespace")
        if not _validate_id(None, self.trigger_id):  # we'll accept all possible identifier types here, other parts of processing can fail on annotation types we don't support
            raise ValueError(f"Invalid EAnnotation target identifier '{self.trigger_id}'")

    @staticmethod
    def parse(line):
        """
        Parses one row taken from an .ann file as a EAnnotation.  Raises BratParseError
        if there is an error parsing the string `line`.  Does minimal, local valiation.
        """
        line = line.strip(" \n\r")  # not removing tabs as perhaps some of our functions rely on the tab separator to correctly parse a line?
        m = re.match(r"^(?P<id>E[0-9]+)\t(?P<type>[^:]+):(?P<trigger_id>[^:]+)(?: (?P<arguments>.*))?$", line)
        if not m:
            raise BratParseError("Error parsing EAnnotation")
        arguments = []
        argument_str = m.group("arguments")
        if argument_str:
            for argument_str in argument_str.split():
                arguments.append(EArgument(*argument_str.split(":")))
        return EAnnotation(m.group("id"), m.group("type"), m.group("trigger_id"), arguments)  # pylint: disable=too-many-function-args

    def serialize(self):
        if self.arguments:
            arguments_str = " ".join(f"{x.role}:{x.id_}" for x in self.arguments)
            return f"{self.id_}\t{self.type_}:{self.trigger_id} {arguments_str}"
        return f"{self.id_}\t{self.type_}:{self.trigger_id}"


def validate_references(id_to_annotation_dict):
    """
    Checks that all annotation ID references (e.g., "A1", "T7", "E5") point to
    defined annotations.
    """
    for annotation in id_to_annotation_dict.values():
        if annotation.id_.startswith("T"):
            pass  # No references for TAnnotations
        elif annotation.id_.startswith("A"):
            if annotation.target_id not in id_to_annotation_dict:
                return False
        elif annotation.id_.startswith("E"):
            if annotation.trigger_id not in id_to_annotation_dict:
                return False
            for x in annotation.arguments:
                if x.id_ not in id_to_annotation_dict:
                    return False
        else:
            raise KeyError(f"Unsupported annotation type with identifier '{annotation.id_}'")
    return True


def parse(f):
    """
    Parses lines from `f`, which is a file-like object or a filename pathlib.Path or
    string.  Returns a dict from the internal annotation IDs to annotation objects.

    One can parse an .ann file that's been read into a string like:

        import io
        parse(io.StringIO("..."))
    """
    id_to_annotation_dict = {}  # relies on maintaining the order of insertion to round-trip
    with contextlib.ExitStack() as stack:
        f = f if callable(getattr(f, "readlines", None)) else stack.enter_context(open(f, "r", encoding="utf-8"))
        for line_num, line in enumerate(f, start=1):
            try:
                annotation_obj = parse_line(line)
                id_to_annotation_dict[annotation_obj.id_] = annotation_obj
            except Exception as e:
                raise BratParseError(f"Parsing error in line {line_num}") from e
    return id_to_annotation_dict


def parse_line(line):
    """
    Parses a single line, what would normally be taken from an .ann file.
    """
    line = line.strip(" \n\r")  # not removing tabs as perhaps some of our functions rely on the tab separator to correctly parse a line?
    if not line:
        raise BratParseError(f"Illegal empty line")

    if line.startswith("T"):
        return TAnnotation.parse(line)
    if line.startswith("A"):
        return AAnnotation.parse(line)
    if line.startswith("E"):
        return EAnnotation.parse(line)
    raise BratParseError(f"Parsing annotation type of '{line[0]}' is not currently supported")


class AnnotationIdBuilder:
    """
    Helper that keeps track of the next free number that can be used for creating
    BRAT annotation IDs.
    """
    def __init__(self):
        self.counter = Counter()

    def __call__(self, type_char):
        type_char = type_char.upper()
        if type_char not in ("T", "A", "E"):
            raise KeyError(f"unsupported annotation type '{type_char}'")
        self.counter[type_char] += 1
        return f"{type_char}{self.counter[type_char]}"


def serialize(f, id_to_annotation_dict):
    """
    Write the annotation objects in `id_to_annotation_dict` to `f`, where `f` is a
    file-like object or is a filename pathlib.Path or string.
    """
    with contextlib.ExitStack() as stack:
        f = f if callable(getattr(f, "write", None)) else stack.enter_context(open(f, "w", encoding="utf-8"))
        for line in gen_serialization(id_to_annotation_dict):
            f.write(line)
            f.write("\n")


def gen_serialization(id_to_annotation_dict):
    """
    Generates lines that would form a valid BRAT .ann file were they joined with
    newlines.
    """
    yield from (x.serialize() for x in id_to_annotation_dict.values())
