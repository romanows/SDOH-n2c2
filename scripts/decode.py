#!/usr/bin/env python3
"""
Decodes a fine-tuned T5 checkpoint on the output of extract-examples.py.

Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from n2c2_ss.utils import path_normalizer, normalize_for_t5_tokenizer, postprocess_t5_decoded
from typing import List
import argparse
import enum
import json
import logging
import textwrap
import time


logger = logging.getLogger(__name__)


# FIXME: we should not be hardcoding these since it's possible to configure the
# structural markup differently
_ARGUMENT_TYPE_TO_SUBTYPES_DICT = {
    "«Alcohol": [],
    "«Amount": [],
    "«Drug": [],
    "«Duration": [],
    "«Employment": [],
    "«Frequency": [],
    "«History": [],
    "«LivingStatus": [],
    "«Method": [],
    "«StatusEmploy": [
        "»employed",
        "»unemployed",
        "»retired",
        "»on_disability",
        "»student",
        "»homemaker"],

    # SubstanceUse triggers are compatible with {None, Current, Past} but LivingStatus
    # is compatible with {Current, Past, Future}.  We'll combine the sets because
    # otherwise we'd have to potentially backtrack if we produce this before producing
    # the SubstanceUse/LivingStatus trigger.
    "«StatusTime": [
        "»none",
        "»current",
        "»past",
        "»future"],

    "«Tobacco": [],
    "«Type": [],
    "«TypeLiving": [
        "»alone",
        "»with_family",
        "»with_others",
        "»homeless"],
}


_CONJ = "€AND"


IncrementalState = enum.Enum("IncrementalState", "START TYPE_OR_EOS TYPE TYPE_LIVING_OR_TOKEN SUBTYPE TOKEN TOKEN_OR_TYPE_OR_CONJ_OR_EOS")
IS = IncrementalState  # alias


def validate_incremental_output(tokenizer, input_text, candidate_output_str, is_eos):
    """
    Validates the `candidate_output_str` with respect to the structural markup and
    the `input_text`.
    """
    current_state = IS.START
    next_subtypes = []
    tokens = []
    if is_eos and not candidate_output_str.endswith(f" {tokenizer.eos_token}"):
        # The </s> token shows up concatenated with the last token, which interferes with
        # our simple method of tokenization.  We'll add a space.
        candidate_output_str = candidate_output_str[:-len(tokenizer.eos_token)] + " " + tokenizer.eos_token
    candidate_output_tokens = candidate_output_str.split()

    def normalize(text):
        return normalize_for_t5_tokenizer(postprocess_t5_decoded(text).replace("\n", " "))

    input_text_normalized_for_matching = normalize(input_text)

    for token_num, output_token in enumerate(candidate_output_tokens, start=1):

        if current_state == IS.START:
            if output_token != tokenizer.pad_token:
                logger.error("In state %s but current token is not the expected '%s' in the candidate string '%s'", current_state, tokenizer.pad_token, candidate_output_str)
                raise Exception()  # this should never happen
            current_state = IS.TYPE_OR_EOS
            continue

        if current_state == IS.TYPE_OR_EOS:
            if is_eos and token_num == len(candidate_output_tokens):
                return True  # done once we hit the EOS
            current_state = IS.TYPE
            # Keep going, we'll process it in one of the blocks below

        if current_state == IS.TYPE_LIVING_OR_TOKEN:
            if output_token.startswith("«Type"):
                # if we're still working on a string that starts with "«Type" after having entered
                # into this state, we're almost certainly working on "«TypeLiving".  Only
                # exception is if that string is in the text and our model wants to produce it as
                # a token.  So unlikely that we'll just reroute the handling back to the TYPE
                # code.
                current_state = IS.TYPE
            else:
                # if we're not working on a TYPE token after having entered this state, we know
                # we're looking at a candidate TOKEN.
                current_state = IS.TOKEN
            # Keep going, we'll process it in one of the blocks below

        if current_state == IS.TOKEN_OR_TYPE_OR_CONJ_OR_EOS:
            if is_eos and token_num == len(candidate_output_tokens):
                return True

            if output_token == _CONJ:
                current_state = IS.TYPE
                tokens = []  # clear tokens as we transition to the next event/event-argument
                continue

            if _CONJ.startswith(output_token):
                tokens = []  # clear tokens as we transition to the next event/event-argument
                continue

            if output_token.startswith("«"):
                # Cheating here a bit, it's possible that we are actually producing tokens and
                # going to produce one starting with the "«" character.  This is so unlikely that
                # we can ignore it in this version of the code.
                current_state = IS.TYPE
                tokens = []  # clear tokens as we transition to the next event/event-argument
            else:
                current_state = IS.TOKEN
            # Keep going, we'll process it in one of the blocks below

        if current_state == IS.TYPE:
            if is_eos and token_num == len(candidate_output_tokens):
                logger.warning("In state %s where it is not permissable to end decoding", current_state)
                return False

            if output_token == "«Type":
                # Very annoying special case.  This could be a whole TYPE or it could be the
                # prefix to "«TypeLiving".  We should change this to "«LivingType" or
                # "«TypeLiving»".  The latter would be the more general solution although it costs
                # an extra subword token in sequence length.
                current_state = IS.TYPE_LIVING_OR_TOKEN
            elif output_token in _ARGUMENT_TYPE_TO_SUBTYPES_DICT:  # exact match
                next_subtypes = _ARGUMENT_TYPE_TO_SUBTYPES_DICT[output_token]
                current_state = IS.SUBTYPE if next_subtypes else IS.TOKEN
            elif any(x.startswith(output_token) for x in _ARGUMENT_TYPE_TO_SUBTYPES_DICT):
                pass # stay in the same state to fill out the type
            else:
                logger.warning("In state %s but candidate string is invalid '%s'", current_state, candidate_output_str)
                return False
            continue

        if current_state == IS.SUBTYPE:
            if is_eos and token_num == len(candidate_output_tokens):
                logger.warning("In state %s where it is not permissable to end decoding", current_state)
                return False

            if output_token in next_subtypes:  # exact match
                current_state = IS.TOKEN
            elif any(x.startswith(output_token) for x in next_subtypes):
                pass # stay in the same state to fill out the subtype
            else:
                logger.warning("In state %s but candidate string is invalid '%s'", current_state, candidate_output_str)
                return False
            continue

        if current_state == IS.TOKEN:
            if is_eos and token_num == len(candidate_output_tokens):
                logger.warning("In state %s where it is not permissable to end decoding", current_state)
                return False

            if output_token in input_text:
                current_state = IS.TOKEN_OR_TYPE_OR_CONJ_OR_EOS
                tokens.append(output_token)
                if len(tokens) > 1:
                    # We could enforce contiguous span here or just left-to-right ordering.  Given
                    # that we're not planning to implement any kind of beam search or backtracking,
                    # and the n2c2 task specifies contiguous spans, it's probably more important to
                    # force contiguous spans.  What this might look like is a substring/suffix to
                    # start, then full tokens, then a prefix to end.  I think we ignore escaped
                    # whitespace like newlines and tabs.
                    #
                    # Easiest way to hack this in may be to be inefficient.  Normalize the input text,
                    # form a string with the output tokens and normalize it too, and then just search
                    # for a substring match.  Normalizing the text means normalizing whitespace and
                    # converting the escaped whitespace back into whitespace (but not converting other
                    # escaped characters because they may be in prefix form).
                    tokens_normalized = normalize(" ".join(tokens))
                    if tokens_normalized not in input_text_normalized_for_matching:
                        logger.warning("In state %s but candidate token string is not a substring of the original text '%s'", current_state, tokens_normalized)
                        return False
            else:
                logger.warning("In state %s but candidate string is invalid '%s'", current_state, candidate_output_str)
                return False
            continue

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--max-length", type=int, default=512, help="Max decoding length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    parser.add_argument("--checkpoint", required=True, type=path_normalizer, help=textwrap.dedent("""\
        Path to directory containing a Huggingface T5 checkpoint"""))

    parser.add_argument("--examples", required=True, type=path_normalizer, help=textwrap.dedent("""\
        Path to evaluation .json output from extract-examples.py or a .txt file with examples on lines"""))

    parser.add_argument("--document-id", required=False, type=str, help=textwrap.dedent("""\
        Document IDs to process, comma-delimited if more than one; otherwise, all
        document IDs will be processed"""))

    parser.add_argument("--special-tokens", required=False, type=path_normalizer, help=textwrap.dedent("""\
        Path to .special-tokens.txt file with the tokens used for event types, values,
        conjunctions, etc."""))

    parser.add_argument("--no-constrain-decoding", action="store_true", help="No constraints enforced during decoding, not even NoBadSubwordsLogitsProcessor; takes precedence over --constrain-decoding")
    parser.add_argument("--constrain-decoding", action="store_true", help="Enforces more constraints during decoding but is much slower")

    parser.add_argument("--verbose", action="store_true", help="Print info-level log messages")
    parser.add_argument("--debug", action="store_true", help="Print debugging-level log messages")

    parser.add_argument("output_filename", type=path_normalizer, help=textwrap.dedent("""\
        Raw decoded output, written in a json-lines format, suitable for passing down to
        tools that will convert this to brat_scoring format for official scoring"""))

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING)

    if args.max_length < 1:
        raise ValueError(f"--max-length must be a positive integer")

    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be a positive integer")

    if not args.examples.is_file():
        raise ValueError(f"--examples is not a file: {args.examples}")

    if not args.checkpoint.is_dir():
        raise ValueError(f"--checkpoint is not a directory: {args.checkpoint}")

    if args.special_tokens and not args.special_tokens.is_file():
        raise ValueError(f"--special-tokens is not a file: {args.special_tokens}")

    args.output_filename.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # lock down output directory permissions so other users can't see protected data

    special_tokens = None
    if args.special_tokens:
        with open(args.special_tokens, "r", encoding="utf-8") as f:
            special_tokens = [x.strip() for x in f if x.strip()]

    if args.document_id:
        args.document_id = args.document_id.split(",")

    with open(args.examples, "r", encoding="utf-8") as f:
        logger.info("reading example text from '%s'", args.examples)
        dev_examples = []
        for line_idx, line in enumerate(f):
            if str(args.examples).endswith(".txt"):
                line_idx = str(line_idx)
                if not args.document_id or line_idx in args.document_id:
                    dev_examples.append({"translation": {"document_id": line_idx, "en": line.strip()}})
            else:
                d = json.loads(line)
                if not args.document_id or d["translation"]["document_id"] in args.document_id:
                    dev_examples.append(d)

    def gen_batch(items, size):
        batch = []
        for x in items:
            batch.append(x)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    dev_example_batches = list(gen_batch(dev_examples, size=args.batch_size))

    logger.info("Loaded %s examples, will process in %s batches of size %s", len(dev_examples), len(dev_example_batches), args.batch_size)

    # Delay import because importing pytorch takes time.  This makes running --help
    # cheaper and also makes it easier for the user to debug certain problems with
    # their command-line usage.
    from n2c2_ss.generate import generate  # pylint: disable=import-outside-toplevel
    from transformers import T5ForConditionalGeneration, T5Tokenizer  # pylint: disable=import-outside-toplevel
    from transformers.generation_logits_process import LogitsProcessorList, LogitsProcessor  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(torch_device)
    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint)

    class NoBadSubwordsLogitsProcessor(LogitsProcessor):
        """
        [`LogitsProcessor`] that implements a per-input-sequence denylist of embedding
        IDs.

        Similar to [`NoBadWordsLogitsProcessor`] except that each sequence will be
        associated with different bad IDs, according to the IDs present in the
        input/source sequence.

        There is no dependence on history, we're only allowlisting the IDs for the
        structure markup and for substrings of tokens in the source/input text.
        """

        def __init__(self, tokenizer, special_tokens, text_batch):
            self.bad_ids: List[List[int]] = []
            self._bad_ids_mask = None  # we can calculate this once for each decoding batch and reuse it for all next-token predictions

            # Always allow the subwords used to create the structure markup and the explicit
            # space character.  We can't produce correct output without this.  Whether or not
            # the remaining embedding IDs should be allowed depends on the contents of each
            # source/input sequence.
            special_token_ids = sorted({y for x in special_tokens for y in tokenizer(x).input_ids})
            special_token_ids.append(tokenizer.get_vocab()["\u2581"])  # this is the sentencepiece explicit space character.  The tokenizer will remove it as whitespace, so we should explicitly call it out as something worth keeping.
            possibly_bad_embedding_ids = set(tokenizer.get_vocab().values()).difference(special_token_ids + [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id])

            # It's not sufficient to tokenize the source/input text and keep those embedding
            # IDs.  It's necessary to be able to separate out subsets of characters from a
            # token, possibly splitting a subword. For example, keep the substring "EtOH" out
            # of the token "+EtOH" or "vodka" out of "etoh(vodka)".
            #
            # The implementation below is very inefficient.  It creates a "sentence" out of
            # all possible subsequences (possibly very large) and then tokenizes it (possibly
            # very slow).

            def gen_contiguous_subsets(items):
                for width in range(1, len(items) + 1):  # `+ 1` because at some point we need items[0:width=len(items)] to occur
                    for i in range(0, len(items) - width + 1):  # `+ 1` because at some point we need items[i=0:width=len(items)] to occur (i.e., i assigned from range(0, 1))
                        yield items[i:(i + width)]

            for text in text_batch:
                contiguous_subsets = set()
                for token in text.split():
                    contiguous_subsets.update(gen_contiguous_subsets(token))
                contiguous_subsets = sorted(contiguous_subsets)
                contiguous_subsets = [contiguous_subsets[0]] + contiguous_subsets + [contiguous_subsets[-1]]  # duplicate the first and last subset to force the tokenizer to deal with the space separation
                contiguous_subset_text = " ".join(contiguous_subsets)
                self.bad_ids.append(sorted(possibly_bad_embedding_ids.difference(tokenizer(contiguous_subset_text).input_ids)))

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if self._bad_ids_mask is None:
                self._bad_ids_mask = torch.zeros(scores.shape)
                for seq_idx, seq_bad_ids in enumerate(self.bad_ids):
                    self._bad_ids_mask[seq_idx, seq_bad_ids] = 1
                self._bad_ids_mask = self._bad_ids_mask.to(scores.device).bool()
            scores = scores.masked_fill(self._bad_ids_mask, -float("inf"))
            return scores

    class GreedyIncrementallyConstrainedLogitsProcessor(LogitsProcessor):
        """
        [`LogitsProcessor`] that is structure and constraint-aware.

        Similar to `NoBadSubwordsLogitsProcessor` except that each output sequence is
        forced to be syntactically valid and to follow as many constraints as possible.

        Only built for greedy decoding strategies; it only makes sure that the highest
        scoring embedding_id is incrementally valid.  It does not constraint the
        sequence to be globally valid (e.g., it will allow output where events do not
        have triggers).
        """

        def __init__(self, tokenizer, special_tokens, text_batch):
            self.tokenizer = tokenizer
            self.special_tokens = special_tokens
            self.text_batch = text_batch

            self._is_finished = [False] * len(self.text_batch)
            self.neg_inf = -float("inf")

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            bad_ids_mask = None  # don't allocate and use unless necessary
            for row_idx in range(len(input_ids)):  # pylint: disable=consider-using-enumerate
                if self._is_finished[row_idx]:
                    continue
                if self.tokenizer.eos_token_id in input_ids[row_idx]:
                    self._is_finished[row_idx] = True
                    continue

                # This isn't bad if the model is picking good next token candidates but this can
                # be _very_ slow in the worst case if it iterates through _all_ vocabulary.  Thus,
                # it may be safest to preprocess the scores with NoBadSubwordsLogitsProcessor
                # first.
                is_valid_found = False
                last_score = None
                for next_token_idx in scores[row_idx].argsort(descending=True):
                    current_score = scores[row_idx][next_token_idx]
                    if current_score == self.neg_inf:
                        break
                    if last_score is not None and last_score != current_score and is_valid_found:
                        break  # no need to keep going, we found at least one valid high-scoring token
                    last_score = current_score
                    candidate_output_str = self.tokenizer.decode(input_ids[row_idx].tolist() + [next_token_idx])  # this is _super_ slow.  If we built our own incremental decoder that could resume, this would be _much_ faster.
                    if validate_incremental_output(self.tokenizer, self.text_batch[row_idx], candidate_output_str, is_eos=next_token_idx == tokenizer.eos_token_id):
                        is_valid_found = True
                    else:
                        if bad_ids_mask is None:
                            bad_ids_mask = torch.zeros(scores.shape)
                        bad_ids_mask[row_idx, next_token_idx] = 1
            if bad_ids_mask is not None:
                scores = scores.masked_fill(bad_ids_mask.to(scores.device).bool(), self.neg_inf)
            return scores

    with open(args.output_filename, "w", encoding="utf-8") as f:
        first_pass = True
        for example_batch_idx, example_batch in enumerate(dev_example_batches):
            text_batch, document_id_batch = zip(*[(d["translation"]["en"], d["translation"]["document_id"]) for d in example_batch])
            input_ids = tokenizer(text_batch, padding=True, return_tensors="pt").input_ids

            logits_processor = LogitsProcessorList()
            if special_tokens and not args.no_constrain_decoding:
                if first_pass:
                    logger.info("Enabling NoBadSubwordsLogitsProcessor")
                logits_processor.append(NoBadSubwordsLogitsProcessor(tokenizer, special_tokens, text_batch))
            if args.constrain_decoding and not args.no_constrain_decoding:
                if first_pass:
                    logger.info("Enabling GreedyIncrementallyConstrainedLogitsProcessor")
                if args.special_tokens:
                    hardcoded_special_tokens = set([_CONJ])
                    hardcoded_special_tokens.update(_ARGUMENT_TYPE_TO_SUBTYPES_DICT.keys())
                    for subtypes in _ARGUMENT_TYPE_TO_SUBTYPES_DICT.values():
                        hardcoded_special_tokens.update(subtypes)
                    missing_special_tokens = set(special_tokens).symmetric_difference(hardcoded_special_tokens)
                    if missing_special_tokens:
                        raise KeyError(f"Missing tokens found in the --special-tokens file: {sorted(missing_special_tokens)}")
                logits_processor.append(GreedyIncrementallyConstrainedLogitsProcessor(tokenizer, special_tokens, text_batch))

            logger.info("decoding batch %s / %s, size %s, with max embedding length %s for document IDs '%s'", example_batch_idx, len(dev_example_batches), len(text_batch), input_ids.size(1), ",".join(document_id_batch))

            start_time = time.time()
            outputs = generate(tokenizer, model, input_ids.to(model.device), max_length=args.max_length, logits_processor=logits_processor, output_attentions=False)
            logger.debug("decoded batch with max length %s in %s", outputs.sequences.size(1), f"{time.time() - start_time:.2f}s")

            for document_id, text, output in zip(document_id_batch, text_batch, outputs.sequences):
                decode_data = {
                    "translation": {
                        "document_id": document_id,
                        "en": text,
                        "hyp": tokenizer.decode(output, skip_special_tokens=True),
                    }
                }
                print(json.dumps(decode_data), file=f, flush=True)
            first_pass = False


if __name__ == "__main__":
    main()
