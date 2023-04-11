# Overview
Notes about some of the important parts of the system.

The system can also be understood through the sequence of scripts, and their
inputs and outputs, described in the [README's "Usage"](README.md#usage)
section.


# Misc text preprocessing
Text is preprocessed before being used in training or in inference.

Most of the whitespace normalization happens first. Beginning and trailing
whitespace is removed from each line, newlines are normalized to `\n`, and all
runs of three or more newlines are normalized to just two newlines. Later, any
remaining whitespace is collapsed to one space character.

Next, we handle special characters that are otherwise out-of-vocabulary (OOV) in
the original T5 models. Rather than attempting to create a new version of the T5
model with extra embeddings and adjusted model weights, we replace the special
characters with special tokens formed from in-vocabulary characters. The
additional characters that are able to be represented are:

* open and close curly brace
* left and right angle bracket (the left angle bracket is OOV by default but the right angle bracket is in-vocabulary, go figure)
* tilde
* newline

> Note: Since line breaks can denote important structure, we decided to explicitly
> capture them for the model.  However, subsequent experiments showed that
> any difference in performance was not significant.

> Note: To round-trip the special tokens back to special characters given the
> sentencepiece encoding, it was necessary to add a special "not-a-space"
> character.  See code for the gory details.

> Note: The preprocessing code could have been simplified had we handled encoding
> the text into embedding identifiers ourselves rather than co-opting the existing
> T5 and PyTorch functions


# Further T5 pretraining on MIMIC III data
Our best performing final models benefited from a few more epochs of T5
pretraining on the MIMIC III clinical text. The pretraining dataset was created
by sliding a window of size 512 sentencepiece elements over the MIMIC III
document text, with as stride of 256 sentencepiece elements. The text captured
under the window was preprocessed as mentioned above.

The official pretraining script, modified to recognize "epochs", was used to run
a few more epochs of pretraining.

> Note: since there is a 256 sentencepiece token overlap in the sliding window,
> the first and last 256 sentencepieces appear in only one example for each
> document, while the remaining "interior" sentencepieces appear twice. Given that
> medical notes are somewhat structured, it's possible that this de-emphasis of
> the leading content biases the pretraining.

> Note: a few MIMIC documents seem to have non-text content.  We filtered
> these out by matching on a small number of telling strings.


# Finetuning on Social Determinants of Health (SDoH)
We wanted to create a system that could operate on the full input text without
needing to bother with additional segmentation. However, initial experiments
seemed to indicate that there was too little data to finetune a T5 model on the
full-original-text training examples.

In a nod towards curriculum learning, we generated easier examples from the
original training data and included them alongside the full-original-text. The
main technique we rely upon is to consider all sub-sequences of sequential, non-
overlapping events in a document as valid training examples. For each of these,
we consider a "tight" and "loose" bound for the original text-- the former
associates annotations with a text span that begins on the first word of
annotated text and ends on the last word of annotated text. By contrast, the
"loose bound" examples can include text before and after any annotated text,
as long as it does not belong to a different event.

After generating all contiguous sub-sequences, and then all unique "tight" and
"loose" variants, we preprocess the resulting text according to the method
described above.

Finetuning was accomplished with the suggested PyTorch scripts.  We used
Deepspeed's implementation of ZeRO to help fit our finetuning within the RAM
capacity of a machine with 4xA100 NVIDIA GPUs.

> Note: Not all edge cases were handled when selecting "easier" examples; it
> should be possible to eke out additional training examples.

Relatively late in development, we noticed that while our model performed well
on the SDoH dev splits, it had a tendency to output false positives on clinical
text with no SDoH content. While this wasn't likely to be a concern for the
competition, it was somewhat unsatisfying to have a model that could only be
practically run on text that had been pre-filtered for SDoH content. In an
attempt to combat this, we reused a subset of the MIMIC III examples from
pretraining by manually deleting SDoH content from them and then introducing
them as negative examples.  This didn't help performance on social history
sections, but it did reduce the rate of false positives on general clinical note
text.


# Constrained decoding
The model outputs a sequence of SDoH events.  The events contain sequences of
TAnnotation and AAnnotation labels, as well as tokens from the input text that
those annotation labels should cover.

This output structure is learned by the sequence-to-sequence model during the
fine-tuning training.  The model output tends to be well-formed sequences;
however, it will sometimes make mistakes.  One amusing class of errors was the
model outputting tokens with spelling mistakes corrected ("illicts" in the input
was corrected to "illicits" in the output).

To improve performance for applications that require well-formed output
sequences (for example, when we want to transform the model output back into a
BRAT-format annotation file for challenge scoring), we can constrain the
decoding process and enforce well-formed output.

To implement this, as we incrementally decode the sequence of sentencepieces
being constructed for the output, we set the probability of all inadmissible
sentencepieces to negative infinity.

An implementation of the constraint process for this specific problem allowed
for us to be in several states, each indicating what was expected to be produced
next:

* the T5 "PAD" symbol
* a TAnnotation or the "EOS" symbol
* a TAnnotation
* the rest of a TypeLiving TAnnotation or a token  (this is a special case mentioned in a note, below)
* an AAnnotation
* a token
* a token, TAnnotation, the "€AND" token, or the "EOS" symbol

The various TAnnotations, their valid AAnnotations (if any), and the tokens
could all be looked up in hardcoded tables or in the original input sequence.

> Note: some constraints above would benefit from backtracking or beam search.
> For example, the constraint that requires a "trigger" to be present in each
> event can only be evaluated when the event ends (i.e., the sequence transitions
> to "€AND" or to the "EOS" symbol).  Events can be forced to continue until a
> trigger is produced, but it may be the case that either the trigger was passed
> over earlier or that the system should not be producing an event in the first
> place.  Systems that order the trigger first in each event do not have this
> particular problem, but then not all arguments are ordered strictly left-to-
> right anymore.

> Note: we had to deal with a slight edge-case because we used the TAnnotation
> labels as given and two share a common prefix: "Type" and "TypeLiving".  After
> the former, we should be producing tokens, whereas after the latter, we should
> be producing one of the acceptable AAnnotation labels.  It's slightly messy and
> the easiest fix would have been to change the TAnnotation labels to avoid the
> ambiguity in the first place.

> Note: the current implementation of constrained decoding is slow.  It could be sped up
> significantly but was sufficient for the challenge datasets.


# Recovering spans
With only the TAnnotation and AAnnotation labels in the output sequence, it is
possible to determine whether a particular SDoH topic was discussed.  In some
cases, it's possible to further determine what kinds of assertions were made.
For example, output with a TAnnotation "Tobacco" and a TAnnotation /
AAnnotations pair "StatusTime=none" implies that a portion of the text is
evidence for the assertion that the patient was not a smoker.

With the tokens in the output sequence, it may be reasonable for a human to
determine more detail about the assertions.  For example, in the output with a
"StatusTime=current" and "TypeLiving=with\_family 'with her husband and 12yr old
daughter'", the tokens provide additional information to a human reader.

When a human reader is interested in using the system output only as a guide to
support their own independent interpretation of the original text, the system
tokens can act like keywords that focus the reader's attention on the regions of
the input text in which they appear.

When there are many events and the original text is long, the loose association
between system output tokens and the original text may no longer suffice.  The
n2c2 challenge task requires that systems identify the precise reference spans
in the original text for tokens that support the event's TAnnotations and
AAnnotations.

Our system outputs tokens, but does not output span offsets or distinguish
between multiple instances of a token.  Especially since our system operates on
the entire input text, which can cover several sentences of a whole SDoH
section, the process to map one token back to correct span offsets is often
ambiguous.

One way to reduce the token ambiguity is to take advantage of the same output
structure constraints used during decoding to narrow down the subsets of the
input text where a particular token in a particular argument of a particular
event could have come from.  When hard constraints don't completely eliminate
the ambiguity, soft constraints and heuristics may be effective in selecting the
same token as is selected in the  reference.

> Note: the use of constraints in span recovery differs from their use in
> constrained decoding because span recovery can take advantage of the full output
> sequence.

We elected to use the constraint solving package [ortools](https://developers.google.com/optimization)
to recover spans.  Each output token is associated with a set of candidate
beginning offsets.  Output tokens are allowed to start at any offset in the
input text; they are not limited to whitespace boundaries because of the need to
handle interior tokens like the hyphen indicating a negative result/presence in
text like "(-)".  Then the ordering relationships at the token, argument, and
event level described above are added as hard constraints that must be satisfied
for any solution.

To deal with remaining ambiguity, soft constraints are also used.  One soft
heuristic assigns a greater cost to solutions that increase the character
distance between the first and last tokens in an argument.  Another soft
heuristic is similar, it increases the cost for solutions that increase the
character distance between the first and last tokens in an event.  These soft
constraints encourage the solver to resolve ambiguities by keeping the tokens
covered by arguments as compact as possible and by keeping events as compact as
possible.

> Note: this system is capable of handling more complex tasks (split spans, >
arbitrary overlaps.
