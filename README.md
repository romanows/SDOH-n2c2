# Overview
A text-to-structured-sequence approach to the [National NLP Clinical Challenges
(n2c2) 2022 Social Determinants of Health (SDoH) track 2 challenge](https://n2c2.dbmi.hms.harvard.edu/2022-track-2).
Results from this challenge, including performance and analysis of this system,
are described in the paper [The 2022 n2c2/UW Shared Task on Extracting Social
Determinants of Health](https://arxiv.org/abs/2301.05571).

Compelling aspects of this approach were:
* Scoring highest in the challenge subtasks
* Leveraging the pretrained T5 model in a translation task
* Supplying the entire social history section text to the model
* Solving training issues likely arising from limited training data
* Use of a constraint solver to reduce token span ambiguity in the output

Separate documents describe some aspects of the [system implementation](SYSTEM.md).

Please see [abachaa/SDOH-n2c2](https://github.com/abachaa/SDOH-n2c2) for the
other, classification-based system we developed.

This code is most useful for researchers interested in the implementation
details behind our JAMIA paper (link TBD).  Unfortunately, there are currently
no plans to continue developing, maintaining, nor heavily supporting this code.
That said, please feel free to create an issue if you would like to point out a
bug.


# Task
The n2c2 2022 Track 2 task is to extract substance-use, employment-status, and
living-status events from a snippet of clinical text.  Consider the example
snippet of text posted on the [n2c2 2022 Track 2 website](https://n2c2.dbmi.hms.harvard.edu/2022-track-2):

```text
SOCIAL HISTORY: Used to be a chef; currently unemployed.
Tobacco Use: quit 7 years ago; 15-20 pack years
Alcohol Use: No     Drug Use: No
```

In this short snippet, a system would label spans of text that inform us about
the patient's employment status and tobacco, alcohol, and drug substance-use.

Subtasks explore how a system scales to different data:
* Extraction -- given [MIMIC III](https://physionet.org/content/mimiciii/1.4) training and development data, the system is evaluated on unseen MIMIC III test data
* Generalization -- with no additional training, the system is evaluated on unseen University of Washington (UW) data
* Learning Transfer -- given new UW training and dev data, the system is evaluated on unseen UW test data

Teams were allowed to use additional data and annotations.


# Method
Idea is to train a seq2seq model to map text to some form of marked-up output.


## Structured sequence output representation
At the moment, the output representation is a 3-tuple with elements:

* TAnnotation label
* AAnnotation label (optional)
* Tokens

Consider the source text "Tob (-)".  There is one event here, that the patient
does not use tobacco.  The "Tob" is annotated as a TAnnotation with the label
"Tobacco".  The "(-)" is annotated as TAnnotation with the label "StatusTime"
and further annotated with an AAnnotation with the label "none".

The target sequence might look something like:

```
«Tobacco Tob «StatusTime »none (-)
```

Where the special character "«", that is rare but still included in the T5
pretrained vocabulary, is used to indicate the TAnnotation label.  The
AAnnotation label is indicated by "»".  An event may have multiple 3-tuples, and
multiple events extracted from the same input text are separated with an "€AND"
token.


# Installation
What follows was how I was able to create an environment that lets me use these
scripts on a CentOS 7 system with CUDA 11.6 installed circa 2022-05-01.  This
includes the dependencies needed for additional pretraining on MIMIC data and
finetuning on n2c2 SDoH data.

The results of running `pip freeze > requirements.txt` in the environment used
for training and analysis are available in the [requirements.txt](requirements.txt)
file.

```sh
# pytorch and jax need a more up-to-date version of gcc
scl enable devtoolset-8 bash

python3.8 -m venv ~/venv-n2c2
source ~/venv-n2c2/bin/activate

# Make sure CUDA, CUDNN, etc. are either installed at the default locations and/or
# that they are installed and in your PATH and LD_LIBRARY_PATH.

pip install --upgrade pip
pip install "jax[cuda]" --find-links https://storage.googleapis.com/jax-releases/jax_releases.html
pip install flax datasets sentencepiece protobuf torch torchvision torchaudio ortools --extra-index-url https://download.pytorch.org/whl/cu113
pip install deepspeed  # had to install this after installing torch separately

git clone https://github.com/huggingface/transformers.git
pip install ./transformers

# Install this package
pip install -e .

# Tooling for making contributions
pip install pylint~=2.12
```

To score the output of the system, install the official n2c2 scoring module:

```sh
# Install the n2c2 track 2 scoring code
git clone git@github.com:Lybarger/brat_scoring.git
pip install -e brat_scoring spacy
python -m spacy download en_core_web_sm
```


# Usage


## Additional T5 pretraining
Best results seem to come from further pretraining the T5 model on MIMIC III
data.  Prepare training examples via a sliding window over documents like:

```sh
#
# If you have more CPUs available, consider taking advantage of the `--split`
# argument to process multiple chunks of NOTEEVENTS.csv.gz in parallel.  To
# process the 42nd chunk out of 1000 chunks, supply: `--split 42/1000`.
#
# Consider setting `--split-num-docs` when processing many splits.  This avoids
# the counting of documents in NOTEEVENTS.csv.gz which is significant when the
# number of documents in each split/chunk is relatively small.  This can change
# when documents are added and removed from NOTEEVENTS.csv.gz; currently, it is:
# `--split-num-docs 2083180`
#
/usr/bin/time -v \
  scripts/extract-pretraining.py \
    /path/to/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz \
    /path/to/mimiciii-t5pretrain/mimiciii-t5pretrain.txt

# If you used the `--split` functionality above, this will concatenate the output
find /path/to/mimiciii-t5pretrain -name 'mimiciii-t5pretrain.txt.split*' -print0 | sort -z | xargs -0 cat >> /path/to/mimiciii-t5pretrain/mimiciii-t5pretrain.txt

# Dedup and shuffle, possibly unnecessary
sort -u /path/to/mimiciii-t5pretrain/mimiciii-t5pretrain.txt > /path/to/mimiciii-t5pretrain/mimiciii-t5pretrain.sort-uniq.txt
shuf /path/to/mimiciii-t5pretrain/mimiciii-t5pretrain.sort-uniq.txt > /path/to/mimiciii-t5pretrain/mimiciii-t5pretrain.sort-uniq.shuffled.txt
```

And then train like the following, which is tuned for an 80Gb A100:

```sh
# Be sure to enter into any SCL or virtualenv environments and to insure that the
# right versions of CUDA and CUDNN will be used by the training script.

# Set LD_LIBRARY_PATH and PATH to point to the correct versions of CUDA / cuDNN as
# needed.

export TOKENIZERS_PARALLELISM=false
/usr/bin/time -v \
  ./scripts/run_t5_mlm_flax.py \
    --model_name_or_path google/t5-v1_1-large \
    --do_train \
    --train_file /path/to/mimiciii-t5pretrain/mimiciii-t5pretrain.sort-uniq.shuffled.txt \
    --output_dir /path/to/mimiciii-t5pretrain/t5-v1_1-large-mimic \
    --cache_dir /path/to/mimiciii-t5pretrain/t5-v1_1-large-mimic/cache \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --per_device_train_batch_size 24 \
    --seed 42 \
    --adafactor \
    --logging_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 5
```


## Extract n2c2 examples
Extract examples from the BRAT annotations and write the serialized `(text,
structured_sequence)` to a JSON-lines format supported by Huggingface's
`run_translation.py` script.

```sh
# Training data is currently generated with the "--explode-document" option.  This
# activates the "training data augmentation" described in our paper.
#
./scripts/extract-examples.py \
  --explode-document \
  /path/to/Annotations/train/mimic \
  /path/to/train.json

# Training data without "exploding" is helpful for debugging
./scripts/extract-examples.py \
  /path/to/Annotations/train/mimic \
  /path/to/train.non-exploded.json

./scripts/extract-examples.py \
  /path/to/Annotations/dev/mimic \
  /path/to/dev.json
```


## Check how structured sequence representation round-trip
When annotations are represented in the structured sequence format, the original
span offsets are lost.  To determine how well these can be recovered in the best
case, one can attempt to recreate the original BRAT annotation files given the
prepared training examples.

```sh
./scripts/roundtrip-extract-examples.py \
  /path/to/Annotations/train/mimic \
  /path/to/train.non-exploded.json \
  /path/to/train.non-exploded.roundtrip

python /path/to/brat_scoring/brat_scoring/run_sdoh_scoring.py \
  /path/to/Annotations/train/mimic \
  /path/to/train.non-exploded.roundtrip \
  /path/to/train.non-exploded.roundtrip.csv \
  --score_trig overlap --score_span exact --score_labeled label
```


## Fine-tune on n2c2 examples
If there has been additional pretraining on MIMIC data, we'll need to convert
that model from Flax format to PyTorch format:

```sh
./scripts/convert-flax-to-pytorch.py \
  /path/to/mimiciii-t5pretrain/t5-v1_1-large-mimic/<CHECKPOINT_DIR> \
  /path/to/mimiciii-t5pretrain/t5-v1_1-large-mimic
```

Create a deepspeed configuration file at a path like `/path/to/deepspeed.json`
with the contents:
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "train_batch_size": "auto",
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "eps": 1e-8
        }
    }
}
```

In the script below, point to `/path/to/mimiciii-t5pretrain/t5-v1_1-large-mimic`
to use the model with additional MIMIC pretraining or just use the string
"google/t5-v1\_1-large" to fine-tune the original model.

The `run_translation.py` script can run on multiple GPUs without deadlocking on
our systems, so the configuration below depends on the compute setup.  We
haven't settled on an optimal recipe, yet:

```sh
# Be sure to enter into any SCL or virtualenv environments and to insure that the
# right versions of CUDA and CUDNN will be used by the training script.

/usr/bin/time -v \
  deepspeed ./transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path /path/to/mimiciii-t5pretrain/t5-v1_1-large-mimic \
    --tokenizer_name "google/t5-v1_1-large" \
    --do_train \
    --source_lang en \
    --target_lang n2c2 \
    --train_file /path/to/train.json \
    --deepspeed /path/to/deepspeed.json \
    --output_dir /path/to/n2c2-2022/t5-v1_1-large-mimic-n2c2 \
    --overwrite_output_dir \
    --max_source_length 448 \
    --max_target_length 448 \
    --learning_rate 1e-4 \
    --weight_decay 3e-06 \
    --per_device_train_batch_size 8 \
    --seed 42 \
    --logging_steps 1 \
    --save_steps 10000 \
    --num_train_epochs 50 &>> /path/to/n2c2-2022/t5-v1_1-large-mimic-n2c2/log.txt
```


## Decode
Decode the dev examples generated with `extract-examples.py` using the fine-
tuned T5 checkpoint.

Note the use of the `--special-tokens` options set to the "special tokens" file
generated alongside the training examples by `extract- examples.py`.  When this
is used, decoding avoids generating tokens that weren't in the input.

Note that `--constrain-decoding` tends to lead to higher-quality outputs, at
least in already highly-performing models, but it does increase the decoding
time significantly.

```sh
./scripts/decode.py \
  --batch-size 16 \
  --examples /path/to/dev.json \
  --checkpoint /path/to/checkpoint-dir \
  --special-tokens /path/to/train.special-tokens.txt \
  --constrain-decoding \
  /path/to/checkpoint-dir/dev.decoded.json
```


## Convert structured sequence system output to BRAT
```sh
./scripts/convert-decoded-to-brat.py \
  /path/to/Annotations/dev/mimic \
  /path/to/checkpoint-dir/dev.decoded.json \
  /path/to/checkpoint-dir/brat-dev
```


## Score with brat\_scoring
```sh
python /path/to/brat_scoring/brat_scoring/run_sdoh_scoring.py \
  /path/to/Annotations/dev/mimic \
  /path/to/checkpoint-dir/brat-dev \
  /path/to/checkpoint-dir/brat-dev.score.csv \
  --include_detailed \
  --score_trig overlap --score_span exact --score_labeled label
```


## Known issues
The constrained decoding implementation is slow.  It should be possible to speed
this up significantly.  Without much effort, one could refactor the code so that
it uses normal decoding by default and the does constrained decoding only in the
rare cases when constraint violations are detected.
