"""
Runs inference.

Largely copied from transformers/src/transformers/generation_utils.py and then
reduced down to the subset of functionality needed for our particular model:

Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)


# pylint: disable=import-outside-toplevel
def generate(tokenizer, model, inputs = None, max_length: Optional[int] = None, logits_processor=None, synced_gpus: Optional[bool] = False, **model_kwargs):
    """
    A very pared-down version of Huggingface Transformers `generate()` model mixin.
    Specialized to the T5 encoder-decoder model.
    """
    # Delay PyTorch import because it would cause a delay for other uses of the
    # module.
    from transformers.generation_logits_process import LogitsProcessorList
    from transformers.generation_stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
    from transformers.utils import ModelOutput
    import torch

    with torch.no_grad():
        inputs_tensor, model_input_name, model_kwargs = inputs, model.main_input_name, {}
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = True
        model_kwargs["output_hidden_states"] = False
        model_kwargs["use_cache"] = True

        model_kwargs["attention_mask"] = inputs.ne(tokenizer.pad_token_id).long()

        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {k: v for k, v in model_kwargs.items() if not any(k.startswith(p) for p in irrelevant_prefix)}
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = model.get_encoder()(**encoder_kwargs)

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=inputs_tensor.device) * model.config.decoder_start_token_id

        # 5. Prepare `max_length` depending on other stopping criteria
        input_ids_seq_length = input_ids.shape[-1]
        if input_ids_seq_length >= max_length:
            logger.warning("decoder_input_ids are greater or equal to max_length, %s >= %s, which can lead to unexpected behavior", input_ids_seq_length, max_length)

        # 7. prepare distribution pre_processing samplers
        if logits_processor is None:
            logits_processor = LogitsProcessorList()

        # 8. prepare stopping criteria
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=max_length))

        # 10. run greedy search
        return _greedy_search(
            tokenizer,
            model,
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            output_scores=True,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )


# pylint: disable=import-outside-toplevel
def _greedy_search(
    tokenizer,
    model,
    input_ids, #: torch.LongTensor,
    logits_processor, #: LogitsProcessorList,
    stopping_criteria, #: StoppingCriteriaList,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    **model_kwargs,
): # -> Union[GreedySearchEncoderDecoderOutput, torch.LongTensor]:
    from transformers.utils import ModelOutput
    import torch
    import torch.distributed as dist

    # init attention / hidden states / scores tuples
    scores, decoder_attentions, cross_attentions, decoder_hidden_states = map(lambda x: () if x else None, (output_scores, output_attentions, output_attentions, output_hidden_states))
    encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
    encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_attentions else None

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]

    this_peer_finished = False  # used by synced_gpus only
    while True:

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = model(**model_inputs, return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states)

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # Store scores, attentions and hidden_states when required
        if output_scores:
            scores += (next_token_logits,)
        if output_attentions:
            decoder_attentions += (outputs.decoder_attentions,)
            cross_attentions += (outputs.cross_attentions,)
        if output_hidden_states:
            decoder_hidden_states += (outputs.decoder_hidden_states,)

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        cur_len = cur_len + 1

        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences.mul((next_tokens != tokenizer.eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            this_peer_finished = True

    @dataclass
    class GreedySearchEncoderDecoderOutput(ModelOutput):
        sequences: torch.LongTensor = None
        scores: Optional[Tuple[torch.FloatTensor]] = None
        encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
        encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
        cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
        decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    return GreedySearchEncoderDecoderOutput(
        sequences=input_ids,
        scores=scores,
        encoder_attentions=encoder_attentions,
        encoder_hidden_states=encoder_hidden_states,
        decoder_attentions=decoder_attentions,
        cross_attentions=cross_attentions,
        decoder_hidden_states=decoder_hidden_states,
    )
