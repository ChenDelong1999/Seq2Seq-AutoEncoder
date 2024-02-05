import torch
import torchvision
import numpy as np
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import logging
logger = logging.get_logger(__name__)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from transformers.utils import ModelOutput
from transformers.cache_utils import Cache, DynamicCache
import time
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union

from .phi.modeling_phi import PhiForCausalLM, PhiModel
from .phi.configuration_phi import PhiConfig

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass



def replace_seg_tokens_in_input_embeddings(
        inputs_embeds,
        segment_tokens,
        segment_position_ids):
    
    if segment_tokens is not None:
        for i in range(len(segment_tokens)):
            if len(segment_tokens[i]) != 0:
                inputs_embeds[i, segment_position_ids[i]] = segment_tokens[i].to(inputs_embeds[i, segment_position_ids[i]].dtype)

    return inputs_embeds


class PhiMultimodalModel(PhiModel):


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        segment_tokens = None,
        segment_position_ids = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.embed_dropout(inputs_embeds)

        # Attention mask.
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # added this
        inputs_embeds = replace_seg_tokens_in_input_embeddings(inputs_embeds, segment_tokens, segment_position_ids)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class SegmentModelingHead(nn.Module):

    def __init__(self, d_llm, d_segment_latent) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(d_llm)
        self.linear_to_latent = nn.Linear(d_llm, d_segment_latent)
        self.linear_to_bbox = nn.Linear(d_llm, 4)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.ln(hidden_states)
        latents = self.linear_to_latent(hidden_states).to(torch.float32)
        bboxes = self.linear_to_bbox(hidden_states).to(torch.float32)

        return latents, bboxes
    
@dataclass
class MultimodalModelingOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    predicted_segment_latents: Optional[torch.FloatTensor] = None
    predicted_segment_bboxes: Optional[torch.FloatTensor] = None
    profiling: Optional[dict] = None


def load_seqae(model, seqae):
    
    model.seqae = seqae    
    model.visual_token_embedding = torch.nn.Linear(model.seqae.config.d_latent, model.config.hidden_size, bias=False)
    model.visual_positional_embedding = torch.nn.Linear(4, model.config.hidden_size, bias=False)
    model.segment_modeling_head = SegmentModelingHead(model.config.hidden_size, model.seqae.config.d_latent)


def prepare_segment_tokens(
        model,
        input_ids,
        segment_sequences,
        bboxes,

    ):

    bs = input_ids.shape[0]
    all_seg_bbox, all_seg_emb, all_seg_tokens, all_seg_position_ids = [], [], [], []
    """
    all_seg_bbox: raw bounding boxes 
        [batch_size, num_segments, 4]
    all_seg_emb: SeqAE encoder output embeddings 
        [batch_size, num_segments, d_seqae_latent]
    all_seg_tokens: LLM input tokens 
        [batch_size, num_segments, d_llm]
    all_seg_position_ids: position of <|seg|> in input_ids 
        [batch_size, num_segments]
    """
    for i in range(bs):
        if len(segment_sequences[i]) != 0: # if there is segments in this sample 
            # flatten the image_num axis, in case there are multiple images in one sample
            seg_seq = segment_sequences[i].flatten(start_dim=0, end_dim=1)
            if model.seqae_batch_size != -1:
                seg_emb = []
                for j in range(0, len(seg_seq), model.seqae_batch_size):
                    seg_emb.append(model.seqae.encode(seg_seq[j:j+model.seqae_batch_size]))
                seg_emb = torch.cat(seg_emb, dim=0)
            else:
                seg_emb = model.seqae.encode(seg_seq)
            
            seg_tokens = model.visual_token_embedding(seg_emb)

            bbox = bboxes[i].flatten(start_dim=0, end_dim=1).to(dtype=model.dtype)
            bbox_positional_embeddings = model.visual_positional_embedding(bbox)

            # TODO: Masking of segments shall be done here
            seg_tokens += bbox_positional_embeddings

            all_seg_tokens.append(seg_tokens)
            all_seg_emb.append(seg_emb)
            all_seg_bbox.append(bbox)

        else:
            all_seg_tokens.append([])
            all_seg_emb.append([])
            all_seg_bbox.append([])

        # find the position of <|seg|> in input_ids
        seg_position_ids = torch.where(input_ids[i].clone().detach() == model.special_token_id_mappinmg["<|seg|>"])[0].tolist()
        assert len(seg_position_ids) == len(all_seg_tokens[i]), f"Number of <|seg|> ({len(seg_position_ids)}) does not match number of segments sequences ({len(all_seg_tokens[i])})"
        all_seg_position_ids.append(seg_position_ids)

    return all_seg_tokens, all_seg_emb, all_seg_bbox, all_seg_position_ids


class PhiForMultimodalModeling(PhiForCausalLM):
    def __init__(
            self,
            config: PhiConfig,
            w_segment_loss: Optional[float] = 1.0,
            w_bbox_loss: Optional[float] = 1/(1000*1000),
            seqae_batch_size: Optional[int] = 256,
                # batch size for seqae inference, -1 for full segments inference for each sample
            seqae_requires_grad: Optional[bool] = False,
            ) -> None:
        super().__init__(config)
        self.model = PhiMultimodalModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()
        
        # TODO: if only the following simple variable setting, why not past it outside?
        self.w_segment_loss = w_segment_loss
        self.w_bbox_loss = w_bbox_loss
        self.seqae_batch_size = seqae_batch_size
        self.seqae_requires_grad = seqae_requires_grad
        # self.lm_ignore_index = self.config.lm_ignore_index


    def forward(
        self,
        segment_sequences: list,
            # a list of num `batch_size` elements, each element is a tensor of [image_num, num_segments, seq_length, input_channels]
        bboxes: list,
            # a list of num `batch_size` elements, each element is a tensor of [image_num, num_segments, 4] representing bounding boxes in x, y, w, h

        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        profiling = {}        
        step_start = time.time()
        start = time.time()

        with torch.no_grad() if not self.seqae_requires_grad else torch.enable_grad():
            all_seg_tokens, all_seg_emb, all_seg_bbox, all_seg_position_ids = prepare_segment_tokens(
                self, input_ids, segment_sequences, bboxes
            )

        profiling['[time]/Segment Embedding'] = time.time() - start
        start = time.time()
            
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            segment_tokens=all_seg_tokens,
            segment_position_ids=all_seg_position_ids,
        )

        profiling['[time]/Transformer'] = time.time() - start
        start = time.time()

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        profiling['[time]/total'] = time.time() - step_start
        # convert all values in profiling to torch float tensor
        profiling = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in profiling.items()}

        return MultimodalModelingOutputWithPast(
            loss=loss, 
            logits=logits, 
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            predicted_segment_latents=None,
            predicted_segment_bboxes=None,
            profiling=profiling,
            )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)

        model_inputs["segment_sequences"] = kwargs.get("segment_sequences", None)
        model_inputs["bboxes"] = kwargs.get("bboxes", None)

        return model_inputs

        