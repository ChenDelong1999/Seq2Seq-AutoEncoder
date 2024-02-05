import torch
import torchvision
import numpy as np
from PIL import Image
from transformers.utils import ModelOutput
import time
import torch.nn as nn

from .phi import PhiForCausalLM, PhiConfig
from .phi.modeling_phi import *

# from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
# from transformers.models.phi.modeling_phi import PhiForCausalLM, PhiModel, PhiPreTrainedModel, PhiConfig, CausalLMHead, InferenceParams
# from transformers.modeling_outputs import CausalLMOutputWithPast


from .seq2seq_autoencoder import Seq2SeqAutoEncoderModel


class PhiModel(PhiPreTrainedModel):
    """Phi model."""

    _keys_to_ignore_on_load_missing = [""]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]

    def __init__(self, config: PhiConfig) -> None:
        super().__init__(config)

        self.embd = Embedding(config)
        self.h = nn.ModuleList([ParallelBlock(config, block_idx=i) for i in range(config.n_layer)])
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embd.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embd.wte = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_tokens: Optional[torch.FloatTensor] = None,
        segment_position_ids: Optional[list] = None,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = self.embd(input_ids)

        if segment_tokens is not None:
            for i in range(len(segment_tokens)):
                if len(segment_tokens[i]) != 0:
                    # hidden_states[i, segment_position_ids[i]] = segment_tokens[i].to(hidden_states[i, segment_position_ids[i]].dtype)
                    pass

        for layer in self.h:
            hidden_states = layer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

        return hidden_states


class CausalLMLoss(nn.Module):
    """Causal Language Modeling loss.
    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.
    """

    def __init__(self, shift_labels: bool = False, ignore_index = -100) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if self.shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss

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
class MultimodalCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    predicted_segment_latents: Optional[torch.FloatTensor] = None
    predicted_segment_bboxes: Optional[torch.FloatTensor] = None
    profiling: Optional[dict] = None
    

class PhiForMultimodal(PhiForCausalLM):
# class PhiForMultimodal(PhiPreTrainedModel):

    def __init__(
            self,
            config: PhiConfig,
            w_segment_loss: float = 1.0,
            w_bbox_loss: float = 1/(1000*1000),
            seqae_batch_size: Optional[int] = 256,
                # batch size for seqae inference, -1 for full segments inference for each sample
            ) -> None:
        super().__init__(config)

        self.transformer = PhiModel(config)
        self.lm_head = CausalLMHead(config)


        self.lm_loss = CausalLMLoss()
        self.mse_loss = torch.nn.MSELoss()

        self.w_segment_loss = w_segment_loss
        self.w_bbox_loss = w_bbox_loss
        self.seqae_batch_size = seqae_batch_size
        self.seqae_loaded = False

        # self.post_init()

    def update_lm_ignore_index(self, ignore_index):
        self.lm_loss = CausalLMLoss(ignore_index=ignore_index)

    def load_seqae(self, seqae_path: str):
        self.seqae = Seq2SeqAutoEncoderModel.from_pretrained(seqae_path)
        self.visual_token_embedding = torch.nn.Linear(self.seqae.config.d_latent, self.config.n_embd, bias=False)
        self.visual_positional_embedding = torch.nn.Linear(4, self.config.n_embd, bias=False)
        self.segment_modeling_head = SegmentModelingHead(self.config.n_embd, self.seqae.config.d_latent)
        self.seqae_loaded = True

    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_sequences: list,
            # a list of num `batch_size` elements, each element is a tensor of [image_num, num_segments, seq_length, input_channels]
        bboxes: list,
            # a list of num `batch_size` elements, each element is a tensor of [image_num, num_segments, 4] representing bounding boxes in x, y, w, h

        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        assert self.seqae_loaded, "Please load seqae model first."
        
        profiling = {}        
        step_start = time.time()
        start = time.time()

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
                if self.seqae_batch_size != -1:
                    seg_emb = []
                    for j in range(0, len(seg_seq), self.seqae_batch_size):
                        seg_emb.append(self.seqae.encode(seg_seq[j:j+self.seqae_batch_size]))
                    seg_emb = torch.cat(seg_emb, dim=0)
                else:
                    seg_emb = self.seqae.encode(seg_seq)
                
                seg_tokens = self.visual_token_embedding(seg_emb)

                bbox = bboxes[i].flatten(start_dim=0, end_dim=1).to(dtype=self.dtype)
                bbox_positional_embeddings = self.visual_positional_embedding(bbox)

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
            seg_position_ids = torch.where(input_ids[i].clone().detach() == self.special_token_id_mappinmg["<|seg|>"])[0].tolist()
            assert len(seg_position_ids) == len(all_seg_tokens[i]), f"Number of <|seg|> ({len(seg_position_ids)}) does not match number of segments sequences ({len(all_seg_tokens[i])})"
            all_seg_position_ids.append(seg_position_ids)

        profiling['[time]/embed segments'] = time.time() - start
        start = time.time()

        hidden_states = self.transformer(
            input_ids, 
            segment_tokens=all_seg_tokens,
            segment_position_ids=all_seg_position_ids,
            past_key_values=past_key_values, 
            attention_mask=attention_mask
            )
        lm_logits = self.lm_head(hidden_states)

        # LLM Transformer last hidden state predicts next text token
        loss = None
        lm_loss = None
        if labels is None:
            # autoregressive modelling: labels is shifted to the right by one based on input_ids
            labels = input_ids[..., 1:].contiguous()
            lm_logits = lm_logits[..., :-1, :].contiguous()

        lm_loss = self.lm_loss(lm_logits, labels)

        profiling['[time]/LLM'] = time.time() - start
        start = time.time()

        # LLM Transformer last hidden state predicts segment latent and bbox
        segment_loss = 0
        bbox_loss = 0
        predicted_seg_emb = []
        predicted_seg_bbox = []
        for i in range(len(all_seg_tokens)):
            if len(all_seg_tokens[i]) != 0:
                seg_emb, seg_bbox = self.segment_modeling_head(hidden_states[i, all_seg_position_ids[i]])
                segment_loss += self.mse_loss(seg_emb, all_seg_emb[i])
                bbox_loss += self.mse_loss(seg_bbox, all_seg_bbox[i])

                predicted_seg_emb.append(seg_emb)
                predicted_seg_bbox.append(seg_bbox)
            else:
                predicted_seg_emb.append([])
                predicted_seg_bbox.append([])

        loss = lm_loss + self.w_segment_loss * segment_loss + self.w_bbox_loss * bbox_loss
            
        profiling['[time]/segment head'] = time.time() - start
        profiling['[time]/total'] = time.time() - step_start

        profiling['[loss]/lm_loss'] = lm_loss.item()
        profiling['[loss]/segment_loss'] = segment_loss.item()
        profiling['[loss]/bbox_loss'] = bbox_loss.item()
        profiling['[loss]/total_loss'] = loss.item()

        # convert all values in profiling to torch float tensor
        profiling = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in profiling.items()}

        output = MultimodalCausalLMOutputWithPast(
            loss=loss, 
            logits=lm_logits, 
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            predicted_segment_latents=predicted_seg_emb,
            predicted_segment_bboxes=predicted_seg_bbox,
            profiling=profiling,
            )
        
        return output
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        segment_sequences: list,
        bboxes: list,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # if past_key_values is None or not (isinstance(past_key_values, InferenceParams)):
        #     past_key_values = InferenceParams(
        #         max_seqlen=self.config.n_positions,
        #         max_batch_size=input_ids.shape[0],
        #         seqlen_offset=0,
        #         batch_size_offset=0,
        #         key_value_memory_dict={},
        #         lengths_per_sample=None,
        #     )
        # else:
        #     # Assume that `past_key_values` has cached all tokens up to the last token in `input_ids`
        #     past_key_values.seqlen_offset = input_ids.shape[1] - 1
        #     input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            'segment_sequences': segment_sequences,
            'bboxes': bboxes,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }