import torch
import torchvision
import numpy as np
from PIL import Image
from transformers.utils import ModelOutput
import time
import pprint

from .phi import PhiForCausalLM, PhiConfig
from .phi.modeling_phi import *
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
            assert segment_position_ids is not None
            for i in range(len(segment_tokens)):
                if len(segment_tokens[i]) != 0:
                    hidden_states[i, segment_position_ids[i]] += segment_tokens[i]

        for layer in self.h:
            hidden_states = layer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

        return hidden_states


class SegmentHead(nn.Module):

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


class RegreessionLoss(nn.Module):

    def __init__(self, shift_labels: bool = True) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = nn.MSELoss()

    def forward(self, prediction: torch.FloatTensor, target: torch.LongTensor) -> torch.FloatTensor:
        if self.shift_labels:
            prediction = prediction[..., :-1, :].contiguous()
            target = target[..., 1:].contiguous()

        loss = self.loss_fct(prediction, target)

        return loss


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

    def __init__(
            self,
            config: PhiConfig,
            w_segment_loss: float = 1.0,
            w_bbox_loss: float = 1/1000,
            seqae_batch_size: Optional[int] = 128,
                # batch size for seqae inference, -1 for full segments inference for each sample
            ) -> None:
        super().__init__(config)

        self.transformer = PhiModel(config)
        self.lm_head = CausalLMHead(config)
        self.lm_loss = CausalLMLoss()
        self.segment_loss = RegreessionLoss(shift_labels=False)
        self.seqae_loaded = False

        self.w_segment_loss=w_segment_loss
        self.w_bbox_loss=w_bbox_loss
        self.seqae_batch_size = seqae_batch_size

        self.post_init()

    def load_seqae(self, seqae_path: str, freeze_seqae_encoder: bool = False, freeze_seqae_decoder: bool = False):
        self.seqae = Seq2SeqAutoEncoderModel.from_pretrained(seqae_path).to(self.device)
        if freeze_seqae_encoder:
            for param in self.seqae.encoder.parameters():
                param.requires_grad = False
        if freeze_seqae_decoder:
            for param in self.seqae.decoder.parameters():
                param.requires_grad = False

        self.visual_token_embedding = torch.nn.Linear(self.seqae.config.d_latent+4, self.config.n_embd, bias=False).to(self.device)
        self.segment_head = SegmentHead(self.config.n_embd, self.seqae.config.d_latent).to(self.device)
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
        
        if not self.seqae_loaded:
            raise ValueError("Please load seqae model first.")
        
        profiling = {}        
        step_start = time.time()
        start = time.time()

        bs = input_ids.shape[0]

        # segment embedding
        batch_segment_tokens = []
        batch_segment_bboxes = []
        batch_segment_latents = []
        batch_segment_position_ids = []
        for sample_idx in range(bs):
            # if there is segments in this sample
            if len(segment_sequences[sample_idx]) != 0: 
                # flatten the image_num axis, in case there are multiple images in one sample
                sample_segment_sequences = segment_sequences[sample_idx].flatten(start_dim=0, end_dim=1).to(self.device)
                sample_bboxes = bboxes[sample_idx].flatten(start_dim=0, end_dim=1).to(self.device, dtype=self.dtype)
                if self.seqae_batch_size != -1:
                    segment_latents = []
                    for i in range(0, len(sample_segment_sequences), self.seqae_batch_size):
                        segment_latents.append(self.seqae.encode(sample_segment_sequences[i:i+self.seqae_batch_size]))
                    segment_latents = torch.cat(segment_latents, dim=0)
                else:
                    segment_latents = self.seqae.encode(sample_segment_sequences)

                latents_and_bboxes = torch.cat((segment_latents, sample_bboxes), dim=1)
                segment_tokens = self.visual_token_embedding(latents_and_bboxes)
                batch_segment_tokens.append(segment_tokens)
                batch_segment_bboxes.append(sample_bboxes)
                batch_segment_latents.append(segment_latents)
            else:
                batch_segment_tokens.append([])
                batch_segment_bboxes.append([])
                batch_segment_latents.append([])

            # find the position of <|seg|> in input_ids
            # seg_pos = torch.where(input_ids[sample_idx] == self.special_token_id_mappinmg["<|seg|>"])[0].tolist()
            seg_pos = torch.where(torch.tensor(input_ids[sample_idx]) == torch.tensor(self.special_token_id_mappinmg["<|seg|>"]))[0].tolist()
            assert len(seg_pos) == len(sample_segment_sequences), f"In the {sample_idx}th sample, number of <|seg|> in input_ids ({len(seg_pos)}) does not match number of segments ({len(sample_segment_sequences)})"
            batch_segment_position_ids.append(seg_pos)

        profiling['[time]/embed segments'] = time.time() - start
        start = time.time()

        hidden_states = self.transformer(
            input_ids, 
            segment_tokens=batch_segment_tokens,
            segment_position_ids=batch_segment_position_ids,
            past_key_values=past_key_values, 
            attention_mask=attention_mask
            )
        lm_logits = self.lm_head(hidden_states)

        profiling['[time]/LLM'] = time.time() - start
        start = time.time()

        # LLM Transformer last hidden state predicts next text token
        loss = None
        lm_loss = None
        if labels is None:
            # autoregressive modelling: labels is shifted to the right by one based on input_ids
            labels = input_ids[..., 1:].contiguous()
            lm_logits = lm_logits[..., :-1, :].contiguous()

        lm_loss = self.lm_loss(lm_logits, labels)

        # LLM Transformer last hidden state predicts segment latent and bbox
        segment_loss = 0
        bbox_loss = 0
        predicted_segment_latents = []
        predicted_segment_bboxes = []
        for sample_idx in range(len(batch_segment_tokens)):
            if len(batch_segment_tokens[sample_idx]) != 0:
                segment_latents, segment_bboxes = self.segment_head(hidden_states[sample_idx, batch_segment_position_ids[sample_idx]])
                segment_loss += self.segment_loss(segment_latents, batch_segment_latents[sample_idx])
                bbox_loss += self.segment_loss(segment_bboxes, batch_segment_bboxes[sample_idx])

                predicted_segment_latents.append(segment_latents)
                predicted_segment_bboxes.append(segment_bboxes)
            else:
                predicted_segment_latents.append([])
                predicted_segment_bboxes.append([])
            
        profiling['[time]/segment head'] = time.time() - start
        profiling['[time]/total'] = time.time() - step_start

        profiling['[loss]/lm_loss'] = lm_loss.item()
        profiling['[loss]/segment_loss'] = segment_loss.item()
        profiling['[loss]/bbox_loss'] = bbox_loss.item()

        # print('-'*80)
        # for k, v in profiling.items():
        #     print(f"\t{k}:\t{v}")

        loss = lm_loss + self.w_segment_loss * segment_loss + self.w_bbox_loss * bbox_loss

        return MultimodalCausalLMOutputWithPast(
            loss=loss, 
            logits=lm_logits, 
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            predicted_segment_latents=predicted_segment_latents,
            predicted_segment_bboxes=predicted_segment_bboxes,
            profiling=profiling,
            )
    