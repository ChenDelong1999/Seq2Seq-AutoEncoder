import torch
import torchvision
import numpy as np

from transformers.utils import ModelOutput

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
    

class PhiForMultimodal(PhiForCausalLM):

    def __init__(
            self,
            config: PhiConfig,
            w_segment_loss: float = 1.0,
            w_bbox_loss: float = 1.0,
            ) -> None:
        super().__init__(config)

        self.transformer = PhiModel(config)
        self.lm_head = CausalLMHead(config)
        self.lm_loss = CausalLMLoss()
        self.segment_loss = RegreessionLoss(shift_labels=False)
        self.seqae_loaded = False

        self.w_segment_loss=w_segment_loss
        self.w_bbox_loss=w_bbox_loss

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

    def preprocess_segments(self, segment_masks, images):

        def resize(segment):
            # segment['patch'] is PIL Image object
            w, h = segment['patch'].size
            if h * w > (self.seqae.config.data_seq_length):
                ratio_to_maxlength = np.sqrt(self.seqae.config.data_seq_length / (h * w))
                h = int(h * ratio_to_maxlength)
                w = int(w * ratio_to_maxlength)
                segment['patch'] = torchvision.transforms.Resize([h, w])(segment['patch'])
                segment['segmentation'] = torchvision.transforms.Resize([h, w])(segment['segmentation'][None, :, :])[0]

            return segment
        
        def encode_to_sequence(segment):
            # segment['patch'] is torch tensor with shape (C, H, W)
            h, w = segment['patch'].shape[1:]
            sequence = []
            for i in range(h):
                for j in range(w):
                    pixel_data = segment['patch'][:, i, j] / 255.0
                    is_rightmost = 1 if j == w - 1 else 0
                    is_non_masked = int(segment['segmentation'][i, j])
                    sequence.append(pixel_data.tolist() + [is_rightmost, is_non_masked])
            sequence = np.array(sequence) 

            # pad the sequence to max_seq_length with zeros
            if len(sequence) < self.seqae.config.data_seq_length:
                sequence = np.concatenate((sequence, np.zeros((self.seqae.config.data_seq_length - len(sequence), self.seqae.config.input_channels))))

            # add the query place holder to the end of the sequence
            sequence = np.concatenate((sequence, np.zeros((self.seqae.config.num_queries, self.seqae.config.input_channels))))
            # add one all zero column to the start 
            sequence = np.concatenate((np.zeros((1, sequence.shape[1])), sequence), axis=0)

            return torch.from_numpy(sequence)
    
        segment_sequences = []
        bboxes = []
        for segment_mask in segment_masks:
            image = images[segment_mask["image_index"]]

            mask = segment_mask["segmentation"]
            bbox = segment_mask["bbox"]
            bbox[2] = 1 if bbox[2] == 0 else bbox[2]
            bbox[3] = 1 if bbox[3] == 0 else bbox[3]
            x, y, w, h = bbox

            segment = {
                "patch": image.crop((x, y, x + w, y + h)),
                "segmentation": mask[y:y+h, x:x+w],
            }

            segment = resize(segment)
            segment['patch'] = torchvision.transforms.ToTensor()(segment['patch'])
            segment['patch'] = segment['patch'] * segment['segmentation'][None, :, :]
            segment_sequence = encode_to_sequence(segment)
            segment_sequence = segment_sequence.to(self.device)
            segment_sequences.append(segment_sequence)
            bboxes.append(bbox)

        segment_sequences = torch.stack(segment_sequences, dim=0).to(self.device, dtype=self.dtype)
        bboxes = torch.tensor(bboxes).to(self.device, dtype=self.dtype)
        return segment_sequences, bboxes

    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_masks: Optional[list] = None,
            # A list of batch_size number of list of dicts, whose key includes 
            #   - "image_index": int
            #   - "mask": binary mask of torch.Tensor with shape (H, W)
            #   - "bbox": bounding box of the object in the image (x, y, w, h)
            # if no segment token in certain sample, the corresponding list should be empty
        images: Optional[list] = None,
            # A list of PIL Images objects
        seqae_batch_size: Optional[int] = 16,
            # batch size for seqae inference, -1 for full segments inference for each sample
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        if not self.seqae_loaded:
            raise ValueError("Please load seqae model first.")
        
        if segment_masks is not None:
            batch_segment_tokens = []
            batch_segment_bboxes = []
            batch_segment_latents = []
            for sample_segment_masks in segment_masks:
                if len(sample_segment_masks)!=0:
                    segment_sequences, bboxes = self.preprocess_segments(sample_segment_masks, images)

                    if seqae_batch_size != -1:
                        segment_latents = []
                        for i in range(0, len(segment_sequences), seqae_batch_size):
                            segment_latents.append(self.seqae.encode(segment_sequences[i:i+seqae_batch_size]))
                        segment_latents = torch.cat(segment_latents, dim=0)
                    else:
                        segment_latents = self.seqae.encode(segment_sequences)

                    latents_and_bboxes = torch.cat((segment_latents, bboxes), dim=1)
                    segment_tokens = self.visual_token_embedding(latents_and_bboxes)
                    batch_segment_tokens.append(segment_tokens)
                    batch_segment_bboxes.append(bboxes)
                    batch_segment_latents.append(segment_latents)
                else:
                    batch_segment_tokens.append([])
                    batch_segment_bboxes.append([])
                    batch_segment_latents.append([])
                
            # find the position of <|seg|> in input_ids
            batch_segment_position_ids = []
            for i in range(len(input_ids)):
                seg_pos = torch.where(input_ids[i] == self.special_token_id_mappinmg["<|seg|>"])[0].tolist()
                assert len(seg_pos) == len(segment_masks[i]), f"number of <|seg|> in input_ids ({len(seg_pos)}) does not match number of segments ({len(segment_masks[i])})"
                batch_segment_position_ids.append(seg_pos)
        else:
            batch_segment_tokens = None
            batch_segment_bboxes = None
            batch_segment_latents = None
            batch_segment_position_ids = None
            
        hidden_states = self.transformer(
            input_ids, 
            segment_tokens=batch_segment_tokens,
            segment_position_ids=batch_segment_position_ids,
            past_key_values=past_key_values, 
            attention_mask=attention_mask
            )
        lm_logits = self.lm_head(hidden_states)

        # LLM Transformer last hidden state predicts next text token
        loss = None
        lm_loss = None
        if labels is not None:
            lm_loss = self.lm_loss(lm_logits, labels)

        # LLM Transformer last hidden state predicts segment latent and bbox
        segment_loss = 0
        bbox_loss = 0
        if segment_masks is not None:
            predicted_segment_latents = []
            predicted_segment_bboxes = []
            for i in range(len(batch_segment_tokens)):
                if len(batch_segment_tokens[i]) != 0:
                    segment_latents, segment_bboxes = self.segment_head(hidden_states[i, batch_segment_position_ids[i]])
                    segment_loss += self.segment_loss(segment_latents, batch_segment_latents[i])
                    bbox_loss += self.segment_loss(segment_bboxes, batch_segment_bboxes[i])

                    predicted_segment_latents.append(segment_latents)
                    predicted_segment_bboxes.append(segment_bboxes)
                else:
                    predicted_segment_latents.append([])
                    predicted_segment_bboxes.append([])
        else:
            predicted_segment_latents = None
            predicted_segment_bboxes = None

        loss = lm_loss + self.w_segment_loss * segment_loss + self.w_bbox_loss * bbox_loss

        return MultimodalCausalLMOutputWithPast(
            loss=loss, 
            logits=lm_logits, 
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            predicted_segment_latents=predicted_segment_latents,
            predicted_segment_bboxes=predicted_segment_bboxes,
            )
    