import torch
from utils import get_params_count_summary
import json
import tqdm
import os
os.environ["NCCL_P2P_LEVEL"] = "NVL"

import sys
sys.path.append('segmentation/')
from segmentation import Segmenter, visualized_masks

from typing import Any, Dict, List, Optional, Tuple, Union 

from data.sharegpt4v import ShareGPT4V
from data.clevr import CLEVR
from model.multimodal_tokenizer import MultimodalTokenizer
from transformers import PreTrainedTokenizerBase

from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from model.seq2seq_autoencoder import Seq2SeqAutoEncoderModel

from model.modeling_multimodal_phi import PhiForMultimodalModeling, load_seqae
from model.phi import PhiForCausalLM, PhiConfig
from dataclasses import dataclass

# fix all random seed
import random
import numpy as np
import torch

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


@dataclass
class DataCollatorForMultimodal:

    tokenizer: PreTrainedTokenizerBase
    
    def __call__(self, features):
        batch = self.tokenizer(features, return_tensors="pt", padding='max_length', truncation=True)
        batch['labels'] = batch['input_ids'].clone()
        return batch
    
class MultimodalTrainer(Trainer):

    def __init__(self, additional_modules_to_save, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_modules_to_save = additional_modules_to_save

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        if 'profiling' in outputs:
            outputs['profiling'] = {k: v.item() for k, v in outputs['profiling'].items()}
            self.log(outputs['profiling'])

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _save(self, output_dir: str | None = None, state_dict=None):
        super()._save(output_dir, state_dict)
        # SAVE self.additional_modules_to_save  
        if self.additional_modules_to_save is not None:
            model_state_dict = self.model.state_dict()
            save_state_dict = {}
            for name, param in self.model.named_parameters():
                if name in self.additional_modules_to_save:
                    save_state_dict[name] = model_state_dict[name]

            torch.save(save_state_dict, os.path.join(output_dir, 'additional_modules.pth'))
        
        model.seqae.save_cached_latents()
    

if __name__ == '__main__':

    # 32x32
    patch_size = 32
    max_seg_per_img = 150  
    square_patches = True
    output_dir="runs/phi-2-multimodal/clevr-patchlmm(32)-5ep-lora(32)-bs64-lr1e-4"
    
    # 80x80
    # patch_size = 80
    # max_seg_per_img = 24 
    # square_patches = True
    # output_dir="runs/phi-2-multimodal/clevr-patchlmm(80)-5ep-lora(32)-bs64-lr1e-4"

    # # segment-based
    # square_patches = False
    # max_seg_per_img = 32
    # output_dir="runs/phi-2-multimodal/clevr-seglmm-5ep-lora(64)-bs16-lr1e-4"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # PhiForMultimodalModeling config
    base_llm_name = "microsoft/phi-2"
    w_segment_loss = 0.000
    w_bbox_loss = 0.000/(1000*1000)
    seqae_batch_size = 512
    seqae_requires_grad = False
    seqae_path = '/home/dchenbs/workspace/Seq2Seq-AutoEncoder/runs/Jan02_11-49-33_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_step4350k'

    # lora tuning config
    rank=64
    target_modules='model.*(q_proj|k_proj|v_proj|dense)$'
    additional_tunable_params_keyword = [
        'visual_token_embedding',
        'visual_positional_embedding',
        'segment_modeling_head',
        'embed_tokens',
    ]

    # [ShareGPT-4V] tokenizer & dataset config 
    # max_seg_per_img = 128
    # model_max_length = 256
    # cached_segments = '/home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/cached_segments/sharegpt4v_instruct_gpt4-vision_cap100k'
    # dataset_path = '/home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V'
    # dataset_identifier = 'sharegpt4v_instruct_gpt4-vision_cap100k.json' # or 'sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json' or 'share-captioner_coco_lcs_sam_1246k_1107.json'

    # [CLEVR] tokenizer & dataset config 
    cached_segments = '/home/dchenbs/workspace/Seq2Seq-AutoEncoder/segmentation/cached_segments/clevr_train'
    dataset_path = '/home/dchenbs/workspace/datasets/CLEVR_v1.0'
    dataset_identifier = 'clevr_train'
    sample_truncation = -1
    model_max_length = max_seg_per_img + 100

    model = PhiForMultimodalModeling.from_pretrained(
        base_llm_name,
        w_segment_loss=w_segment_loss,
        w_bbox_loss=w_bbox_loss,
        seqae_batch_size=seqae_batch_size,
        seqae_requires_grad=seqae_requires_grad,
        )

    peft_config = LoraConfig(
        r=rank,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    seqae = Seq2SeqAutoEncoderModel.from_pretrained(seqae_path).eval()
    seqae.enable_caching_latents(
        cache_dir='segmentation/cached_segments/seqae_cache_dir',
        use_existing=True
        )

    load_seqae(model.base_model.model, seqae)
    print(f'Loaded SeqAE: {model.seqae.config}')

    additional_modules_to_save = []
    for name, param in model.named_parameters():
        if 'seqae' in name:
            param.requires_grad = False
        for keyword in additional_tunable_params_keyword:
            if keyword in name:
                param.requires_grad = True
                additional_modules_to_save.append(name)

    print(model)
    print(get_params_count_summary(model, max_name_len=96))
    if model.modules_to_save is not None:
        model.modules_to_save += additional_modules_to_save
    else:
        model.modules_to_save = additional_modules_to_save
    
    if square_patches:
        segmenter = Segmenter(model_name='square_patches', patch_size=patch_size, do_mask_expansion=False)
    else:
        segmenter = None

    tokenizer = MultimodalTokenizer.from_pretrained(
        base_llm_name, 
        trust_remote_code=True, 
        segmenter=segmenter, 
        seqae_config=seqae.config,
        max_seg_per_img=max_seg_per_img,
        model_max_length=model_max_length,
        )

    if not square_patches:
        tokenizer.load_cached_segments(cached_segments=cached_segments)

    # model.resize_token_embeddings(len(tokenizer))
    model.base_model.model.special_token_id_mapping = {
        "<|startofimage|>": tokenizer.convert_tokens_to_ids("<|startofimage|>"),
        "<|endofimage|>": tokenizer.convert_tokens_to_ids("<|endofimage|>"),
        "<|seg|>": tokenizer.convert_tokens_to_ids("<|seg|>"),
        "<|endoftext|>": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        "[PAD]": tokenizer.convert_tokens_to_ids("[PAD]"),
    }
    model.base_model.model.lm_ignore_index = tokenizer.convert_tokens_to_ids("<|seg|>")

    if 'sharegpt4v' in dataset_identifier:
        annotation_file = os.path.join(dataset_path, dataset_identifier+'.json')
        dataset = ShareGPT4V(annotation_file)
        failed_samples = dataset.validate_exist(valid_img_paths=tokenizer.cache)
    elif 'clevr' in dataset_identifier:
        split = dataset_identifier.split('_')[-1]
        dataset = CLEVR(dataset_path=dataset_path, split=split, sample_truncation=sample_truncation)
        if not square_patches:
            failed_samples = dataset.validate_exist(valid_img_paths=tokenizer.cache)

    n_token = []
    n_seg = []
    for i in tqdm.tqdm(range(10)):
        inputs = tokenizer([dataset[i]])
        n_token.append(len(inputs['input_ids'][0]))
        n_seg.append(len(inputs['segment_sequences'][0][0]))
    print(f'Max token length: {max(n_token)}, Max segment tokens: {max(n_seg)}')

    inputs = tokenizer([dataset[0]])
    print(tokenizer.decode(inputs['input_ids'][0]))

    data_collator = DataCollatorForMultimodal(tokenizer)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=200,
        logging_steps=1,
        # max_steps=100000,
        num_train_epochs=5,
        push_to_hub=False,
        bf16=True,
    )

    trainer = MultimodalTrainer(
        model=model,
        train_dataset=dataset,
        # eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_arguments,
        additional_modules_to_save=additional_modules_to_save
    )
    torch.cuda.empty_cache()
    trainer.train()
