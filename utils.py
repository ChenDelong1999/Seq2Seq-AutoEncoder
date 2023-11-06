
from torchvision import datasets
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.distributed import init_process_group

def ddp_setup(rank: int, world_size: int, MASTER_PORT="12355"):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = MASTER_PORT
  init_process_group(backend="nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)


def get_params_count(model, max_name_len: int = 64):
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  return params, total_trainable_params, total_nontrainable_params


def get_params_count_summary(model, max_name_len: int = 64):
  padding = 64
  params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
  param_counts_text = ''
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  for name, param_count, shape, trainable in params:
      truncated_name = name[:max_name_len]  # Truncate the name if it's too long
      param_counts_text += f'| {truncated_name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
  param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  return param_counts_text


def save_hf_pretrained_model(model, checkpoint_dir):
  model.module.save_pretrained(checkpoint_dir)
  print(f'Saved huggingface model: {checkpoint_dir}')