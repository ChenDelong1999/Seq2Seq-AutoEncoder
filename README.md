# Sequence-to-sequence Autoencoder

### Install

```bash
conda create -n seq2seq-ae python=3.11 -y
conda activate seq2seq-ae
pip install -r requirements.txt
```

```bash
# For RepViT-SAM
git clone https://github.com/THU-MIG/RepViT
cd RepViT/sam
pip install -e .
mkdir weights && cd weights
wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt
```


```bash
# TRL - Transformer Reinforcement Learning
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .
```

### Data

#### SA-1B

- xiaobing: /cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/sa1b/SA-1B/EXTRACTED
- 116@hkust: /home/dchenbs/workspace/datasets/sa1b

#### LVIS

https://www.lvisdataset.org/dataset

116@hkust: /home/dchenbs/workspace/datasets/lvis,/home/dchenbs/workspace/datasets/coco2017

#### ShareGPT4V

https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md


### Pre-Training

```bash
# 2023.11.14
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --model_config 'configs/model_config/model.json' \
    --training_config 'configs/training_config/sa1b.json'
```


```bash
# 2023.11.28
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --model_config 'runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step1000k' \
    --training_config 'configs/training_config/sa1b.json'
```

```bash
# 2023.12.28 / 2024.01.02 updated encoder attention mask
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --model_config 'runs/Nov28_20-50-04_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep2_step3200k' \
    --training_config 'configs/training_config/sa1b.json'
```



### High-resolution Continual Pre-training

```bash
# 2023.11.22
# model_config not endwith `.json` means load model from pretrained
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --model_config 'runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step1000k' \
    --training_config 'configs/training_config/sa1b-HD2304@xiaobing739.json' \
    --new_data_seq_length 2304
```


### Evaluation

```bash
conda activate seq2seq-ae
cd /home/dchenbs/workspace/Seq2Seq-AutoEncoder

CUDA_VISIBLE_DEVICES=4 python evaluation.py \
    --model_dir "runs/Nov28_20-50-04_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep1_step3000k" \
    --loss-evaluation --loss-step 20 --loss-batch-size 50 \
    --reconstruction-evaluation --reconstruction-step 20 --reconstruction-batch-size 50 --reconstruction-num-visualization 100 \
    --representation-evaluation --representation-truncation 30000 \
```


```bash

CUDA_VISIBLE_DEVICES=4 python evaluation.py \
    --model_dir "/home/dchenbs/workspace/Seq2Seq-AutoEncoder/runs/Jan02_11-49-33_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_step50k" \
    --representation-evaluation --representation-truncation 30000  
```


