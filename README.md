# Sequence-to-sequence Autoencoder

### Install

```bash
conda create -n seq2seq-ae python=3.11 -y
conda activate seq2seq-ae
pip install -r requirements.txt
```

### Data

#### SA-1B

- xiaobing: /cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/sa1b/SA-1B/EXTRACTED
- 116@hkust: /home/dchenbs/workspace/datasets/sa1b

#### LVIS

https://www.lvisdataset.org/dataset

116@hkust: /home/dchenbs/workspace/datasets/lvis,/home/dchenbs/workspace/datasets/coco2017

### Pre-Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --model_config 'configs/model_config/model.json' \
    --training_config 'configs/training_config/sa1b.json'
```


### High-resolution Continual Pre-training

```bash
# model_config not endwith `.json` means load model from pretrained
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --model_config 'runs/Nov14_17-31-06_host19-SA1B-[327MB-16queries-1024]-[lr1e-05-bs16x1step-8gpu]/checkpoints/checkpoint_ep0_step1000k' \
    --training_config 'configs/training_config/sa1b-HD2304@xiaobing739.json' \
    --new_data_seq_length 2304
```