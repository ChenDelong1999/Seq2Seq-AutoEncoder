# Sequence-to-sequence Autoencoder

### Install

```bash
conda create -n seq2seq-ae python=3.11 -y
conda activate seq2seq-ae
pip install -r requirements.txt
```

### Data

#### SA-1B

xiaobing: /cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/sa1b/SA-1B/EXTRACTED
116@hkust: /home/dchenbs/workspace/datasets/sa1b

#### LVIS

https://www.lvisdataset.org/dataset

116@hkust: /home/dchenbs/workspace/datasets/lvis,/home/dchenbs/workspace/datasets/coco2017

### Training


```bash
CUDA_VISIBLE_DEVICES=4,5 python main.py \
    --model_config 'configs/model_config/model.json' \
    --training_config 'configs/training_config/sa1b.json'
```