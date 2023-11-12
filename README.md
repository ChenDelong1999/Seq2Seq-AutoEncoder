# Sequence-to-sequence Autoencoder

### Install

```bash
conda create -n seq2seq-ae python=3.11 -y
conda activate seq2seq-ae
pip install -r requirements.txt
```

### Data

#### LVIS

https://www.lvisdataset.org/dataset

### Training on Images

```bash
# on 3090ti GPU, MNIST
CUDA_VISIBLE_DEVICES=1,2 python main.py \
    --dataset mnist --img_size 28 \
    --eval_interval 1000 --save_interval=10000 \
    --batch_size=32 --gradient_accumulation_steps 1 --lr=5e-5  --n_generation=3 \
    --d_model 512 --encoder_layers 6 --decoder_layers 6 \
    --encoder_attention_heads 8 --decoder_attention_heads 8 \
    --encoder_ffn_dim 512 --decoder_ffn_dim 512 \
    --num_queries 16 --d_latent 128
```


```bash
# on A100/A800 80G GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --dataset stl10 --img_size 32 --min_resize_ratio 0.2 \
    --epochs 50 \
    --pretrained "/home/dchenbs/workspace/Seq2Seq-AutoEncoder/runs/Nov08_09-09-00_eez116-stl10-[model=0.73B-16queries]-[lr1e-05-bs8x1step-4gpu]/checkpoints/checkpoint_ep28_step95k" \
    --eval_interval 5000 --save_interval=5000 \
    --batch_size=8 --gradient_accumulation_steps 1 --lr=1e-5  --n_generation=1 \
    --d_model 1024 --encoder_layers 24 --decoder_layers 24 \
    --encoder_attention_heads 8 --decoder_attention_heads 8 \
    --encoder_ffn_dim 4096 --decoder_ffn_dim 4096 \
    --num_queries 16 --d_latent 1024
```

### Training on Segments

on A100/A800 80G GPU, [COCO/LVIS], HKUST, small-scale
```bash
# --dataset coco --data_dir '/home/dchenbs/workspace/datasets/coco2017' \
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --master_port '12345' \
    --dataset lvis --data_dir '/home/dchenbs/workspace/datasets/lvis,/home/dchenbs/workspace/datasets/coco2017' \
    --img_size 32 --min_resize_ratio 0.8 \
    --epochs 10 \
    --eval_interval 5000 --save_interval=5000 \
    --batch_size=16 --gradient_accumulation_steps 1 --lr=1e-5  --n_generation=1 \
    --d_model 1024 --encoder_layers 12 --decoder_layers 12 \
    --encoder_attention_heads 8 --decoder_attention_heads 8 \
    --encoder_ffn_dim 4096 --decoder_ffn_dim 4096 \
    --num_queries 16 --d_latent 1024
```


on A100/A800 80G GPU, [SA-1B], Xiaobing, large-scale
```bash
# --dataset sa1b --data_dir '/home/dchenbs/workspace/datasets/sa1b' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --master_port '12345'  --torch_compile \
    --dataset sa1b --data_dir '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/sa1b/SA-1B/EXTRACTED' \
    --img_size 48 --min_resize_ratio 0.8 --size_warmup_steps 10000 --size_warmup_min 0.25 \
    --epochs 3 \
    --eval_interval 10000 --save_interval=10000 \
    --batch_size=4 --num_workers 16 --gradient_accumulation_steps 2 --lr=1e-5  --n_generation=1 \
    --d_model 768 --encoder_layers 24 --decoder_layers 24 \
    --encoder_attention_heads 8 --decoder_attention_heads 8 \
    --encoder_ffn_dim 4096 --decoder_ffn_dim 4096 \
    --num_queries 16 --d_latent 1024
```