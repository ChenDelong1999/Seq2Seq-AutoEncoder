# Sequence-to-sequence Autoencoder

### Install

```bash
conda create -n seq2seq-ae python=3.11
conda activate seq2seq-ae
pip install -r requirements.txt
```


### Training

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
    --eval_interval 5000 --save_interval=5000 \
    --batch_size=8 --gradient_accumulation_steps 1 --lr=1e-5  --n_generation=1 \
    --d_model 1024 --encoder_layers 24 --decoder_layers 24 \
    --encoder_attention_heads 8 --decoder_attention_heads 8 \
    --encoder_ffn_dim 4096 --decoder_ffn_dim 4096 \
    --num_queries 16 --d_latent 1024
```

```bash
# on A100/A800 80G GPU, [COCO]
CUDA_VISIBLE_DEVICES=4,5 python main.py --master_port '12345' \
    --dataset coco --img_size 32 --min_resize_ratio 0.2 \
    --data_dir '/home/dchenbs/workspace/datasets/coco2017' \
    --epochs 50 \
    --eval_interval 100000000 --save_interval=5000 \
    --batch_size=16 --gradient_accumulation_steps 1 --lr=1e-5  --n_generation=1 \
    --d_model 1024 --encoder_layers 12 --decoder_layers 12 \
    --encoder_attention_heads 8 --decoder_attention_heads 8 \
    --encoder_ffn_dim 4096 --decoder_ffn_dim 4096 \
    --num_queries 16 --d_latent 1024
```
