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
# on 10GB GPU, CIFAR10
CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --dataset cifar10 --img_size 32 --master_port 12354 \
    --epochs 100 \
    --pretrained '/home/delong/workspace/seq2seq_autoencoder/runs/Nov04_19-30-05_black-rack-0-cifar10-[model=0.03B-16queries]-[lr5e-05-bs16x1step-3gpu]/checkpoints/checkpoint_ep307_step320k' \
    --eval_interval 1000 --save_interval=10000 \
    --batch_size=16 --gradient_accumulation_steps 1 --lr=5e-5  --n_generation=3 \
    --d_model 512 --encoder_layers 6 --decoder_layers 6 \
    --encoder_attention_heads 8 --decoder_attention_heads 8 \
    --encoder_ffn_dim 512 --decoder_ffn_dim 512 \
    --num_queries 16 --d_latent 128
```

```bash
# on A100 80G GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --dataset cifar10 --img_size 64 --min_resize_ratio 0.2 \
    --epochs 50 \
    --eval_interval 10000 --save_interval=10000 \
    --batch_size=1 --gradient_accumulation_steps 8 --lr=1e-5  --n_generation=1 \
    --d_model 1024 --encoder_layers 24 --decoder_layers 24 \
    --encoder_attention_heads 8 --decoder_attention_heads 8 \
    --encoder_ffn_dim 4096 --decoder_ffn_dim 4096 \
    --num_queries 16 --d_latent 1024

```
