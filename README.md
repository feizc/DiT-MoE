## Scaling Diffusion Transformers with Mixture of Experts <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv-2407.11633-b31b1b.svg)](https://arxiv.org/abs/2407.11633)

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper scaling Diffusion Transformers to 16 billion parameters (DiT-MoE).
DiT-MoE as a sparse version of the diffusion Transformer, is scalable and competitive with dense networks while exhibiting highly optimized inference. 

![DiT-MoE framework](visuals/framework.png) 


* 🪐 A PyTorch [implementation](models.py) of DiT-MoE and pre-trained checkpoints in paper
* 🌋 **Rectified flow**-based training and sampling scripts 
* 💥 A [sampling script](sample.py) for running pre-trained DiT-MoE 
* 🛸 A DiT-MoE training script using PyTorch [DDP](train.py) and [deepspeed](train_deepspeed.py)
* ⚡️  A **upcycle** [scripts](https://github.com/feizc/DiT-MoE/blob/main/upcycle.py) to convert dense to MoE ckpts referring [link](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/toolkits/model_checkpoints_convertor/qwen/hf2mcore_qwen1.5_dense_to_moe_convertor.sh) 


### To-do list

- [x] training / inference scripts
- [x] experts routing analysis
- [x] huggingface ckpts

### 1. Training 

You can refer to the [link](https://github.com/facebookresearch/DiT/blob/main/environment.yml) to build the running environment.

To launch DiT-MoE-S/2 (256x256) in the latent space training with `N` GPUs on one node with pytorch DDP:
```bash
torchrun --nnodes=1 --nproc_per_node=N train.py \
--model DiT-S/2 \
--num_experts 8 \
--num_experts_per_tok 2 \
--data-path /path/to/imagenet/train \
--image-size 256 \
--global-batch-size 256 \
--vae-path /path/to/vae
```


For multiple node training, we solve the [bug](https://github.com/facebookresearch/DiT/blob/main/train.py#L149) at original DiT repository, and you can run with 8 nodes as: 
```bash
torchrun --nnodes=8 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr="10.0.0.0" \
    --master_port=1234 \
    train.py \
    --model DiT-B/2 \
    --num_experts 8 \
    --num_experts_per_tok 2 \
    --global-batch-size 1024 \
    --data-path /path/to/imagenet/train \
    --vae-path /path/to/vae
```


For larger model size training, we recommand to use deepspeed with flash attention scripts, and different stage settings including zero2 and zero3 can be seen in config file. 
You can run as:
```bash
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_deepspeed.py \
--deepspeed_config config/zero2.json \
--model DiT-XL/2 \
--num_experts 8 \
--num_experts_per_tok 2 \
--data-path /path/to/imagenet/train \
--vae-path /path/to/vae \
--train_batch_size 32
```

For rectified flow training as [FLUX](https://github.com/black-forest-labs/flux) and [SD3](https://stability.ai/news/stable-diffusion-3), you can run as: 
```bash
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_deepspeed.py \
--deepspeed_config config/zero2.json \
--model DiT-XL/2 \
--rf True \
--num_experts 8 \
--num_experts_per_tok 2 \
--data-path /path/to/imagenet/train \
--vae-path /path/to/vae \
--train_batch_size 32
```
Our experiments show that rectified flow training leads to a better performance as well as faster convergence. 


We also provide all shell scripts for different model size training in file folder *scripts*. 

### 2. Inference 

We include a [`sample.py`](sample.py) script which samples images from a DiT-MoE model. Take care that we use torch.float16 for large model inference. 
```bash
python sample.py \
--model DiT-XL/2 \
--ckpt /path/to/model \
--vae-path /path/to/vae \
--image-size 256 \
--cfg-scale 1.5
```


### 3. Download Models and Data 

We are processing it as soon as possible, the model weights, data and used scripts for results reproduce will be released within two weeks continuously :) 

We use sd vae in this [link](https://huggingface.co/feizhengcong/DiT-MoE/tree/main/sd-vae-ft-mse). 


| DiT-MoE Model     | Image Resolution | Url | Scripts | Loss curve |
|---------------|------------------|---------|---------|-------|
| DiT-MoE-S/2-8E2A | 256x256          | [link](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_s_8E2A.pt)  | DDIM | -|
| DiT-MoE-S/2-16E2A | 256x256         | [link](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_s_16E2A.pt)  | DDIM| -|
| DiT-MoE-B/2-8E2A | 256x256         | [link](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_b_8E2A.pt)  | DDIM | -|
| DiT-MoE-XL/2-8E2A | 256x256         | [link](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_xl_8E2A.pt)   | RF|-|
| DiT-MoE-G/2-16E2A | 512x512         | [link](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_g_16E2A.pt)  | RF|-|


### 4. Expert Specialization Analysis Tools

We provide all the analysis scripts used in the paper.  
You can use [`expert_data.py`](analysis/expert_data.py) to sample data points towards experts ids across different class-conditional.  
Then, a series of files headmap_xx.py are used to visualize the frequency of expert selection for different scenarios.  
Quick validation can be achieved by adjusting the number of sampled data and the save path. 



### 5. BibTeX

```bibtex
@article{FeiDiTMoE2024,
  title={Scaling Diffusion Transformers to 16 Billion Parameters},
  author={Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Jusnshi Huang},
  year={2024},
  journal={arXiv preprint},
}
```


### 6. Acknowledgments

The codebase is based on the awesome [DiT](https://github.com/facebookresearch/DiT) and [DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE) repos. 


