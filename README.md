## Scaling Diffusion Transformers with Mixture of Experts <br><sub>Official PyTorch Implementation</sub>

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper scaling Diffusion Transformers to 16 billion parameters (DiT-MoE).
DiT-MoE as a sparse version of the diffusion Transformer, is scalable and competitive with dense networks while exhibiting highly optimized inference. 

![DiT-MoE framework](visuals/framework.png) 


* 🪐 A PyTorch [implementation](models.py) of DiT-MoE
* ⚡️ Pre-trained checkpoints in paper
* 💥 A [sampling script](sample.py) for running pre-trained DiT-MoE 
* 🛸 A DiT-MoE [training script](train.py) using PyTorch DDP 


### To do list

1. training / inference scripts
2. huggingface ckpts
3. experts routing analysis
4. synthesized data

### 1. Training 


### 2. CKPTs and Data 


### 3. Expert Specialization Analysis Tools


### 4. BibTeX

```bibtex
@article{FeiDiTMoE2024,
  title={Scaling Diffusion Transformers to 16 Billion Parameters},
  author={Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Jusnshi Huang},
  year={2024},
  journal={arXiv preprint},
}
```


### 5. Acknowledgments

The codebase is based on the awesome [DiT](https://github.com/facebookresearch/DiT) and [DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE) repos. 


