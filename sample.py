# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.image_size in [256, 512]
    assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8 

    if args.model == "DiT-XL/2" or args.model == "DiT-G/2": 
        pretraining_tp=1
        use_flash_attn=True 
        dtype = torch.float16
    else:
        pretraining_tp=2
        use_flash_attn=False 
        dtype = torch.float32

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_experts=args.num_experts, 
        num_experts_per_tok=args.num_experts_per_tok,
        pretraining_tp=pretraining_tp,
        use_flash_attn=use_flash_attn
    ).to(device)

    if dtype == torch.float16:
        model = model.half()
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    
    if args.ckpt is None: 
        print('only for testing middle ckpts')
        if args.model == "DiT-S/2":
            ckpt_path = "results/002-DiT-S-2/checkpoints/ckpt.pt" 
        elif args.model == "DiT-B/2":
            ckpt_path = "results/003-DiT-B-2/checkpoints/ckpt.pt" 
        elif args.model == "DiT-XL/2": 
            ckpt_path = "results/deepspeed-DiT-XL-2/checkpoints/ckpt.pt" 
        else:
            pass 
    else:
        ckpt_path = args.ckpt 


    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device) 
    
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    if dtype == torch.float16: 
        with torch.autocast(device_type='cuda'):
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
    else:
        samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
    
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images: 
    if args.model == "DiT-S/2":
        save_image(samples, "sample_s.png", nrow=4, normalize=True, value_range=(-1, 1))
    elif args.model == "DiT-B/2":
        save_image(samples, "sample_b.png", nrow=4, normalize=True, value_range=(-1, 1))
    elif args.model == "DiT-XL/2":
        save_image(samples, "sample_xl.png", nrow=4, normalize=True, value_range=(-1, 1)) 
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae-path", type=str, default="/maindata/data/shared/multimodal/zhengcong.fei/ckpts/sd-vae-ft-mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument('--num_experts', default=8, type=int,) 
    parser.add_argument('--num_experts_per_tok', default=2, type=int,) 
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--ckpt", type=str, default=None, )
    args = parser.parse_args()
    main(args)
