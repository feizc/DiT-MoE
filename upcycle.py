from models import DiT_models
import argparse
import torch
from collections import OrderedDict
import re 
from copy import deepcopy 

def main(args):
    print("convert dense dit to moe dit")
    assert args.image_size in [256, 512]
    assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8 

    pretraining_tp=1
    use_flash_attn=True 

    
    moe_model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_experts=args.num_experts, 
        num_experts_per_tok=args.num_experts_per_tok,
        pretraining_tp=pretraining_tp,
        use_flash_attn=use_flash_attn
    )
    param = sum(p.numel() for p in moe_model.parameters())
    print("DiT Parameters: ", param)
    
    # test for loading
    # moe_state_dict = torch.load("upcycle.pt", map_location=lambda storage, loc: storage) 
    # moe_model.load_state_dict(moe_state_dict)
    # print('load success!')
    
    dense_model = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    
    moe_model_dict = OrderedDict()
    with torch.no_grad():
        for k, p in moe_model.named_parameters(): 
            # print(k, p.size())
            if k in dense_model.keys():
                if p.size() == dense_model[k].size(): 
                    moe_model_dict[k] = dense_model[k]
                else: 
                    # rounting network initialize with norm (0, 0.02)
                    print('initialize with norm:', k) 
                    moe_model_dict[k] = p.normal_(0, 0.02)
            else:
                tgt = deepcopy(k)
                for num in range(args.num_experts): 
                    pattern = "experts." + str(num) + "."
                    tgt = tgt.replace(pattern, 'experts.0.')
                print(k, tgt)
                moe_model_dict[k] = dense_model[tgt]
    
    torch.save(moe_model_dict, 'upcycle.pt')
    
            

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-G/2")
    parser.add_argument("--ckpt", type=str, default="results/deepspeed-DiT-G-2-rf-recycle/checkpoints/tmp.pt") 
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument('--num_experts', default=16, type=int,) 
    parser.add_argument('--num_experts_per_tok', default=2, type=int,) 
    args = parser.parse_args()
    main(args)