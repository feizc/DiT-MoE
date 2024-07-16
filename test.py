import torch 
from download import find_model
from models import DiT_models
from thop import profile
from diffusion import create_diffusion 



def calculate_params_and_flops(): 
    image_size = 256 
    model = "DiT-G/2" 
    num_classes = 1000 
    device = "cuda" 
    ckpt_path = None # "results/002-DiT-S-2/checkpoints/1350000.pt" 

    latent_size = image_size // 8
    model = DiT_models[model](
        input_size=latent_size,
        num_classes=num_classes,
        num_experts=16, 
        num_experts_per_tok=2,
    ).to(device)

    if ckpt_path is not None: 
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
    
    model.eval() 

    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(1, 4, 32, 32).cuda()
    t = torch.randint(1, 1000, (1,)).cuda()
    y = torch.randint(1, 1000, (1,)).cuda()
    flops, _ = profile(model, inputs=(x, t, y)) 
    print('FLOPs = ' + str(flops * 2/1000**3) + 'G')
    # plot for expert routing 
    

def image_class_expert_ratio(): 
    import os 
    import json 
    from models import selected_ids_list 

    image_size = 256 
    model = "DiT-S/2" 
    num_classes = 1000 
    device = "cuda" 
    ckpt_path = "results/002-DiT-S-2/checkpoints/1700000.pt" 
    num_sampling_steps = 250 
    cfg_scale = 4.0 
    every_class_sample = 50

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)

    latent_size = image_size // 8
    model = DiT_models[model](
        input_size=latent_size,
        num_classes=num_classes,
        num_experts=8, 
        num_experts_per_tok=2,
    ).to(device)

    if ckpt_path is not None:  
        print('load from: ', ckpt_path)
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)

    model.eval() 
    diffusion = create_diffusion(str(num_sampling_steps))
    
    for i in range(1000): 
        experts_ids = []
        for j in range(every_class_sample):
            class_labels = [i]
            # Create sampling noise:
            n = len(class_labels)
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = torch.tensor(class_labels, device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            print(i, j)
            print(len(selected_ids_list), len(selected_ids_list[0]), len(selected_ids_list[0][0]))
            tmp_ids_list = selected_ids_list[-3000:]
            print(len(tmp_ids_list), len(tmp_ids_list[0]), len(tmp_ids_list[0][0]))
            print(tmp_ids_list[0][0])
            experts_ids.append(tmp_ids_list)
            #break 
        #continue 
        print(len(experts_ids))
        tgt_path = os.path.join('experts', str(i)+'.json')
        with open(tgt_path, 'w') as f:
            json.dump(experts_ids, f,)
        


# image_class_expert_ratio()
calculate_params_and_flops() 
