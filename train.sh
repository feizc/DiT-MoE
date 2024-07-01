torchrun --nnodes=1 --nproc_per_node=8 train.py \
--model DiT-L/2 \
--data-path /maindata/data/shared/multimodal/public/dataset_img_only/imagenet/data/train