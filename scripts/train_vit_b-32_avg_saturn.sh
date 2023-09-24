export CUDA_VISIBLE_DEVICES=4,5,6,7,8,9

# python train_net_auto.py --num-gpus 4 --config-file configs/lvis-base_r50_4x_clip_ours_vit-b32_5_scale.yaml

# python train_net_auto.py --num-gpus 4 --config-file configs/lvis-base_r50_4x_clip_ours_vit-b32_multi-modal_agg.yaml

# python train_net_auto.py --num-gpus 4 --config-file configs/lvis-base_r50_4x_clip_ours_vit-b32_multi-modal_avg.yaml

# python train_net_auto.py --num-gpus 2 --config-file configs/lvis-base_r50_4x_clip_ours_vit-b32_classname.yaml
python train_net_auto.py --num-gpus 6 --config-file configs/saturn_lvis-base_r50_4x_clip_multi_modal_avg.yaml
