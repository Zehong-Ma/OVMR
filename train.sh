
# seed 1
# sleep 4.5h
# datasets, random_seed, prompt_token_num, gpu_id
bash scripts/mm_cls/train_ovmr.sh imagenet_21k_P 1 2 2

# sleep 3s
datasets="oxford_flowers eurosat fgvc_aircraft stanford_cars imagenet caltech101 oxford_pets food101 sun397 dtd ucf101"

# for dataset in $datasets; do
#     # datasets, random_seed, sample_set, prompt_token_num, eval_mode, eval_tau, gpu_id
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 1 base 2 fusion 10 0
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 2 base 2 fusion 10 0
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 3 base 2 fusion 10 0
# done
# for dataset in $datasets; do
#     # datasets, random_seed, sample_set, prompt_token_num, eval_mode, eval_tau, gpu_id
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 1 base 2 multimodal 10 0
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 2 base 2 multimodal 10 0
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 3 base 2 multimodal 10 0
# done

# for dataset in $datasets; do
#     # datasets, random_seed, sample_set, prompt_token_num, eval_mode, eval_tau, gpu_id
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 1 base 2 vision 10 0
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 2 base 2 vision 10 0
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 3 base 2 vision 10 0
# done

