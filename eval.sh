# datasets="eurosat oxford_flowers  fgvc_aircraft stanford_cars imagenet caltech101 oxford_pets food101 sun397 dtd ucf101"
datasets="eurosat"
for dataset in $datasets; do
    # datasets, random_seed, sample_set, prompt_token_num, eval_mode, eval_tau, gpu_id
    bash scripts/mm_cls/eval_ovmr.sh $dataset 1 base 2 fusion 10 2
    bash scripts/mm_cls/eval_ovmr.sh $dataset 2 base 2 fusion 10 2
    bash scripts/mm_cls/eval_ovmr.sh $dataset 3 base 2 fusion 10 2
done
for dataset in $datasets; do
    # datasets, random_seed, sample_set, prompt_token_num, eval_mode, eval_tau, gpu_id
    bash scripts/mm_cls/eval_ovmr.sh $dataset 1 base 2 multimodal 10 2
    bash scripts/mm_cls/eval_ovmr.sh $dataset 2 base 2 multimodal 10 2
    bash scripts/mm_cls/eval_ovmr.sh $dataset 3 base 2 multimodal 10 2
done

# for dataset in $datasets; do
#     # datasets, random_seed, sample_set, prompt_token_num, eval_mode, eval_tau, gpu_id
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 1 base 2 vision 10 0
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 2 base 2 vision 10 0
#     bash scripts/mm_cls/eval_ovmr.sh $dataset 3 base 2 vision 10 0
# done

