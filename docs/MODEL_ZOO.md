# Open-Vocabulary Recognition with Multi-Modal References

## Introduction

This file documents a collection of models reported in our paper.
Training in all cases is done with 4 24GB 3090 GPUs.

#### How to Read the Tables

The "Name" column contains a link to the config file. 
To train a model, run 

```
python train_net_auto.py --num-gpus 4 --config-file /path/to/config/name.yaml
``` 

To evaluate a model with a trained/ pretrained model, run 

```
python train_net_auto.py --num-gpus 4 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
```

#### Fast Demo
```shell
# image detection
python demo.py --config-file configs/mm_classifier_swin_b_box_visualization.yaml --input datasets/coco/val2017/*.jpg --output visualization_imgs --vocabulary lvis --confidence-threshold 0.5 --opts MODEL.WEIGHTS checkpoints/model_final_swinb.pth

# video detection
python demo.py --config-file configs/mm_classifier_swin_b_box_visualization.yaml --input visualization_img_input/1.mp4 --output visualize_videos --vocabulary lvis --confidence-threshold 0.5 --opts MODEL.WEIGHTS checkpoints/model_final_swinb.pth
```
## Open-vocabulary LVIS

| Name                                                                                                                   |  APr |  mAP | Weights                                                          |
|------------------------------------------------------------------------------------------------------------------------|:----:|:----:|------------------------------------------------------------------|
| [lvis-RN50](../configs/mm_classifier.yaml)                     | 21.2 | 30.0 | [model](https://drive.google.com/file/d/19fknyb6MlrgnjWGFYedgB4FRgQwInQQw/view?usp=drive_link) |
| [lvis-SwinB](../configs/mm_classifier_swin_b_imagenet.yaml)                 | 34.4 | 40.9 | [model](https://drive.google.com/file/d/1SM3G4N_ZYnQ5CGH_K36gSTpEvM6O2yvd/view?usp=drive_link) |


#### Note

- The open-vocabulary LVIS setup is LVIS without rare class annotations in training. We evaluate rare classes as novel classes in testing.

- The models with `in-l` use the overlap classes between ImageNet-21K and LVIS as image-labeled data.

