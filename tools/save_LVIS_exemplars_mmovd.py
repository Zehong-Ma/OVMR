#!/usr/bin/env python
# coding: utf-8

import clip
import json
from PIL import Image
import sys
import argparse
from lvis import LVIS
import os
# from pprint import pprint
import random
import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision.transforms as tvt
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# PATHS = {
#     "imagenet21k": "",
#     "visual_genome": "/scratch/local/hdd/prannay/datasets/VisualGenome/",
#     "lvis": "/scratch/local/hdd/prannay/datasets/coco/",
# }


def _convert_image_to_rgb(image: Image.Image):
    return image.convert("RGB")


def get_crop(img, bb, context=0.0, square=True):
    # print(bb)
    x1, y1, w, h = bb
    W, H = img.size
    y, x = y1 + h / 2.0, x1 + w / 2.0
    h, w = h * (1. + context), w * (1. + context)
    if square:
        w = max(w, h)
        h = max(w, h)
    # print(x, y, w, h)
    x1, x2 = x - w / 2.0, x + w / 2.0
    y1, y2 = y - h / 2.0, y + h / 2.0
    # print([x1, y1, x2, y2])
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    # print([x1, y1, x2, y2])
    bb_new = [int(c) for c in [x1, y1, x2, y2]]
    # print(bb_new)
    crop = img.crop(bb_new)
    return crop


def run_crop(d, paths, context=0.4, square=True):
    dataset = d['dataset']
    file_name = os.path.join(paths[dataset], d['file_name'])
    # with open(file_name, "rb") as f:
    img = Image.open(file_name)
    if dataset == "imagenet21k":
        bb = [0, 0, 0, 0]
        return img
    elif dataset == "lvis":
        bb = [
            int(c)
            for c in [
                d['bbox'][0] // 1,
                d['bbox'][1] // 1,
                d['bbox'][2] // 1 + 1,
                d['bbox'][3] // 1 + 1
            ]
        ]
    elif dataset == "visual_genome":
        bb = [int(c) for c in [d['x'], d['y'], d['w'], d['h']]]
    return get_crop(img, bb, context=context, square=square)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann-path",
        type=str,
        default= "datasets/metadata/lvis_image_exemplar_dict_K-030_own.json"#"datasets/metadata/lvis_image_exemplar_dict_K-005_author.json"
    )
    parser.add_argument(
        "--train_ann_path",
        type=str,
        default= "datasets/metadata/lvis_image_exemplar_dict_K-030_own.json"#"datasets/metadata/lvis_image_exemplar_dict_K-016_own.json"
    )
    
    parser.add_argument(
        "--lvis-img-dir",
        type=str,
        default="datasets/coco/"
    )
    parser.add_argument(
        "--imagenet-img-dir",
        type=str,
        default="datasets/imagenet/imagenet21k_P"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets/lvis_exemplars_mmovod_K30"
    )
    parser.add_argument(
        "--visual-genome-img-dir",
        type=str,
        default="datasets/VisualGenome/"
    )
    parser.add_argument("--num-augs", type=int, default=5)
    parser.add_argument("--nw", type=int, default=8)

    args = parser.parse_args()
    return args


def main(args):

    anns_path = args.ann_path

    run(anns_path, args)


class CropDataset(Dataset):
    def __init__(
        self,
        exemplar_dict,
        train_exemplar_dict,
        args,
    ):
        self.exemplar_dict = exemplar_dict
        # with open("datasets/metadata/lvis_image_exemplar_dict_K-005_author.json", "r") as fp:
        #     self.exemplar_dict_mmovd = json.load(fp)
        self.train_exemplar_dict = train_exemplar_dict
        self.paths = {
            "imagenet21k": args.imagenet_img_dir,
            "visual_genome": args.visual_genome_img_dir,
            "lvis": args.lvis_img_dir,
        }
        self.save_dir = args.save_dir
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)
            # os.makedirs(os.path.join(self.save_dir, "train"))
            os.makedirs(os.path.join(self.save_dir, "val"))
            # os.makedirs(os.path.join(self.save_dir, "all"))

    def __len__(self):
        return len(self.exemplar_dict)

    def __getitem__(self, idx):
        val_anns = self.exemplar_dict[idx]
        train_anns = self.train_exemplar_dict[idx]
        val_crops = [run_crop(ann, self.paths) for ann in val_anns]
        train_crops = [run_crop(train_ann, self.paths) for train_ann in train_anns]
        # add the tta in here somewhere
        # num_per_class = len(val_crops)
        category_id = idx
        save_folder = os.path.join(self.save_dir, "all", str(category_id))

        if os.path.exists(os.path.join(self.save_dir, "val", str(category_id))) is False:
            os.makedirs(os.path.join(self.save_dir, "val", str(category_id)))
        if os.path.exists(os.path.join(self.save_dir, "train", str(category_id))) is False:
            os.makedirs(os.path.join(self.save_dir, "train", str(category_id)))
        # train_file_names = [ann_['file_name'].split("/")[-1] for ann_ in train_anns]
        val_num_ins = 0
        for i in range(len(val_crops)):
            save_file_name = "val_%d_%s_%s"%(val_num_ins, val_anns[i]['dataset'], val_anns[i]['file_name'].split("/")[-1])
            if os.path.exists(os.path.join(self.save_dir, "val", str(category_id))) is False:
                os.makedirs(os.path.join(self.save_dir, "val", str(category_id)))
            val_crops[i].save(os.path.join(self.save_dir, "val", str(category_id), save_file_name))
            val_num_ins +=1
        train_num_ins = 0
        for j in range(len(train_crops)):
            save_file_name = "train_%d_%s_%s"%(train_num_ins, train_anns[j]['dataset'], train_anns[j]['file_name'].split("/")[-1])
            if os.path.exists(os.path.join(self.save_dir, "train", str(category_id))) is False:
                os.makedirs(os.path.join(self.save_dir, "train", str(category_id)))
            train_crops[j].save(os.path.join(self.save_dir, "train", str(category_id), save_file_name))
            train_num_ins += 1
        return category_id, val_num_ins


def run(anns_path,  args):
    random.seed(100000 )
    torch.manual_seed(100000 )
    
    with open(anns_path, "r") as fp:
        exemplar_dict = json.load(fp)
    
    with open(args.train_ann_path, "r") as fp2:
        train_exemplar_dict = json.load(fp2)
    
    dataset = CropDataset(exemplar_dict, train_exemplar_dict, args)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    # chosen_anns_all = []
    all_samples_num = 0
    for crops, num_ins in tqdm(dataloader, total=len(dataloader)):
     
        print(crops[0], "num:", num_ins)
        all_samples_num+=num_ins
    print("total_num: ", all_samples_num)


if __name__ == "__main__":
    args = get_args()
    main(args)
