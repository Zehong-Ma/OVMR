import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='8'
import torch
import sys
import json
import torchvision.transforms as T
import skimage.io as io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0,"./")
from models.classifier import build_classifier
from models.clip.clip import tokenize
from models.clip.model import build_visual_model
from preprocess.class_category import COCO_CLASSES, LVIS_CLASSES
import shutil
import util.misc as utils

def img_transform():
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    height = 224
    width = 224
    preprocess = T.Compose(
        [
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return preprocess


class CustomDataset(Dataset):
    def __init__(self, class_name, dataset_name="COCO"):
 
        self.preprocess = img_transform()
        self.img_root = "/data0/mzh/OVD/datasets/retrieval_data"
        self.img_dir = os.path.join(self.img_root, dataset_name, class_name)
        self.img_paths = os.listdir(self.img_dir)
        self.img_paths = [os.path.join(self.img_dir, im_path) for im_path in self.img_paths if im_path.endswith("jpg") or im_path.endswith("png")]
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.fromarray(io.imread(self.img_paths[idx])).convert("RGB")
        # print(image)
        image = self.preprocess(image)     
        filename =  self.img_paths[idx]   
        return image, filename

def get_args_parser():
    parser = argparse.ArgumentParser('Class_Embed', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--backbone', default="clip_RN50", type=str)
    parser.add_argument('--text_len', default=77, type=int)
    parser.add_argument('--classifier_cache', default='', type=str)
    parser.add_argument('--output_path', default="./data/class_img_embed_all.pt", type=str)
    return parser

def generate_class_embed(positional_embed):
    
    model_name = "clip_RN50"
    state_dict = torch.jit.load(os.path.expanduser("~/.cache/clip")+"/%s.pt"%model_name[5:], map_location="cpu").state_dict()
    model_img = build_visual_model(state_dict)
    
    model_img = model_img.float().cuda()
    model_img.eval()
    model_img.attnpool.positional_embedding.data = positional_embed 
   
    mean_img_embeds = []
  
    with torch.no_grad():
        for cat_id, class_name in enumerate(COCO_CLASSES):
            dataset = CustomDataset(class_name)
            dataloader = DataLoader(dataset, batch_size=200, drop_last=False, num_workers=6, shuffle=False)
            img_embeds = []
            filenames = []
            for i, (img, filename) in enumerate(dataloader):
                img = img.cuda()
                img_embed = model_img(img)
                img_embed = img_embed/img_embed.norm(dim=-1, p=2, keepdim=True)
                img_embeds.append(img_embed)
            img_embeds = torch.cat(img_embeds)
            mean_img_embed = img_embeds.mean(dim=0, keepdim=True)
            # mean_img_embed = select_img_embeds.mean(dim=0, keepdim=True)
            mean_img_embed = mean_img_embed/mean_img_embed.norm(dim=-1, p=2, keepdim=True)
    
            mean_img_embeds.append(mean_img_embed)
          
        mean_img_embeds = torch.cat(mean_img_embeds)
        if utils.is_main_process():
            torch.save(
                {
                "mean_img_embeds": mean_img_embeds,
            
                },
                "./data/class_img_embed_roi_box.pt"
            )
            print("save new class_embed")
    return mean_img_embeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Class_Relation", parents=[get_args_parser()])
    args = parser.parse_args()
    model_name = args.backbone
    state_dict = torch.jit.load(os.path.expanduser("~/.cache/clip")+"/%s.pt"%model_name[5:], map_location="cpu").state_dict()
    model_img = build_visual_model(state_dict)
    
    model_img = model_img.cuda()
    model_img.eval()
    model_text = build_classifier(args)
    model_text = model_text.cuda()
    model_text.eval()
    
    cat2id = {}
    mean_img_embeds = []
    filenames_list = []
    LVIS_CLASSES.extend(COCO_CLASSES)
    all_classes = LVIS_CLASSES
    with torch.no_grad():
        
        for cat_id, class_name in enumerate(all_classes):
            if class_name in COCO_CLASSES:
                dataset_name = "COCO"
            else:
                dataset_name = "LVIS"
            dataset = CustomDataset(class_name, dataset_name=dataset_name)
            dataloader = DataLoader(dataset, batch_size=200, drop_last=False, num_workers=8, shuffle=False)
            text_embed = model_text.forward_feature([class_name]).float()
            img_embeds = []
            filenames = []
            # print(class_name)
            for i, (img, filename) in enumerate(dataloader):
                img = img.cuda()
                img_embed = model_img(img)
                img_embed = img_embed/img_embed.norm(dim=-1, p=2, keepdim=True)
                img_embeds.append(img_embed)
                filenames.extend(np.array(filename))
                
            filenames = np.array(filenames)
            img_embeds = torch.cat(img_embeds).float()
        
            text2img_sim = torch.einsum("pc, qc->pq", text_embed, img_embeds)

            score, index = torch.topk(text2img_sim, dim=-1, k=img_embeds.shape[0]//2)
            select_img_embeds = torch.gather(img_embeds, dim=0, index=index.reshape(-1, 1).repeat(1, img_embeds.shape[-1]))
        
            # mean_img_embed = (score.reshape(-1, 1)* select_img_embeds).sum(dim=0, keepdim=True).float()#.mean(dim=0, keepdim=True)
            # mean_img_embed = img_embeds.mean(dim=0, keepdim=True).float()
            mean_img_embed = select_img_embeds.mean(dim=0, keepdim=True).float()
            mean_img_embed = mean_img_embed/mean_img_embed.norm(dim=-1, p=2, keepdim=True)
            text2meanImg_sim = text_embed@mean_img_embed.t()
            print("%s: "%class_name, text2meanImg_sim.item())
            filenames = filenames[index.squeeze(0).cpu()]
            cat2id[class_name] = cat_id
            mean_img_embeds.append(mean_img_embed)
            cat2id[class_name+'_filenames'] = list(filenames) 
            cat2id[class_name+'_sim'] = text2meanImg_sim.item()
            # if cat_id>2:
            #     break
            class_img_root = "/data0/mzh/OVD/datasets/retrieval_data/all_data_filtered/%s"%class_name
            if os.path.exists(class_img_root) is False:
                os.makedirs(class_img_root)
            for file_name in filenames:
                shutil.copy(file_name, file_name.replace("COCO", "all_data_filtered").replace("LVIS", "all_data_filtered"))
            # print(class_name)
        mean_img_embeds = torch.cat(mean_img_embeds)
        torch.save(
            {
            "mean_img_embeds": mean_img_embeds,
           
            },
            args.output_path
        )
        with open("./data/class_img_embed_cat2id_all.json", "w") as f:
            json.dump(cat2id, f, indent=1)
        

            
        
