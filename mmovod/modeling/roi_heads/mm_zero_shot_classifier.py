# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from copy import deepcopy

class MMZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        zs_weight_dim: int = 768,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        classifier_name: str = "multimodal",
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = 1024 # input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.classifier_name = classifier_name
        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias_t = nn.Parameter(torch.ones(1) * use_bias)
            self.cls_bias_v = deepcopy(self.cls_bias_t)
            self.cls_bias_mm = deepcopy(self.cls_bias_t)

        self.linear_t = nn.Linear(input_size, zs_weight_dim)
        self.linear_v = nn.Linear(input_size, zs_weight_dim)
        self.linear_mm = nn.Linear(input_size, zs_weight_dim)
        prompt_model_params = torch.load(zs_weight_path, map_location="cpu") 

        
        t_zs_weight = torch.tensor(
            prompt_model_params["text_classifier"],
            dtype=torch.float32).permute(1, 0).contiguous()  # D x C
        v_zs_weight = torch.tensor(
                prompt_model_params["vision_classifier"],
                dtype=torch.float32).permute(1, 0).contiguous()
        mm_zs_weight = torch.tensor(
                prompt_model_params["mm_classifier"],
                dtype=torch.float32).permute(1, 0).contiguous()
        
        fusion_weight = torch.tensor(
                prompt_model_params["fusion_weight"],
                dtype=torch.float32) # 
        
        t_zs_weight = torch.cat(
            [t_zs_weight, t_zs_weight.new_zeros((zs_weight_dim, 1))],
            dim=1)  # D x (C + 1)
        v_zs_weight = torch.cat(
            [v_zs_weight, v_zs_weight.new_zeros((zs_weight_dim, 1))],
            dim=1)  # D x (C + 1)
        mm_zs_weight = torch.cat(
            [mm_zs_weight, mm_zs_weight.new_zeros((zs_weight_dim, 1))],
            dim=1)  # D x (C + 1)
        fusion_weight = torch.cat([fusion_weight, fusion_weight.new_ones((1,3))], dim=0)

        self.register_buffer("v_zs_weight", v_zs_weight)
        self.register_buffer("mm_zs_weight", mm_zs_weight)
        self.register_buffer("fusion_weight", fusion_weight)
    
        self.register_buffer('t_zs_weight', t_zs_weight)
        # new_fusion_weight = torch.load(zs_weight_path.replace("mm_classifiers", "fusion_weight_map_swin"), map_location="cpu")["fusion_weight"] 
        new_fusion_weight = torch.load(zs_weight_path.replace("coco_mm_classifiers_K30", "fusion_random"), map_location="cpu")["fusion_weight"]
        new_fusion_weight = torch.cat([new_fusion_weight, new_fusion_weight.new_ones((1,3))], dim=0)
        self.register_buffer('new_fusion_weight', new_fusion_weight)

        assert self.t_zs_weight.shape[1] == num_classes + 1, self.t_zs_weight.shape

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS, # -2
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT, # True
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP, # 50
            'classifier_name': cfg.CLASSIFIER_NAME,
        }

    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        
        x_t = self.linear_t(x)
        x_v = self.linear_v(x)
        x_mm = self.linear_mm(x)
        
        x_t = self.norm_temperature * F.normalize(x_t, p=2, dim=1)
        x_v = self.norm_temperature * F.normalize(x_v, p=2, dim=1)
        x_mm = self.norm_temperature * F.normalize(x_mm, p=2, dim=1)

        logits_t = torch.mm(x_t, self.t_zs_weight)
        logits_v = torch.mm(x_v, self.v_zs_weight)
        logits_mm = torch.mm(x_mm, self.mm_zs_weight)
        if self.use_bias:
            logits_t = logits_t + self.cls_bias_t
            logits_v = logits_v + self.cls_bias_v
            logits_mm = logits_mm + self.cls_bias_mm
        
        
        if self.classifier_name == "text":
            return logits_t, logits_v, logits_t, logits_t
        elif self.classifier_name == "vision":
            return logits_v, logits_v, logits_t, logits_v
        elif self.classifier_name == "multimodal":
            return logits_mm, logits_v, logits_t, logits_mm

        if self.classifier_name == "fusion":
            logits_fuse = torch.cat([logits_mm.sigmoid().unsqueeze(-1), 
                                 logits_v.sigmoid().unsqueeze(-1),
                                 logits_t.sigmoid().unsqueeze(-1)], dim=-1)
            logits_fuse = torch.einsum("bcn,cn->bcn", logits_fuse, self.new_fusion_weight).sum(dim=-1)
            logits_fuse = inverse_sigmoid(logits_fuse)
            return logits_fuse, logits_v, logits_t, logits_fuse
    
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
