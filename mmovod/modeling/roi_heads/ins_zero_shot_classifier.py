# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from mmovod.modeling.clip.model import TransformerDropout
from mmovod.modeling.text.text_encoder import build_text_encoder
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class InsZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        stage: int = 1,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = 1024 #input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.topk_candidate = 5
        self.num_classes = num_classes
        self.use_bias = use_bias < 0
        self.stage = stage
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)
            self.ins_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)
        self.ins_linear = nn.Linear(input_size, zs_weight_dim)
        prompt_model_params = torch.load(zs_weight_path, map_location="cpu") 

        # self.ins_token = nn.Parameter(prompt_model_params["ins_token"], requires_grad=True)
        self.ins_aggregator = TransformerDropout(
            width = zs_weight_dim,
            layers = 4,
            heads = zs_weight_dim//64,
            dropout=0.0
        )
        # for v in self.ins_aggregator.parameters():
        #     v.requires_grad = False

        # self.sample_linear = nn.Linear(zs_weight_dim, zs_weight_dim)

        self.ins_aggregator.load_state_dict(prompt_model_params["ins_aggregator"])
        self.token_seq = prompt_model_params["token_seq"].cuda()
        assert self.token_seq.shape[0] == num_classes
        self.exemplar_image_features = prompt_model_params["exemplar_image_features"].cuda()
        self.eos_index = prompt_model_params["eos_index"].cuda()

        zs_weight = torch.tensor(
            prompt_model_params["text_features"],
            dtype=torch.float32).permute(1, 0).contiguous()  # D x C

        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))],
            dim=1)  # D x (C + 1)

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)

        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape
        self.text_encoder = build_text_encoder(pretrain=True, model_name="ViT-B/32")
        for v in self.text_encoder.parameters():
            v.requires_grad = False
        for v in self.ins_aggregator.parameters():
            v.requires_grad = False

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
            'stage': cfg.TRAINING_STAGE, # [1 or 2]
        }

    def forward(self, input_x, classifier=None, labels=None, objectness_scores=None, iteration=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        
        x = self.linear(input_x)
        
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()  # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        logits = torch.mm(x, zs_weight)
        if self.use_bias:
            logits = logits + self.cls_bias
        # return logits, (None, None, None, logits)
        if self.stage==1:
            return logits, (None, None, None, logits)
        # instance learning
        if self.training:
            sample_num = 32 # due to limited memory, only sample 128 boxes from 512*8
        else:
            sample_num = 32
        objectness_scores = torch.cat(objectness_scores)
    
        
        # if self.training:
        #     proposal_with_object = (labels!=self.num_classes).nonzero().squeeze(1)

        topk_objectness, topk_objectness_index = torch.topk(objectness_scores, k=sample_num, dim=-1)
        select_index = topk_objectness_index
        
        select_labels = labels[select_index]
        if len(select_index)<1:
            return logits, (None, None, None, logits)
            # select_index = torch.arange(logits.shape[0])

        topk_logits, topk_index = torch.topk(logits[:,:-1].detach()[select_index], k=self.topk_candidate, dim=-1)
        
        rerank_logits = []
        binary_labels = []
        # sampled_x = self.sample_linear(x[select_index].detach())
        ins_x = self.ins_linear(input_x[select_index].detach())

        if self.norm_weight:
            ins_x = F.normalize(ins_x, dim=-1, p=2)
        
        box_num_per_iter = 8
        for ind in range(ins_x.shape[0]//box_num_per_iter):
            topk_ind = topk_index[box_num_per_iter*ind:box_num_per_iter*(ind+1)]
            # print("topk_ind: ", topk_ind)
            topk_prompt = torch.cat([self.token_seq[topk_ind_] for topk_ind_ in topk_ind], dim=0)
            # print(topk_ind)
            # print("select:", select_index)
            # print("labels:", labels)
            if self.training:
                binary_labels.append(
                    (topk_ind==select_labels[box_num_per_iter*ind:box_num_per_iter*(ind+1)].unsqueeze(1).repeat(1, self.topk_candidate))
                .float())

            # if self.training:
            #     ins_agg_input = torch.cat([
            #         self.ins_token.unsqueeze(1).repeat(1, self.topk_candidate*2+1, 1) +\
            #             (sampled_x[ind:ind+1]/self.norm_temperature).unsqueeze(0).repeat(1, self.topk_candidate*2+1, 1),
            #         self.exemplar_image_features[topk_ind].permute(1,0,2),
            #                                 ], dim=0).float()
            # else:
            ins_agg_input = torch.cat([
                # self.ins_token.unsqueeze(1).repeat(1, self.topk_candidate, 1) +\
            (ins_x[box_num_per_iter*ind:box_num_per_iter*(ind+1)]).unsqueeze(1).repeat(1, self.topk_candidate, 1).flatten(0,1).unsqueeze(0),
            self.exemplar_image_features[box_num_per_iter*ind:box_num_per_iter*(ind+1)].unsqueeze(1).repeat(1, self.topk_candidate, 1, 1).flatten(0,1).permute(1,0,2),
            ], dim=0).float()
                
            ins_tokens = self.ins_aggregator(ins_agg_input).permute(1,0,2)[:, 0:1, :]

            updated_prompt = self.update_ins_prompts(topk_prompt, ins_tokens)
            # updated_prompt = self.prompt_learner.forward_ins(topk_prompt, exemplar_image_features[topk_ind], ins_tokens.repeat(self.topk_candidate,1,1))
            ins_text_feats = self.text_encoder.forward_embedding(updated_prompt, self.eos_index[topk_ind].flatten(0,1)+1)

            # topk_prompt_wo_exem = self.prompt_learner.prompt_tokens[exemplar_label][topk_ind]
            # updated_prompt = self.prompt_learner.update_ins_prompts(topk_prompt_wo_exem, ins_tokens.repeat(self.topk_candidate,1,1))
            # ins_text_feats = self.text_encoder(updated_prompt, tokenized_prompts[topk_ind].argmax(dim=-1)+self.prompt_learner.n_ctx)
            ins_text_feats = (ins_text_feats/ins_text_feats.norm(dim=-1, keepdim=True)).float()
            ins_rerank_logits = torch.einsum("bc, bnc->bn", ins_x[box_num_per_iter*ind:box_num_per_iter*(ind+1)], ins_text_feats.reshape(box_num_per_iter, self.topk_candidate, -1))
            ins_rerank_logits = self.norm_temperature*ins_rerank_logits
            rerank_logits.append(ins_rerank_logits)
        if self.training:
            binary_labels = torch.cat(binary_labels, dim=0)
        # print(binary_labels)
        rerank_logits = torch.cat(rerank_logits, dim=0)
        if self.use_bias:
            rerank_logits = rerank_logits + self.ins_bias
        selected_topk_logits = torch.gather(logits[select_index].detach().clone(), dim=1, index=topk_index)
        # import pdb
        # pdb.set_trace()
        # reranked_topk_logits =  (((selected_topk_logits)+(rerank_logits))/2).sigmoid()
        reranked_topk_logits =  inverse_sigmoid(((selected_topk_logits.sigmoid())**0.75)*((rerank_logits.sigmoid())**0.25))
        # reranked_topk_logits =  inverse_sigmoid((3*(selected_topk_logits).sigmoid()+(rerank_logits).sigmoid())/4)
        logits_clone = logits[select_index].detach().clone() # * rerank_logits.mean(dim=-1, keepdim=True)
        logits_clone.scatter_(dim=1, index=topk_index, src=reranked_topk_logits)
        
        if torch.rand(1)[0]<0.001:
            print("rerank logits: ", rerank_logits[0].sigmoid())
            print("ori_logits: ", selected_topk_logits[0].sigmoid())
            print("reranked_logits: ", reranked_topk_logits[0].sigmoid())
        if self.training is False:
            logits_final = logits.clone()
            logits_final[select_index] = logits_clone
        else:
            logits_final = logits_clone
        return logits, (rerank_logits, binary_labels, select_index, logits_final)
    
    def update_ins_prompts(self, prompt_tokens, ins_tokens):
        return torch.cat([prompt_tokens[:,:2], ins_tokens.type(prompt_tokens.dtype), prompt_tokens[:, 2:-1]], dim=1)
    
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
