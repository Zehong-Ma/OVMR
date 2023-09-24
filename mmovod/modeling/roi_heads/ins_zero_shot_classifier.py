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
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)
      
        prompt_model_params = torch.load(zs_weight_path, map_location="cpu") 

        self.ins_token = nn.Parameter(prompt_model_params["ins_token"], requires_grad=True)
        self.ins_aggregator = TransformerDropout(
            width = zs_weight_dim,
            layers = 4,
            heads = zs_weight_dim//64,
            dropout=0.1
        )
        # for v in self.ins_aggregator.parameters():
        #     v.requires_grad = False

        self.sample_linear = nn.Linear(zs_weight_dim, zs_weight_dim)

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
        }

    def forward(self, x, classifier=None, labels=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
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

        sample_num=5
        if self.training:
            proposal_with_object = (labels!=self.num_classes).nonzero().squeeze(1)
            if proposal_with_object.shape[0]>=sample_num:
                entropy_val = (-logits[proposal_with_object].sigmoid()*torch.log(logits[proposal_with_object].sigmoid())).sum(dim=-1)
                topk_entropy, topk_entro_index = torch.topk(entropy_val, k=sample_num, dim=-1)
                select_index = proposal_with_object[topk_entro_index]
            else:
                if proposal_with_object.shape[0]>0:
                    select_index = proposal_with_object
                else:
                    select_index = torch.randperm(logits.shape[0])[:sample_num-1]
                    print("easy_proposals!!!!")
            # select_index_rand = torch.randperm(logits.shape[0])[:1].to(x.device)
            # proposal_with_object = (labels!=self.num_classes).nonzero().squeeze(1)
            # if proposal_with_object.shape[0]>=sample_num-1:
            #     random_index = torch.randperm(proposal_with_object.shape[0])[:sample_num-1]
            #     select_index_obejct = proposal_with_object[random_index]
            # else:
            #     if proposal_with_object.shape[0]>0:
            #         select_index_obejct = proposal_with_object
            #     else:
            #         select_index_obejct = torch.randperm(logits.shape[0])[:sample_num-1]
            #         print("easy_proposals!!!!")
            # select_index = torch.cat([select_index_obejct, select_index_rand], dim=0)
            # print(select_index)
        else:
            entropy_val = (-logits.sigmoid()*torch.log(logits.sigmoid())).sum(dim=-1)
            topk_entropy, topk_entro_index = torch.topk(entropy_val, k=32, dim=-1)
            select_index = topk_entro_index
            # select_index = torch.arange(logits.shape[0])

        topk_logits, topk_index = torch.topk(logits[:,:-1][select_index], k=self.topk_candidate, dim=-1)
        
        if self.training:
            gt_index = labels[select_index]
            # replace 
            # if (gt_index==self.num_classes).sum()>0:
            #     gt_index_zero = (gt_index==self.num_classes).nonzero().squeeze(1)
            #     gt_index_zero_replaced = torch.randperm(self.num_classes-1).to(labels.device)[:gt_index_zero.shape[0]]
            #     gt_index[gt_index_zero] = gt_index_zero_replaced
            random_ind = torch.randperm(logits.shape[1]-1)[:self.topk_candidate].to(topk_index.device)
            topk_index = torch.cat([gt_index.unsqueeze(1), topk_index, random_ind.unsqueeze(0).repeat(sample_num, 1)], dim=1)

        rerank_logits = []
        binary_labels = []
        # import pdb
        # pdb.set_trace()
        # print(select_index.shape)
        sampled_x = self.sample_linear(x[select_index].detach())
        for ind in range(sampled_x.shape[0]):
            topk_ind = topk_index[ind]
            # print("topk_ind: ", topk_ind)
            topk_prompt = self.token_seq[topk_ind]
            # print(topk_ind)
            # print("select:", select_index)
            # print("labels:", labels)
            if self.training:
                binary_labels.append((topk_ind==labels[select_index][ind]).float())

            if self.training:
                ins_agg_input = torch.cat([
                    self.ins_token.unsqueeze(1).repeat(1, self.topk_candidate*2+1, 1) +\
                        (sampled_x[ind:ind+1]/self.norm_temperature).unsqueeze(0).repeat(1, self.topk_candidate*2+1, 1),
                    self.exemplar_image_features[topk_ind].permute(1,0,2),
                                            ], dim=0).float()
            else:
                ins_agg_input = torch.cat([
                    self.ins_token.unsqueeze(1).repeat(1, self.topk_candidate, 1) +\
                        (sampled_x[ind:ind+1]/self.norm_temperature).unsqueeze(0).repeat(1, self.topk_candidate, 1),
                    self.exemplar_image_features[topk_ind].permute(1,0,2),
                                            ], dim=0).float()
                
            ins_tokens = self.ins_aggregator(ins_agg_input).permute(1,0,2)[:, 0:self.ins_token.shape[0], :]

            updated_prompt = self.update_ins_prompts(topk_prompt, ins_tokens)
            # updated_prompt = self.prompt_learner.forward_ins(topk_prompt, exemplar_image_features[topk_ind], ins_tokens.repeat(self.topk_candidate,1,1))
            ins_text_feats = self.text_encoder.forward_embedding(updated_prompt, self.eos_index[topk_ind]+2*self.ins_token.shape[0])

            # topk_prompt_wo_exem = self.prompt_learner.prompt_tokens[exemplar_label][topk_ind]
            # updated_prompt = self.prompt_learner.update_ins_prompts(topk_prompt_wo_exem, ins_tokens.repeat(self.topk_candidate,1,1))
            # ins_text_feats = self.text_encoder(updated_prompt, tokenized_prompts[topk_ind].argmax(dim=-1)+self.prompt_learner.n_ctx)
            ins_text_feats = (ins_text_feats/ins_text_feats.norm(dim=-1, keepdim=True))
            ins_rerank_logits = sampled_x[ind:ind+1] @ ins_text_feats.t()
            rerank_logits.append(ins_rerank_logits)
        if self.training:
            binary_labels = torch.stack(binary_labels)
        # print(binary_labels)
        rerank_logits = torch.cat(rerank_logits, dim=0)
        selected_topk_logits = torch.gather(logits[select_index].detach().clone(), dim=1, index=topk_index)
        # import pdb
        # pdb.set_trace()
        # reranked_topk_logits =  (((selected_topk_logits)+(rerank_logits))/2).sigmoid()
        reranked_topk_logits =  inverse_sigmoid(((selected_topk_logits.sigmoid())**0.75)*((rerank_logits.sigmoid())**0.25))
        # reranked_topk_logits =  inverse_sigmoid((3*(selected_topk_logits).sigmoid()+(rerank_logits).sigmoid())/4)
        logits_clone = logits[select_index].detach().clone() # * rerank_logits.mean(dim=-1, keepdim=True)
        logits_clone.scatter_(dim=1, index=topk_index, src=reranked_topk_logits)
        
        if torch.rand(1)[0]<0.001 and self.training:
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
        return torch.cat([prompt_tokens[:,:2], ins_tokens.type(prompt_tokens.dtype), prompt_tokens[:, 2:-self.ins_token.shape[0]]], dim=1)
    
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
