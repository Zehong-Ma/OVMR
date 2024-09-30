import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from typing import Optional, List
from torch import Tensor

import random
import numpy as np
from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import time
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import Transformer, LayerNorm, TransformerDropout, VisionTransformer
from copy import deepcopy
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def load_clip_visual(model_name):
    backbone_name = model_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model.visual

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float16
        # self.set_short_attn_mask()
    
    def set_short_attn_mask(self, short_len=40):
        mask = torch.empty(short_len, short_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        for block in self.transformer.resblocks:
            block.attn_mask = mask

    def forward(self, prompts, eos_index):
        x = prompts.type(self.dtype)  + self.positional_embedding.type(self.dtype)[:prompts.shape[1]]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_index] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.cfg = cfg
        n_ctx = cfg.TRAINER.COCOOP.N_CTX # default is 1 or 4
        dtype = torch.float16
        self.dtype = dtype
        vis_dim = clip_model.visual.output_dim
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.num_class = len(classnames)
        self.zero_shot_classifier = None
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        # our prompt: a class name
        prompts = ["a " + name+ "." for name in classnames] # clip zero shot baseline
        visual_template = ["a ."]
        visual_template_tokenized_prompts = torch.cat([clip.tokenize(p) for p in visual_template])
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # generate zero_shot_classifier for evaluation
        if len(prompts)<5000:  
            with torch.no_grad():
                self.zero_shot_classifier = []
                for tokens in tokenized_prompts.reshape(self.num_class,-1, tokenized_prompts.shape[-1]):
                    clip_model = clip_model.cuda()
                    text_feats = clip_model.encode_text(tokens.cuda())
                    self.zero_shot_classifier.append(F.normalize(text_feats.mean(dim=0), dim=-1, p=2))
                self.zero_shot_classifier = torch.stack(self.zero_shot_classifier) # text classifier
                clip_model = clip_model.cpu()

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.visual_prompt_temp = clip_model.token_embedding(visual_template_tokenized_prompts).type(dtype).cuda()
        self.prompt_tokens = embedding
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts.cuda()
        self.name_lens = name_lens

        # our key component visual token generator
        self.aggregator = TransformerDropout(
            width = vis_dim,
            layers = 4,
            heads = vis_dim//64,
            dropout=0.1
        )
        print(self.aggregator)
        proj_std = (self.aggregator.width ** -0.5) * ((2 * self.aggregator.layers) ** -0.5)
        attn_std = self.aggregator.width ** -0.5
        fc_std = (2 * self.aggregator.width) ** -0.5
        
        for block in self.aggregator.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        self.cls_token = nn.Parameter(F.normalize(torch.randn(self.n_ctx, vis_dim), dim=-1, p=2), requires_grad=True)

    def update_prompts(self, prompt_tokens, ins_tokens):
        return torch.cat([prompt_tokens[:,:2], ins_tokens.type(self.prompt_tokens.dtype), prompt_tokens[:, 2:-self.n_ctx]], dim=1)

    def forward(self, exemplar_img_feats, label, ori_text_len):
        num_class, num_ins, dim = exemplar_img_feats.shape
        prompts = self.prompt_tokens[label]
        mm_prompts_list = []
        mm_lens = ori_text_len+self.n_ctx
        v_prompts_list = []
        v_lens = torch.ones_like(ori_text_len, dtype=torch.int32)+self.n_ctx # a v_token .
        
        cls_token = self.cls_token.unsqueeze(1).repeat(1, num_class, 1)
        aggregator_input = torch.cat([cls_token, exemplar_img_feats.permute(1,0,2)], dim=0)
        agg_img_token_ = self.aggregator(aggregator_input)[0:self.n_ctx, :, :].permute(1,0,2)

        new_mm_prompts = self.update_prompts(prompts, agg_img_token_)
        mm_prompts_list.append(new_mm_prompts)
        new_v_prompts = self.update_prompts(self.visual_prompt_temp.repeat(num_class,1,1), agg_img_token_)
        v_prompts_list.append(new_v_prompts)
    
        return mm_prompts_list, mm_lens, v_prompts_list, v_lens, agg_img_token_


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.train_bs = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.num_ins = cfg.DATALOADER.TRAIN_X.N_INS# 16
        self.test_num_ins = cfg.DATASET.NUM_SHOTS
        self.aug_times = cfg.DATALOADER.K_TRANSFORMS 
        self.visual_encoder_list = [self.image_encoder]

        self.zero_shot_classifier = self.prompt_learner.zero_shot_classifier
        self.mm_classifier = None
        self.fusion_weight = None
        self.device = None

    def get_mm_v_feats(self, mm_prompts, mm_lens, v_prompts, v_lens):
        mm_features_list, v_features_list = [], []
        for mm_prompt, v_prompt in zip(mm_prompts, v_prompts):
            mm_features = self.text_encoder(mm_prompt, mm_lens)
            mm_features = mm_features / mm_features.norm(dim=-1, keepdim=True)
            mm_features_list.append(mm_features.unsqueeze(1))

            v_features = self.text_encoder(v_prompt, v_lens)
            v_features = v_features / v_features.norm(dim=-1, keepdim=True)
            v_features_list.append(v_features.unsqueeze(1))
        mm_features_list = F.normalize(torch.cat(mm_features_list, dim=1).mean(dim=1), dim=-1, p=2)
        v_features_list = F.normalize(torch.cat(v_features_list, dim=1).mean(dim=1), dim=-1, p=2)
        return mm_features_list, v_features_list

    @torch.no_grad()
    def forward_prompt(self, eval_set_loader):
        self.mm_classifier = torch.randn((len(self.tokenized_prompts), self.image_encoder.output_dim), dtype=torch.float16, device=self.device)
        self.inference_text_initialized = torch.zeros(len(self.tokenized_prompts), dtype=torch.int32, device=self.device)
        self.visual_tokens = torch.ones((len(self.tokenized_prompts), self.prompt_learner.n_ctx, self.image_encoder.output_dim), dtype=torch.float16, device=self.device)
        self.exemplar_image_features = torch.zeros((len(self.tokenized_prompts), self.test_num_ins, self.image_encoder.output_dim), dtype=torch.float16, device=self.device)
        self.fusion_weight = torch.ones((len(self.tokenized_prompts), 3), dtype=torch.float32, device=self.device)
        
        total_cls_num = len(self.tokenized_prompts)
        cross_valid_num = self.test_num_ins
        self.eval_feat4cls = torch.randn((len(self.tokenized_prompts), cross_valid_num, self.image_encoder.output_dim), dtype=torch.float16, device=self.device)
        self.visual_classifer = torch.randn((len(self.tokenized_prompts), self.image_encoder.output_dim), dtype=torch.float16, device=self.device)
        for batch_idx, batch in enumerate(eval_set_loader):
            image = batch["img"]
            label = batch["label"]
            if isinstance(image, list):
                image_ = []
                for im_ in image:
                    image_.append(im_.to(self.device).unsqueeze(1))
                image_ = torch.cat(image_, dim=1)
                image = image_.flatten(0,1)
            else:
                image = image.to(self.device)
            label = label.to(self.device)
            num_cls = image.shape[0]//(self.test_num_ins)
            logit_scale = self.logit_scale.exp()
            exemplar_label = label.reshape(num_cls, self.test_num_ins)[:, 0]
            tokenized_prompts = self.tokenized_prompts[exemplar_label.cpu()] 
            with torch.no_grad():
                exemplar_features = self.image_encoder(image.type(self.dtype))
                exemplar_features = exemplar_features / exemplar_features.norm(dim=-1, keepdim=True)
                exemplar_features = exemplar_features.reshape(num_cls, self.test_num_ins, -1)
                fusion_weights = []
                self.eval_feat4cls[exemplar_label] = exemplar_features
            mm_prompts, mm_lens, v_prompts, v_lens, visual_tokens = self.prompt_learner(exemplar_features, exemplar_label, tokenized_prompts.argmax(dim=-1))
            mm_features_list, v_features_list = self.get_mm_v_feats(mm_prompts, mm_lens, v_prompts, v_lens)

            self.mm_classifier[exemplar_label] = mm_features_list.half()
            self.visual_classifer[exemplar_label] = v_features_list.half()
            
            self.inference_text_initialized[exemplar_label] = 1
            self.visual_tokens[exemplar_label] = visual_tokens.half()
               
            print("NO. inference prompt batch_id %d generation"%(batch_idx))
      
        assert self.inference_text_initialized.bool().all()
        
        eval_labels = torch.arange(len(self.tokenized_prompts)).cuda().reshape(-1, 1).repeat(1, self.test_num_ins).flatten(0,1)

        eval_mm_logits = logit_scale*torch.einsum("bmc, pc->bmp", self.eval_feat4cls, self.mm_classifier).flatten(0,1)
        eval_v_logits = logit_scale*torch.einsum("bmc, pc->bmp", self.eval_feat4cls, self.visual_classifer).flatten(0,1)
        eval_t_logits = logit_scale*torch.einsum("bmc, pc->bmp", self.eval_feat4cls, self.zero_shot_classifier).flatten(0,1)
        
      
        eval_mm_ce = multiclass_f1_score(eval_mm_logits, eval_labels, num_classes=total_cls_num, average=None).reshape(total_cls_num)
        eval_v_ce = multiclass_f1_score(eval_v_logits, eval_labels, num_classes=total_cls_num, average=None).reshape(total_cls_num)
        eval_t_ce = multiclass_f1_score(eval_t_logits, eval_labels, num_classes=total_cls_num, average=None).reshape(total_cls_num)
        
        ce_ = (torch.cat([eval_mm_ce.unsqueeze(-1), eval_v_ce.unsqueeze(-1), eval_t_ce.unsqueeze(-1)], dim=-1)).float()
        fusion_weights = (self.cfg.EVAL_TAU*ce_).softmax(dim=-1)
        self.fusion_weight = fusion_weights
        print(fusion_weights)
        torch.save(
            {
                "text_classifier": self.zero_shot_classifier.float(),
                "vision_classifier": self.visual_classifer.float(),
                "mm_classifier": self.mm_classifier.float(),
                "fusion_weight": self.fusion_weight.float()
             },
             osp.join(self.cfg.OUTPUT_DIR,"mm_classifiers.pt")

        )
        torch.save(
            {
                "visual_tokens": self.visual_tokens
            },
            osp.join(self.cfg.OUTPUT_DIR,"visual_tokens.pt")
        )
        return self.mm_classifier, self.visual_classifer, self.fusion_weight

    def forward(self, image, label=None, eval_set_loader=None, scale_no=None):
        
        logit_scale = self.logit_scale.exp()
        num_cls = image.shape[0]//self.num_ins
        train_num_ins = self.num_ins
        if eval_set_loader is None:
            split_point = torch.randint(train_num_ins//4,(3*train_num_ins//4),(1,))[0]
            exemplar_image = image.reshape(num_cls, self.num_ins, image.shape[1], image.shape[2], image.shape[3])[:, split_point:train_num_ins].flatten(0,1)
            input_image = image.reshape(num_cls, self.num_ins, image.shape[1], image.shape[2], image.shape[3])[:, :split_point].flatten(0,1)
        else:
            input_image = image
        with torch.no_grad():
            image_features = self.image_encoder(input_image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        self.device = image_features.device
        if self.prompt_learner.training: # training
            label_group = label.reshape(num_cls, self.num_ins)
            exemplar_label = label_group[:, 0]
            with torch.no_grad():    
                exemplar_features = self.image_encoder(exemplar_image.type(self.dtype))
                exemplar_features = exemplar_features / exemplar_features.norm(dim=-1, keepdim=True)
                exemplar_features = exemplar_features.reshape(num_cls, train_num_ins-split_point, -1)

            train_image_features = image_features
            train_labels = torch.arange(num_cls).reshape(num_cls,-1).repeat(1, split_point).reshape(-1).cuda()
            tokenized_prompts = self.tokenized_prompts[exemplar_label.cpu()]

            mm_prompts, mm_lens, v_prompts, v_lens, visual_tokens = self.prompt_learner(exemplar_features, exemplar_label, tokenized_prompts.argmax(dim=-1))

            mm_features_list, v_features_list = [], []
            for mm_prompt, v_prompt in zip(mm_prompts, v_prompts):
                mm_features = self.text_encoder(mm_prompt, mm_lens)
                mm_features = mm_features / mm_features.norm(dim=-1, keepdim=True)
                mm_features_list.append(mm_features.unsqueeze(1))
                v_features = self.text_encoder(v_prompt, v_lens)
                v_features = v_features / v_features.norm(dim=-1, keepdim=True)
                v_features_list.append(v_features.unsqueeze(1))
            mm_features_list = F.normalize(torch.cat(mm_features_list, dim=1).mean(dim=1), dim=-1, p=2)
            v_features_list = F.normalize(torch.cat(v_features_list, dim=1).mean(dim=1), dim=-1, p=2)
            mm_logits = (logit_scale * train_image_features @ mm_features_list.t()).float()
            v_logits = (logit_scale * train_image_features @ v_features_list.t()).float()
            cls_loss = F.cross_entropy(mm_logits, train_labels) + F.cross_entropy(v_logits, train_labels)
            if self.prompt_learner.training:
                return cls_loss
            
        else: # evaluation
            if self.mm_classifier is None:
                mm_classifier, v_classifier, fusion_weight = self.forward_prompt(eval_set_loader)#
            else:
                mm_classifier = self.mm_classifier
                v_classifier = self.visual_classifer
                fusion_weight = self.fusion_weight
            
            if self.cfg.EVAL_MODE=="text":
                t_logits = (logit_scale * image_features @ self.zero_shot_classifier.t()).float().softmax(dim=-1)
                return t_logits
            if self.cfg.EVAL_MODE=="vision":
                v_logits = (logit_scale * image_features @ v_classifier.t()).float().softmax(dim=-1)
                return v_logits
            if self.cfg.EVAL_MODE=="multimodal":
                mm_logits = (logit_scale * image_features @ mm_classifier.t()).float().softmax(dim=-1)
                return mm_logits
            if self.cfg.EVAL_MODE=="fusion":
                t_logits = (logit_scale * image_features @ self.zero_shot_classifier.t()).float().softmax(dim=-1)
                v_logits = (logit_scale * image_features @ v_classifier.t()).float().softmax(dim=-1)
                mm_logits = (logit_scale * image_features @ mm_classifier.t()).float().softmax(dim=-1)
                three_logits = torch.cat([mm_logits.unsqueeze(-1), v_logits.unsqueeze(-1), t_logits.unsqueeze(-1)], dim=-1)
                logits = torch.einsum("bmn,mn->bmn", three_logits, fusion_weight).sum(-1)
                return logits
        return logits
    

@TRAINER_REGISTRY.register()
class MM_CLS_OP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        random.seed(cfg.SEED)
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!") 
            # dist.init_process_group(backend='nccl')
            # torch.cuda.set_device(dist.get_rank())
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            if torch.cuda.device_count()>1:
                loss.sum().backward()
            else:
                loss.backward()
            optim.step()
        if torch.cuda.device_count()>1:
            loss_summary = {"loss": loss.sum().item()}
        else:
            loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        # directory = False
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
