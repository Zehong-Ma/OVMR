import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts_list, tokenized_prompts, is_imagenet=False, prompt_ind=0):
        text_features = []
        if is_imagenet:
            prompts = prompts_list[prompt_ind]
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            if prompt_ind <= 1: # mm and visual
                indexes = tokenized_prompts.argmax(dim=-1) + 2
            else:
                indexes = tokenized_prompts.argmax(dim=-1)
            x = x[torch.arange(x.shape[0]), indexes] @ self.text_projection
            x = x/x.norm(dim=-1, keepdim=True)
            text_features.append(x)
        else:
            for ind, prompts in enumerate(prompts_list):
                x = prompts + self.positional_embedding.type(self.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.ln_final(x).type(self.dtype)

                # x.shape = [batch_size, n_ctx, transformer.width]
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                if ind <= 1: # mm and visual
                    indexes = tokenized_prompts.argmax(dim=-1) + 2
                else:
                    indexes = tokenized_prompts.argmax(dim=-1)

                x = x[torch.arange(x.shape[0]), indexes] @ self.text_projection
                x = x/x.norm(dim=-1, keepdim=True)
                text_features.append(x)
            # text_features = torch.cat(text_features, dim=1).mean(dim=1)
            # text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        return text_features


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # self.ctx_mm = deepcopy(ctx_init)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        visual_template = [prompt_prefix + "."]
        visual_tokenized_templates = torch.cat([clip.tokenize(p) for p in visual_template])
        with torch.no_grad():
            visual_template =  clip_model.token_embedding(visual_tokenized_templates).type(dtype)
        self.register_buffer("visual_template", visual_template)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        visual_tokens = torch.load(cfg.TRAINER.COOP.VISUAL_TOKEN_PATH, map_location="cpu")["visual_tokens"]
        self.visual_tokens_len = visual_tokens.shape[1]
        self.register_buffer("token_visual", visual_tokens)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        visual_tokens = self.token_visual
        prompts_list = []
        if self.class_token_position == "end":
            # for i in range(self.visual_tokens_len):
            #     visual_token = visual_tokens[:, i:i+1,:]
            #     prompts = torch.cat(
            #         [
            #             prefix,  # (n_cls, 1, dim)
            #             ctx,     # (n_cls, n_ctx, dim),
            #             visual_token,
            #             suffix[:, :-1, :],  # (n_cls, *, dim)
            #         ],
            #         dim=1,
            #     )
            #     prompts_list.append(prompts)
            
            visual_token = visual_tokens
            mm_prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim),
                    visual_token,
                    suffix[:, :-2, :],  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_list.append(mm_prompts)
            v_prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim),
                    visual_token,
                    self.visual_template[:, 1+self.n_ctx:-2, :].repeat(prefix.shape[0], 1, 1),
                    # suffix[:, :-1, :],  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_list.append(v_prompts)
            t_prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim),
                    # visual_token,
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_list.append(t_prompts)
            


        else:
            raise ValueError

        return prompts_list


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.fusion_weight = None 
        self.test_num_ins = cfg.DATALOADER.TEST.N_INS
        self.device="cuda:0"

    def get_fusion_weight(self, eval_set_loader, mm_classifier, v_classifier, t_classifier):
        
        self.fusion_weight = torch.ones((len(self.tokenized_prompts), 3), dtype=torch.float32).cuda()
        
        total_cls_num = len(self.tokenized_prompts)
        eval_num = 1
        cross_valid_num = self.test_num_ins

        self.eval_feat4cls = torch.randn((len(self.tokenized_prompts), cross_valid_num, self.image_encoder.output_dim), dtype=torch.float16).cuda()
        
        # self.visual_classifer = torch.randn((len(self.tokenized_prompts), self.image_encoder.output_dim), dtype=torch.float16, device=self.device)
        eval_feat_lists = []
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
            # image = image.to(self.device)
            label = label.to(self.device)
            num_cls = image.shape[0]//(self.test_num_ins)
            logit_scale = self.logit_scale.exp()
            exemplar_label = label.reshape(num_cls, self.test_num_ins)[:, 0]
            with torch.no_grad():
                exemplar_features = self.image_encoder(image.type(self.dtype))
                exemplar_features = exemplar_features / exemplar_features.norm(dim=-1, keepdim=True)
                exemplar_features = exemplar_features.reshape(num_cls, self.test_num_ins, -1)
                fusion_weights = []
                self.eval_feat4cls[exemplar_label] = exemplar_features
    

            # self.inference_img_token[exemplar_label] = image_features
        # print(self.inference_text_initialized)
        # self.inference_text_features = F.normalize(self.inference_text_features, dim=-1, p=2)
        # assert self.inference_text_initialized.bool().all()
        self.eval_feat4cls = self.eval_feat4cls.permute(1,0,2)
        eval_labels = torch.arange(len(self.tokenized_prompts)).cuda().reshape(-1, 1).repeat(1, self.test_num_ins).flatten(0,1)

        # shot16_feats = torch.load("./data/16shot_feats.pt", map_location="cuda:0")
        # self.eval_feat4cls = shot16_feats["eval_feat4cls"]
        # eval_labels = shot16_feats['eval_labels']

        eval_mm_logits = logit_scale*torch.einsum("bmc, pc->bmp", self.eval_feat4cls, mm_classifier).permute(1,0,2).flatten(0,1)
        eval_v_logits = logit_scale*torch.einsum("bmc, pc->bmp", self.eval_feat4cls, v_classifier).permute(1,0,2).flatten(0,1)
        eval_t_logits = logit_scale*torch.einsum("bmc, pc->bmp", self.eval_feat4cls, t_classifier).permute(1,0,2).flatten(0,1)
        
        # import pdb
        # pdb.set_trace()
        # eval_mm_ce = F.cross_entropy(eval_mm_logits, eval_labels, reduction="none").reshape(cross_valid_num,total_cls_num, -1).mean(dim=2)
        # eval_v_ce = F.cross_entropy(eval_v_logits, eval_labels, reduction="none").reshape(cross_valid_num,total_cls_num, -1).mean(dim=2)
        # eval_t_ce = F.cross_entropy(eval_t_logits, eval_labels, reduction="none").reshape(cross_valid_num,total_cls_num, -1).mean(dim=2)
        eval_mm_ce = multiclass_f1_score(eval_mm_logits, eval_labels, num_classes=total_cls_num, average=None).reshape(total_cls_num)
       
        eval_v_ce = multiclass_f1_score(eval_v_logits, eval_labels, num_classes=total_cls_num, average=None).reshape(total_cls_num)
        
        
        eval_t_ce = multiclass_f1_score(eval_t_logits, eval_labels, num_classes=total_cls_num, average=None).reshape(total_cls_num)
        
        ce_ = (torch.cat([eval_mm_ce.unsqueeze(-1), eval_v_ce.unsqueeze(-1), eval_t_ce.unsqueeze(-1)], dim=-1)).float()
        fusion_weights = (10*ce_).softmax(dim=-1)
        self.fusion_weight = fusion_weights
       
        # import pdb
        # pdb.set_trace()
        print(self.fusion_weight)
        return self.fusion_weight

    def forward(self, image, label=None, eval_set_loader=None, scale_no=None):
        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        
            # text_features_list = self.text_encoder(prompts, tokenized_prompts, is_imagenet=is_imagenet, ind=ind)
        # else:
        if self.prompt_learner.training:
            is_imagenet = prompts[0].shape[0]>400
            if is_imagenet:
                ind = torch.randint(0,3,(1,))[0]
                text_features_list = self.text_encoder(prompts, tokenized_prompts, is_imagenet=is_imagenet, prompt_ind=ind)
                logits = logit_scale * image_features @ text_features_list[0].t()
                # if ind == 0:
                return F.cross_entropy(logits, label), logits
                
            else:

                text_features_list = self.text_encoder(prompts, tokenized_prompts)
                mm_features = text_features_list[0]
                v_features = text_features_list[1]
                t_features = text_features_list[2]

                logits_mm = logit_scale * image_features @ mm_features.t()
                logits_v = logit_scale * image_features @ v_features.t()
                logits_t = logit_scale * image_features @ t_features.t()

                return F.cross_entropy(logits_mm, label) + F.cross_entropy(logits_v, label) + F.cross_entropy(logits_t, label), logits_mm
        
        else:
            text_features_list = self.text_encoder(prompts, tokenized_prompts)
            mm_features = text_features_list[0]
            v_features = text_features_list[1]
            t_features = text_features_list[2]

            logits_mm = logit_scale * image_features @ mm_features.t()
            logits_v = logit_scale * image_features @ v_features.t()
            logits_t = logit_scale * image_features @ t_features.t()
            if self.fusion_weight is None:
                fusion_weight = self.get_fusion_weight(eval_set_loader, mm_features, v_features,t_features)
            else:
                fusion_weight = self.fusion_weight
    
            three_logits = torch.cat([logits_mm.softmax(dim=-1).unsqueeze(-1), logits_v.softmax(dim=-1).unsqueeze(-1), logits_t.softmax(dim=-1).unsqueeze(-1)], dim=-1)
            logits = torch.einsum("bmn,mn->bmn", three_logits.float(), fusion_weight.float()).sum(-1)
            return logits



@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss, output = self.model(image, label)
            # loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

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
        # directory=False
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
