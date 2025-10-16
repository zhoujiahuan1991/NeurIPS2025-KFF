import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Mlp
from timm.models.helpers import checkpoint_seq
import math
from functools import reduce
from operator import mul
import numpy as np
import copy

class PromptViT(nn.Module):
    '''
    Vision Transformer with added prompts at the input layer
    '''
    def __init__(self, vit, cfg):
        super().__init__()
        self.vit = vit
        num_prompts = cfg.OPTIM.PROMPT_NUM
        self.num_prompts = num_prompts
        if isinstance(vit, nn.DataParallel):
            self.prompt_dim = vit.module.embed_dim
            self.patchsize = vit.module.patch_embed.patch_size
        else:
            self.prompt_dim = vit.embed_dim
            self.patchsize = vit.patch_embed.patch_size
        self.cls_prompt = None

        if num_prompts > 0:
            self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
            # self.prompts_cls = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
            # initialization adopted from vpt, https://arxiv.org/abs/2203.12119
            val = math.sqrt(6. / float(3 * reduce(mul, self.patchsize, 1) + self.prompt_dim)) # noqa
            self.val = val
            nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization
            # nn.init.uniform_(self.prompts_cls.data, -val, val)
            self.prompt_copy = copy.deepcopy(self.prompts)
        self.domain_extractor = DomainExtractor()

        self.n_c = cfg.OURS.N_C
        self.thr_c = cfg.OURS.THR_C
        self.thr_ent = cfg.OURS.THR_ENT
        self.cls_prompt_generator = cls_prompt_gen(val     = self.val, 
                                                   dim     = self.prompt_dim, 
                                                   num     = self.n_c, 
                                                   thr_c   = self.thr_c, 
                                                   thr_ent = self.thr_ent,
                                                   alpha_c = cfg.OURS.ALPHA_C)
    
    def reset(self):
        """
        Reset domain prompts to their initial state.
        """
        # val = math.sqrt(6. / float(3 * reduce(mul, self.vit.patch_embed.patch_size, 1) + self.prompt_dim)) # noqa
        # nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization
        self.prompts = copy.deepcopy(self.prompt_copy)

    def reset_init(self):
        """
        Reset the model to its initial state, including domain prompts and cls prompt.
        """
        self.prompts = copy.deepcopy(self.prompt_copy)
        self.cls_prompt_generator.reset()

    def prompt_injection(self, x):
        if self.num_prompts > 0:
            x = torch.cat((
                x[:,:1,:], # cls token
                self.prompts.expand(x.shape[0],-1,-1),
                x[:,1:,:]  # img tokens
            ), dim=1)
        return x
    
    def cls_prompt_injection(self, x):
        assert self.cls_prompt is not None, "cls_prompt is not initialized"
        assert self.cls_prompt.shape[0] == x.shape[0], f"cls_prompt shape {self.cls_prompt.shape} does not match input shape {x.shape}"
        if self.num_prompts > 0:
            x = torch.cat((
                x[:,:1,:], # cls token
                # self.cls_prompt.expand(x.shape[0],-1,-1),
                self.cls_prompt.to(x.device),
                x[:,1:,:]  # img tokens
            ), dim=1)
        return x
    
    def _collect_layers_features(self, x):
        # collecting features for each layer
        cls_features = []
        for i in range(len(self.vit.blocks)):
            x = self.vit.blocks[i](x)
            if i < len(self.vit.blocks) - 1:
                cls_features.append(self.vit.blocks[i+1].norm1(x[:, 0]))
            else:
                cls_features.append(self.vit.norm(x[:, 0]))
        # cls_features = torch.cat(cls_features, dim=1)
        return cls_features

    
    def forward(self, x):
        x = self.forward_features(x)
        if isinstance(self.vit, nn.DataParallel):
            x = self.vit.module.forward_head(x)
            return x
        x = self.vit.forward_head(x)
        return x
    
    def layers_cls_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
    
    
    def layers_cls_features_with_prompts(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x)
        # !!end
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
    
    
    def forward_features(self, x):
        '''
        Forwarding a batch of samples with prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        if isinstance(self.vit, nn.DataParallel):
            x = self.vit.module.patch_embed(x)
            x = self.vit.module._pos_embed(x)
            # inject prompts
            x = self.prompt_injection(x)
            x = self.cls_prompt_injection(x)
            # !!end
            x = self.vit.module.norm_pre(x)
            x = self.vit.module.blocks(x)
            x = self.vit.module.norm(x)
            return x
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x)
        x = self.cls_prompt_injection(x)
        # !!end
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x
    

    def forward_features_ncls(self, x):
        '''
        Forwarding a batch of samples with prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x)
        # !!end
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x
    
    
    def forward_raw_features(self, x):
        '''
        Forwarding a batch of samples without prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        if isinstance(self.vit, nn.DataParallel):
            x = self.vit.module.patch_embed(x)
            x = self.vit.module._pos_embed(x)
            x = self.vit.module.norm_pre(x)
            x = self.vit.module.blocks(x)
            x = self.vit.module.norm(x)
            return x
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # x = self.cls_prompt_injection(x)

        # !!end
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x
    

    def get_cls_prompt(self, output = None, size = 64):
        if output is None:
            self.cls_prompt = nn.Parameter(torch.zeros(size, 1, self.prompt_dim))
            nn.init.uniform_(self.cls_prompt.data, -self.val, self.val)
        else:
            self.cls_prompt = self.cls_prompt_generator.get_cls_prompt(output)
        # pass
    
    def update_cls_prompt(self, outputs):
        # pass
        self.cls_prompt_generator.update_cls_prompt(outputs, self.cls_prompt)

class DomainExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        batch_std, batch_mean = torch.std_mean(features, dim=0)
        x = torch.cat((batch_mean, batch_std), dim=-1)
        return x
    

class cls_prompt_gen:
    def __init__(self, val, dim = 768, num = 100, thr_c = 0.005, thr_ent = 2, alpha_c = 0.1):
        self.max_num = num
        self.num = 0
        self.set = []
        self.record = []
        self.val = val
        self.dim = dim
        self.id = 0
        self.length = 1
        self.thr = thr_c
        self.thr_e = thr_ent
        self.alpha_c = alpha_c


    def get_cls_prompt(self, output):
        output = output.softmax(dim=-1)
        res = []
        n=0
        for i in range(output.shape[0]):
            p = None
            if self.num == 0:
                p = nn.Parameter(torch.zeros(1, self.length, self.dim))
                nn.init.uniform_(p.data, -self.val, self.val)
                self.record.append([-1])
            else:
                n += 1
                key_tensor = torch.stack([self.set[j][0].squeeze(0) for j in range(len(self.set))], dim=0)
                key_match = torch.cosine_similarity(key_tensor, output[i], dim=-1)
                key_match = torch.where(key_match > self.thr, key_match, -torch.inf)
                max = torch.max(key_match)

                if max < self.thr:
                    p = nn.Parameter(torch.zeros(1, self.length, self.dim))
                    nn.init.uniform_(p.data, -self.val, self.val)
                    n -= 1
                    self.record.append([-1])
                else:
                    weights = torch.softmax(key_match, dim=-1).detach().cpu()
                    p = torch.stack([w * p[1] for w, p in zip(weights, self.set)], dim=0).sum(dim=0)
                    self.record.append(weights)
            res.append(p)
        return nn.Parameter(torch.cat(res, dim=0))
    

    def update_cls_prompt(self, output, prompts):
        ent = softmax_entropy(output)
        output = output.softmax(dim=-1)
        assert len(self.record) == output.shape[0], f"len(self.record): {len(self.record)}, output.shape: {output.shape}"
        for i in range(len(self.record)):
            if ent[i] > self.thr_e:
                continue
            if self.record[i][0] == -1:
                self.set.append([output[i], prompts[i].unsqueeze(0), self.id])
                self.num += 1
                self.id += 1
            else:
                for idx in range(len(self.record[i])):
                    self.set[idx][0] += (output[i] - self.set[idx][0]) * self.record[i][idx] * self.alpha_c
                    self.set[idx][1] += (prompts[i].unsqueeze(0) - self.set[idx][1]) * self.record[i][idx]
        self.record = []
        self.resize()
                
    
    def resize(self, num = 0):
        if num > 0:
            self.max_num = num
        while self.num > self.max_num:
            self.delete_cls_prompt()

    def delete_cls_prompt(self):
        tmp = []
        for i in range(len(self.set)):
            for j in range(i+1, len(self.set)):
                tmp.append([torch.dot(self.set[i][0], self.set[j][0]), i, j])
        tmp = sorted(tmp, key=lambda x: x[0], reverse=True)

        n = self.num
        used = []
        cluster = []
        for i in range(len(tmp)):
            if n <= self.max_num:
                break
            if tmp[i][1] not in used and tmp[i][2] not in used:
                cluster.append([tmp[i][1], tmp[i][2]])
                used.append(tmp[i][1])
                used.append(tmp[i][2])
                n -= 1
            elif tmp[i][1] not in used:
                for j in range(len(cluster)):
                    if tmp[i][2] in cluster[j]:
                        cluster[j].append(tmp[i][1])
                        used.append(tmp[i][1])
                        n -= 1
                        break
            elif tmp[i][2] not in used:
                for j in range(len(cluster)):
                    if tmp[i][1] in cluster[j]:
                        cluster[j].append(tmp[i][2])
                        used.append(tmp[i][2])
                        n -= 1
                        break
            else:
                for j in range(len(cluster)):
                    b = False
                    if tmp[i][1] in cluster[j] and tmp[i][2] not in cluster[j]:
                        for k in range(j+1, len(cluster)):
                            if tmp[i][2] in cluster[k]:
                                if j != k:
                                    cluster[j].extend(cluster[k])
                                    del cluster[k]
                                    n -= 1
                                    b = True
                                    break
                    elif tmp[i][2] in cluster[j] and tmp[i][1] not in cluster[j]:
                        for k in range(j+1, len(cluster)):
                            if tmp[i][1] in cluster[k]:
                                if j != k:
                                    cluster[j].extend(cluster[k])
                                    del cluster[k]
                                    n -= 1
                                    b = True
                                    break
                    elif tmp[i][1] in cluster[j] and tmp[i][2] in cluster[j]:
                        b = True
                    if b:
                        break
            
        todel = []
        for i in range(len(cluster)):
            tmp1 = self.set[cluster[i][0]][0]
            tmp2 = self.set[cluster[i][0]][1]
            for j in range(1, len(cluster[i])):
                if cluster[i][j] not in todel:
                    todel.append(cluster[i][j])
                tmp1 += self.set[cluster[i][j]][0]
                tmp2 += self.set[cluster[i][j]][1]
            self.set[cluster[i][0]][0] = tmp1 / len(cluster[i])
            self.set[cluster[i][0]][1] = tmp2 / len(cluster[i])
            self.num -= (len(cluster[i]) - 1)
        todel = sorted(todel, reverse=True)
        for i in todel:
            del self.set[i]

    def reset(self):
        self.set = []
        self.record = []
        self.num = 0
        self.id = 0


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x