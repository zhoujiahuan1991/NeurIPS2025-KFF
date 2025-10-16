import logging
logger = logging.getLogger(__name__)

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from vpt import PromptViT
import numpy as np
import math

global _corruption_type
global _dataset_name


class PromptBase:
    def __init__(self, max_len = 20, ema_alpha=0.1, tau=3.0, thr_d = 25.):
        self.set = []
        self.num = 0
        self.id = 0
        self.ema_alpha = ema_alpha
        self.tau = tau
        self.max_len = max_len
        self.thr = thr_d


    def __str__(self):
        pass
    

    def delete(self):
        # Find the pair of prompts with the minimum distance and merge them
        min_distance = float('inf')
        min_pair = None
        for i in range(len(self.set)):
            for j in range(len(self.set)):
                if i != j:
                    distance = torch.norm(self.set[i][0] - self.set[j][0], p=2)
                    if distance < min_distance:
                        min_distance = distance
                        min_pair = (i, j)
        assert min_pair is not None, f'No pair found to delete'

        i, j = min_pair
        ca, cb = self.set[i][2], self.set[j][2]
        max_id = max(self.set[i][3], self.set[j][3])
        ida, idb = self.set[i][3], self.set[j][3]
        self.set[i][0] = (self.set[i][0] + self.set[j][0]) / 2
        self.set[i][1] = (self.set[i][1] + self.set[j][1]) / 2

        for item in self.set[j][2]:
            if item not in self.set[i][2]:
                self.set[i][2].append(item)

        self.set[i][3] = max_id
        self.set.pop(j)
        self.num -= 1
        # logger.info(f"merge {ca}: {ida} and {cb}: {idb} to {max_id}, distance: {min_distance:.2f}")


    def __getlen__(self):
        return self.num


    def __len__(self):
        return self.num
    

    def add_prompt(self, key, prompt, note):
        self.id += 1
        self.set.append([key, prompt, note, self.id])
        self.num += 1
        while len(self.set) > self.max_len:
            self.delete()


    def get_weighted_prompts(self, key):
        """Calculate weights and weighted prompts for the promptbase."""
        key_tensor = torch.stack([p[0] for p in self.set])
        assert key_tensor.shape[0] == len(self.set)
        is_ID = True
        key_match = torch.norm(key - key_tensor, p=2, dim=1)
        key_match = torch.where(key_match <= self.thr, key_match, torch.inf)
        if torch.min(key_match).item() > self.thr:
            is_ID = False
        weights = torch.nn.functional.softmax(-key_match/self.tau, dim=0).detach().cpu()
        weighted_prompts = torch.stack([w * p[1] for w, p in zip(weights, self.set)], dim=0).sum(dim=0)
        assert weighted_prompts.shape == self.set[0][1].shape, f'{weighted_prompts.shape} != {self.set[0][1].shape}'

        return weighted_prompts, weights, is_ID
    

    def update(self, weights, prompt, key=None):
        """Update the promptbase with new statistics."""
        for p_idx in range(len(self.set)):
            if key is not None:
                self.set[p_idx][0] += weights[p_idx] * (key - self.set[p_idx][0]) * self.ema_alpha
            self.set[p_idx][1] += weights[p_idx] * (prompt - self.set[p_idx][1])


class Ours(nn.Module):
    def __init__(self, model:PromptViT, optimizer, cfg, tau=3.0, ema_alpha=0.1, E_OOD=50):
        super().__init__()
        self.tau = tau
        self.ema_alpha = ema_alpha
        self.E_ID = 1
        self.E_OOD = E_OOD
        self.cfg = cfg

        self.model = model
        self.optimizer = optimizer
        
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        self.prompt_base = PromptBase(max_len=cfg.OURS.N_D, ema_alpha=self.ema_alpha, tau=self.tau, thr_d=cfg.OURS.THR_D)
        # self.corruptions = {}
        self.last = ""
        self.counter = 0
            

    @torch.no_grad()
    def _eval_coreset(self, x):
        """Evaluate the coreset on a batch of samples."""
        loss, key, output = self.forward_and_get_loss(x, self.model, self.train_info, with_prompt=False)
        is_ID = False
        weights = None
        weighted_prompts = None
        if len(self.prompt_base) > 0:
            weighted_prompts, weights, is_ID = self.prompt_base.get_weighted_prompts(key)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.get_cls_prompt(output)
        else:
            self.model.get_cls_prompt(output)
        return is_ID, key, weighted_prompts, weights

    def forward(self, x):
        if _corruption_type != self.last:
            self.counter = 0
            self.last = _corruption_type
        else:
            self.counter += 1

        is_ID, key, weighted_prompts, weights = self._eval_coreset(x)
        if is_ID:
            self.model.prompts = torch.nn.Parameter(weighted_prompts.cuda())
            self.model.prompts.requires_grad = True
            optimizer = torch.optim.AdamW([self.model.prompts], lr=self.cfg.OPTIM.LR_DOMAIN)
            self.optimizer = torch.optim.AdamW([self.model.cls_prompt], lr=self.cfg.OPTIM.LR)
            outputs, loss = self.forward_and_adapt(x, self.model, optimizer, self.train_info, self.E_ID)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.prompt_base.update(weights, self.model.prompts.clone().detach().cpu(), key)
            
        else:
            self.model.reset()
            self.model.prompts.requires_grad = True
            optimizer = torch.optim.AdamW([self.model.prompts], lr=self.cfg.OPTIM.LR_DOMAIN)
            self.optimizer = torch.optim.AdamW([self.model.cls_prompt], lr=self.cfg.OPTIM.LR)
            
            outputs, loss = self.forward_and_adapt(x, self.model, optimizer, self.train_info, self.E_OOD)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.prompt_base.add_prompt(key, self.model.prompts.clone().detach().cpu(), [])
            # self.prompt_base.add_prompt(key, self.model.prompts.clone().detach().cpu(), [_corruption_type])

            # len_coreset = len(self.prompt_base)
            # if _corruption_type not in self.corruptions:
            #     self.corruptions[_corruption_type] = 1
            # else:
            #     self.corruptions[_corruption_type] += 1
            # logger.info(f'New prompt added at {self.counter}. Coreset size: {len_coreset}')

        self.model.update_cls_prompt(outputs)
            
        return outputs
    
    def obtain_src_stat(self, data_path, num_samples=5000, train_info=None):
        if train_info is not None:
            self.train_info = torch.load(train_info)
            print(f'Loaded train info from {train_info}')
            return

        assert num_samples > 0, f"num_samples must be greater than 0, got {num_samples}"
        print(f"Obtaining source statistics from {data_path} with {num_samples} samples for {_dataset_name}")
        num = 0
        features = []
        import timm
        from torchvision.datasets import ImageNet, CIFAR10, CIFAR100
        from torchvision import transforms
        if _dataset_name == 'imagenet':
            net = timm.create_model('vit_base_patch16_224', pretrained=False)
            data_config = timm.data.resolve_model_data_config(net)
            src_transforms = timm.data.create_transform(**data_config, is_training=False)
            src_dataset = ImageNet(root=data_path, split='train', transform=src_transforms)
            src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=64, shuffle=True)
        elif _dataset_name == 'cifar10':
            src_transforms = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            src_dataset = CIFAR10(root=data_path, train=True, transform=src_transforms)
            src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=64, shuffle=True)
        elif _dataset_name == 'cifar100':
            src_transforms = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            src_dataset = CIFAR100(root=data_path, train=True, transform=src_transforms)
            src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=64, shuffle=True)
        
        with torch.no_grad():
            for _, dl in enumerate(src_loader):
                images = dl[0].cuda()
                if _dataset_name == 'cifar10' or _dataset_name == 'cifar100':
                    import torch.nn.functional as F
                    images = F.interpolate(images, size=(384, 384), mode='bilinear', align_corners=False)
                feature = self.model.forward_raw_features(images)
                
                output = self.model(images)
                ent = self.softmax_entropy(output)
                selected_indices = torch.where(ent < math.log(1000)/2-1)[0]
                feature = feature[selected_indices]
                
                features.append(feature[:, 0])
                # features.append(feature)
                num += feature.shape[0]
                # print(feature.shape)
                if num >= num_samples:
                    break

            features = torch.cat(features, dim=0)
            features = features[:num_samples, :]
            logger.info(f'Obtained {num_samples} features from {num} samples')
            self.train_info = torch.std_mean(features, dim=0)
        del features

    
    @torch.no_grad()
    def forward_and_get_loss(self, images, model:PromptViT, train_info, with_prompt=False):
        model.get_cls_prompt(size=images.shape[0])
        if with_prompt:
            features = model.forward_features(images)
        else:
            features = model.forward_raw_features(images)
        
        cls_features = features[:, 0]

        """discrepancy loss"""
        key = model.domain_extractor(cls_features)
        if isinstance(model.vit, nn.DataParallel):
            output = model.vit.module.forward_head(features)
        else:
            output = model.vit.forward_head(features)
        # todo
        loss = self.distribution_loss(cls_features, train_info)

        return loss, key, output
    

    @torch.enable_grad()
    def forward_and_adapt(self, x, model: PromptViT, optimizer, train_info, iteration=1):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # print_trainable_params(model)
        loss = 0
        loss_ = 0
        output = None
        cls_features = None
        for i in range(iteration):
            features = model.forward_features(x)
            cls_features = features[:, 0]
            loss = self.distribution_loss(cls_features, train_info)
            
            # output = model.vit.head(cls_features)
            if isinstance(model.vit, nn.DataParallel):
                output = model.vit.module.forward_head(features)
            else:
                output = model.vit.forward_head(features)

            # todo
            loss_ = 3 * self.softmax_entropy(output).mean(0) # classification loss

            loss += loss_
            
            optimizer.zero_grad()
            if i == iteration - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer.step()
            
        # logger.info(f"loss: {loss:.2f}, entropy_loss: {loss_:.2f}")
        return output, loss
        

    def reset(self):
        # self.model.load_state_dict(self.model_state)
        # self.optimizer.load_state_dict(self.optimizer_state)
        # self.model.reset()
        self.model.reset_init()
        self.prompt_base = PromptBase(max_len   = self.cfg.OURS.N_D,
                                      ema_alpha = self.ema_alpha,
                                      tau       = self.tau,
                                      thr_d     = self.cfg.OURS.THR_D)

    
    def distribution_loss(self, x, train_info):
        std, mean = torch.std_mean(x, dim=0)
        ls = torch.norm(std - train_info[0].cuda(), p=2)
        lm = torch.norm(mean - train_info[1].cuda(), p=2)
        return ls + self.cfg.OPTIM.LAMDA * lm
    
    @staticmethod
    @torch.jit.script
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        temprature = 1
        x = x / temprature
        x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
        return x
        


def configure_model(model, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PromptViT(model, cfg)
    model.to(device)
    model.train()
    return model


def collect_params(model):
    for param in model.parameters():
        param.requires_grad = False
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.get_cls_prompt()
    model.prompts.requires_grad = True
    model.cls_prompt.requires_grad = True
    print_trainable_params(model)
    return [model.prompts], [model.cls_prompt]


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def print_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = sum(p.numel() for p in model.parameters())
    print(f"Adaptating {trainable_params / 1e3:.2f}K parameters in {params / 1e6:.2f}M parameters, "+
          f"which is {trainable_params / params:.2%} of total")
