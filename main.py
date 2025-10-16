import logging
logger = logging.getLogger(__name__)

import torch
import torch.optim as optim
import numpy as np
import random

from robustbench.data import load_imagenetc, load_cifar10c, load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import ours

from conf import cfg, load_cfg_fom_args

def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    # base_model = torch.nn.DataParallel(base_model)
    if cfg.TEST.ckpt is not None:
        logger.info(f"loading checkpoint from {cfg.TEST.ckpt}")
        base_model = torch.nn.DataParallel(base_model)
        checkpoint = torch.load(cfg.TEST.ckpt)
        base_model.load_state_dict(checkpoint['model'], strict=False)
        # if you do not use DataParallel, if you want to use DataParallel, simple comment the line below
        base_model = base_model.module if isinstance(base_model, torch.nn.DataParallel) else base_model
        del checkpoint
    base_model = base_model.cuda()

    if cfg.MODEL.ADAPTATION == "ours":
        model = setup_ours(base_model)
    logger.info(f"test-time adaptation: {cfg.MODEL.ADAPTATION}")
    # evaluate on each severity and type of corruption in turn
    All_error = []
    Average_error = []
    corruption_map = {}
    all_err_save = {}
    for _, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
        if corruption_type not in corruption_map:
            corruption_map[corruption_type] = 0
            all_err_save[corruption_type] = []

    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            ours._corruption_type = corruption_type

            if corruption_type == "reset":
                model = reset_model(model)
                ours._corruption_type = None
                continue
            if i_x == 0:
                model = reset_model(model)

            if len(All_error) == 15:
                avg = sum(All_error) / len(All_error)
                logger.info(f"average error {avg:.2%}")
                Average_error.append(avg)
                All_error = []
            
            if cfg.CORRUPTION.DATASET == 'imagenet':
                x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, cfg.SHUFFLE, [corruption_type])
            elif cfg.CORRUPTION.DATASET == 'cifar10':
                x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, cfg.SHUFFLE, [corruption_type])
            elif cfg.CORRUPTION.DATASET == 'cifar100':
                x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, cfg.SHUFFLE, [corruption_type])
            else:
                raise NotImplementedError(f"Dataset {cfg.CORRUPTION.DATASET} not supported for corruption loading.")
       
            if cfg.CORRUPTION.DATASET == 'cifar10' or cfg.CORRUPTION.DATASET == 'cifar100':
                x_test = torch.nn.functional.interpolate(x_test, size=(384, 384), mode='bilinear', align_corners=False)
                mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
                x_test = (x_test - mean) / std
            
            # x_test, y_test = x_test.cuda(), y_test.cuda()
            # acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            offset = 0
            n = 10 * cfg.TEST.BATCH_SIZE
            acc_num = 0.
            num = x_test.shape[0]
            while offset + n <= num:
                x_test_ = x_test[offset:offset + n].cuda()
                y_test_ = y_test[offset:offset + n].cuda()
                acc = accuracy(model, x_test_, y_test_, cfg.TEST.BATCH_SIZE)
                offset += n
                acc_num += acc * n
            if offset < num:
                x_test_ = x_test[offset:num].cuda()
                y_test_ = y_test[offset:num].cuda()
                acc = accuracy(model, x_test_, y_test_, cfg.TEST.BATCH_SIZE)
                acc_num += acc * (num - offset)
            acc = acc_num / num

            err = 1. - acc
            All_error.append(err)
            logger.info(f"error % [{cfg.CORRUPTION.DATASET} {corruption_type}{severity}]: {err:.2%}")
            all_err_save[corruption_type].append(err)
            # corruption_map[corruption_type] += cfg.CORRUPTION.NUM_EX
    
    for k, v in all_err_save.items():
        logger.info(f"error % [{k}]: {', '.join([f'{e:.2%}' for e in v])}")

    if len(Average_error) > 0:
        if len(All_error) > 0:
            avg = sum(All_error) / len(All_error)
            Average_error.append(avg)
            All_error = []
        avg_err_res = ', '.join([f"{e:.2%}" for e in Average_error])
        logger.info(f"AVG error: {avg_err_res}")
        logger.info(f"Mean of AVG error: {sum(Average_error) / len(Average_error):.2%}")
    elif len(All_error) > 0:
        avg = sum(All_error) / len(All_error)
        all_err_res = ', '.join([f"{e:.2%}" for e in All_error])
        logger.info(f"All error: {all_err_res}")
        logger.info(f"average error {avg:.2%}")


def reset_model(model):
    """Reset the model to its initial state, as well as the random seeds."""
    logger.info("resetting model")
    # reset random seeds
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
    # reset model
    try:
        model.reset()
    except:
        logger.warning("exception in resetting model")
    return model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    elif cfg.OPTIM.METHOD == 'AdamW':
        return optim.AdamW(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError

    
def setup_ours(model):
    model = ours.configure_model(model, cfg)
    ours._dataset_name = cfg.CORRUPTION.DATASET
    domain_prompts, class_prompts = ours.collect_params(model)
    optimizer = setup_optimizer(class_prompts)
    model = ours.Ours(model   = model,
                    optimizer = optimizer,
                    cfg       = cfg,
                    tau       = cfg.OPTIM.TAU,
                    ema_alpha = cfg.OPTIM.EMA_ALPHA,
                    E_OOD     = cfg.OPTIM.STEPS,)
    model.obtain_src_stat(data_path=cfg.SRC_DATA_DIR, num_samples=cfg.SRC_NUM_SAMPLES, 
                          train_info=cfg.OURS.TRAIN_INFO)
    return model

if __name__ == '__main__':
    evaluate('Imagenet-C evaluation.')