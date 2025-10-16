# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH = 'Standard'

_C.MODEL.ADAPTATION = 'ours'

# By default tent is online, with updates persisting across batches.
# To make adaptation episodic, and reset the model for each batch, choose True.
_C.MODEL.EPISODIC = False

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'imagenet'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate 
# The 5000 val images defined by Robustbench were actually used:
# Please see https://github.com/RobustBench/robustbench/blob/7af0e34c6b383cd73ea7a1bbced358d7ce6ad22f/robustbench/data/imagenet_test_image_ids.txt
_C.CORRUPTION.NUM_EX = 5000

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3
_C.OPTIM.LR_DOMAIN = 0.1

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 64

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# ---------------------------------- Misc options --------------------------- #

# Optional description of a config
_C.DESC = ""

# Note that non-determinism is still present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Output directory
_C.SAVE_DIR = "./output"

# Data directory
_C.DATA_DIR = "imagenet-c"
_C.SRC_DATA_DIR = "imagenet1k"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# parameters(dpcore and ours)
_C.OPTIM.PROMPT_NUM = 8
_C.OPTIM.EMA_ALPHA = 0.1
_C.OPTIM.TAU = 3.0
_C.SRC_NUM_SAMPLES = 300
_C.OPTIM.LAMDA = 1.0
_C.SHUFFLE = False

# ------------------ hyperparameters added for ours ---------------------- #
_C.OURS = CfgNode()
_C.OURS.N_C = 100
_C.OURS.N_D = 20
_C.OURS.THR_D = 25.
_C.OURS.THR_C = 5e-3
_C.OURS.THR_ENT = 2.
_C.OURS.ALPHA_C = 0.1

_C.OURS.TRAIN_INFO = None

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", nargs='+', type=str, default=[],
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    parser.add_argument("--data_dir", default="/path/to/imagenet-c", type=str)
    parser.add_argument("--src_data_dir", default="/path/to/imagenet1k", type=str)
    parser.add_argument("--random_seed", default=1, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--log_name", default="ours", type=str)

    parser.add_argument("--gamma_c", default=0, type=float)
    parser.add_argument("--gamma_d", default=0, type=float)
    parser.add_argument("--gamma_ent", default=0, type=float)
    parser.add_argument("--alpha_c", default=-1, type=float)
    parser.add_argument("--alpha_d", default=-1, type=float)
    parser.add_argument("--N_c", default=0, type=int)
    parser.add_argument("--N_d", default=0, type=int)

    parser.add_argument("--train_info", default=None, type=str,
                        help="the path of the train info of source data")

    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # merge_from_file("./cfgs/corruption.yaml")
    # merge_from_file(args.cfg_file)
    for cfg_file in args.cfg_file:
        if not g_pathmgr.exists(cfg_file):
            raise FileNotFoundError(f"Config file {cfg_file} does not exist.")
        merge_from_file(cfg_file)
    cfg.merge_from_list(args.opts)

    if args.gamma_c != 0:
        cfg.OURS.THR_C = args.gamma_c
    if args.gamma_d != 0:
        cfg.OURS.THR_D = args.gamma_d
    if args.gamma_ent != 0:
        cfg.OURS.THR_ENT = args.gamma_ent
    if args.N_c > 0:
        cfg.OURS.N_C = args.N_c
    if args.N_d > 0:
        cfg.OURS.N_D = args.N_d
    if args.alpha_d >= 0:
        cfg.OPTIM.EMA_ALPHA = args.alpha_d
    if args.alpha_c >= 0:
        cfg.OURS.ALPHA_C = args.alpha_c
        
    if args.train_info is not None:
        cfg.OURS.TRAIN_INFO = args.train_info

    cfg.DATA_DIR = args.data_dir
    cfg.SRC_DATA_DIR = args.src_data_dir
    cfg.TEST.ckpt = args.checkpoint
    cfg.RNG_SEED = args.random_seed

    log_dest = f"{args.log_name}_{current_time}.txt" if args.log_name else f"log_{current_time}.txt"

    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    # Set random seeds for reproducibility
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(f"\n{cfg}")
    return