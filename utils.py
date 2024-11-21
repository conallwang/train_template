import torch
import os
import random

import numpy as np


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # If you want to reproduce fully, this item should be set False
    # But that will reduce the performance
    torch.backends.cudnn.benchmark = True


def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


def update_lambda(lambda_start, lambda_slope, lambda_end, global_step, interval):
    res = lambda_start
    if lambda_slope > 0:
        res = min(lambda_end, global_step // interval * lambda_slope + lambda_start)
    elif lambda_slope < 0:
        res = max(lambda_end, global_step // interval * lambda_slope + lambda_start)
    return res


def restore_model(model_path, models, optimizer, logger):
    """Restore checkpoint

    Args:
        model_path (str): checkpoint path
        models (dict): model dict
        optimizer (optimizer): torch optimizer
        logger (logger): logger
    """
    if model_path is None:
        if logger:
            logger.info("Not using pre-trained model...")
        return 1

    assert os.path.exists(model_path), "Model %s does not exist!"

    logger.info("Loading ckpts from {} ...".format(model_path))
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())

    for key, model in models.items():
        _state_dict = {
            k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict[key].items()
        }
        # Check if there is key mismatch:
        missing_in_model = set(_state_dict.keys()) - set(model.state_dict().keys())
        missing_in_ckp = set(model.state_dict().keys()) - set(_state_dict.keys())

        if logger:
            logger.info("[MODEL_RESTORE] missing keys in %s checkpoint: %s" % (key, missing_in_ckp))
            logger.info("[MODEL_RESTORE] missing keys in %s model: %s" % (key, missing_in_model))

        model.load_state_dict(_state_dict, strict=False)

    # load optimizer
    optimizer.load_state_dict(state_dict["optimizer"])

    current_epoch = state_dict["epoch"] if "epoch" in state_dict else 1
    global_step = state_dict["global_step"] if "global_step" in state_dict else 0

    return current_epoch, global_step


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class CUDA_Timer(object):
    def __init__(self, label, logger=None, valid=True, warmup_steps=10):
        self.valid = valid
        if not valid:
            return
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.label = label
        self.logger = logger
        self.counter = 0
        self.val = 0.0
        self.warmup_steps = warmup_steps

    def start(self, step):
        if self.valid and step > self.warmup_steps:
            self.starter.record()

    def end(self, step):
        if self.valid and step > self.warmup_steps:
            self.ender.record()
            self._update_val()

    def _update_val(self):
        torch.cuda.synchronize()
        time = self.starter.elapsed_time(self.ender)
        self.val = self.val * self.counter + time
        self.counter += 1
        self.val /= self.counter

        if self.logger:
            self.logger.info("[{}] ".format(self.label) + "{val " + str(time) + "ms} {avg " + str(self.val) + "ms}")
        else:
            print("[{}] ".format(self.label) + "{val " + str(time) + "ms} {avg " + str(self.val) + "ms}")

        # reset timer
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)

    def __str__(self):
        if self.valid:
            fmtstr = "[{}] " + "{avg " + str(self.val) + "ms}"
        else:
            fmtstr = "[{}] " + "\{avg -1ms\}"
        return fmtstr.format(self.label)

    def __enter__(self):
        if self.valid:
            self.starter.record()

    def __exit__(self, exc_type, exc_value, tb):
        if self.valid:
            self.ender.record()
            torch.cuda.synchronize()
            if self.logger:
                self.logger.info(self.label + " : {}ms".format(self.starter.elapsed_time(self.ender)))
            else:
                print(self.label + " : {}ms".format(self.starter.elapsed_time(self.ender)))
