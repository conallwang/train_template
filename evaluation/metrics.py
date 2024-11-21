import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.example_dataset import ExampleData
from trainer import Trainer
from utils import CUDA_Timer, seed_everything, directory

parser = argparse.ArgumentParser("METRICS")
parser.add_argument("--checkpoint", type=str, required=True, help="path of model checkpoint")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("-bz", "--batch_size", type=int, default=6)
parser.add_argument("--time", action="store_true", help="evaluate time when set true.")

args = parser.parse_args()

# make sure params.yaml in the same directory with checkpoint
dir_name = os.path.dirname(args.checkpoint)
config_path = os.path.join(dir_name, "params.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["train.pretrained_checkpoint_path"] = args.checkpoint
config["local_workspace"] = dir_name

seed_everything(42)


def get_dataset(logger, datatype="example"):
    data_dict = {"example": ExampleData}
    assert datatype in data_dict.keys(), "Not Supported Datatype: {}".format(datatype)
    Data = data_dict[datatype]

    batch_size = args.batch_size
    test_set = Data(config, split=args.split)
    if logger:
        logger.info("number of test images: {}".format(len(test_set)))
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["train.num_workers"],
        drop_last=False,
    )

    return test_loader


def render(trainer, test_loader, logger, skip_render=False):
    logdir = os.path.join(config["local_workspace"], "{}_eval".format(args.split))
    directory(logdir)

    gt_dir = os.path.join(logdir, "gt")
    renders_dir = os.path.join(logdir, "renders")
    directory(gt_dir)
    directory(renders_dir)

    if skip_render:
        logger.info("Skipping rendering...")
        return gt_dir, renders_dir

    bar = tqdm(range(len(test_loader)))
    # time test
    show_time = args.time
    warmup_steps = 10
    ld_timer = CUDA_Timer("load data", logger, valid=show_time, warmup_steps=warmup_steps)
    render_timer = CUDA_Timer("render", logger, valid=show_time, warmup_steps=warmup_steps)
    ld_timer.start(0)

    for step, items in enumerate(test_loader):
        step += 1

        # 1. Set data for trainer
        trainer.set_data(items)
        if show_time and step > warmup_steps:
            ld_timer.end(step - 1)

        # 2. Run the network
        render_timer.start(step)
        outputs = trainer.network_forward(is_val=True)
        render_timer.end(step)

        for i in range(trainer.img.shape[0]):
            render_path = os.path.join(renders_dir, trainer.name[i] + ".png")
            gt_path = os.path.join(gt_dir, trainer.name[i] + ".png")

            # TODO: save useful results
            # visimg(render_path, outputs["render_fuse"][i : i + 1])
            # visimg(gt_path, trainer.img[i : i + 1])

        bar.update()

        if show_time and step > warmup_steps:
            ld_timer.start(step)

    logger.info("Rendering Finished.\n\n")

    return gt_dir, renders_dir


def metrics(gt_dir, renders_dir, logger, skip_metric=False):
    if skip_metric:
        logger.info("Skipping computing metrics...")
        return

    logger.info("Evaluating metrics...")

    basedir = os.path.dirname(renders_dir)

    metrics = []

    bar = tqdm(range(len(os.listdir(renders_dir))), desc="Metric evaluation progress")
    for fname in os.listdir(renders_dir):
        render = Image.open(os.path.join(renders_dir, fname))
        gt = Image.open(os.path.join(gt_dir, fname))

        # TODO: compute & save metrics
        # metrics.append()

        bar.update()

    mean_metrics = np.mean(metrics)

    logger.info("  METRICS : {}".format(mean_metrics))
    print("")

    savepath = os.path.join(basedir, "metrics.txt")
    with open(str(savepath), "w") as f:
        f.write("METRICS : {}\n".format(mean_metrics))


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    # If you want to reproduce fully, this item should be set False
    # But that will reduce the performance
    torch.backends.cudnn.benchmark = False

    # Config logging and tb writer
    logger = None
    import logging

    # logging to file and stdout
    # config["log_file"] = os.path.join(dir_name, 'test_image.log')
    logger = logging.getLogger("METRICS")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
    stream_handler.setFormatter(formatter)
    # file_handler = logging.FileHandler(config["log_file"])
    # file_handler.setFormatter(formatter)
    # logger.handlers = [file_handler, stream_handler]
    logger.handlers = [stream_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    config["logger"] = logger

    logger.info("Config: {}".format(config))

    if not args.skip_render:
        test_loader = get_dataset(logger, datatype=config["data.datatype"])
        trainer = Trainer(config, logger, is_val=True)
        trainer.set_eval()

        torch.set_grad_enabled(False)
    else:
        trainer, test_loader = None, None

    gt_dir, renders_dir = render(trainer, test_loader, logger, skip_render=args.skip_render)
    metrics(gt_dir, renders_dir, logger, skip_metric=args.skip_metric)
