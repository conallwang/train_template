import os

import numpy as np
import torch

from utils import (
    AverageMeter,
    CUDA_Timer,
    directory,
    restore_model,
    update_lambda,
)


class Trainer:
    def __init__(self, config, logger, is_val=False):
        # DEBUG
        # torch.autograd.set_detect_anomaly(True)

        self.config = config
        self.img_h, self.img_w = config["data.img_h"], config["data.img_w"]
        self.rate_h, self.rate_w = self.img_h / 802.0, self.img_w / 550.0  # TODO: change the constants
        self.rate = min(self.rate_h, self.rate_w)
        self.nan_detect = False
        self.is_val = is_val
        self.lr = config["train.learning_rate"]

        self.models = {}
        self.parameters_to_train = []
        self._init_nets()

        # set optimizer
        self.optimizer = torch.optim.Adam(self.parameters_to_train, eps=1e-15)  # TODO: can be modified
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config["training.step"], gamma=0.1
        )

        # Restore checkpoint
        checkpoint_path = (
            os.path.join(config["local_workspace"], "checkpoint_latest.pth")
            if config["train.pretrained_checkpoint_path"] is None
            else config["train.pretrained_checkpoint_path"]
        )

        self.current_epoch = 1
        self.global_step = 0
        if os.path.exists(checkpoint_path):
            self.current_epoch, self.global_step = restore_model(checkpoint_path, self.models, self.optimizer, logger)

        self.logger = logger
        self.tb_writer = config.get("tb_writer", None)

        self._init_data()
        self._init_losses()

        # find all lambda
        self.all_lambdas = {}
        prelen = len("train.lambda_")
        for k, v in self.config.items():
            if "lambda" not in k or "lambda_update_list" in k:
                continue
            self.all_lambdas[k[prelen:]] = v

    def _freeze(self, label):
        for group in self.optimizer.param_groups:
            if label in group["name"] or label == "all":
                group["params"][0].requires_grad = False

    def _unfreeze(self, label):
        for group in self.optimizer.param_groups:
            if label in group["name"] or label == "all":
                group["params"][0].requires_grad = True

    def _init_nets(self):
        # TODO: initialize all networks you needed, and add optimizable parameters to 'self.parameters_to_train'
        self.models[""] = None

        self.parameters_to_train = None  # TODO: modify

    def _init_data(self):
        # TODO: initialize all needed data here
        pass

    def _init_losses(self):
        # TODO: initialize all needed losses, using 'AverageMeter' class
        self.train_losses = {
            "loss": AverageMeter("train_loss"),
        }
        self.val_losses = {
            "loss": AverageMeter("val_loss"),
        }

    def set_train(self):
        """Convert models to training mode"""
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert models to evaluation mode"""
        for m in self.models.values():
            m.eval()

    def train(self, train_loader, val_loader, show_time=False):
        torch.cuda.empty_cache()
        while self.current_epoch <= self.config["train.epochs"]:
            success = self.train_epoch(train_loader, val_loader, show_time)
            if not success:
                return

            self.scheduler.step()
            self.logger.info("Epoch finished, average losses: ")
            for v in self.train_losses.values():
                self.logger.info("    {}".format(v))
            self.current_epoch += 1

    def set_data(self, items):
        # TODO: set data according to the current batch
        pass

    def network_forward(self, is_val=False):
        # TODO: network forward process, return the predicted outputs
        outputs = {}

        return outputs

    def update_x(self, lambda_name):
        return update_lambda(
            self.config["train.lambda_{}".format(lambda_name)],
            self.config["train.lambda_{}.slope".format(lambda_name)],
            self.config["train.lambda_{}.end".format(lambda_name)],
            self.global_step,
            self.config["train.lambda_{}.interval".format(lambda_name)],
        )

    def update_lambda(self):
        update_names = self.config["training.lambda_update_list"]
        for k, _ in self.all_lambdas.items():
            if k in update_names:
                self.all_lambdas[k] = self.update_x(k)

    def get_lambda(self, key):
        return self.all_lambdas.get(key, 0.0)

    def compute_loss(self, outputs):
        # update hyper-parameters, if some lambdas are set to change according to time
        self.update_lambda()

        # TODO: compute loss to perform loss.backward(), using the outputs from network_forward() and the ground truth
        loss = 0.0

        # TODO: save some useful losses
        loss_dict = {
            "loss": loss,
        }

        return loss_dict

    def log_training(self, epoch, step, global_step, dataset_length, loss_dict):
        loss = loss_dict["loss"]
        loss_rgb = loss_dict["loss_pho/rgb"]

        lr = self.scheduler.get_last_lr()[0]
        self.logger.info(
            "stage [%s] epoch [%.3d] step [%d/%d] global_step = %d loss = %.4f lr = %.6f\n"
            "        rgb = %.4f                     w: %.4f\n"
            % (
                self.stage,
                epoch,
                step,
                dataset_length,
                self.global_step,
                loss.item(),
                lr,
                loss_rgb.item(),
                self.get_lambda("rgb"),
            )
        )

        # Write losses to tensorboard
        # Update avg meters
        for key, value in self.train_losses.items():
            if self.tb_writer:
                self.tb_writer.add_scalar(key, loss_dict[key].item(), global_step)
            value.update(loss_dict[key].item())

    def run_eval(self, val_loader):
        self.logger.info("Start running evaluation on validation set:")
        self.set_eval()

        # clear train losses average meter
        for val_loss_item in self.val_losses.values():
            val_loss_item.reset()

        batch_count = 0
        with torch.no_grad():
            for step, items in enumerate(val_loader):
                batch_count += 1
                if batch_count % 20 == 0:
                    self.logger.info("    Eval progress: {}/{}".format(batch_count, len(val_loader)))

                self.set_data(items)
                outputs = self.network_forward(is_val=True)
                loss_dict = self.compute_loss(outputs)

                # TODO: you can change the metrics
                mse, psnr = self.compute_metrics(outputs)

                loss_dict["metrics/mse"] = mse
                loss_dict["metrics/psnr"] = psnr

                self.log_val(step, loss_dict)

            # log evaluation result
            self.logger.info("Evaluation finished, average losses: ")
            for v in self.val_losses.values():
                self.logger.info("    {}".format(v))

            # Write val losses to tensorboard
            if self.tb_writer:
                for key, value in self.val_losses.items():
                    self.tb_writer.add_scalar(key + "//val", value.avg, self.global_step)

        self.set_train()

    def log_val(self, step, loss_dict):
        B = self.batch_size
        for key, value in self.val_losses.items():
            value.update(loss_dict[key].item(), n=B)

        # TODO: visualize some useful results

    def compute_metrics(self, outputs):
        if outputs["fullmask"] is None:
            return np.array(0.0), np.array(0.0)

        valid_mask = outputs["fullmask"] * self.mask["full"]

        gt_img = (self.img[0] * valid_mask[0, ..., None]).detach().cpu().numpy() * 255
        pred_img = (outputs["render_fuse"][0] * valid_mask[0, ..., None]).detach().cpu().numpy() * 255
        mse = ((pred_img - gt_img) ** 2).mean()
        psnr = 10 * np.log10(65025 / mse)

        return mse, psnr

    def visualization(self, outputs, step, label="log"):
        # create dirs
        logdir = os.path.join(self.config["local_workspace"], label)
        directory(logdir)
        if label == "log":
            savedir = os.path.join(logdir, "it{}".format(step))
            directory(savedir)
        elif label == "eval":
            savedir = os.path.join(logdir, self.name[0])
            directory(savedir)

        # TODO: visualize some middle results

        # compute metrics, if eval mode
        # TODO: change to some metrics you need
        if label == "eval":
            savepath = os.path.join(savedir, "metrics.txt")
            mse, psnr = self.compute_metrics(outputs)

            with open(savepath, "w") as f:
                f.write("MSE: {}\n".format(mse))
                f.write("PSNR: {}\n".format(psnr))

            print("MSE: {}\nPSNR: {}\n".format(mse, psnr))

    def clip_grad(self, max_norm=0.01):
        for dict in self.parameters_to_train:
            torch.nn.utils.clip_grad_norm_(dict["params"], max_norm)

    def save_ckpt(self, savepath):
        save_dict = {
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        for k, m in self.models.items():
            save_dict[k] = m.state_dict()

        torch.save(save_dict, savepath)

    def nan_debug(self, loss):
        # DEBUG: save the checkpoint before NaN Loss
        if not self.nan_detect and loss.isnan().any():
            self.nan_detect = True
            checkpoint_path = os.path.join(self.config["local_workspace"], "nan_break.pth")
            self.save_ckpt(checkpoint_path)
            print("NaN break checkpoint has saved at {}".format(checkpoint_path))
            print("Data {}".format(self.name))
            return False

    def train_epoch(self, train_loader, val_loader, show_time=False):
        # convert models to traning mode
        self.set_train()

        warmup_steps = 10
        ld_timer = CUDA_Timer("load data", self.logger, valid=show_time, warmup_steps=warmup_steps)
        sd_timer = CUDA_Timer("set data", self.logger, valid=show_time, warmup_steps=warmup_steps)
        f_timer = CUDA_Timer("forward", self.logger, valid=show_time, warmup_steps=warmup_steps)
        cl_timer = CUDA_Timer("compute loss", self.logger, valid=show_time, warmup_steps=warmup_steps)
        b_timer = CUDA_Timer("backward", self.logger, valid=show_time, warmup_steps=warmup_steps)
        up_timer = CUDA_Timer("update params", self.logger, valid=show_time, warmup_steps=warmup_steps)
        ld_timer.start(0)

        for step, items in enumerate(train_loader):
            step += 1
            self.global_step += 1

            if show_time and self.global_step > warmup_steps:
                ld_timer.end(step - 1)

            # 1. Set data for trainer
            sd_timer.start(step)
            self.set_data(items)
            sd_timer.end(step)

            # 2. Run the network
            f_timer.start(step)
            outputs = self.network_forward()
            f_timer.end(step)

            # 3. Compute losses
            cl_timer.start(step)
            loss_dict = self.compute_loss(outputs)
            cl_timer.end(step)
            loss = loss_dict["loss"]

            self.nan_debug(loss)  # detact if has nan values

            # 4. Backprop
            b_timer.start(step)
            loss.backward()
            b_timer.end(step)

            # 5. update parameters
            up_timer.start(step)
            self.clip_grad()  # you can disable this. used to avoid gradient explodation
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            up_timer.end(step)

            # logging
            if step > 0 and (step % self.config["train.log_freq"] == 0 or step == len(train_loader)):
                self.log_training(
                    self.current_epoch,
                    step,
                    self.global_step,
                    len(train_loader),
                    loss_dict,
                )

            # Visualize
            if self.stage_step == 1 or self.global_step % self.config["train.vis_freq"] == 0:
                self.visualization(outputs, self.global_step)

            if self.global_step % self.config["train.save_freq"] == 0:
                # Save model
                checkpoint_path = os.path.join(self.config["local_workspace"], "checkpoint_latest.pth")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                self.save_ckpt(checkpoint_path)
                self.logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

            if self.global_step > 0 and self.global_step % self.config["train.eval_freq"] == 0:
                self.run_eval(val_loader)

                # Save model
                checkpoint_path = os.path.join(
                    self.config["local_workspace"],
                    "checkpoint_%012d.pth" % self.global_step,
                )
                self.save_ckpt(checkpoint_path)

            if show_time and self.global_step > warmup_steps:
                ld_timer.start(step)

            #   DEBUG: CUDA footprint
            # free, total = torch.cuda.mem_get_info()
            # self.logger.info('free and total mem: {}GB / {}GB'.format(free/1024/1024/1024, total/1024/1024/1024))

        return True
