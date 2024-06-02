import logging
import numpy as np
import torch
import math
import os
import model.ESRGAN_EESN_FRCNN_Model as ESRGAN_EESN
from utils import (
    save_img,
    tensor2img,
    mkdir,
)
from tqdm import tqdm
import glob
import json
import shutil
import sys

logger = logging.getLogger("base")
"""
python train.py -c config_GAN.json
modified from ESRGAN repo
"""
COLOUR_DICT = {
    0: [0, 0, 0, 1.0],
    1: [0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0],
    2: [1.0, 0.4980392156862745, 0.054901960784313725, 1.0],
    3: [0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0],
    4: [0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0],
    5: [0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0],
    6: [0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0],
    7: [0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0],
    8: [0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0],
    9: [0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0],
    10: [0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0],
    11: [0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0],
}

class MyDataMaskRCNNTrainer:
    """
    Trainer class
    """

    def __init__(self, config, data_loader, valid_data_loader=None):
        self.config = config
        self.data_loader = data_loader
        # print(f"Data Loader: {data_loader}")
        # print(next(iter(data_loader)))
        # sys.exit()
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if n_gpu > 0 else "cpu")
        print(self.device)
        # self.device = torch.device("cpu")
        self.train_size = int(
            math.ceil(
                self.data_loader.length
                / int(config["data_loader"]["args"]["batch_size"])
            )
        )
        self.total_iters = int(config["train"]["niter"])
        self.total_epochs = int(math.ceil(self.total_iters / self.train_size))
        print(self.total_epochs)
        print(f'totalepochs: {self.total_epochs}')
        self.model = ESRGAN_EESN.ESRGAN_EESN_FRCNN_Model(config, self.device)

    def test(self):
        self.model.test(self.data_loader, train=False, testResult=True)

    def train(self):
        """
        Training logic for an epoch
        for visualization use the following code (use batch size = 1):

        category_id_to_name = {1: 'car'}
        for batch_idx, dataset_dict in enumerate(self.data_loader):
            if dataset_dict['idx'][0] == 10:
                print(dataset_dict)
                visualize(dataset_dict, category_id_to_name) #--> see this method in util

        #image size: torch.Size([10, 3, 256, 256]) if batch_size = 10
        """
        logger.info(
            "Number of train images: {:,d}, iters: {:,d}".format(
                self.data_loader.length, self.train_size
            )
        )
        logger.info(
            "Total epochs needed: {:d} for iters {:,d}".format(
                self.total_epochs, self.total_iters
            )
        )
        # tensorboard logger
        if self.config["use_tb_logger"] and "debug" not in self.config["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir=os.path.join(self.config["train"]["save_dir"], "tb_logger/", self.config["name"]))
        ## Todo : resume capability
        current_step = 0
        start_epoch = 0

        #### training
        logger.info(
            "Start training from epoch: {:d}, iter: {:d}".format(
                start_epoch, current_step
            )
        )
        print(f"{'#'*20}")
        print(len(self.valid_data_loader))
        print(f"{'#'*20}")
        for epoch in range(start_epoch, self.total_epochs + 1):
            for _, (image, targets) in tqdm(
                enumerate(self.data_loader), desc=f"Epoch {epoch}: ", total=len(self.data_loader)
            ):
                current_step += 1
                if current_step > self.total_iters:
                    break
                #### update learning rate
                self.model.update_learning_rate(
                    current_step, warmup_iter=self.config["train"]["warmup_iter"]
                )

                #### training
                self.model.feed_data(image, targets)
                self.model.optimize_parameters(current_step)

                #### log
                if current_step % self.config["logger"]["print_freq"] == 0:
                    logs = self.model.get_current_log()
                    message = "Train:::Epoch: {:3d}, Iter:{:6d}, lr:{:.8f} ".format(
                        epoch, current_step, self.model.get_current_learning_rate()
                    )
                    for k, v in logs.items():
                        message += "{:s}: {:.4f} ".format(k, v)
                        # tensorboard logger
                        if (
                            self.config["use_tb_logger"]
                            and "debug" not in self.config["name"]
                        ):
                            tb_logger.add_scalar(f"train/{k}", v, current_step)

                    logger.info(message)

                # validation
                
                if current_step % self.config["train"]["val_freq"] == 0:
                    self.model.test(self.valid_data_loader)
                    # validation logs
                    
                    logs = self.model.get_current_log()
                    message = "Val:::Epoch: {:3d}, Iter:{:8d}, lr:{:.3f} ".format(
                        epoch, current_step, self.model.get_current_learning_rate()
                    )
                    for k, v in logs.items():
                        message += "{:s}: {:.4f} ".format(k, v)
                        # tensorboard logger
                        if (self.config["use_tb_logger"] and "debug" not in self.config["name"]):
                            tb_logger.add_scalar(f"val/{k}", v, current_step)

                    # Saving the model
                    val_loss = 0
                    for k, val in logs.items():
                        val_loss += self.config["train"]["mask_rcnn_loss"][k.split("/")[-1]] * val

                    # tensorboard logger
                    if (self.config["use_tb_logger"] and "debug" not in self.config["name"]):
                        tb_logger.add_scalar(f"val/MaskRCNN", val_loss, current_step)

                    message += "{:s}: {:.4f} ".format("MaskRCNN", val_loss)

                    logger.info(message)

                    save_path = self.config["path"]["models"]
                    if len(glob.glob(save_path + "/checkpoints_*")) < self.config["train"]["num_of_saved_models"]:
                        count = len(glob.glob(save_path + "/checkpoints_*"))
                        save_path = os.path.join(save_path, f"checkpoints_{count}")
                        os.makedirs(save_path, exist_ok=True)
                        self.model.save(current_step, save_path)
                        json.dump({"val_loss": val_loss, "current_step": current_step}, open(os.path.join(save_path, "status.json"), "w"))
                    else:
                        high_val_loss = None
                        for path in glob.glob(save_path + "/checkpoints_*"):
                            status = json.load(open(os.path.join(path, "status.json"), "r"))
                            if status["val_loss"] > val_loss:
                                high_val_loss = path
                        if high_val_loss is not None:
                            # NOTE: this dont work ---> os.removedirs(high_val_loss)
                            shutil.rmtree(high_val_loss,  ignore_errors=True)
                            save_path = high_val_loss
                            os.makedirs(save_path, exist_ok=True)
                            self.model.save(current_step, save_path)
                            json.dump({"val_loss": val_loss, "current_step": current_step}, open(os.path.join(save_path, "status.json"), "w"))

                    print("Saved the model!!!!!!!!!!!")

                #### save models and training states
                if current_step % self.config["logger"]["save_checkpoint_freq"] == 0:
                    # logger.info("Saving models and training states.")
                    # self.model.save(current_step)
                    # self.model.save_training_state(epoch, current_step)
                    logger.info("Saving Intermediate Images.")

                    # saving SR_images
                    for _, (image, targets) in enumerate(self.valid_data_loader):
                        # print(image)
                        img_name = os.path.splitext(
                            os.path.basename(image["LQ_path"][0])
                        )[0]
                        img_dir = os.path.join(
                            self.config["path"]["val_images"], img_name
                        )
                        mkdir(img_dir)

                        self.model.feed_data(image, targets)
                        self.model.test(self.valid_data_loader, train=False)

                        visuals = self.model.get_current_visuals()
                        sr_img = tensor2img(visuals["SR"])  # uint8
                        gt_img = tensor2img(visuals["GT"])  # uint8
                        lap_learned = tensor2img(visuals["lap_learned"])  # uint8
                        lap = tensor2img(visuals["lap"])  # uint8
                        lap_HR = tensor2img(visuals["lap_HR"])  # uint8
                        final_SR = tensor2img(visuals["final_SR"])  # uint8

                        # Save SR images for reference
                        save_img_path = os.path.join(
                            img_dir, "{:s}_{:d}_SR.png".format(img_name, current_step)
                        )
                        save_img(sr_img, save_img_path)
                        # Save GT images for reference
                        save_img_path = os.path.join(
                            img_dir, "{:s}_{:d}_GT.png".format(img_name, current_step)
                        )
                        save_img(gt_img, save_img_path)
                        # Save final_SR images for reference
                        save_img_path = os.path.join(
                            img_dir,
                            "{:s}_{:d}_final_SR.png".format(img_name, current_step),
                        )
                        save_img(final_SR, save_img_path)
                        # Save lap_learned images for reference
                        save_img_path = os.path.join(
                            img_dir,
                            "{:s}_{:d}_lap_learned.png".format(img_name, current_step),
                        )
                        save_img(lap_learned, save_img_path)
                        # Save lap images for reference
                        save_img_path = os.path.join(
                            img_dir, "{:s}_{:d}_lap.png".format(img_name, current_step)
                        )
                        save_img(lap, save_img_path)
                        # Save lap images for reference
                        save_img_path = os.path.join(
                            img_dir,
                            "{:s}_{:d}_lap_HR.png".format(img_name, current_step),
                        )
                        save_img(lap_HR, save_img_path)

                        # Saving masks for reference
                        results = visuals["FRCNN"]
                        masks = results[0]["masks"]
                        labels = results[0]["labels"]
                        masks = masks.squeeze(1)
                        masks = masks >= 0.5
                        masks = masks.float()
                        masks = get_masks(masks, labels)
                        import matplotlib.pyplot as plt
                        mask_path = os.path.join(
                            img_dir, "{:s}_{:d}_mask.jpg".format(img_name, current_step)
                        )
                        plt.imsave(mask_path, masks, cmap="gray")

        logger.info("Saving the final model.")
        self.model.save("latest")
        json.dump({"val_loss": val_loss, "current_step": current_step}, 
                  open(os.path.join(self.config["path"]["models"], "status.json"), "w"))
        logger.info("End of training.")


def get_masks(masks, labels):
    _, H, W = masks.shape
    req_mask = np.zeros((H, W, 4))
    req_mask[:, :, 3] = np.ones((H, W))
    labels = labels.cpu().numpy()
    unique_values, counts = np.unique(labels, return_counts=True)
    unique_counts_dict = dict(zip(unique_values, counts))
    color_dict = {}
    for i in unique_values:
        color_dict[i] = []
        intensity = np.linspace(0.1, 1.0, unique_counts_dict[i])
        for j in range(unique_counts_dict[i]):
            color = COLOUR_DICT[i]
            color[3] = intensity[j]
            color_dict[i].append(color)
    masks = masks.cpu().numpy()
    masks = list(masks)
    for label, mask in zip(labels, masks):
        color = color_dict[label].pop(0)
        new_mask = np.zeros((H, W, 4))
        new_mask[mask > 0] = color
        req_mask = req_mask + new_mask

    np.clip(req_mask, 0, 1, out=req_mask)
    return req_mask