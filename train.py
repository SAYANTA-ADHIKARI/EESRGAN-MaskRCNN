import logging
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
from parse_config import ConfigParser
from trainer import MyDataMaskRCNNTrainer
from utils import setup_logger

"""
python train.py -c config_GAN_original.json
"""

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # logger = config.get_logger('train')
    # config loggers. Before it, the log will not work
    setup_logger(
        "base",
        config["path"]["log"],
        "train_" + config["name"],
        level=logging.INFO,
        screen=False,
        tofile=True,
    )
    setup_logger(
        "val",
        config["path"]["log"],
        "val_" + config["name"],
        level=logging.INFO,
        screen=False,
        tofile=True,
    )
    logger_train = logging.getLogger("base")
    logger_val = logging.getLogger("val")
    # logger.info(dict2str(config))

    # setup data_loader instances
    # train_data_loader = config.init_obj("train_data_loader", module_data)
    # val_data_loader = config.init_obj("val_data_loader", module_data)
    # train_data_loader = config.init_obj("data_loader", module_data)
    # val_data_loader = config.init_obj("data_loader", module_data)
    # change later this valid_data_loader using init_obj
    # train_data_loader = module_data.COWCGANFrcnnDataLoader(
    #     data_dir_GT="./DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/",
    #     data_dir_LQ="./DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/",
    #     batch_size=8,
    #     training=True,
    # )
    # val_data_loader = module_data.COWCGANFrcnnDataLoader(
    #     data_dir_GT="./DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/val_dir/HR",
    #     data_dir_LQ="./DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/val_dir/LR",
    #     batch_size=8,
    #     training=False,
    # )
    
    data_loader = config.init_obj('data_loader', module_data)
    # print(data_loader)
    val_data_loader = config.init_obj('val_data_loader', module_data)
    # exit()
    # print(val_data_loader)
    # exit()
    #change later this valid_data_loader using init_obj
    # valid_data_loader = module_data.COWCGANFrcnnDataLoader(
    #     "./DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/val_dir/HR/",
    #     "./DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/val_dir/LR/",
    #     12,
    #     training = False)
    # print("CANNY Filter Base")
    print(f"{'#'*20}")
    print(f"Train len outside trainer: {len(data_loader)}")
    print(f"Val len outside trainer: {len(val_data_loader)}")
    print(f"{'#'*20}")
    # build model architecture, then print to console
    # model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    # metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    """
    trainer = COWCGANTrainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    """
    """
    trainer = COWCGANTrainer(config=config,data_loader=data_loader,
                     valid_data_loader=valid_data_loader
                     )
    """

    # trainer = COWCGANFrcnnTrainer(
    #     config=config, data_loader=data_loader, valid_data_loader=val_data_loader
    # )
    trainer = MyDataMaskRCNNTrainer(
        config=config, data_loader=data_loader, valid_data_loader=val_data_loader
    )
    trainer.train()
    """
    trainer = COWCFRCNNTrainer(config=config)
    trainer.train()
    """


if __name__ == "__main__":
    # def force_cudnn_initialization():
    #     s = 32
    #     dev = torch.device('cuda')
    #     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    # force_cudnn_initialization()
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    # print(config.init_obj("data_loader",module_data))
    # print(config)
    main(config)
    # print(module_data)
