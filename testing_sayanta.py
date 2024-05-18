from parse_config import ConfigParser
import argparse
import collections
from model.ESRGAN_EESN_FRCNN_Model import ESRGAN_EESN_FRCNN_Model
import torch

CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
options = [
    CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
    CustomArgs(
        ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
    ),
]
args = argparse.ArgumentParser(description="PyTorch Template")
args.add_argument(
    "-c",
    "--config",
    default="/raid/ai22mtech13004/EESRGAN/config_GAN.json",
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

config = ConfigParser.from_args(args, options)

device = "cuda:7" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
model = ESRGAN_EESN_FRCNN_Model(config, device)

model.print_network()