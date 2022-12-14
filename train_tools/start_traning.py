import sys
import glob
import os
import torch
import json
import subprocess
import time
import logging

sys.path.insert(0, "../")
from dataset.Dataset import SemData
from train_tools.Train import train
import util.Transforms as transform

def start(CONFIG_PATH, script):
    bashCommand = [script, CONFIG_PATH]
    list_files = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')
        
    CONFIG_PATH = sys.argv[1]
    CONFIG = json.load(open(CONFIG_PATH, "r+"))

    if(CONFIG["LOG_MODE"] == "DEBUG"):
        logging.basicConfig(level=logging.DEBUG, filename=CONFIG['LOG_PATH'])
    elif(CONFIG["LOG_MODE"] == "INFO"):
        logging.basicConfig(level=logging.INFO, filename=CONFIG['LOG_PATH'])
    
    args_dataset = CONFIG['DATASET']

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transform = transform.Compose([
        transform.Crop([args_dataset["train_h"], args_dataset["train_w"]], crop_type='center', padding=mean, ignore_label=args_dataset["ignore_label"]),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    val_loader = torch.utils.data.DataLoader(
        dataset=SemData(
            split='val',
            data_root=CONFIG['DATA_PATH'],
            data_list=CONFIG['DATASET']['val_list'],
            transform=val_transform
        ),
        batch_size=16,
        num_workers=8,
        pin_memory=True
    )

    train_transform = transform.Compose([
        transform.RandScale([args_dataset["scale_min"], args_dataset["scale_max"]]),
        transform.RandRotate([args_dataset["rotate_min"], args_dataset["rotate_max"]], padding=mean, ignore_label=args_dataset["ignore_label"]),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args_dataset["train_h"], args_dataset["train_w"]], crop_type='rand', padding=mean, ignore_label=args_dataset["ignore_label"]),
        transform.ToTensor(),
        transform.Normalize(mean, std)])

    train_data = SemData(
        split='train',
        data_root=CONFIG['DATA_PATH'],
        data_list=CONFIG['DATASET']["train_list"],
        transform=train_transform
    )

    train_data_set_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        drop_last=True
    )

    try:
        train(CONFIG_PATH, CONFIG, train_data_set_loader, val_loader, start)
    except Exception as e:
        logging.exception(str(e))
