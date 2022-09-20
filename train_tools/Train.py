import glob
import sys
import torch
import os
import time
import numpy as np
import json
import time
import logging

sys.path.insert(0, "../")

from util.Optimizer import poly_learning_rate
from models.Model import get_model
from util.Metrics import intersectionAndUnion
from util.WBLogger import LogerWB

def train(CONFIG_PATH, CONFIG, train_loader, val_loader_, start):
    logger = LogerWB(CONFIG["WB_LOG"], print_messages=CONFIG["PRINT_LOG"])
    
    model = get_model(CONFIG['DEVICE_TRAIN'])
    optimizer = torch.optim.SGD(
        [{'params': model.layer0.parameters()},
        {'params': model.layer1.parameters()},
        {'params': model.layer2.parameters()},
        {'params': model.layer3.parameters()},
        {'params': model.layer4.parameters()},
        {'params': model.ppm.parameters(), 'lr': CONFIG['LEARNING_RATE'] * 10},
        {'params': model.cls.parameters(), 'lr': CONFIG['LEARNING_RATE'] * 10},
        {'params': model.aux.parameters(), 'lr': CONFIG['LEARNING_RATE'] * 10}],
        lr=CONFIG['LEARNING_RATE'], momentum=CONFIG['MOMENTUM'], weight_decay=CONFIG['WEIGHT_DECAY'])
    print("Traning started.....")
    
    max_iter = CONFIG["EPOCHS"] * train_loader.__len__()

    train_loader_len = int(CONFIG["TRAIN_DATASET_SIZE"] / CONFIG["TRAIN_BATCH_SIZE"])
    val_loader_len = int(CONFIG["VAL_DATASET_SIZE"] / CONFIG["TRAIN_BATCH_SIZE"])
    
    current_iter = 1
    
    for e in range(CONFIG["EPOCHS"]):
        model = model.train()

        loss_train_epoch = 0
        iou_train_epoch = 0
        acc_train_epoch = 0

        logging.info(f"Train Adversarial loader length:{train_loader_len}")
        logging.info(f"Val Adversarial loader length:{val_loader_len}")

        batch_id = 0
        
        for data in train_loader:
            image_normal, target_normal = data
            image_normal = image_normal.to(CONFIG['DEVICE_TRAIN'])
            target_normal = target_normal.to(CONFIG['DEVICE_TRAIN'])

            poly_learning_rate(optimizer, CONFIG['LEARNING_RATE'], current_iter, max_iter, power=CONFIG['POWER'])
            optimizer.zero_grad()

            output_normal, main_loss, aux_loss, _ = model(image_normal, target_normal)
            loss = main_loss + CONFIG['AUX_WEIGHT'] * aux_loss
            
            loss.backward()
            optimizer.step()

            intersection_normal, union_normal, target_normal = intersectionAndUnion(output_normal, target_normal, CONFIG['CALSSES'], CONFIG['IGNOR_LABEL'])
            intersection_normal, union_normal, target_normal = intersection_normal.cpu().numpy(), union_normal.cpu().numpy(), target_normal.cpu().numpy()
            
            iou = np.mean(intersection_normal / (union_normal + 1e-10))
            acc = sum(intersection_normal) / sum(target_normal)

            logger.log_loss_batch_train_adversarial(train_loader_len, e, batch_id + 1, loss.item())
            logger.log_iou_batch_train_adversarial(train_loader_len, e, batch_id + 1, iou)
            logger.log_acc_batch_train_adversarial(train_loader_len, e, batch_id + 1, acc)

            iou_train_epoch += iou
            loss_train_epoch += loss.item()
            acc_train_epoch += acc
            
            batch_id += 1
            current_iter += 1
            
            logger.log_current_iter_epoch(current_iter)
            logger.log_epoch(int(e))

        loss_train_epoch = loss_train_epoch / batch_id
        iou_train_epoch = iou_train_epoch / batch_id
        acc_train_epoch = acc_train_epoch / batch_id

        logger.log_loss_epoch_train_adversarial(e, loss_train_epoch)
        logger.log_iou_epoch_train_adversarial(e, iou_train_epoch)
        logger.log_acc_epoch_train_adversarial(e, acc_train_epoch)

        torch.save(model.state_dict(), CONFIG["MODEL_SAVE"] + CONFIG["MODEL_NAME"] + "_" + str(e) + ".pt")
        torch.save(optimizer.state_dict(), CONFIG["MODEL_SAVE"] + CONFIG["MODEL_NAME"] + "_optimizer" + str(e) + ".pt")

        model = model.eval()

        loss_val_epoch = 0
        iou_val_epoch = 0
        acc_val_epoch = 0

        val_status = 0
        
        logging.info("Set val...")
        logging.info(f"Val finished:{str(val_status / val_loader_len)[:5]}%")

        loss_val_epoch = 0
        iou_val_epoch = 0
        acc_val_epoch = 0
        
        batch_id = 0
        
        for data in val_loader_:
            with torch.no_grad():
                image_val, target = data
                image_val = image_val.to(CONFIG['DEVICE_TRAIN'])
                target = target.to(CONFIG['DEVICE_TRAIN'])
                
                output, _, loss = model(image_val, target)

                intersection, union, target = intersectionAndUnion(output, target, CONFIG['CALSSES'], CONFIG['IGNOR_LABEL'])
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()

                iou = np.mean(intersection / (union + 1e-10))
                acc = sum(intersection) / (sum(target) + 1e-10)

                iou_val_epoch += iou
                loss_val_epoch += loss
                acc_val_epoch += acc
                batch_id += 1
            logging.info(f"Val Normal Finished:{str(batch_id * 100 / val_loader_.__len__())}")
                
        loss_val_epoch = loss_val_epoch / val_loader_.__len__()
        iou_val_epoch = iou_val_epoch / val_loader_.__len__()
        acc_val_epoch = acc_val_epoch / val_loader_.__len__()

        logger.log_loss_epoch_val(e, loss_val_epoch)
        logger.log_iou_epoch_val(e, iou_val_epoch)
        logger.log_acc_epoch_val(e, acc_val_epoch)
