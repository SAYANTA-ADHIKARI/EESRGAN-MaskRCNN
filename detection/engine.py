import math
import sys
import time
import torch
import os
import numpy as np
import cv2
import copy
import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset, get_coco_api_from_dataset_base
from .coco_eval import CocoEvaluator
from .utils import MetricLogger, SmoothedValue, warmup_lr_scheduler, reduce_dict
from utils import tensor2img
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger("val")

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print(images)
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    # iou_types = ["bbox"]
    iou_types = []
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

'''
Draw boxes on the test images
'''
def draw_detection_boxes(new_class_conf_box, config, file_name, image, actual_image):
    print(config['path']['output_images'])
    print(file_name)
    
    source_image_path = os.path.join("./DetectionPatches_256x256/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4", file_name + '.jpg')
    print(source_image_path)
    dest_image_path = os.path.join(config['path']['Test_Result_SR'], file_name+'.png')
    dest_image_path_actual = os.path.join(config['path']['Test_Result_SR'], file_name+'_actual.png')
    source_image = cv2.imread(source_image_path, 1)
    #image = actual_image
    #print(new_class_conf_box)
    #print(len(new_class_conf_box))
    for i in range(len(new_class_conf_box)):
        clas,con,x1,y1,x2,y2 = new_class_conf_box[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Car: '+ str((int(con*100))) + '%', (x1+5, y1+8), font, 0.2,(0,255,0),1,cv2.LINE_AA)
    #cv2.imshow('image', image)
    cv2.imwrite(dest_image_path, image)
    for i in range(len(new_class_conf_box)):
        clas,con,x1,y1,x2,y2 = new_class_conf_box[i]
        cv2.rectangle(source_image, (x1, y1), (x2, y2), (0,0,255), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(source_image, 'Car: '+ str((int(con*100))) + '%', (x1+5, y1+8), font, 0.2,(0,255,0),1,cv2.LINE_AA)
    #cv2.imshow('image', image)
    cv2.imwrite(dest_image_path_actual, source_image)
'''
for generating test boxes
'''

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

def get_masks(masks, labels):
    _, H, W = masks.shape
    req_mask = np.zeros((H, W, 4))
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

def get_prediction(outputs, file_path, config, file_name, image, actual_image, fileptr ,threshold=0.5):
    print(file_path)
    print(file_name)
    fileptr.write(file_name + '\n')
    new_class_conf_box = list()
    pred_class = [i for i in list(outputs[0]['labels'].detach().cpu().numpy())] # Get the Prediction Score
    text_boxes = [ [i[0], i[1], i[2], i[3] ] for i in list(outputs[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
    pred_score = list(outputs[0]['scores'].detach().cpu().numpy())
    masks = get_masks(outputs[0]['masks'].squeeze(1), outputs[0]['labels'])
    dest_image_path = os.path.join(config['path']['Test_Result_SR'], "test_generated_masks" ,file_name+'.png')
    plt.imsave(dest_image_path, masks)
    for pred_scor, label, text_box in zip(pred_score, pred_class, text_boxes):
        fileptr.write(str(label) + ' ' + str(pred_scor) + ' [' + str(int(text_box[0])) + ' ' + str(int(text_box[1])) + ' ' + str(int(text_box[2])) + ' ' + str(int(text_box[3]))+ ']\n')
    #print(pred_score)
    for i in range(len(text_boxes)):
        new_class_conf_box.append([pred_class[i], pred_score[i], int(text_boxes[i][0]), int(text_boxes[i][1]), int(text_boxes[i][2]), int(text_boxes[i][3])])
    draw_detection_boxes(new_class_conf_box, config, file_name, image, actual_image)
    new_class_conf_box1 = np.matrix(new_class_conf_box)
    #print(new_class_conf_box)
    if(len(new_class_conf_box))>0:
        np.savetxt(file_path, new_class_conf_box1, fmt="%i %1.3f %i %i %i %i")


@torch.no_grad()
def evaluate_save(model_G, model_FRCNN, data_loader, device, config):
    i = 0
    print("Detection started........")
    os.makedirs(os.path.join(config['path']['Test_Result_SR'], "test_generated_masks"), exist_ok=True)
    f = open(os.path.join(config['path']['Test_Result_SR'], "test_generated_masks" ,'test_results.txt'), 'a')
    f.write("Detection Results\n")
    for image, targets in data_loader:
        #print(targets)
        actual_image = copy.deepcopy(image['image_lq'][0])
        actual_image = tensor2img(actual_image)
        image['image_lq'] = image['image_lq'].to(device)

        _, img, _, _ = model_G(image['image_lq'])
        img_count = img.size()[0]
        images = [img[i] for i in range(img_count)]
        outputs = model_FRCNN(images)
        file_name = os.path.splitext(os.path.basename(image['LQ_path'][0]))[0]
        file_path = os.path.join(config['path']['Test_Result_SR'], file_name+'.txt')
        i=i+1
        print(i)
        img = img[0].detach()[0].float().cpu()
        img = tensor2img(img)
        get_prediction(outputs, file_path, config, file_name, img, actual_image, f)
    print("successfully generated the results!")

'''
This evaluate method is changed to pass the generator network and evalute
the FRCNN with generated SR images
'''
@torch.no_grad()
def evaluate(model_G, model_FRCNN, data_loader, device, train):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    if train:
        model_FRCNN.train()
    #model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model_FRCNN)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    log_dict = {}
    counter = 0
    logger.info("Evaluating the model")
    for image, targets in tqdm(metric_logger.log_every(data_loader, 100, header), desc="Evaluating: ", total=len(data_loader)):
        image['image_lq'] = image['image_lq'].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        # model_g_copy = copy.deepcopy(model_G.module)
        # model_g_copy.eval()
        # model_FRCNN_copy = copy.deepcopy(model_FRCNN.module)
        # model_FRCNN_copy.eval()
        # model_g_copy.to(device)
        # model_FRCNN_copy.to(device)
        # #model_G.to(device)
        #model_FRCNN.to(device)
        _, image, _, _ = model_G(image['image_lq'])
        
        #print(model_G.device_ids)
        #print(model_FRCNN.device_ids)
        img_count = image.size()[0]
        image = [image[i] for i in range(img_count)]
        #image = [img.cuda(device=model_FRCNN.device_ids[0]) for img in image]
        loss_dict = None
        outputs = None
        metric_flag = True
        if train:
            loss_dict = model_FRCNN(image, targets)
            loss_dict = reduce_dict(loss_dict)
        else:
            outputs = model_FRCNN(image)

        if outputs is not None:
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time) 
        if loss_dict is not None:
            metric_flag  =False
            for k in loss_dict.keys():
                if k in log_dict.keys():
                    log_dict[k] += loss_dict[k].item()
                else:
                    log_dict[k] = loss_dict[k].item()
            message = "Val:::Counter: {:1d}\t".format(counter)
            for k, v in loss_dict.items():
                message += "{:s}: {:.4f} ".format(k, v)

            logger.info(message)
            counter += 1

        # Check for NaN loss
            losses = 0
            for k, val in loss_dict.items():
                losses = losses + val
            if torch.isnan(losses):
                print(f" Evaluate Loss is Nan, {loss_dict}, Image ID: {targets[0]['image_id']}")
                sys.exit()

            for k in log_dict.keys():
                log_dict[k] = log_dict[k]/len(data_loader)
            
    if metric_flag:
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        return coco_evaluator, res, log_dict   # NOTE: Change made here to return res
    else:
        return None, None, log_dict

@torch.no_grad()
def evaluate_base(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset_base(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)
        #print(outputs)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
