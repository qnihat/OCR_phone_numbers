import argparse
import os
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from time import sleep

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import json
from rabbit_producer import publish_message_to
from config import RABBIT_QUEUE_NAME
from config import FOLDERS_TO_SAVE_NUM_IMGS
import uuid
import datetime

model_path = 'weights/number_detect.pt'


def detect_num(img_path, img_class='nomre'):
    global num_img
    imgsz = 640
    source = img_path
    weights = model_path

    set_logging()
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment='store_true')[0]

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment='store_true')[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=opt.classes, agnostic='store_true')

        # Apply Classifier


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            #if webcam:  # batch_size >= 1
            #    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            #else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                print('TOTAL NUMBERS: ', len(det))
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    crop_img = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    year_mon_day = datetime.datetime.now().strftime("%Y%m%d")
                    full_path = FOLDERS_TO_SAVE_NUM_IMGS + year_mon_day
                    uuid_name_ = str(uuid.uuid4()) + '.jpg'
                    if not os.path.exists(full_path):
                        os.makedirs(full_path)
                    cv2.imwrite(full_path + '/' + uuid_name_, crop_img)
                    data = {'file_name': full_path + '/' + uuid_name_}
                    message = json.dumps(data)
                    publish_message_to(RABBIT_QUEUE_NAME, message)

            return crop_img


#if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
opt = parser.parse_args()
'''
dir_ = 'images_from_milana/'

for file in os.listdir(dir_):
    if '.jpg' in file:
        file = dir_ + file
        detect_num(file)
    sleep(2)
'''