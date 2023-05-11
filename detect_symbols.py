import torch
from numpy import random
import cv2
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from sk_img_class_predict import predict_symbols

model_path='weights/simbol_detect.pt'
num_img=0

def detect_sym(image):
    global num_img
    imgsz=640
    source=image
    weights=model_path
    # Initialize
    set_logging()
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader

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
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment='store_true')[0]

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment='store_true')[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic='store_true')


    # Process detections
    for i, det in enumerate(pred):  # detections per image
        arr_of_nums=[]
        temp_dict={}

        im0 = im0s
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            #print('TOTAL NUMBERS: ',len(det))
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                #print(label)
                crop_img=plot_one_box(xyxy, im0, label='', color=colors[int(cls)], line_thickness=0)
                #print('shape: ', crop_img.shape[1]/crop_img.shape[0])
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                #print('coor: ',xywh[0])
                #with open(img_path.split('.')[0] + '.txt', 'a') as f:
                #    f.write(f'0 {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')
                #cv2.imshow('img',im0)
                #cv2.waitKey(0)
                #cv2.imwrite('extracted_symbols/'+str(num_img)+'.jpg',crop_img)
                num_img=num_img+1

                #cv2.imwrite('result/'+img_class+'/'+str(time.time())+'.png',crop_img)
                sym_=predict_symbols(crop_img,xywh[0])
                arr_of_nums.append(sym_)
                temp_dict[xywh[0]]=sym_
                #print(temp_dict)
        return temp_dict