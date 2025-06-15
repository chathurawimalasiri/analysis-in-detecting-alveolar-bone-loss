from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
# import numpy as np
import os
# os.environ["MKL_THREADING_LAYER"] = "GNU"


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov9e.pt')


model.train(data='config.yaml', epochs=200, imgsz=640,  lr0 = 0.0001,device=[0] ,optimizer ='Adam', batch = 4, patience=30,resume = False)
