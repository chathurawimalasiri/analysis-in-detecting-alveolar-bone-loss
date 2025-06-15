from ultralytics import YOLO


import os
# os.environ["MKL_THREADING_LAYER"] = "GNU"

# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov9e.pt')

model.train(data='config.yaml', epochs=200, imgsz=640,  lr0 = 0.0001,device=[1] ,optimizer ="Adam", batch = 4, patience=25,resume = False,)