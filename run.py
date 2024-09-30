from ultralytics import YOLO
import os
#os.environ['WANDB_DISABLED'] = 'true'
model = YOLO('yolov10s.pt')
model.train(data="SKU-110K.yaml",epochs=100,imgsz=416)
#model.export(format = "onnx")