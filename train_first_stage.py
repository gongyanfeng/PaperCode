from ultralytics import YOLO
from PIL import Image

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128_skyline.yaml", epochs=300, imgsz=128, resume=False)  # train the model for detecting skyline on image

