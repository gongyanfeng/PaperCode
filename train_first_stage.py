from ultralytics import YOLO
from PIL import Image

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
#model  = YOLO("/home/gyf/e/yolov8-sahi/yolov8/runs/detect/train28/weights/last.pt")

# Use the model
model.train(data="coco128_skyline.yaml", epochs=300, imgsz=128, resume=False)  # train the model for detecting skyline on image

'''
model = YOLO("best_128.pt")
results = model("./result_slice/slice_0_0_128_24.jpg")  # predict on an image
print("gyf:results={}".format(results))
for r in results:
    im_array = r.plot(conf=False,line_width=1)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image
#path = model.export(format="onnx")  # export the model to ONNX format
'''
