from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from sahi.utils.cv import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    crop_object_predictions,
    cv2,
    get_video_reader,
    read_image_as_pil,
)

from ultralytics import YOLO
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img',type=str,default="", help='input the image for test')

opt = parser.parse_args()

model = YOLO("best_skyline_on_image_128.pt")

# 初始化检测模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='best_ship_on_slice_128.pt',
    confidence_threshold=0.3,
    device="cuda:0", #or "cpu"
)

assert opt.img!="","please input the image for test!"

base_name = os.path.basename(opt.img)
base_prefix = base_name.split(".")[0]
skyline_result = base_prefix+"_skyline."+base_name.split(".")[-1]
export_dir = "result"

#results = model("7_5.jpg")  # predict on an image
results = model(opt.img)  # predict on an image
print("gyf:results={}".format(results))
#img = cv2.imread("7_5.jpg",1)
img = cv2.imread(opt.img,1)
for r in results:
    # print("gyf:r.boxes={}".format(r.boxes.xyxy))
    # print("gyf:r.boxes.shape={}".format(r.boxes.xyxy.shape))
    boxes_list = r.boxes.xyxy.tolist()
    #print("gyf:len(boxes_list) from first_stage={}".format(len(boxes_list)))
    if len(boxes_list)>0:      
        for bbox in boxes_list:
            #print("gyf:bbox of skyline={}".format(bbox))
            int_bbox = list(map(int,bbox))
            dst = img[int_bbox[1]:int_bbox[3], int_bbox[0]:int_bbox[2]]   # 裁剪坐标为[y0:y1, x0:x1]
            #dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR) #gyf: 从img裁剪下来的dst区域为BGR的，需要转成RBG
            cv2.imwrite(skyline_result,dst)
            result = get_sliced_prediction(
                img, #added by gyf
                dst,
                detection_model,
                slice_height=dst.shape[0],
                slice_width = 128,
                overlap_height_ratio = 0.2,
                overlap_width_ratio = 0.2,
                #perform_standard_pred = True, #annotated by gyf
                perform_standard_pred = False,
            )
            for pred in result.object_prediction_list:
                bbox = pred.bbox  # 标注框BoundingBox对象，可以获得边界框的坐标、面积
                category = pred.category  # 类别Category对象，可获得类别id和类别名
                score = pred.score.value  # 预测置信度
                print("gyf:bbox-BoundingBox from detection results={}".format(bbox))
                print("gyf:bbox from detection results:x1={},y1={},x2={},y2={}".format(bbox.minx,bbox.miny,bbox.maxx,bbox.maxy))

            # 保存文件结果
            file_name = base_prefix
            #font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            #font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * 35 + 0.5).astype('int32'))
            #thickness   = int(max((64 + 35) // np.mean(self.input_shape), 1))
            #thickness   = int(max((64 + 35) // 128, 1))
            thickness = 1
            text_size = 1
            text_th = 1
            skyline_bbox = int_bbox

            result.export_visuals(export_dir=export_dir, file_name=file_name, rect_th=thickness, text_size=text_size,text_th=text_th,bbox_base=skyline_bbox, hide_labels=True)
    
    else:
        cv2.imwrite(os.path.join(export_dir,base_name),img)

