# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_package_minimum_version, check_requirements

#the following added by gyf
from nets.yolo import YoloBody
from utils.utils import (cvtColor, preprocess_input,
                         resize_image)
import torch
import torch.nn as nn

from  PIL import Image #added by gyf
import cv2 #added by gyf

logger = logging.getLogger(__name__)


class Yolov5DetectionModel(DetectionModel):

    
    ###added by gyf=={{{
    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi, backbone = self.backbone, input_shape = self.input_shape)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def perform_inference(self, image):##gyf:image is ndarray
        print("gyf:image.shape input perform_inference={}".format(image.shape))#gyf:(14, 64, 3)#ndarray直接就是RGB
        print("gyf:image type input perform_inference={}".format(type(image)))#gyf: ndarray
        # print("gyf:np-transpose.shape={}".format(np.transpose(np.array(image, dtype='float32'), (2, 0, 1)).shape))
        # print("gyf:np-transpose={}".format(np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))))
        #print("gyf:np-transpose2={}".format(np.transpose(image, (2, 0, 1))))
        #image_one = image[...,0]
        #image_shape = image.shape[0:2]
        #print("gyf:image_shape input perform_inference={}".format(image_shape))
        #print("gyf:image input perform_inference={}".format(image))
        #image = image
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        #image       = cvtColor(image) #gyf:#ndarray直接就是RGB,所以无需转换
        image_pil = Image.fromarray(image)
        # image_data=cv2.resize(image,(128,128),interpolation=cv2.INTER_CUBIC)
        # image_data=image_data.astype('float32')
        # print("gyf:image_data.shape after resize={}".format(image_data))
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        #print("gyf:type of image={}".format(type(image)))
        print("gyf:image_pil.size before resize={}".format(image_pil.size))#(64, 14)
        # pixels = image_pil.load()
        # for w in range(image_pil.size[0]):
        #     for h in range(image_pil.size[1]):
        #         print("gyf:pixels[{},{}]={}".format(w,h,pixels[w,h]))
        #image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        self.letterbox_image = True
        print("gyf:self.input_shape[1]={}, self.input_shape[0]={}, self.letterbox_image={}".format(self.input_shape[1], self.input_shape[0], self.letterbox_image))
        
        image_data  = resize_image(image_pil, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #print("gyf:image_data after resize={}".format(image_data))#(128, 128)
        # pixels = image_data.load()
        # for w in range(image_data.size[0]):
        #     for h in range(image_data.size[1]):
        #         print("gyf:pixels[{},{}]={}".format(w,h,pixels[w,h]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        #image_data  = np.expand_dims(np.transpose(preprocess_input(image_data), (2, 0, 1)), 0)
        #print("gyf:image_data after transpose and preprocess={}".format(image_data))
        print("gyf:image_data shape after transpose and preprocess={}".format(image_data.shape))

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            print("gyf:image_shape input backbone={}".format(images.shape))
            #print("gyf:images input net={}".format(images))
            #outputs = self.net(images.to(torch.float32))
            outputs = self.net(images)
            # for output in outputs:
            #     print("gyf:output of my yolov5.shape={}".format(output.shape))
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                self._original_predictions = results[0] #added by gyf,一定要加，否则前一张识别结果永远储存在self._original_predictions成员变量中。
                return image
                
            for res in results:
                print("gyf:res.shape={}".format(res.shape))#gyf:一张图中的所有目标tensor,每个目标tensor的shape=[7]
            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
            print("gyf:top_label={}".format(top_label))
            print("gyf:top_conf={}".format(top_conf))
            print("gyf:top_boxes={}".format(top_boxes))
            print("gyf:results[0]={}".format(results[0]))
            for pred in results[0]:
                print("gyf:pred={}".format(pred))


            # for pred_box,pred_conf,pred_label in zip(list(top_boxes),list(top_conf),list(top_label)):
            # #for pred in zip(list(top_boxes),list(top_conf),list(top_label)):
            #     print("gyf:pred={}".format(pred))


        self._original_predictions = results[0]
        #print("gyf:self._original_predictions={}".format(self._original_predictions))#shape=(n,7),n表示n个目标
        print("gyf:type of self._original_predictions={}".format(type(self._original_predictions)))#<class 'numpy.ndarray'>
        if not self._original_predictions is None:
            for p in self._original_predictions:
                print("gyf:p={}".format(p))
    ###}}} #added by gyf
    



    def check_dependencies(self) -> None:
        check_requirements(["torch", "yolov5"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        import yolov5

        try:
            model = yolov5.load(self.model_path, device=self.device)
            print("gyf:model.__class__={}".format(model.__class__))#<class 'yolov5.models.common.AutoShape'>
            print("gyf:model.__class__.__module__={}".format(model.__class__.__module__))
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov5 model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv5 model.
        Args:
            model: Any
                A YOLOv5 model
        """

        if model.__class__.__module__ not in ["yolov5.models.common", "models.common"]:
            raise Exception(f"Not a yolov5 model: {type(model)}")

        model.conf = self.confidence_threshold
        self.model = model

        # set category_mapping
        print("gyf:self.category_mapping={}".format(self.category_mapping))#None
        print("gyf:self.category_names={}".format(self.category_names))
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping
    '''
    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        print("gyf:image input yolov5={}".format(image.shape))#gyf:(256, 256, 3)---h,w,c,一张切片图
        image = torch.tensor(np.transpose(image,(2,0,1))).unsqueeze(0)
        print("gyf:image input yolov5 after={}".format(image.shape))
        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        if self.image_size is not None:
            prediction_result = self.model(image, size=self.image_size)
        else:
            prediction_result = self.model(image)
        print("gyf:type of prediction_result={}".format(type(prediction_result)))#gyf:<class 'yolov5.models.common.Detections'>
        #print("gyf:all attributes of prediction_result={}".format(dir(prediction_result)))
        #print("gyf:only own attributes of prediction_result={}".format(vars(prediction_result)))
        #print("gyf:only xyxy attribute of prediction_result={}".format(prediction_result.xyxy))
        print("gyf:len(prediction_result.xyxy)={}".format(len(prediction_result.xyxy)))#gyf:1,因为输入只有一张图，不管是原图还是切片图，都只输入一张
        # for pred in prediction_result.xyxy:
        #     for index, pred_i in enumerate(pred):#gyf:遍历该图中各个目标，每个目标tensor结果size为６：[x1,y1,x2,y2,score,category_id]
        #         print("gyf:pred_{} of pred={}".format(index,pred_i))

        self._original_predictions = prediction_result
        #print("gyf:_original_predictions={}".format(self._original_predictions))#gyf:for instance: image 1/1: 256x256 6 cars Speed: 2.4ms pre-process, 124.3ms inference, 1.0ms NMS per image at shape (1, 3, 640, 640)
    '''    
    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        import yolov5
        from packaging import version

        if version.parse(yolov5.__version__) < version.parse("6.2.0"):
            return False
        else:
            return False  # fix when yolov5 supports segmentation models

    @property
    def category_names(self):
        if check_package_minimum_version("yolov5", "6.2.0"):
            return list(self.model.names.values())
            
        else:
            return self.model.names
          

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        print("gyf:shift_amount_list={}".format(shift_amount_list))#gyf:实际上是切片的偏移位置左上角坐标，因此，该list只有２个元素
        print("gyf:full_shape_list={}".format(full_shape_list))
        # compatilibty for sahi v0.8.15
        #shift_amount_list = fix_shift_amount_list(shift_amount_list)#gyf:仅增加了一个维度，表示图片个数，这里只有一张切片图，因此，shift_amount_list长度为１
        #full_shape_list = fix_full_shape_list(full_shape_list)
        #print("gyf:shift_amount_list after fix={}".format(shift_amount_list))
        
        # handle all predictions
        object_prediction_list_per_image = []
        #print("gyf:len(prediction_result.xyxy)={}".format(len(prediction_result.xyxy)))#gyf:1,因为输入只有一张图，不管是原图还是切片图，都只输入一张
        #for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions.xyxy):#annotated by gyf
        shift_amount = shift_amount_list
        full_shape = full_shape_list
        object_prediction_list = []
        object_prediction_list_per_image = []
        # for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
        #     print("gyf:image_predictions_in_xyxy_format={}".format(image_predictions_in_xyxy_format))
            # shift_amount = shift_amount_list[image_ind]
            # full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            # object_prediction_list = []

            # process predictions
            #for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():#annotated by gyf
        print("gyf:original_predictions is Nonetype={}".format(original_predictions is None))
        if not original_predictions is None:
            for prediction in original_predictions:
                # x1 = prediction[0]
                # y1 = prediction[1]
                # x2 = prediction[2]
                # y2 = prediction[3]
                y1 = prediction[0]
                x1 = prediction[1]
                y2 = prediction[2]
                x2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]*prediction[5]
                category_id = int(prediction[6])
                print("gyf:category_id={}".format(category_id))
                print("gyf:self.category_mapping={}".format(self.category_mapping))
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])
                print("gyf:bbox={}".format(bbox))
                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
        object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image#gyf:返回的是一张原图或者一张切片图的所有目标list
