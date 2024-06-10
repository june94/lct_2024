import torch
import os
import cv2

import numpy as np

from sahi.predict import get_sliced_prediction
#from sahi.utils.yolov8 import download_yolov8n_model
from sahi.utils.cv import Colors
from sahi import AutoDetectionModel

from config import *


class Model:
    def __init__(self, model_path="/home/Документы/lct_2024/lct_2024/weights/best.pt"):
        #download_yolov8s_model(model_path)
        self.model = AutoDetectionModel.from_pretrained(
                                                        model_type='yolov8',
                                                        model_path=model_path,
                                                        confidence_threshold=0.25,
                                                        device="cuda:0" if torch.cuda.is_available() else "cpu" 
                                                        )
        
    def __call__(self, image):
        result = get_sliced_prediction(
                                        image = image,
                                        detection_model = self.model,
                                        slice_height = 640,
                                        slice_width = 640,
                                        overlap_height_ratio = 0.1,
                                        overlap_width_ratio = 0.1,
                                        perform_standard_pred = True,
                                        postprocess_type = "NMS",
                                        postprocess_class_agnostic = True,

        )
        return result
    
    @staticmethod
    def voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
        x = ((x2 + x1)/(2*image_w))
        y = ((y2 + y1)/(2*image_h))
        w = (x2 - x1)/image_w
        h = (y2 - y1)/image_h
        return [x, y, w, h]
    
    @staticmethod
    def yolo_to_voc(image_w, image_h, x, y, w, h):
        xmax = int((x*image_w) + (w * image_w)/2.0)
        xmin = int((x*image_w) - (w * image_w)/2.0)
        ymax = int((y*image_h) + (h * image_h)/2.0)
        ymin = int((y*image_h) - (h * image_h)/2.0)
        return [xmin, xmax, ymin, ymax]
    
    @staticmethod
    def save_preds(result, image_height, image_width, image_name, 
                   save_path):
        os.makedirs(save_path, exist_ok=True) # 
        file_name = os.path.splitext(image_name) + '.txt'
        with open(os.path.join(save_path, file_name), "w") as f:
            for detection in result.object_prediction_list:
                tmp_bb = Model.voc_to_yolo(*detection.bbox.to_xyxy(), image_height, image_width)
                f.write(f"{detection.category.id} {' '.join([str(i) for i in tmp_bb])}\n")

    @staticmethod
    def plot_preds(image, 
                   bboxes,
                   classes,
                   file_name = None,
                   save_path = None, 
                   select_classes = [],
                   rect_th: int = None,
                   text_size: float = None,
                   text_th: float = None,
                   color: tuple = None,
                   hide_labels: bool = False,
        ):
            if color is None:
                colors = Colors()
            else:
                colors = None

            rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
            text_th = text_th or max(rect_th - 1, 1)
            text_size = text_size or rect_th / 3
            for bbox, cl in zip(bboxes, classes):
                if cl not in select_classes and len(select_classes):
                    continue
                if colors is not None:
                    color = colors(cl)
                bbox = Model.yolo_to_voc(image.shape[1], image.shape[0], bbox[0], bbox[1], bbox[2], bbox[3])
                point1, point2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(
                    image,
                    point1,
                    point2,
                    color=color,
                    thickness=rect_th,
                )

                if not hide_labels:
                    label = f"{CATEGORIES[cl]}"

                    box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
                        0
                    ]  
                    outside = point1[1] - box_height - 3 >= 0  
                    point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
                    cv2.rectangle(image, point1, point2, color, -1, cv2.LINE_AA) 
                    cv2.putText(
                        image,
                        label,
                        (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                        0,
                        text_size,
                        (255, 255, 255),
                        thickness=text_th,
                    )

            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                full_save_path = os.path.join(save_path, file_name) ## check ext!!!
                cv2.imwrite(full_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
