import torch
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
import numpy as np


class Model:
    def __init__(self, model_path="/home/Документы/lct_2024/lct_2024/weights/best.pt"):
        self.model_path = model_path
        #download_yolov8s_model(model_path)
        self.model = AutoDetectionModel.from_pretrained(
                                                                model_type='yolov8',
                                                                model_path=self.model_path,
                                                                confidence_threshold=0.25,
                                                                device="cuda:0" if torch.cuda.is_available() else "cpu" 
                                                            )
        
    def __call__(self, image_path):
        result = get_sliced_prediction(
                                        image = image_path,
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
        return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]
    
    @staticmethod
    def save_preds(result, image_height, image_width, image_name, 
                   save_path="/home/Документы/lct_2024/lct_2024/stremlit_utils/logs/testset"):
        os.makedirs(save_path, exist_ok=True) # 
        file_name = os.path.splitext(image_name) + '.txt'
        with open(os.path.join(save_path, file_name), "w") as f:
            for detection in result.object_prediction_list:
                tmp_bb = Model.voc_to_yolo(*detection.bbox.to_xyxy(), image_height, image_width)
                f.write(f"{detection.category.id} {" ".join([str(i) for i in tmp_bb])}\n")

    @staticmethod
    def plot_preds(result, image_path, save_path):
        
        #export_visuals


import os


img_dir = "/home/Документы/lct_2024/small_lct/test/images"
img_paths = [i for i in os.listdir(img_dir)]

model=Model()
for img in img_paths:
    if isinstance(img, str):
        obj = f"{img_dir}/{img}"
    else:
        obj = np.frombuffer(image_bytes, np.uint8)
    r = model(obj)
    model.save_preds(r, img)
    break

""""# Function to process and visualize incoming images on-the-fly
def process_and_visualize(image):
    result = get_prediction(image, detection_model)
    result.export_visuals(export_dir="path/to/save/visuals/")
    # Display or further process the visualized results as needed

# Here, `image` could be frames from a video stream or any real-time image source

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform sliced inference on each frame
    result = get_sliced_prediction(
        image = frame,
        detection_model = model,
        slice_height = 256,
        slice_width = 256,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )
    
    # Here you can add code to visualize/save the detection results for each frame
    # ...

cap.release()"""