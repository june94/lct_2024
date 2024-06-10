import torch
from ultralytics import YOLO

torch.backends.cudnn.enabled = False

exp_name = "yolo_crop" # upd
project = "/home/Документы/lct_2024/logs/"
model_path = "/home/Документы/lct_2024/weights/yolov8n.pt" # yolov8n.pt
config_path = "/home/Документы/lct_2024/lct_2024/train_utils/configs/yolo8p2_n.yaml"
data_path = "/home/Документы/lct_2024/lct_2024/train_utils/configs/data.yaml"
train_config_path = f"/home/Документы/lct_2024/lct_2024/train_utils/configs/{exp_name}.yaml" 

model = YOLO(config_path) 
model = YOLO(model_path)
results = model.train(data=data_path,
                      cfg=train_config_path, 
                      project=project,
                      name=exp_name,)