import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from skimage.measure import label, regionprops, find_contours

# make yolo for all data, nessess classes

# 0 - copter
# 1 - plane
# 2 - heli
# 3 - bird
# 4 - milit

load_paths = ["D:\\Docs\\lct_2024\\data\\robo_1\\test\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_1\\train\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_1\\valid\\labels",
               "D:\\Docs\\lct_2024\\data\\birds_kaggle\\32-birds-01\\mask",
               "D:\\Docs\\lct_2024\\data\\robo_2\\test\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_2\\train\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_2\\valid\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_3\\test\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_3\\train\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_3\\valid\\labels",
               "D:\\Docs\\lct_2024\\data\\kaggle_4k_drones\\Database1",
               "D:\\Docs\\lct_2024\\data\\droneset\\Drone_TestSet_XMLs",
               "D:\\Docs\\lct_2024\\data\\droneset\\Drone_TrainSet_XMLs",
                "D:\\Docs\\lct_2024\\data\\mil_drones\\train\\labels",
               "D:\\Docs\\lct_2024\\data\\mil_drones\\valid\\labels",
               "D:\\Docs\\lct_2024\\data\\mil_drones\\test\\labels",
               ]


save_paths = ["D:\\Docs\\lct_2024\\data\\labels\\yolo\\labels\\yolo\\robo_1\\test",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_1\\train",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_1\\valid",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\birds_kaggle",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_2\\test",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_2\\train",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_2\\valid",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_3\\test",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_3\\train",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_3\\valid",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\kaggle_4k_drones",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\droneset\\Drone_TestSet",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\droneset\\Drone_TrainSet",
                "D:\\Docs\\lct_2024\\data\\labels\\yolo\\train",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\valid",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\test",
               ]

def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

# birds_kaggle need to extract bboxes from masks
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))
    contours = find_contours(mask, 0) #128
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255
    return border

def mask_to_bbox(mask):
    bboxes = []
    h, w = mask.shape[:2]
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bbox = pascal_voc_to_yolo(x1, y1, x2, y2, w, h)
        bboxes.append([3] + bbox) # attention to class index!!
    return bboxes

load_path = "D:\\Docs\\lct_2024\\data\\birds_kaggle\\32-birds-01\\mask"
save_path = "D:\\Docs\\lct_2024\\data\\labels\\yolo\\birds_kaggle"
os.makedirs(save_path, exist_ok=True)

for mask_path in os.listdir(load_path):
    full_mask_path = os.path.join(load_path, mask_path)
    file_name = mask_path.split('.')[0]+'.txt'
    file_name = file_name.replace("mask", "image")
    full_save_path = os.path.join(save_path, file_name)
    mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
    bboxes = mask_to_bbox(mask)
    with open(full_save_path, "w") as f_out:
        for l in bboxes:
            f_out.write(" ".join([str(i) for i in l])+"\n") 


# droneset get from xml
def droneset_upd(load_paths, save_paths):
    for load_path, save_path in zip(load_paths, save_paths):
        os.makedirs(save_path, exist_ok=True)
        for voc in os.listdir(load_path):
            full_load_path = os.path.join(load_path, voc)
            full_save_path = os.path.join(save_path, voc.split('.')[0]+'.txt')
            root = ET.parse(full_load_path).getroot()
            size = root.find('size')
            w, h = size.find('width').text, size.find('height').text
            res = []
            for type_tag in root.findall('object'):
                value = type_tag.find('bndbox')
                tmp = [0] + pascal_voc_to_yolo(int(value.find('xmin').text), 
                                               int(value.find('ymin').text), 
                                               int(value.find('xmax').text),
                                               int(value.find('ymax').text),
                                               int(w), int(h))
                res.append(tmp)
                with open(full_save_path, "w") as f_out:
                    for l in res:
                        f_out.write(" ".join([str(i) for i in l])+"\n")          

load_paths = [
               "D:\\Docs\\lct_2024\\data\\droneset\\Drone_TestSet_XMLs",
               "D:\\Docs\\lct_2024\\data\\droneset\\Drone_TrainSet_XMLs",
               ]
save_paths = [
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\droneset\\Drone_TestSet",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\droneset\\Drone_TrainSet",
               ]
droneset_upd(load_paths, save_paths)

# kaggle_4k - remove empty (later remove very big bboxes?)
load_path = "D:\\Docs\\lct_2024\\data\\kaggle_4k_drones\\Database1"
save_path = "D:\\Docs\\lct_2024\\data\\labels\\yolo\\kaggle_4k_drones"
txts = [f for f in os.listdir(load_path) if f.endswith(".txt")]

os.makedirs(save_path, exist_ok=True)
for txt in txts:
    full_load_path = os.path.join(load_path, txt)
    full_save_path = os.path.join(save_path, txt)
    if os.stat(full_load_path).st_size != 0:
        shutil.copyfile(full_load_path, full_save_path)

# robo_1 robo_2 robo_3 change labels
def robo_upd(load_paths, save_paths, map_names):
    for load_path, save_path in zip(load_paths, save_paths):
        os.makedirs(save_path, exist_ok=True)
        for yolo in os.listdir(load_path):
            full_load_path = os.path.join(load_path, yolo)
            full_save_path = os.path.join(save_path, yolo)
            with open(full_load_path, "r") as f_in:
                init_lines = [l.strip() for l in f_in.readlines()]
                fin_lines = []
                for l in init_lines:
                    try:
                        c, x, y, h, w = l.split(" ")
                    except ValueError:
                        continue
                    c = map_names[int(c)]
                    if c == -1:
                        continue
                    fin_lines.append([str(c), x, y, h, w])
            with open(full_save_path, "w") as f_out:
                for l in fin_lines:
                    f_out.write(" ".join([str(i) for i in l])+"\n")

# robo_1
names = ['airplane', 'bird', 'drone', 'helicopter'] 
map_names = {0:1, 1:3, 2:0, 3:0}

load_paths = ["D:\\Docs\\lct_2024\\data\\robo_1\\test\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_1\\train\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_1\\valid\\labels",
               ]
save_paths = ["D:\\Docs\\lct_2024\\data\\labels\\yolo\\labels\\yolo\\robo_1\\test",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_1\\train",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_1\\valid",
               ]
robo_upd(load_paths, save_paths, map_names)

# robo 2
names = ['Aircraft', 'Bird', 'Drone', 'Helo', 'Jet', 'JetContrail']
map_names = {0:1, 1:3, 2:0, 3:2, 4:1, 5:-1}

load_paths = ["D:\\Docs\\lct_2024\\data\\robo_2\\test\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_2\\train\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_2\\valid\\labels",
               ]
save_paths = [
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_2\\test",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_2\\train",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_2\\valid",
               ]
robo_upd(load_paths, save_paths, map_names)

# robo 3
names = ['Helicopter', 'UAV', 'airplane', 'birds', 'drone']
map_names = {0:2, 1:4, 2:1, 3:3, 4:0}

load_paths = [
               "D:\\Docs\\lct_2024\\data\\robo_3\\test\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_3\\train\\labels",
               "D:\\Docs\\lct_2024\\data\\robo_3\\valid\\labels",
               ]

save_paths = [
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_3\\test",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_3\\train",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\robo_3\\valid",
               ]
robo_upd(load_paths, save_paths, map_names)

#mil_drones
names = ['UAV', 'hedef']
map_names = {0:-1, 1:4}
load_paths = [
                "D:\\Docs\\lct_2024\\data\\mil_drones\\train\\labels",
               "D:\\Docs\\lct_2024\\data\\mil_drones\\valid\\labels",
               "D:\\Docs\\lct_2024\\data\\mil_drones\\test\\labels",
               ]

save_paths = [
                "D:\\Docs\\lct_2024\\data\\labels\\yolo\\train",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\valid",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\test",
               ]
robo_upd(load_paths, save_paths, map_names)
