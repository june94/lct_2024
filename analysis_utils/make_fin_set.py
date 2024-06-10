import os
import random
import cv2
import shutil
import pandas as pd
import numpy as np
import albumentations as A
from tqdm import tqdm

# get image and bbox
# 1 - resize image and bbox - get bbox sizes - if (any bbox size) less than 3 pixs - get random crops to 640x640 - apply to all data
# 2 - all train init data - ref, + random transform - rain, snow, fog
# 3- don't forget to upd df! + fix name of init image

random.seed(16)

df_path = "D:\\Docs\\lct_2024\\data\\labels\\ds.csv"
save_df = "D:\\Docs\\lct_2024\\data\\labels\\ds_fin.csv"
df = pd.read_csv(df_path, header=0)

save_root = "D:\\Docs\\lct_2024\\data\\fin_lct"
save_labels = "labels"
save_images = "images"

new_df = []
new_columns = ["init_image_path", "init_label_path", "new_name", "label", "test"]


def copy_data(row):
    sub_path = "test" if row["test"] else "train"
    full_image_path = os.path.join(save_root, sub_path, save_images)
    full_label_path = os.path.join(save_root, sub_path, save_labels)
    os.makedirs(full_image_path, exist_ok=True)
    os.makedirs(full_label_path, exist_ok=True)
    name = f"{row["image_name_no_ext"]}_{row["prefix"]}"
    shutil.copyfile(row["image_path"], os.path.join(full_image_path, name+".png"))
    shutil.copyfile(row["label_path"], os.path.join(full_label_path, name+".txt"))
    
    one_image_df = df[df["image_path"]==row["image_path"]]
    for _, lab in one_image_df.iterrows():
        new_df.append([row["image_path"], row["label_path"], name, lab["label"], row["test"]])


def save_data(img, bbox, classes, row, suffix="", test=0):
    sub_path = "test" if test else "train"
    full_image_path = os.path.join(save_root, sub_path, save_images)
    full_label_path = os.path.join(save_root, sub_path, save_labels)
    os.makedirs(full_image_path, exist_ok=True)
    os.makedirs(full_label_path, exist_ok=True)
    name = f"{row["image_name_no_ext"]}_{row["prefix"]}_{suffix}" 
    with open(os.path.join(full_label_path, name+".txt"), "w") as f:
        for c, b in zip(classes, bbox):
            f.write(f"{c} {" ".join([str(i) for i in b])}")
            new_df.append([row["image_path"], row["label_path"], name, c, test])
    cv2.imwrite(os.path.join(full_image_path, name+".png"), img)
    

def get_image_w_labels(image_path, label_path):
    bboxes, classes = [], []
    with open(label_path, "r") as f:
        for line in f.readlines():
            cl, x, y, w, h = line.strip().split(" ")
            cl, x, y, w, h = int(float(cl)), float(x), float(y), float(w), float(h)
            if np.min([x, y, w, h]) <= 0. or np.max([x, y, w, h]) > 1.:
                continue
            classes.append(cl)
            bboxes.append([x, y, w, h])
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image, bboxes, classes


#img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
img_size = 640
resize_trans = A.Compose(
    [A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR, p=1.)],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
crop_trans = [A.Compose(
    [A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR, p=1.)],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),]
    #  A.Compose(
    #[A.RandomSizedBBoxSafeCrop(height=1.5*img_size, width=1.5*img_size, interpolation=cv2.INTER_LINEAR, p=1.)],
    #bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))]
weather_trans = A.Compose([A.OneOf([
    A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, num_flare_circles_lower=1, num_flare_circles_upper=6, src_radius=250, p=1),
    A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.08, p=1)
    ], p=1.)])

thresh_size = 1280
subdf = df.drop_duplicates(subset=["image_path"])
for _, row in tqdm(subdf.iterrows(), total=len(subdf)):
    try:
        image, bboxes, class_labels = get_image_w_labels(row["image_path"], row["label_path"])
        if not len(bboxes):
            continue
    except cv2.error as e:
        print(e, row["image_path"])
        continue
    try:
        transformed = resize_trans(image=image, bboxes=bboxes, class_labels=class_labels)
    except ValueError as e:
        print(e, row["image_path"], bboxes, class_labels)
        continue
    transformed_bboxes = transformed['bboxes']
    for box in transformed_bboxes:
        size = box[2]*box[3]*image.shape[0]*image.shape[1]
        if size < 6.: # init 3
            for scale in range(1): #2
                transformed = crop_trans[scale](image=image, bboxes=bboxes, class_labels=class_labels)
                image = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
                save_data(image, bboxes, class_labels, row, suffix=f"crop_{scale}", test=row["test"])
            break
        elif box == transformed_bboxes[-1]:
            min_side = np.min([image.shape[0], image.shape[1]])//thresh_size
            if min_side > 0:
                tmp_resize_trans = A.Compose(
                                [A.Resize(height=image.shape[0]//(min_side+1), width=image.shape[1]//(min_side+1), 
                                          interpolation=cv2.INTER_LINEAR, p=1.)],
                                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                transformed = tmp_resize_trans(image=image, bboxes=bboxes, class_labels=class_labels)
                image = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
                save_data(image, bboxes, class_labels, row, suffix="downscaled", test=row["test"])
            else:
                copy_data(row)
    if row["init_dataset"]:
        if np.min([image.shape[0], image.shape[1]]) > thresh_size//2:
            continue
        transformed = weather_trans(image=image)
        image = transformed['image']
        save_data(image, bboxes, class_labels, row, suffix="weather", test=0)        


new_df = pd.DataFrame(new_df, columns=new_columns)
print(new_df["label"].value_counts())
print(new_df["test"].value_counts())
print(len(new_df["init_image_path"].drop_duplicates()))
print(new_df.head(10))
new_df.to_csv(save_df)