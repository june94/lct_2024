import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

image_paths = ["D:\\Docs\\lct_2024\\data\\data_init\\images",
                "D:\\Docs\\lct_2024\\data\\robo_1\\test\\images",
               "D:\\Docs\\lct_2024\\data\\robo_1\\train\\images",
               "D:\\Docs\\lct_2024\\data\\robo_1\\valid\\images",
               "D:\\Docs\\lct_2024\\data\\birds_kaggle\\32-birds-01\\720p",
               "D:\\Docs\\lct_2024\\data\\robo_2\\test\\images",
               "D:\\Docs\\lct_2024\\data\\robo_2\\train\\images",
               "D:\\Docs\\lct_2024\\data\\robo_2\\valid\\images",
               "D:\\Docs\\lct_2024\\data\\robo_3\\test\\images",
               "D:\\Docs\\lct_2024\\data\\robo_3\\train\\images",
               "D:\\Docs\\lct_2024\\data\\robo_3\\valid\\images",
               "D:\\Docs\\lct_2024\\data\\kaggle_4k_drones\\Database1",
               "D:\\Docs\\lct_2024\\data\\droneset\\Drone_TestSet",
               "D:\\Docs\\lct_2024\\data\\droneset\\Drone_TrainSet",
               "D:\\Docs\\lct_2024\\data\\mil_drones\\train\\images",
               "D:\\Docs\\lct_2024\\data\\mil_drones\\valid\\images",
               "D:\\Docs\\lct_2024\\data\\mil_drones\\test\\images",
               ]

prefix_path = "D:\\Docs\\lct_2024\\data\\labels\\yolo\\"
label_paths = ["D:\\Docs\\lct_2024\\data\\data_init\\labels",
               "D:\\Docs\\lct_2024\\data\\labels\\yolo\\labels\\yolo\\robo_1\\test",
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

image_list_path = "D:\\Docs\\lct_2024\\data\\labels\\valid_images.txt"

col_names = ["init_dataset", "image_path", "label_path", "image_name_no_ext", "prefix", "label"]
df = []

with open(image_list_path, "r") as f:
    valid_images = [l.strip() for l in f.readlines()]
    for image_path in tqdm(valid_images, total=len(valid_images)):
        image_dir, image_name = image_path.rsplit("\\", 1)
        init_dataset = 1 if image_dir == "D:\\Docs\\lct_2024\\data\\data_init\\images" else 0
        ind = image_paths.index(image_dir)
        label_path = label_paths[ind]
        image_no_ext = image_name.rsplit(".", 1)[0]
        full_label_path = os.path.join(label_path, image_no_ext+".txt")
        prefix = "_".join(label_path.lstrip(prefix_path).split("\\"))
        try:
            with open(full_label_path, "r") as f:
                for l in f.readlines():
                    cl = l.strip().split(" ", 1)[0]
                    df.append([init_dataset, image_path, full_label_path, image_no_ext, prefix, cl])
        except FileNotFoundError:
            continue

df = pd.DataFrame(df, columns=col_names)
df["label"] = df["label"].astype(float).astype(int)

sub_df = df[df["init_dataset"] == 1].drop_duplicates(subset="image_path")
train, test, _, _ = train_test_split(sub_df["image_path"], 
                                         sub_df["label"], 
                                         stratify=sub_df["label"], 
                                         test_size=0.5)

# additional part for small data batch
"""_, small_batch, _, _ = train_test_split(sub_df["image_path"], 
                                         sub_df["label"], 
                                         stratify=sub_df["label"], 
                                         test_size=0.05)
small_df = sub_df[sub_df["image_path"].isin(small_batch)]
small_train, small_test, _, _ = train_test_split(small_df["image_path"], 
                                         small_df["label"], 
                                         stratify=small_df["label"], 
                                         test_size=0.25)
######
df["test_small"] = df["image_path"].isin(small_test)
df["train_small"] = df["image_path"].isin(small_train)"""

df["test"] = df["image_path"].isin(test)

df.to_csv("D:\\Docs\\lct_2024\\data\\labels\\ds.csv")

print(df["label"].value_counts())
print(df["test"].value_counts())
print(len(df["image_path"].drop_duplicates()))
print(df.head(10))