import os
import shutil
import pandas as pd

df_path = "D:\\Docs\\lct_2024\\data\\labels\\ds.csv"
df = pd.read_csv(df_path, header=0)

## small subset 
train_data = df[df["train_small"]==1]
test_data = df[df["test_small"]==1]

save_root = "D:\\Docs\\lct_2024\\data\\small_lct"
save_labels = "labels"
save_images = "images"
save_train = "train"
save_test = "test"

for sub_path, sub_df in zip([save_train, save_test], 
                            [train_data, test_data]):
    full_image_path = os.path.join(save_root, sub_path, save_images)
    full_label_path = os.path.join(save_root, sub_path, save_labels)
    os.makedirs(full_image_path, exist_ok=True)
    os.makedirs(full_label_path, exist_ok=True)
    for index, row in sub_df.iterrows():
        name = f"{row["image_name_no_ext"]}_{row["prefix"]}"
        shutil.copyfile(row["image_path"], os.path.join(full_image_path, name+".png"))
        shutil.copyfile(row["label_path"], os.path.join(full_label_path, name+".txt"))

