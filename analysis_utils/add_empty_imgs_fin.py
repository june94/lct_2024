import os
import shutil

from_root = "D:\\Docs\\lct_2024\\data\\other_objs\\images"
save_root = "D:\\Docs\\lct_2024\\data\\fin_lct\\train"
save_labels = "labels"
save_images = "images"

for img_name in os.listdir(from_root):
    full_image_path = os.path.join(save_root, save_images)
    full_label_path = os.path.join(save_root, save_labels)
    image_name_no_ext = img_name.split(".")[0] + "_no_object"
    with open(os.path.join(full_label_path, image_name_no_ext+".txt"), "w") as f:
        f.write("\n")
    shutil.copyfile(os.path.join(from_root, img_name), 
                    os.path.join(full_image_path, image_name_no_ext+".png"))
