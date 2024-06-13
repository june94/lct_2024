import os
import cv2
import hashlib
from tqdm import tqdm

main_path = "D:\\Docs\\lct_2024\\data\\data_init\\images"

check_paths = ["D:\\Docs\\lct_2024\\data\\robo_1\\test\\images",
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

hsh_func = cv2.img_hash.PHash_create()

def file_as_bytes(file):
    with file:
        return file.read()
    
def get_hash_dict(root_path, files):
    res = {}
    for file in tqdm(files, total=len(files)):
        if file.endswith(".txt"):
            continue
        full_path = os.path.join(root_path, file)
        #hsh = hashlib.md5(file_as_bytes(open(full_path, 'rb'))).hexdigest()

        #img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        #try: 
        #    hsh = hashlib.md5(img).hexdigest()
        #except TypeError:
        #    continue
        
        try:
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            hsh = hsh_func.compute(img)
            hsh = int.from_bytes(hsh.tobytes(), byteorder='big', signed=False)
        except cv2.error:
            continue
        
        res[hsh] = full_path
    return res


main_hash = dict()
for path in tqdm(check_paths, total=len(check_paths)):
    files_check = os.listdir(path)
    check_hash = get_hash_dict(path, files_check)
    main_hash.update(check_hash)
    
files_main = os.listdir(main_path)
new_hash = get_hash_dict(main_path, files_main)
main_hash.update(new_hash)

with open("D:\\Docs\\lct_2024\\data\\labels\\valid_images.txt", "w") as f:
    for _, i in main_hash.items():
        f.write(f"{i}\n")
        