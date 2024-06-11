import os
import cv2
import numpy as np

import sys
sys.path.append("/home/Документы/lct_2024/lct_2024/stremlit_utils")
from config import *
from model import Model


class Video:
    def __init__(self, video, video_root=SAVE_VIDEO, n_frame=3):
        self.name_no_ext = os.path.splitext(video.name)[0]

        self.root_dir = f"{video_root}/{self.name_no_ext}"
        self.init_video_path = f"{self.root_dir}/{video.name}"
        self.init_images_path = f"{self.root_dir}/init_images"
        self.labels_path = f"{self.root_dir}/labels"
        self.res_path = f"{self.root_dir}/{self.name_no_ext}_result.mp4"
        
        os.makedirs(self.init_images_path, exist_ok=True)
        os.makedirs(self.labels_path, exist_ok=True)

        self.n_frame = n_frame

        if not len(os.listdir(self.init_images_path)):
            self.get_frames(video)
        else:
            tmp_img = cv2.imread(f"{self.init_images_path}/{os.listdir(self.init_images_path)[0]}", -1)
            self.height, self.width = tmp_img.shape[:2]


    def get_frames(self, video):
            with open(self.init_video_path, mode='wb') as f:
                f.write(video.read()) # save video to disk

            vidcap = cv2.VideoCapture(self.init_video_path) # load video from disk
            count = 0
            success = True
            
            while vidcap.isOpened():
                success, image = vidcap.read()
                if success:
                    if count % self.n_frame == 0: 
                        cv2.imwrite(f"{self.init_images_path}/{count}.png", image)
                    if count == 0:
                        self.height, self.width = image.shape[0], image.shape[1]
                    count += 1
                else:
                    break
            
            cv2.destroyAllWindows()
            vidcap.release()
    
    @staticmethod
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    
    @staticmethod
    def get_timestamps(classes, nframes):
        init_tags = []
        for b in Video.batch(classes, nframes):
            init_tags.append(np.max(b))

        timestamps, tags = [0], [2]
        for ind, ch in enumerate(init_tags[1:]):
            if ch != tags[-1]:
                timestamps.append(ind+1)
                tags.append(ch)
        timestamps = np.array(timestamps) 
        tags = np.array(tags)
        timestamps = timestamps[tags!=-1]
        tags = tags[tags!=-1]
        
        res_dict = dict()
        res_dict.update({f"{MAP_DANGER[tag]}_{c}":tm for c, (tm, tag) in enumerate(zip(timestamps, tags))})
        return res_dict
    

    def make_video(self, select_classes, nframes=20.):
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        video = cv2.VideoWriter(self.res_path, fourcc, nframes, (self.width, self.height), True)
        
        timeline = []

        sorted_names = os.listdir(self.init_images_path)
        sorted_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for img_name in sorted_names:
            name_no_ext = os.path.splitext(img_name)[0]
            img = cv2.imread(f"{self.init_images_path}/{img_name}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            
            bboxes, classes = [], []
            with open(f"{self.labels_path}/{name_no_ext}.txt", "r") as f:
                for line in f.readlines():
                    cl, x, y, w, h = line.strip().split(" ")
                    cl, x, y, w, h = int(float(cl)), float(x), float(y), float(w), float(h)
                    bboxes.append([x, y, w, h])
                    classes.append(cl)

            img = Model.plot_preds(img, bboxes, classes, select_classes = select_classes, 
                                   #save_path="/home/Документы/lct_2024/lct_2024/stremlit_utils/logs/debug", #
                                   #file_name=img_name,
                                    )
            video.write(img)
            
            if len(set(classes) & DANGER_SET & set(select_classes)):
                timeline.append(1)
            elif len(set(classes) & OTHER_SET & set(select_classes)):
                timeline.append(0)
            else:
                timeline.append(-1) 

        video.release()
        cv2.destroyAllWindows()

        # for stremlit 
        #os.system('ffmpeg -i {} -vcodec libx264 {}'.format(self.res_path, self.res_path.replace('tmp', '')))
        os.system(f'ffmpeg -y -i "{self.res_path}" -vcodec libx264 "{self.res_path}.mp4"')
        os.system(f'mv -f "{self.res_path}.mp4" "{self.res_path}"')
    
        time_dict = Video.get_timestamps(timeline, int(nframes))
        
        return time_dict

    def approximate_results():
        pass
        