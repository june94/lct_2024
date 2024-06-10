import os
import cv2
import numpy as np
from .config import *
from .model import Model


class Video:
    def __init__(self, video, video_root=SAVE_VIDEO, n_frame=3):
        self.name_no_ext = os.path.splitext(video.name)[0]

        self.root_dir = f"{video_root}/{self.name_no_ext}"
        self.init_video_path = f"{self.root_dir}/{video.name}"
        self.init_images_path = f"{self.root_dir}/init_images"
        self.labels_path = f"{self.root_dir}/labels"
        self.res_path = f"{self.root_dir}/{self.name_no_ext}_result.avi"
        
        os.makedirs(self.init_images_path, exists_ok=True)
        os.makedirs(self.labels_path, exists_ok=True)
        os.makedirs(self.vis_path, exists_ok=True)

        self.n_frame = n_frame
        self.height, self.width = None, None

        if not len(os.listdir(self.init_images_path)):
            self.height, self.width = self.get_frames(video)

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
                        cv2.imwrite(os.path.join(self.init_images_path, '%d.png') % count, image)
                    count += 1
                else:
                    break
            
            cv2.destroyAllWindows()
            vidcap.release()
            return image.shape[:2]
    
    @staticmethod
    def get_timestamps(classes, nframes):
        timestamps, tags = [0], [classes[0]]
        for ind, ch in enumerate(classes[1:]):
            if ch != tags[-1]:
                timestamps.append(ind)
                tags.append(ch)
        timestamps = np.array(timestamps)/nframes 
        tags = np.array([MAP_DANGER.get(t, None) for t in tags])    
        timestamps = timestamps[tags!=None]
        tags = tags[tags!=None]
        return {f"{MAP_DANGER[tag]}_{c+1}":tm for c, tag, tm in enumerate(zip(timestamps, tags))}
    

    def make_video(self, select_classes, nframes=30):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timeline = []

        sorted_names = sorted(os.listdir(self.init_images_path))
        for c, img_name in enumerate(sorted_names):
            name_no_ext = os.path.splitext(img_name)[0]
            img = cv2.imread(f"{self.init_images_path}/{img_name}", -1) 
            h, w = img.shape[:2]
            
            bboxes, classes = [], []
            with open(f"{self.labels_path}/{name_no_ext}.txt", "r") as f:
                for line in f.readlines():
                    cl, x, y, w, h = line.strip().split(" ")
                    cl, x, y, w, h = int(float(cl)), float(x), float(y), float(w), float(h)
                    bboxes.append([x, y, w, h])
                    classes.append(cl)

            img = Model.plot_preds(img, bboxes, classes, select_classes = select_classes)
            
            if len(set(classes) & DANGER_SET ):
                timeline.append(0)
            elif len(set(classes) & OTHER_SET):
                timeline.append(1)
            else:
                timeline.append(-1)

            if c == 0:
                video = cv2.VideoWriter(self.res_path, fourcc, nframes, (w, h))
            
            video.write(img)

        time_dict = Video.get_timestamps(timeline, float(nframes))
        
        return time_dict

    def approximate_results():
        pass
        
