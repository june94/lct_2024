import os
import cv2
import numpy as np
from typing import Optional, Any, List, Dict

from streamlit.runtime.uploaded_file_manager import UploadedFile

from streamlit_utils.config import *
from streamlit_utils.model import Model


class Video:
    def __init__(
        self, video: UploadedFile, video_root: str = SAVE_VIDEO, get_fps: int = 10
    ):
        """Класс видеоданных.

        Args:
            video (UploadedFile): загруженные данные из Streamlit
            video_root (str, optional): путь до директории сохранения логов и результатов
                                                детекции по видео. Defaults to SAVE_VIDEO.
            get_fps (int, optional): кол-во кадров в секунду, которые нужно обрабатывать. Defaults to 10.
        """
        self.name_no_ext = os.path.splitext(video.name)[0]

        self.root_dir = f"{video_root}/{self.name_no_ext}"
        self.init_video_path = f"{self.root_dir}/{video.name}"
        self.init_images_path = f"{self.root_dir}/init_images"
        self.labels_path = f"{self.root_dir}/labels"
        self.res_path = f"{self.root_dir}/{self.name_no_ext}_result.mp4"
        self.tmp_path = f"{self.root_dir}/tmp_result.avi"  # for streamlit

        os.makedirs(self.init_images_path, exist_ok=True)
        os.makedirs(self.labels_path, exist_ok=True)

        self.get_fps = get_fps

        if not len(os.listdir(self.init_images_path)):
            self.get_frames(video)
        else:
            tmp_img = cv2.imread(
                f"{self.init_images_path}/{os.listdir(self.init_images_path)[0]}", -1
            )
            self.height, self.width = tmp_img.shape[:2]

    def __len__(self):
        return len(os.listdir(self.init_images_path))

    def get_frames(self, video: UploadedFile):
        """Раскадровка видео и сохранение файлов локально.

        Args:
            video (UploadedFile): загруженные данные из Streamlit
        """
        with open(self.init_video_path, mode="wb") as f:
            f.write(video.read())

        vidcap = cv2.VideoCapture(self.init_video_path)
        count = 0
        success = True

        init_fps = vidcap.get(cv2.CAP_PROP_FPS)
        self.get_fps = init_fps if self.get_fps > init_fps else self.get_fps
        n_frame = init_fps // self.get_fps

        while vidcap.isOpened():
            success, image = vidcap.read()
            if success:
                if count % n_frame == 0:
                    cv2.imwrite(f"{self.init_images_path}/{count}.png", image)
                if count == 0:
                    self.height, self.width = image.shape[0], image.shape[1]
                count += 1
            else:
                break

        cv2.destroyAllWindows()
        vidcap.release()

    @staticmethod
    def batch(iterable: Any, n: int = 1):
        """Итерация по батчу.

        Args:
            iterable (Any): итерируемый объект
            n (int, optional): размер батча. Defaults to 1.

        Yields:
            _type_: подмножество объекта
        """
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    @staticmethod
    def get_timestamps(classes: List[int], nframes: int) -> Dict[str, int]:
        """Получение ключевых точек таймлайна, где обнаружены интересующие детекции.
        Трекинг не предусмотрен, поэтому за новую точку принимается та, до которой не
        было обноружено объектов такого типа. Типы обобщены на Опасные и Прочие.

        Args:
            classes (List[int]): лист длинной == числу проанализированных кадров, состоящий из
            0, -1, 1, где 1 - опасность, 0 - прочее, -1 - на кадре нет (интересующих) детекций
            nframes (int): число кадров в секунду для финального видео

        Returns:
            Dict[str, int]: словарь вида тег_индекс: секунда на видео, где тег -
                    опасность или прочее, а индекс соответствует индексу фрагмента
        """
        init_tags = []
        for b in Video.batch(classes, nframes):
            init_tags.append(np.max(b))

        timestamps, tags = [0], [init_tags[0]]
        for ind, ch in enumerate(init_tags[1:]):
            if ch != tags[-1]:
                timestamps.append(ind + 1)
                tags.append(ch)

        timestamps = np.array(timestamps)
        tags = np.array(tags)
        timestamps = timestamps[tags != -1]
        tags = tags[tags != -1]

        res_dict = dict()
        res_dict.update(
            {
                f"{MAP_DANGER[tag]}_{c}": tm
                for c, (tm, tag) in enumerate(zip(timestamps, tags))
            }
        )
        return res_dict

    def make_video(self, select_classes: List[int]) -> Dict[str, int]:
        """Создание видео с отрисовкой ббоксов на сохраненных фото. Данный метод не запускает
        инференс модели, а использует сохраненные в файлах предсказания.

        Args:
            select_classes (List[int]): лист индексов классов, которые требуется отрисовать

        Returns:
            Dict[str, int]: словарь вида тег_индекс: секунда для Streamlit
        """
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        resize_condition = self.width * self.height

        if resize_condition >= IMAGE_SIZE_THRESH:
            res_width = self.width // 2
            res_height = self.height // 2
        else:
            res_width, res_height = self.width, self.height

        video = cv2.VideoWriter(
            self.tmp_path, fourcc, self.get_fps, (res_width, res_height), True
        )
        timeline = []

        sorted_names = os.listdir(self.init_images_path)
        sorted_names.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        for img_name in sorted_names:
            name_no_ext = os.path.splitext(img_name)[0]
            img = cv2.imread(f"{self.init_images_path}/{img_name}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bboxes, classes = [], []
            with open(f"{self.labels_path}/{name_no_ext}.txt", "r") as f:
                for line in f.readlines():
                    cl, x, y, w, h = line.strip().split(" ")
                    cl, x, y, w, h = (
                        int(float(cl)),
                        float(x),
                        float(y),
                        float(w),
                        float(h),
                    )
                    bboxes.append([x, y, w, h])
                    classes.append(cl)

            img = Model.plot_preds(
                img,
                bboxes,
                classes,
                select_classes=select_classes,
                # save_path="/home/Документы/lct_2024/lct_2024/stremlit_utils/logs/debug", #debug
                # file_name=img_name, # debug
                #
            )
            # huge images downscaled for fin visualization to save memory
            if resize_condition:
                img = cv2.resize(img, (res_width, res_height))

            video.write(img)

            if len(set(classes) & DANGER_SET & set(select_classes)):
                timeline.append(1)
            elif len(set(classes) & OTHER_SET & set(select_classes)):
                timeline.append(0)
            else:
                timeline.append(-1)

        video.release()
        cv2.destroyAllWindows()

        # for streamlit (because of streamlit bug with reading opencv created videos)
        os.system(f'ffmpeg -y -i "{self.tmp_path}" -vcodec libx264 "{self.res_path}"')
        os.system(f'rm "{self.tmp_path}"')

        time_dict = Video.get_timestamps(timeline, int(self.get_fps))

        return time_dict
