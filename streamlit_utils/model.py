import torch
import os
import cv2
import numpy as np

from typing import Optional, Any, List, Union

from ultralytics import YOLO
from ultralytics.engine.results import Results
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import Colors
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.prediction import PredictionResult

from streamlit_utils.config import *


class Model:
    def __init__(self, model_path: str = MODEL_PATH):
        """Инициализация модели детекции.

        Args:
            model_path (str, optional): Путь до модели. Defaults to MODEL_PATH.
        """
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CustomYolov8DetectionModel(
            model_path=model_path,
            confidence_threshold=MAIN_THRESH,
            device=self.device,
            category_mapping=ENG_CATEGORIES,
        )
        self.warmup()

    def warmup(self):
        """Warmup модели для ускорения первых итераций инференса."""
        test_img = np.random.rand(640, 640, 3)
        for _ in range(100):
            self.model.model(test_img)

    def __call__(
        self,
        data: Any,
        slice_infer: Optional[bool] = True,
        slice_height: Optional[int] = 1024,
        slice_width: Optional[int] = 1024,
        y_ratio: Optional[float] = 0.1,
        x_ratio: Optional[float] = 0.1,
    ):
        """Инференс модели с пре- и постпроцессингом. Стандартный и с использованием скользящего окна.

        Args:
            data (Any): входное(ые) из-ние(я)
            slice_infer (Optional[bool], optional): флаг использования скользящего инференса. Defaults to True.
            slice_height (Optional[int], optional): высота скользящего окна. Defaults to 1024.
            slice_width (Optional[int], optional): ширина скользящего окна. Defaults to 1024.
            y_ratio (Optional[float], optional): Доля наложения скользящего окна (на другое) по высоте. Defaults to 0.1.
            x_ratio (Optional[float], optional): Доля наложения скользящего окна (на другое) по ширине. Defaults to 0.1.

        Returns:
            Union[PredictionResult, Results]: результат детекции в формате библиотеки ultralytics или sahi
        """
        try:
            if slice_infer:
                result = get_sliced_prediction(
                    image=data,
                    detection_model=self.model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=y_ratio,
                    overlap_width_ratio=x_ratio,
                    auto_slice_resolution=False,
                    perform_standard_pred=True,
                    postprocess_type="NMS",
                    postprocess_class_agnostic=True,
                    postprocess_match_metric="IOS",
                )
            else:  # classic YOLO inference
                result = self.model.model.predict(
                    data,
                    iou=0.6,
                    max_det=50,
                    agnostic_nms=True,
                    conf=MAIN_THRESH,
                    device=self.device,
                )
        except TypeError as e:
            print(e)
            return None
        return result

    @staticmethod
    def voc_to_yolo(
        x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int
    ) -> List[float]:
        """Конвертация ббокса из формата Pascal VOC в Yolo.

        Args:
            x1 (float): xmin bbox
            y1 (float): ymin bbox
            x2 (float): xmax bbox
            y2 (float): ymax bbox
            image_w (int): ширина изображения
            image_h (int): высота изображения

        Returns:
            List[float]: нормализованные координаты в виде x_center, y_center, w, h
        """
        box_x_center = (x1 + x2) / 2.0 - 1
        box_y_center = (y1 + y2) / 2.0 - 1
        box_w = x2 - x1
        box_h = y2 - y1

        box_x = box_x_center * 1.0 / image_w
        box_w = box_w * 1.0 / image_w
        box_y = box_y_center * 1.0 / image_h
        box_h = box_h * 1.0 / image_h
        return [box_x, box_y, box_w, box_h]

    @staticmethod
    def yolo_to_voc(
        x: float, y: float, w: float, h: float, image_w: int, image_h: int
    ) -> List[float]:
        """Конвертация ббокса из формата Yolo в Pascal VOC.

        Args:
            x (float): x_center normalized
            y (float): y_center normalized
            w (float): width normalized
            h (float): height normalized
            image_w (int): ширина изображения
            image_h (int): высота изображения

        Returns:
            List[float]: координаты в виде xmin, ymin, xmax, ymax
        """
        xmin = int(image_w * max(float(x) - float(w) / 2, 0))
        xmax = int(image_w * min(float(x) + float(w) / 2, 1))
        ymin = int(image_h * max(float(y) - float(h) / 2, 0))
        ymax = int(image_h * min(float(y) + float(h) / 2, 1))
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def save_preds(
        result: Any,  # Union[PredictionResult, Results],
        image_height: int,
        image_width: int,
        image_name: str,
        save_path: str,
        save_thresh: Optional[List] = None,
    ):
        """Получение результатов детекций и сохранение в файлы в YOLO формате.

        Args:
            result (Union[PredictionResult, Results]): результат детекции в формате библиотеки ultralytics или sahi
            image_height (int): высото изображения
            image_width (int):ширина изображения
            image_name (str): имя фото (или номер кадра, если сохраняется детекция для видео)
            save_path (str): путь до директории сохранения
            save_thresh (Optional[List], optional): лист длинной == кол-во классов с порогами уверенности
                                                    детекции для каждого из классов, детекции ниже этого
                                                    порога в документ не сохраняются. Defaults to None.
        """

        os.makedirs(save_path, exist_ok=True)
        file_name = os.path.splitext(image_name)[0] + ".txt"
        with open(os.path.join(save_path, file_name), "w") as f:
            if type(result) == PredictionResult:
                for detection in result.object_prediction_list:
                    if (
                        save_thresh is not None
                        and not detection.score.is_greater_than_threshold(
                            save_thresh[detection.category.id]
                        )
                    ):
                        continue
                    tmp_bbox = Model.voc_to_yolo(
                        *detection.bbox.to_xyxy(), image_width, image_height
                    )
                    f.write(
                        f"{detection.category.id} {' '.join([str(i) for i in tmp_bbox])}\n"
                    )
            elif type(result) == Results:
                bboxes = result.boxes.cpu().numpy()
                for bbox in bboxes:
                    if (
                        save_thresh is not None
                        and bbox.conf[0] < save_thresh[int(bbox.cls[0])]
                    ):
                        continue
                    f.write(
                        f"{int(bbox.cls[0])} {' '.join([str(i) for i in bbox.xywhn[0]])}\n"
                    )

    @staticmethod
    def plot_preds(
        image: np.ndarray,
        bboxes: List[List],
        classes: List[str],
        file_name: Optional[str] = None,
        save_path: Optional[str] = None,
        select_classes: Optional[List] = [],
        rect_th: Optional[int] = None,
        text_size: Optional[float] = None,
        text_th: Optional[float] = None,
        color: Optional[tuple] = None,
    ) -> np.ndarray:
        """Отрисовка ббоксов на фото, модифицированный метод sahi.cv.visualize_prediction.

        Args:
            image (np.ndarray): фото для отрисовки
            bboxes (List[List]): лист координат ббоксов
            classes (List[str]): лист классов
            file_name (Optional[str], optional): имя файла, если требуется его сохранить. Defaults to None.
            save_path (Optional[str], optional): путь до директории сохранения, если требуется. Defaults to None.
            select_classes (Optional[List], optional): лист индексов классов, которые требуется отрисовать. Defaults to [].
            rect_th (Optional[int], optional): ширина рабок ббокса. Defaults to None.
            text_size (Optional[float], optional): размер шрифта. Defaults to None.
            text_th (Optional[float], optional): ширина шрифта. Defaults to None.
            color (Optional[tuple], optional): лист цветов для ббоксов. Defaults to None.

        Returns:
            np.ndarray: исходное фото с отрисованными на нем ббоксами
        """
        if color is None:
            colors = Colors()
        else:
            colors = None

        rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
        text_th = text_th or max(rect_th - 1, 1)
        text_size = text_size or rect_th / 3
        for bbox, cl in zip(bboxes, classes):
            if cl not in select_classes and len(select_classes):
                continue
            if colors is not None:
                color = colors(cl)
            bbox = Model.yolo_to_voc(*bbox, image.shape[1], image.shape[0])
            point1, point2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(
                image,
                point1,
                point2,
                color=color,
                thickness=rect_th,
            )

            label = f"{CATEGORIES[cl]}"
            box_width, box_height = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_COMPLEX, fontScale=text_size, thickness=text_th
            )[0]
            outside = point1[1] - box_height - 3 >= 0
            point2 = point1[0] + box_width, (
                point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
            )
            cv2.rectangle(image, point1, point2, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                label,
                (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                cv2.FONT_HERSHEY_COMPLEX,
                text_size,
                (255, 255, 255),
                thickness=text_th,
            )

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            full_save_path = os.path.join(save_path, file_name)  ## check ext!!!
            cv2.imwrite(full_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class CustomYolov8DetectionModel(Yolov8DetectionModel):
    """Обертка Yolov8 модели из SAHI для возможности использования
    onnx и tensorrt инференса с батчом размером > 1."""

    def load_model(self):
        """Загрузка претрена."""
        try:
            model = YOLO(self.model_path)
            if not self.model_path.endswith((".onnx", ".engine")):
                model.to(self.device)
                model.fuse()
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov8 model path: ", e)
