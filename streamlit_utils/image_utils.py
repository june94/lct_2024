import cv2
import os
import io
import tarfile
import shutil
import numpy as np
from typing import Optional, Any, Union, List

from streamlit_utils.config import *
from streamlit.runtime.uploaded_file_manager import UploadedFile

from sahi.slicing import (
    calc_resolution_factor,
    calc_aspect_ratio_orientation,
    calc_ratio_and_slice,
)


class ImageObj:
    def __init__(self, path: str, name: str):
        """Структура метаинформации фото для удобства работы с архивом.

        Args:
            path (str): путь до фото (распакованного архива)
            name (str): имя фото
        """
        self.path = f"{path}/{name}"
        self.name = name
        self.size = os.path.getsize(self.path)


class ImageTar:
    def __init__(
        self, data: UploadedFile, save_path: Optional[str] = f"{SAVE_ROOT}/tmp"
    ):
        """Обертка для работы с tar-архивом.

        Args:
            data (UploadedFile): загруженные данные из Streamlit
            save_path (Optional[str], optional): путь по директории
                   распаковки архива. Defaults to f"{SAVE_ROOT}/tmp".
        """
        self.save_path = save_path
        self.extract_archive(data)

    def __len__(self):
        return len(os.listdir(self.save_path))

    def extract_archive(self, data: UploadedFile):
        """Распаковка архива.

        Args:
            data (UploadedFile): данные из Streamlit
        """
        os.makedirs(self.save_path, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data.getvalue())) as tar:
            tar.extractall(path=self.save_path)

    def get_paths(self) -> List[ImageObj]:
        """Получение метаинформации для всех изображений из архива.

        Returns:
            List[ImageObj]: лист ImageObj
        """
        return [ImageObj(self.save_path, f) for f in os.listdir(self.save_path)]

    def remove_dir(self):
        """Удаление директории, куда был распакован архив."""
        if os.path.isdir(self.save_path):
            shutil.rmtree(self.save_path, ignore_errors=True)


def read_image(img_file: Union[ImageObj, UploadedFile]) -> np.ndarray:
    """Чтение (конвертация) изображения в формат opencv.

    Args:
        img_file (Union[ImageObj, UploadedFile]): входные данные от Streamlit
                    или метаинформация о фото (в формате ImageObj) из архива

    Returns:
        np.ndarray: изображение
    """
    if isinstance(img_file, ImageObj):
        with open(img_file.path, "rb") as image:
            bytes_data = image.read()
    else:
        bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), -1)
    return img


def get_auto_slice_params(height: int, width: int) -> Union[float, int]:
    """Урезанный одноименный метод из SAHI.

    Args:
        height (int): высота изображения
        width (int): ширина изображения

    Returns:
        Union[float, int]: лист с долями наложения скользящего окна и его размерами
    """
    resolution = height * width
    factor = calc_resolution_factor(resolution)
    if 21 <= factor < 24:
        return get_resolution_selector("high", height=height, width=width)
    else:
        return get_resolution_selector("ultra-high", height=height, width=width)


def get_resolution_selector(res: str, height: int, width: int) -> Union[float, int]:
    """Одноименный метод из SAHI с использованием переопределенных методов.

    Args:
        res (str): строковое описание размера изображения ("high", "ultra-high")
        height (int): высота изображения
        width (int): ширина изображения

    Returns:
        Union[float, int]: лист с долями наложения скользящего окна и его размерами
    """
    orientation = calc_aspect_ratio_orientation(width=width, height=height)
    x_overlap, y_overlap, slice_width, slice_height = calc_slice_and_overlap_params(
        resolution=res, height=height, width=width, orientation=orientation
    )

    return x_overlap / slice_width, y_overlap / slice_height, slice_width, slice_height


def calc_slice_and_overlap_params(
    resolution: str, height: int, width: int, orientation: str
) -> List[int]:
    """Одноименный метод из SAHI с новыми коэффициентами для
    уменьшения количества частей (окон) для больших фото.

    Args:
        resolution (str): строковое описание размера изображения ("high", "ultra-high")
        height (int): высота изображения
        width (int): ширина изображения
        orientation (str): строковое описание ориентации изображения
                        (горизонтальное, вертикальное или квадратное)

    Returns:
        List[int]: лист с размерами скользящего окна и частей его наложения
    """

    if resolution == "high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = (
            calc_ratio_and_slice(orientation, slide=1, ratio=0.8)
        )

    elif resolution == "ultra-high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = (
            calc_ratio_and_slice(orientation, slide=2, ratio=0.4)
        )
    else:  # low condition
        split_col = 1
        split_row = 1
        overlap_width_ratio = 1
        overlap_height_ratio = 1

    slice_height = height // split_col
    slice_width = width // split_row

    x_overlap = int(slice_width * overlap_width_ratio)
    y_overlap = int(slice_height * overlap_height_ratio)

    return x_overlap, y_overlap, slice_width, slice_height
