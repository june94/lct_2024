SAVE_ROOT = "/home/Документы/lct_2024/lct_2024/streamlit_utils/logs"  # root to save all image and video results
SAVE_VIDEO = f"{SAVE_ROOT}/videos"  # path to save videos
SAVE_IMAGES_TXT = f"{SAVE_ROOT}/testset"  # path to save images
MODEL_PATH = "/home/Документы/lct_2024/lct_2024/weights/best.pt"  # path to model, .onnx or .engine also supported

CATEGORIES = [
    "БПЛА (коптер)",
    "Самолет",
    "Вертолет",
    "Птица",
    "БПЛА (самолет)",
]  # list of categories in russian

ENG_CATEGORIES = {
    "0": "copter",
    "1": "plane",
    "2": "heli",
    "3": "bird",
    "4": "uav",
}  # dict of categories in eng (important if onnx model used)

MAIN_THRESH = 0.25  # base confiedence threshold fo model
THRESH = [
    0.3,
    0.45,
    0.4,
    0.25,
    0.25,
]  # spec confiedence thresholds for each class (for video)

MAP_DANGER = {
    1: "ОПАСНОСТЬ",
    0: "Прочее",
}  # dict to map result danger_tags to namings for final display
DANGER_SET = set([0, 4])  # sets of class indicies considered as danger
OTHER_SET = set([1, 2, 3])  # sets of class indicies considered as normal

IMAGE_SIZE_THRESH = (
    3840 * 2160
)  # min image size threshold to be predicted with sahi (default is 4k)
IMG_BATCH_SIZE = 64  # batch size for image inference
VIDEO_BATCH_SIZE = 30  # batch size for video inference (if frames size > IMAGE_SIZE_THRESH each VIDEO_BATCH_SIZE+1 frame predicted with sahi, others - with stadart yolo inference)
