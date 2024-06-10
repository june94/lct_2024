SAVE_ROOT = "/home/Документы/lct_2024/lct_2024/stremlit_utils/logs"
SAVE_VIDEO = f"{SAVE_ROOT}/videos"
SAVE_IMAGES_TXT = f"{SAVE_ROOT}/testset"
MODEL_PATH = "/home/Документы/lct_2024/lct_2024/weights/best.pt"

CATEGORIES = ["БПЛА (коптер)",
              "Самолет",
              "Вертолет",
              "Птица",
              "БПЛА (самолет)"]

MAP_DANGER = {0: "Опасность",
              1: "Прочее"}

DANGER_SET = set(0, 4)
OTHER_SET = set(1, 2, 3)