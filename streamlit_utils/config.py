SAVE_ROOT = "/home/Документы/lct_2024/lct_2024/streamlit_utils/logs"
SAVE_VIDEO = f"{SAVE_ROOT}/videos"
SAVE_IMAGES_TXT = f"{SAVE_ROOT}/testset"
MODEL_PATH = "/home/Документы/lct_2024/lct_2024/weights/best.pt"

CATEGORIES = ["БПЛА (коптер)",
              "Самолет",
              "Вертолет",
              "Птица",
              "БПЛА (самолет)"]

MAP_DANGER = {1: "ОПАСНОСТЬ",
              0: "Прочее",
              2: "Начало"}

MAIN_THRESH = 0.25
THRESH = [0.3, 0.45, 0.4, 0.25, 0.25]

DANGER_SET = set([0, 4])
OTHER_SET = set([1, 2, 3])