import os

BASE_LABEL_DIR = "work_dirs/PU/EXP1"


def get_label_path(round_num, img_id):
    return os.path.join(BASE_LABEL_DIR, f"cycle{round_num})/annfile/queried/{img_id}.txt")
