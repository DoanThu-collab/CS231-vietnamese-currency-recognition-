import os
import json

import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from PIL import Image


# =========================
# Letterbox Resize
# =========================

class LetterboxResize:
    def __init__(self, size=224, fill=128):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size

        scale = self.size / max(w, h)

        nw = int(w * scale)
        nh = int(h * scale)

        img = img.resize((nw, nh), Image.BILINEAR)

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            (self.fill, self.fill, self.fill)
        )

        canvas.paste(
            img,
            (
                (self.size - nw) // 2,
                (self.size - nh) // 2
            )
        )

        return canvas


# =========================
# Build ResNet18
# =========================

def build_model(num_classes):

    model = models.resnet18(
        weights=None
    )

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    return model


# =========================
# Load checkpoint
# =========================

BASE_DIR = os.path.dirname(__file__)

CKPT_PATH = os.path.join(
    BASE_DIR,
    "best_model.pth"
)

JSON_PATH = os.path.join(
    BASE_DIR,
    "mapping.json"
)

ckpt = torch.load(
    CKPT_PATH,
    map_location="cpu"
)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    DESCRIPTIONS = json.load(f)

DEBUG_DEMO = os.getenv("DEBUG_DEMO", "0") == "1"

CLASS_NAMES = ckpt["class_names"]
DENOM_NAMES = ckpt["denom_names"]

MEAN = ckpt["mean"]
STD = ckpt["std"]

NUM_CLASSES = len(CLASS_NAMES)

model = build_model(NUM_CLASSES)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()


# =========================
# Mapping denomination
# =========================

denom_to_indices = {}

for i, c in enumerate(CLASS_NAMES):

    d = c.rsplit("_", 1)[0]

    denom_to_indices.setdefault(
        d,
        []
    ).append(i)


SIDE_LABEL = {
    "truoc": "Mặt trước",
    "sau": "Mặt sau",
    "front": "Mặt trước",
    "back": "Mặt sau"
}


def normalize_denom_key(denom):
    return str(denom).strip().zfill(6)


def get_description_entry(denom):
    candidates = [
        str(denom).strip(),
        normalize_denom_key(denom)
    ]

    for key in candidates:
        if key in DESCRIPTIONS:
            return key, DESCRIPTIONS[key]

    return candidates[0], {}


# =========================
# Transform
# =========================

transform = T.Compose([
    LetterboxResize(224),
    T.ToTensor(),
    T.Normalize(
        mean=MEAN,
        std=STD
    )
])


# =========================
# Predict
# =========================

def predict(image):

    image = Image.fromarray(image)

    x = transform(image)

    x = x.unsqueeze(0)

    with torch.no_grad():

        probs = torch.softmax(
            model(x),
            dim=1
        )[0]

    # =====================
    # Tổng xác suất theo mệnh giá
    # =====================

    denom_probs = {}

    for d in DENOM_NAMES:

        denom_probs[d] = probs[
            denom_to_indices[d]
        ].sum().item()

    best_denom = max(
        denom_probs,
        key=denom_probs.get
    )

    conf_denom = denom_probs[best_denom]

    # =====================
    # Xác định mặt trước/sau
    # =====================

    side_probs = {}

    for idx in denom_to_indices[best_denom]:

        side = CLASS_NAMES[idx].rsplit("_", 1)[1]

        side_probs[side] = probs[idx].item()

    best_side = max(
        side_probs,
        key=side_probs.get
    )

    side_vn = SIDE_LABEL.get(
        best_side,
        best_side
    )

    info_key, info = get_description_entry(best_denom)

    # =====================
    # Lấy thông tin từ JSON
    # =====================

    denomination = info.get(
        "denomination",
        str(int(best_denom))
    )

    currency_type = info.get(
        "type",
        "Không có dữ liệu"
    )

    place_name = info.get(
        "place",
        "Không có dữ liệu"
    )

    description = info.get(
        "description",
        "Không có mô tả."
    )

    if best_side == "truoc":
        side_description = info.get(
            "front_side",
            "Không có dữ liệu"
        )
    else:
        side_description = info.get(
            "back_side",
            "Không có dữ liệu"
        )

    if DEBUG_DEMO:
        print("=" * 50)
        print("best_denom =", repr(best_denom))
        print("info_key =", repr(info_key))
        print("best_side =", repr(best_side))
        print("json_key_exists =", info_key in DESCRIPTIONS)
        print("json_keys =", list(DESCRIPTIONS.keys()))
        print("info =", info)
        print("=" * 50)

    # =====================
    # Kết quả hiển thị
    # =====================

    result = f"""
🏆 Mệnh giá: {int(denomination):,} VND

📄 {side_vn}

🎯 Độ tin cậy: {conf_denom:.2%}

💵 Loại tiền:
{currency_type}

🏛️ Hình ảnh trên {side_vn.lower()}:
{side_description}

📍 Địa danh / Công trình:
{place_name}

📝 Ý nghĩa:
{description}
"""

    # =====================
    # Top-5
    # =====================

    top5_vals, top5_idx = probs.topk(5)

    top5 = {}

    for score, idx in zip(
        top5_vals.tolist(),
        top5_idx.tolist()
    ):
        top5[CLASS_NAMES[idx]] = float(score)

    return result, top5


# =========================
# Gradio UI
# =========================

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(
            label="Kết quả",
            lines=18
        ),
        gr.Label(
            label="Top-5 Prediction"
        )
    ],
    title="🇻🇳 Vietnamese Currency Recognition",
    description="Nhận diện tiền Việt Nam bằng ResNet18 Fine-Tuning"
)

demo.launch()