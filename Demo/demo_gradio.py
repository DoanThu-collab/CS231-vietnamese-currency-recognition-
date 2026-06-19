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

CKPT_PATH = "best_model.pth"

ckpt = torch.load(
    CKPT_PATH,
    map_location="cpu"
)

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
    "front": "Mặt trước",
    "back": "Mặt sau"
}


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

    # Tổng xác suất theo mệnh giá

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

    # mặt trước / sau

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

    # Top 5

    top5_vals, top5_idx = probs.topk(5)

    result = f"""
🏆 Mệnh giá: {int(best_denom):,} VND

📄 {side_vn}

🎯 Độ tin cậy: {conf_denom:.2%}
"""

    top5 = {}

    for score, idx in zip(
        top5_vals.tolist(),
        top5_idx.tolist()
    ):
        top5[CLASS_NAMES[idx]] = score

    return result, top5


# =========================
# Gradio UI
# =========================

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(label="Kết quả"),
        gr.Label(label="Top-5 Prediction")
    ],
    title="Vietnam Currency Recognition",
    description="ResNet18 Fine-Tuning"
)

demo.launch()