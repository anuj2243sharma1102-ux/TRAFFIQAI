"""Train a small vehicle-type classifier on the crops you sorted by hand.

Usage:
    python tools\\train_vehicle_type.py [crops_folder] [test_from_second]

The sub-folder names are the classes (car, taxi, auto, tempo, bus, truck, motorcycle ...).
File names must contain the video time, like cctv_023.4s_id12.jpg.
Crops from before test_from_second (default 20 s) are used for training and the
later ones only for testing, so the test uses vehicles the model has never seen.
After the test, the model is trained again on ALL crops and saved to
models\\vehicle_type.pt .
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

PROJECT_ROOT = Path(__file__).resolve().parent.parent
crops_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "datasets" / "vehicle_crops"
test_from = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TIME_RE = re.compile(r"_(\d+(?:\.\d+)?)s_")

torch.manual_seed(0)

# ---------------------------------------------------------------- find the crops
if not crops_dir.exists():
    sys.exit(f"Folder not found: {crops_dir}")
classes = sorted(p.name for p in crops_dir.iterdir() if p.is_dir() and any(p.glob("*.jpg")))
if len(classes) < 2:
    sys.exit("Need at least 2 class folders that contain .jpg crops.")

items = []   # (path, class index, video time)
for index, name in enumerate(classes):
    for path in sorted((crops_dir / name).glob("*.jpg")):
        match = TIME_RE.search(path.name)
        items.append((path, index, float(match.group(1)) if match else 0.0))

print("Crops per class (all | train | test):")
for index, name in enumerate(classes):
    total = sum(1 for _, c, _ in items if c == index)
    train = sum(1 for _, c, t in items if c == index and t < test_from)
    print(f"  {name:<11} {total:>4} | {train:>4} | {total - train:>4}")
    if total < 15:
        print(f"    (warning: only {total} crops, this class will be weak)")

# ---------------------------------------------------------------- feature extractor
net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()


def prepare(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def features(images):
    out = []
    for i in range(0, len(images), 32):
        batch = torch.stack([prepare(im) for im in images[i:i + 32]])
        with torch.no_grad():
            out.append(net.avgpool(net.features(batch)).flatten(1))
    return torch.cat(out)


images, labels, times = [], [], []
for path, index, t in items:
    img = cv2.imread(str(path))
    if img is not None:
        images.append(img)
        labels.append(index)
        times.append(t)
labels = torch.tensor(labels)
times = torch.tensor(times)

print("Extracting features (this takes a little while) ...")
feats = features(images)
feats_flipped = features([cv2.flip(im, 1) for im in images])   # mirror images = extra training data

# ---------------------------------------------------------------- training function
n_classes = len(classes)


def fit(X, y, epochs=400):
    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True) + 1e-6
    Xn = (X - mean) / std
    counts = torch.bincount(y, minlength=n_classes).clamp(min=1).float()
    weights = (counts.sum() / counts) / n_classes          # rare classes count more
    layer = torch.nn.Linear(X.shape[1], n_classes)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss_fn(layer(Xn), y).backward()
        optimizer.step()
    return layer, mean, std


def predict(layer, mean, std, X):
    with torch.no_grad():
        return torch.softmax(layer((X - mean) / std), dim=1)


# ---------------------------------------------------------------- 1) honest test
train_mask = times < test_from
test_mask = ~train_mask
if test_mask.sum() == 0 or train_mask.sum() == 0:
    print("Cannot split by time (all crops on one side). Skipping the test.")
else:
    X_train = torch.cat([feats[train_mask], feats_flipped[train_mask]])
    y_train = torch.cat([labels[train_mask], labels[train_mask]])
    layer, mean, std = fit(X_train, y_train)

    probs = predict(layer, mean, std, feats[test_mask])
    pred = probs.argmax(1)
    truth = labels[test_mask]
    print(f"\nTEST on {int(test_mask.sum())} crops from {test_from:g} s onwards "
          f"(vehicles the model has never seen):")
    print(f"  overall accuracy: {(pred == truth).float().mean().item():.0%}")
    for index, name in enumerate(classes):
        total = int((truth == index).sum())
        if total:
            correct = int(((pred == truth) & (truth == index)).sum())
            print(f"  {name:<11} {correct}/{total} correct")
    print("  confusion (row = real class, column = what the model said):")
    print("  " + " " * 12 + "".join(f"{name[:6]:>8}" for name in classes))
    for r, name in enumerate(classes):
        row = [int(((truth == r) & (pred == c)).sum()) for c in range(n_classes)]
        print(f"  {name:<12}" + "".join(f"{v:>8}" for v in row))

# ---------------------------------------------------------------- 2) final model on everything
X_all = torch.cat([feats, feats_flipped])
y_all = torch.cat([labels, labels])
layer, mean, std = fit(X_all, y_all)

out_path = PROJECT_ROOT / "models" / "vehicle_type.pt"
torch.save({"classes": classes, "weight": layer.weight.detach(), "bias": layer.bias.detach(),
            "mean": mean, "std": std}, out_path)
print("\nFinal model (trained on all crops) saved:", out_path)