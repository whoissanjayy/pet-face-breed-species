import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from utils.labels import species_from_breed

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/best_model.pth not found. Run train.py first."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names

def predict(image_path, top_k=5):
    model, class_names = load_model()

    weights = ResNet18_Weights.DEFAULT
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    x = tfm(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0]
        values, indices = torch.topk(
            probabilities, k=min(top_k, len(class_names))
        )

    results = []
    for value, index in zip(values.cpu(), indices.cpu()):
        breed = class_names[index.item()]
        results.append({
            "breed": breed,
            "species": species_from_breed(breed),
            "confidence": float(value),
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = predict(args.image, args.top_k)

    print("\nTop predictions:")
    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result['breed']} "
            f"({result['species']}) - "
            f"{result['confidence']*100:.2f}%"
        )
