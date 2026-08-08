import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from utils.labels import species_from_breed

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Pet Face Breed & Species Classifier",
    page_icon="🐾",
    layout="wide",
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, checkpoint

def preprocess(image):
    weights = ResNet18_Weights.DEFAULT
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std
        ),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

st.title("🐾 Pet Face Breed & Species Classifier")
st.caption(
    "Upload a cat or dog image to classify its breed and species "
    "using Deep Learning (PyTorch ResNet18 Transfer Learning)."
)

with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.write(f"Device: `{DEVICE}`")
    top_k = st.slider("Top Predictions (Top-K)", 1, 5, 5)

    st.divider()
    st.header("📊 Dataset & Model Info")
    st.write("**Dataset:** Oxford-IIIT Pet Dataset")
    st.write("**Categories:** 37 breeds (12 cats, 25 dogs)")
    st.write("**Architecture:** ResNet18 Transfer Learning")
    st.write("**Framework:** PyTorch")

model, class_names, checkpoint = load_model()

if model is None:
    st.error(
        "No trained model found. Please run `python train.py` first."
    )
    st.stop()

uploaded = st.file_uploader(
    "Choose a cat or dog image...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("📤 Uploaded Image")
        st.image(image, use_container_width=True)

    x = preprocess(image)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(
            probabilities,
            k=min(top_k, len(class_names))
        )

    predictions = []
    for value, index in zip(values.cpu(), indices.cpu()):
        breed = class_names[index.item()]
        predictions.append(
            (breed, species_from_breed(breed), float(value))
        )

    top_breed, top_species, top_confidence = predictions[0]

    with right:
        st.subheader("🎯 Classification Results")
        st.metric("Species", f"🐱 {top_species}" if top_species == "Cat" else f"🐶 {top_species}")
        st.subheader("Top Prediction")
        st.success(top_breed)
        st.write(f"**Confidence:** {top_confidence*100:.2f}%")

        st.subheader(f"Top-{len(predictions)} Prediction Probabilities")

        for breed, species, confidence in predictions:
            st.write(
                f"**{breed}** ({species}) — "
                f"{confidence*100:.2f}%"
            )
            st.progress(confidence)

        st.caption(
            "Species is derived from the predicted Oxford-IIIT Pet breed. "
            "Confidence is the softmax probability for the selected class."
        )
else:
    st.info("Upload a cat or dog image to start classification.")
