from pathlib import Path
import sys

import numpy as np
import pennylane as qml
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


PROJECT_ROOT = (
    Path(sys._MEIPASS)
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[1]
)
MODEL_PATH = PROJECT_ROOT / "models" / "ultrasound_resnet18.pt"
QUANTUM_MODEL_PATH = PROJECT_ROOT / "models" / "quantum_head.pt"
DEFAULT_CLASSES = ["benign", "malignant", "normal"]
QUANTUM_DEVICE = qml.device("default.qubit", wires=4)


@qml.qnode(QUANTUM_DEVICE, interface="torch")
def quantum_layer(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(4), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(index)) for index in range(4)]


class QuantumHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(2, 4, 3))
        self.output = nn.Linear(4, len(DEFAULT_CLASSES))

    def forward(self, inputs):
        quantum_outputs = torch.stack([
            torch.stack(quantum_layer(item, self.weights)) for item in inputs
        ]).float()
        return self.output(quantum_outputs)


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint.get("classes", DEFAULT_CLASSES)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, classes


@st.cache_resource
def load_quantum_model():
    checkpoint = torch.load(QUANTUM_MODEL_PATH, map_location="cpu", weights_only=True)
    model = QuantumHead()
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    return model, checkpoint.get("classes", DEFAULT_CLASSES)


def prepare_image(image):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return pipeline(image.convert("RGB")).unsqueeze(0)


RECOMMENDATIONS = {
    "malignant": ("Suspicious", "Further clinical evaluation recommended."),
    "benign": ("Not suspicious", "Low suspicion; routine follow-up recommended."),
    "normal": ("Not suspicious", "No abnormality detected; continue routine screening."),
}


def compute_gradcam(model, image_tensor, target_class):
    activations = {}
    gradients = {}
    target_layer = model.layer4[-1]

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(image_tensor)
        output[0, target_class].backward()
    finally:
        handle_f.remove()
        handle_b.remove()

    weights = gradients["value"][0].mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * activations["value"][0]).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy()


def overlay_important_region(image, cam):
    cam_resized = np.array(
        Image.fromarray(np.uint8(cam * 255)).resize(image.size, resample=Image.BILINEAR)
    ) / 255.0
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red_layer = np.zeros_like(base)
    red_layer[..., 0] = 1.0
    alpha = (cam_resized ** 1.5)[..., None] * 0.6
    blended = base * (1 - alpha) + red_layer * alpha
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))


st.set_page_config(page_title="Breast Ultrasound ML", page_icon="ML", layout="centered")
st.title("Breast Ultrasound ML Project")
st.caption("Classical and hybrid quantum image classification")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model, classes = load_model()
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    image_tensor = prepare_image(image)
    with torch.inference_mode():
        classical_logits = model(image_tensor)
        probabilities = torch.softmax(classical_logits[0], dim=0)
    top_index = int(probabilities.argmax())
    top_class = classes[top_index].lower()
    label, recommendation = RECOMMENDATIONS.get(top_class, ("Unknown", "Manual review recommended."))

    st.subheader("Prediction Summary")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {probabilities[top_index].item():.0%}")
    st.write(f"**Model recommendation:** {recommendation}")

    cam = compute_gradcam(model, image_tensor, top_index)
    highlighted_image = overlay_important_region(image, cam)
    st.image(highlighted_image, caption="Important image region: highlighted", use_container_width=True)

    st.subheader(f"Classical result: {classes[top_index].title()}")
    st.write(f"Classical probability: {probabilities[top_index].item():.1%}")
    st.bar_chart({name.title(): float(probability) for name, probability in zip(classes, probabilities)})

    if QUANTUM_MODEL_PATH.exists():
        quantum_model, quantum_classes = load_quantum_model()
        with torch.inference_mode():
            quantum_features = torch.nn.functional.pad(classical_logits, (0, 1))
            quantum_probabilities = torch.softmax(quantum_model(torch.tanh(quantum_features)), dim=1)[0]
        quantum_index = int(quantum_probabilities.argmax())
        st.subheader(f"Hybrid quantum result: {quantum_classes[quantum_index].title()}")
        st.write(f"Quantum-head probability: {quantum_probabilities[quantum_index].item():.1%}")
    else:
        st.info("Quantum head is still training. The classical result is available now.")
    st.info("Academic demonstration using a public dataset. Results require expert review.")