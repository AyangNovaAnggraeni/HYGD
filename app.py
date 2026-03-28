import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torchvision import models
import sys

st.write(sys.version)

# =========================
# Grad-CAM (Manual)
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)

        loss = output[:, class_idx]
        loss.backward()

        gradients = self.gradients[0].detach().numpy()
        activations = self.activations[0].detach().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam


def overlay_cam(image, cam):
    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam).resize(image.size)

    heatmap = np.array(cam) / 255.0
    image_np = np.array(image) / 255.0

    overlay = image_np.copy()
    overlay[:, :, 0] += heatmap * 0.4  # red highlight

    overlay = np.clip(overlay, 0, 1)

    return overlay


# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()


# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =========================
# UI
# =========================
st.title("👁️ Glaucoma Detection App")
st.write("Upload a fundus image to predict glaucoma.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad = True

    # Prediction
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]

    prob_glaucoma = probs[1].item()

    # Threshold
    threshold = 0.669

    if prob_glaucoma > threshold:
        prediction = "Glaucoma (GON+)"
    else:
        prediction = "Normal (GON-)"

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Probability: {prob_glaucoma:.4f}")

    # =========================
    # Grad-CAM
    # =========================
    target_layer = model.layer4[-1]
    cam_generator = GradCAM(model, target_layer)

    pred_class = torch.argmax(outputs, dim=1).item()
    cam = cam_generator.generate(input_tensor, pred_class)

    overlay_img = overlay_cam(image.resize((224, 224)), cam)

    st.image(overlay_img, caption="Grad-CAM", use_container_width=True)