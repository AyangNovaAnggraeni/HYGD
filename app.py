import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from torchvision import models
# from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
import streamlit as st

st.write(sys.version)

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
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# UI
# =========================
st.title(" Glaucoma Detection App")
st.write("Upload a fundus image to predict glaucoma.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
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
    cam = GradCAM(model=model, target_layers=[target_layer])

    img_np = np.array(image.resize((224,224))) / 255.0

    grayscale_cam = cam(input_tensor=input_tensor)
    cam_image = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)

    st.image(cam_image, caption="Grad-CAM", use_container_width=True)