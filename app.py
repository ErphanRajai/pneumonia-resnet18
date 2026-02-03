# force rebuild
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import matplotlib.cm as cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumonia_final_resnet18.pth")
EXAMPLE_DIR = os.path.join(BASE_DIR, "data") 
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        st.error(f"Model file not found at {MODEL_PATH}")
        
    model.to(device)
    model.eval()
    return model, device

def get_gradcam_overlay(model, input_tensor, original_img, device):
    """Generates Grad-CAM heatmap and overlays it using PIL/Matplotlib"""
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output)
        
    handle = model.layer4.register_forward_hook(hook_fn)

    model.zero_grad()
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)
    
    score = outputs[:, pred.item()]
    

    gradients = torch.autograd.grad(score, feature_maps[0], retain_graph=False)[0]
    
    activations = feature_maps[0].detach()
    handle.remove()
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0) # ReLU
    heatmap /= (np.max(heatmap) + 1e-8) # Normalize

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(original_img.size, resample=Image.BICUBIC)
    
    colormap = cm.get_cmap('jet')
    heatmap_rgba = colormap(np.array(heatmap_img)) 
    heatmap_rgb = np.uint8(heatmap_rgba[:, :, :3] * 255)
    
    original_rgb = np.array(original_img.convert('RGB'))
    overlay = Image.blend(Image.fromarray(original_rgb), Image.fromarray(heatmap_rgb), alpha=0.4)
    
    return overlay, CLASS_NAMES[pred.item()], conf.item()

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁")
st.title("🫁 Pneumonia Detection via ResNet-18")
st.markdown("""
Upload a Chest X-ray or select an **Example Image**. This AI identifies potential pneumonia 
and uses **Grad-CAM** to highlight the regions of interest.
""")

model, device = load_model()

st.write("### 🖼️ Quick Test: Click an Example")
if os.path.exists(EXAMPLE_DIR):
    example_files = [f for f in os.listdir(EXAMPLE_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))]
    if example_files:
        cols = st.columns(len(example_files))
        selected_img = None
        for i, file in enumerate(example_files):
            img_path = os.path.join(EXAMPLE_DIR, file)
            img = Image.open(img_path)
            cols[i].image(img, use_container_width=True)
            if cols[i].button(f"Select #{i+1}", key=f"btn_{i}"):
                selected_img = img
    else:
        st.info("No images found in the data folder.")
else:
    st.warning("Example data folder not found.")

uploaded_file = st.file_uploader("📂 Or Upload a Chest X-ray...", type=["jpg", "jpeg", "png"])

final_img = None
if uploaded_file:
    final_img = Image.open(uploaded_file).convert('RGB')
elif selected_img:
    final_img = selected_img.convert('RGB')

if final_img:
    st.divider()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(final_img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True # Required for Grad-CAM gradients

    with st.spinner("Analyzing Pathology & Generating Focus Heatmap..."):
        overlay_img, label, confidence = get_gradcam_overlay(model, input_tensor, final_img, device)

    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Original X-ray")
        st.image(final_img, use_container_width=True)
        
        if label == 'PNEUMONIA':
            st.error(f"**Result: {label}**")
        else:
            st.success(f"**Result: {label}**")
        st.metric("Confidence", f"{confidence*100:.2f}%")

    with col_right:
        st.subheader("AI Focus Area")
        st.image(overlay_img, caption="Grad-CAM Heatmap (Explainable AI)", use_container_width=True)
        st.info("Red/Yellow areas show where the model detected opacity or signs of infection.")

st.divider()
st.caption("Disclaimer: This tool is for educational purposes and portfolio demonstration only.")