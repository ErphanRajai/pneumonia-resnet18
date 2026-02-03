import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumonia_final_resnet18.pth")
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']



@st.cache_resource
def load_trained_model(model_path, device):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():  # Speed up inference
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return CLASS_NAMES[predicted.item()], confidence.item()

st.title("🫁 Pneumonia Detection System")
st.write("Upload a Chest X-ray image to get an instant diagnosis.")

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
try:
    model = load_trained_model(MODEL_PATH, device)

    # File Uploader Widget - This is the "Upload" feature you wanted
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded X-ray', use_column_width=True)

        # Run prediction
        with st.spinner('Analyzing image...'):
            label, score = predict(image, model, device)

        # Display Result
        if label == 'PNEUMONIA':
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        st.metric("Confidence Score", f"{score * 100:.2f}%")

        st.info("Note: This is an AI-assisted tool and should not replace professional medical advice.")

except Exception as e:
    st.error(f"Error: {e}. Ensure your model is at {MODEL_PATH}")