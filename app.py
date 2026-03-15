import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

from model import build_dual_path_model
from gradcam import make_gradcam_heatmap, overlay_heatmap

# Configuration
st.set_page_config(page_title="Cancer Detection Interpretability", layout="wide", page_icon="🔬")

@st.cache_resource
def load_trained_model():
    model_path = 'models/best_model.h5'
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.warning(f"Failed to load full model structure natively, falling back to weights: {e}")
            model = build_dual_path_model()
            model.load_weights(model_path)
            # Recompile or prepare model here if needed
            return model
    else:
        st.error(f"Cannot find trained model at {model_path}. Please make sure you have trained a model and it's located in the models/ directory.")
        return None

def preprocess_image(image: Image.Image):
    image = image.resize((96, 96))
    img_array = np.array(image)
    if img_array.shape[-1] == 4: # Handle RGBA
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

st.title("🔬 Histopathologic Cancer Detection")
st.markdown("""
This dashboard provides predictions for metastatic cancer in small image patches taken from larger digital pathology scans. 
It leverages our custom **Dual-Path CNN** and uses **Grad-CAM** to explain the model's focus areas in both streams.
""")

model = load_trained_model()

if model is None:
    st.info("Upload a trained model `.h5` file using the sidebar or train a model using `train.py`.")
    st.stop()

# Sidebar for inputs
with st.sidebar:
    st.header("Input Selection")
    
    st.markdown("Upload a pathology patch (96x96 pixels) to analyze.")
    uploaded_file = st.file_uploader("Upload Image", type=["tif", "png", "jpg", "jpeg"])
    
    st.divider()
    
    st.markdown("Or select a sample from the test set:")
    if st.button("Load Random Test Image"):
        test_dir = 'test'
        if os.path.exists(test_dir):
            import glob
            import random
            test_files = glob.glob(os.path.join(test_dir, '*.tif'))
            if test_files:
                sample_path = random.choice(test_files)
                st.session_state['sample_image'] = sample_path
            else:
                st.error("No test files found in the test directory.")
        else:
            st.error("Test directory not found.")
            
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif 'sample_image' in st.session_state:
    image = Image.open(st.session_state['sample_image'])
else:
    image = None

if image is not None:
    # Preprocess
    img_tensor = preprocess_image(image)
    
    # Inference
    prediction = model.predict(img_tensor)[0][0]
    is_cancer = prediction > 0.5
    confidence = prediction if is_cancer else 1 - prediction
    
    # Display Result
    st.header("Analysis Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original 96x96 Image", use_container_width=False, width=300)
        
    with col2:
        st.subheader("Prediction")
        if is_cancer:
            st.error(f"**Malignant (Tumor Detected)**")
            st.markdown("The model detected metastatic cancer cells in the center 32x32 region.")
        else:
            st.success(f"**Benign (No Tumor Detected)**")
            st.markdown("The model did not detect metastatic cancer cells.")
            
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.divider()
    
    # Interpretability
    st.header("🧠 Model Interpretability (Grad-CAM)")
    st.markdown("These heatmaps show which parts of the image the model focused on to make its prediction for both spatial streams.")
    
    try:
        # Context Stream Heatmap
        full_path_hm = make_gradcam_heatmap(img_tensor, model, 'full_path_last_conv')
        
        # Center Stream Heatmap
        center_path_hm = make_gradcam_heatmap(img_tensor, model, 'center_path_last_conv')
        
        # Array for overlay
        original_img_np = np.array(image.resize((96, 96)))
        if original_img_np.shape[-1] == 4:
            original_img_np = original_img_np[..., :3]
            
        overlay_full = overlay_heatmap(full_path_hm, original_img_np)
        
        # The center crop mechanism crops [32:64, 32:64]
        center_img_np = original_img_np[32:64, 32:64]
        overlay_center = overlay_heatmap(center_path_hm, center_img_np)
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Context Stream (Full 96x96 View)")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(overlay_full)
            ax.axis('off')
            st.pyplot(fig)
            st.markdown("*Looks at the overall tissue architecture and broader context.*")
            
        with col4:
            st.subheader("Center Stream (32x32 Focus)")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(overlay_center)
            ax2.axis('off')
            st.pyplot(fig2)
            st.markdown("*Focuses minutely on the cell nuclei in the exact center.*")
            
    except Exception as e:
        st.error(f"An error occurred while generating Grad-CAM heatmaps: {e}")

else:
    st.info("Please upload an image or load a sample from the sidebar to begin analysis.")
