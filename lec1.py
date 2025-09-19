import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np

# ✅ Load CLIP model with progress message
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# ✅ Class labels
CLASS_LABELS = [
    "Plastic waste - complaint sent to Waste Dept.",
    "Paper waste - complaint sent to Waste Dept.",
    "Metal waste - complaint sent to Waste Dept.",
    "Organic waste - complaint sent to Waste Dept.",
    "Overflowing garbage - complaint sent to Sanitation Dept.",
    "Public washroom issue - complaint sent to Health Dept.",
    "Pothole - complaint sent to Road Dept.",
    "Damaged road - complaint sent to Public Works Dept.",
    "Broken streetlight - complaint sent to Electrical Dept.",
    "Exposed wires - complaint sent to Electrical Safety Dept.",
    "Leaking pipe - complaint sent to Water Dept.",
    "Dirty water - complaint sent to Water Quality Dept.",
    "Vandalism - complaint sent to Security Dept.",
    "Damaged park - complaint sent to Parks Dept."
]

# ✅ Main App
def main():
    st.set_page_config(page_title="ViT-B/32 Classifier", layout="centered")
    st.title("🧠 ViT-B/32-Based Issue Classifier")
    st.markdown("Upload a photo of an issue (like potholes, waste, etc.) and the model will classify it.")

    # ✅ Model loading with spinner
    with st.spinner("Loading CLIP model..."):
        model, preprocess, device = load_model()

    # ✅ Image uploader
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Classifying..."):
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_inputs = clip.tokenize(CLASS_LABELS).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).squeeze(0)
                probs = similarity.softmax(dim=0).cpu().numpy()

            # ✅ Prediction result
            best_idx = np.argmax(probs)
            predicted_label = CLASS_LABELS[best_idx]
            confidence = probs[best_idx]

            st.subheader("✅ Prediction Result")
            st.success(f"Predicted Class: **{predicted_label}**")
            st.info(f"Confidence: **{confidence:.4f}**")

            # Optional: show all class scores
            with st.expander("📊 All class probabilities"):
                for label, prob in zip(CLASS_LABELS, probs):
                    st.write(f"{label}: {prob:.4f}")
    else:
        st.warning("👆 Upload an image to get started.")

# ✅ Run the app
if __name__ == "__main__":
    main()
