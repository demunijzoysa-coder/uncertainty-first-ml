import sys
from pathlib import Path
# Add project root to Python path (Streamlit fix)
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import streamlit as st
import torch
from PIL import Image

from demo.utils import (

 
    CIFAR10_CLASSES,
    load_ensemble,
    preprocess_image,
    predict,
)

st.set_page_config(page_title="Self-Aware ML Demo", layout="centered")

st.title("🧠 Self-Aware ML: Uncertainty-First Demo")
st.write(
    "This model predicts **only when confident**. "
    "If uncertainty is high, it refuses to answer."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load ensemble once ----
@st.cache_resource
def load_models():
    return load_ensemble(
        checkpoint_dir="experiments/runs/20260104_062257_deep_ensemble",
        device=device,
    )

models = load_models()

# ---- Upload image ----
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", width=256)

    x = preprocess_image(img)
    pred_idx, confidence, entropy = predict(models, x, device)

    st.subheader("Prediction")
    st.write(f"**Class:** {CIFAR10_CLASSES[pred_idx]}")
    st.write(f"**Confidence:** {confidence:.3f}")
    st.write(f"**Uncertainty (entropy):** {entropy:.3f}")

    # ---- Abstention rule ----
    THRESHOLD = 1.2  # same logic as training
    if entropy > THRESHOLD:
        st.error("❌ I DON'T KNOW\n\nReason: High predictive uncertainty")
    else:
        st.success("✅ ANSWER\n\nModel is confident")
