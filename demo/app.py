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
    # Portable path inside repo (works on Streamlit Cloud)
    return load_ensemble(
        checkpoint_dir="demo/checkpoints/ensemble",
        device=device,
    )

# Load models with user-friendly error handling (prevents redacted traceback pain)
with st.spinner("Loading ensemble checkpoints (first run may download files)..."):
    try:
        models = load_models()
    except Exception as e:
        st.error("❌ Failed to load model checkpoints.")
        st.write(
            "This usually happens on Streamlit Cloud when checkpoints are missing or the "
            "**GitHub Release download URL** is not set correctly."
        )
        st.write("### Fix checklist")
        st.write("- Ensure `demo/utils.py` has a valid `ENSEMBLE_ZIP_URL` (GitHub Releases direct download link).")
        st.write("- Ensure the app loads from `demo/checkpoints/ensemble` (portable path).")
        st.write("- After updating, **commit + push**, then **Reboot app** on Streamlit Cloud.")
        st.write("### Error details (for debugging)")
        st.code(str(e))
        st.stop()

st.success("✅ Model loaded. Upload an image to test uncertainty + abstention.")

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
    THRESHOLD = 1.2  # tune later; same as config logic
    st.subheader("Decision")

    if entropy > THRESHOLD:
        st.error("❌ I DON'T KNOW")
        st.caption("Reason: High predictive uncertainty (predictive entropy above threshold).")
    else:
        st.success("✅ ANSWER")
        st.caption("Reason: Predictive entropy is low, model is confident enough.")
else:
    st.info("Upload an image to see prediction + uncertainty + abstention behavior.")
