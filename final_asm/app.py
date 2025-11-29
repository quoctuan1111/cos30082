import os, sys
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
from collections import OrderedDict

# Environment & local imports

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))  # allow local imports
from model_definitions import FaceNetSoftmax, FaceNetTriplet  # <- your file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = BASE_DIR / "models"
REGISTRY_DIR = BASE_DIR / "artifacts" / "registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# Default model file names you produced earlier
SOFTMAX_CKPT = MODEL_DIR / "classifier_embed_resnet18_softmax_cpu.pt"
TRIPLET_CKPT = MODEL_DIR / "triplet_embed_resnet18_cpu.pt"
LIVENESS_CKPT = MODEL_DIR / "liveness_detector_zalo.pt"
EMOTION_CKPT = MODEL_DIR / "emotion_detector_fer2013.pt"

st.set_page_config(page_title="Smart Facial System", layout="wide")
st.title("Facial Recognition • Liveness • Emotion")
st.caption("CPU mode • Resilient loaders for your saved checkpoints")

# Helpers: transforms

embed_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

emo_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Robust loaders (handle prefixes, head sizes, sequential vs linear FC)

def _strip_backbone_prefix(state_dict: dict) -> dict:
    """Remove 'backbone.' prefix if present."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k.replace("backbone.", "")
        new_sd[nk] = v
    return new_sd

@st.cache_resource
def load_softmax_model():
    model = FaceNetSoftmax(num_classes=4000, emb_dim=128)
    sd = torch.load(SOFTMAX_CKPT, map_location=DEVICE)
    # allow both pure state_dict and {"state_dict": ...}
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

@st.cache_resource
def load_triplet_model():
    model = FaceNetTriplet(emb_dim=128)
    sd = torch.load(TRIPLET_CKPT, map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def _build_resnet18_fc(out_dim: int, seq: bool):
    """Build resnet18 with either Linear or Sequential head."""
    m = models.resnet18(weights=None)
    if seq:
        m.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, out_dim)
        )
    else:
        m.fc = nn.Linear(512, out_dim)
    return m

@st.cache_resource
def load_liveness_model():
    """
    Try to load as:
      - resnet18 + Linear(512,2)  OR Linear(512,1)
      - handle backbone.* prefix
      - fallback to strict=False
    """
    sd = torch.load(LIVENESS_CKPT, map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # remove 'backbone.' if present
    sd_stripped = _strip_backbone_prefix(sd)

    # decide output dim by inspecting fc.* shapes
    out_dim = 2
    if "fc.weight" in sd_stripped:
        out_dim = sd_stripped["fc.weight"].shape[0]
    elif "fc.3.weight" in sd_stripped:
        out_dim = sd_stripped["fc.3.weight"].shape[0]
    elif "fc.0.weight" in sd_stripped and "fc.3.weight" in sd_stripped:
        out_dim = sd_stripped["fc.3.weight"].shape[0]

    # decide if sequential head was used:
    use_seq = any(k.startswith("fc.0.") for k in sd_stripped.keys()) or any(k.startswith("fc.3.") for k in sd_stripped.keys())

    live = _build_resnet18_fc(out_dim=out_dim, seq=use_seq)
    live.load_state_dict(sd_stripped, strict=False)
    live.eval()
    return live, out_dim

@st.cache_resource
def load_emotion_model():
    """
    Try to load FER-2013 head (7 classes), linear or sequential.
    """
    sd = torch.load(EMOTION_CKPT, map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = _strip_backbone_prefix(sd)

    # decide if seq head
    use_seq = any(k.startswith("fc.0.") for k in sd.keys()) or any(k.startswith("fc.3.") for k in sd.keys())
    emo = _build_resnet18_fc(out_dim=7, seq=use_seq)
    emo.load_state_dict(sd, strict=False)
    emo.eval()
    return emo

# Load all models

softmax_model = load_softmax_model()
triplet_model = load_triplet_model()
liveness_model, LIVENESS_OUT = load_liveness_model()   # 1 or 2
emotion_model = load_emotion_model()

# Inference utils

def extract_embedding(model, img_pil: Image.Image):
    x = embed_tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        e = model(x, return_embedding=True)
    return e.cpu().numpy().squeeze()

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def predict_liveness(frame_bgr: np.ndarray) -> float:
    """
    Returns probability of REAL face in [0,1].
    Supports 1-logit (sigmoid) or 2-logit (softmax) heads.
    """
    x = cv2.resize(frame_bgr, (160, 160))
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        logits = liveness_model(x)
        if LIVENESS_OUT == 1:
            prob_real = torch.sigmoid(logits).item()
        else:  # assume index 1 = real
            probs = torch.softmax(logits, dim=1)
            prob_real = probs[0, 1].item()
    return float(prob_real)

EMO_LABELS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def predict_emotion(img_pil: Image.Image):
    x = emo_tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = emotion_model(x)
        probs = torch.softmax(logits, dim=1)
        idx = int(torch.argmax(probs).item())
        return EMO_LABELS[idx], float(probs[0, idx].item())

# Sidebar controls

st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Register", "Verify"], horizontal=True)
recog_model_name = st.sidebar.selectbox("Recognition Model", ["Softmax", "Triplet"])
SIM_THR = st.sidebar.slider("Verification threshold (cosine)", 0.0, 1.0, 0.60, 0.01)
LIVE_THR = st.sidebar.slider("Liveness threshold (prob real)", 0.0, 1.0, 0.50, 0.01)
input_type = st.sidebar.radio("Input", ["Camera", "Upload"], horizontal=True)

recog_model = softmax_model if recog_model_name == "Softmax" else triplet_model
REG_SUBDIR = REGISTRY_DIR / recog_model_name.lower()
REG_SUBDIR.mkdir(parents=True, exist_ok=True)

# Small UI helpers

def emoji_for_emotion(lbl: str) -> str:
    mapping = {
        "Happy": "😄", "Sad": "😢", "Angry": "😠", "Disgust": "🤢",
        "Fear": "😱", "Surprise": "😲", "Neutral": "😐"
    }
    return mapping.get(lbl, "🙂")

def load_image_from_streamlit(input_type: str):
    if input_type == "Camera":
        snap = st.camera_input("Capture face")
        if snap is None: return None
        return Image.open(snap).convert("RGB")
    else:
        up = st.file_uploader("Upload a face image", type=["png","jpg","jpeg"])
        if up is None: return None
        return Image.open(up).convert("RGB")

# Registration

if mode == "Register":
    st.subheader("Register a new user")
    username = st.text_input("Username (folder-safe name)", "")
    img = load_image_from_streamlit(input_type)

    if st.button("Save Registration", disabled=(img is None or not username.strip())):
        if not username.strip():
            st.error("Please enter a username.")
        else:
            emb = extract_embedding(recog_model, img)
            user_path = REG_SUBDIR / f"{username.strip()}.npy"
            np.save(user_path, emb)
            st.success(f"Registered '{username}' for model: {recog_model_name}")
            st.caption(f"Saved → {user_path}")


# Verification

else:
    st.subheader("Verify • Liveness • Emotion")
    # list registered users
    users = sorted([p.stem for p in REG_SUBDIR.glob("*.npy")])
    if not users:
        st.warning(f"No users registered yet for model '{recog_model_name}'. Switch to Register.")
    else:
        colA, colB = st.columns([2, 1])
        with colB:
            selected_user = st.selectbox("Select registered user", users)

        img = load_image_from_streamlit(input_type)

        if img is not None and selected_user:
            # embeddings
            ref_path = REG_SUBDIR / f"{selected_user}.npy"
            ref_emb = np.load(ref_path)
            now_emb = extract_embedding(recog_model, img)
            sim = cosine_sim(ref_emb, now_emb)

            # liveness (use BGR frame for cv2)
            frame = np.array(img)[:, :, ::-1]  # RGB->BGR
            prob_real = predict_liveness(frame)

            # emotion
            emo_label, emo_prob = predict_emotion(img)

            with colA:
                st.image(img, caption=f"Emotion: {emo_label} {emoji_for_emotion(emo_label)} ({emo_prob:.2f})", use_container_width=True)

            # metrics
            with colB:
                st.metric("Cosine similarity", f"{sim:.3f}")
                st.progress(min(max(sim, 0.0), 1.0), text="Match score")

                st.metric("Liveness (prob real)", f"{prob_real:.3f}")
                st.progress(min(max(prob_real, 0.0), 1.0), text="Liveness confidence")

                st.metric("Emotion", f"{emo_label} ({emo_prob:.2f})")

            # decisions
            st.divider()
            if sim >= SIM_THR and prob_real >= LIVE_THR:
                st.success("Verified • Real face detected")
            elif sim >= SIM_THR and prob_real < LIVE_THR:
                st.warning("Face matched but possible spoof")
            else:
                st.error("Verification failed")


# Footer
st.caption(f"Device: {DEVICE} • Models dir: {MODEL_DIR}")
