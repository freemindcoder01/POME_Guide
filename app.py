"""
Oil Palm FFB Grading System - Streamlit Cloud Compatible Version
Uses ONNX Runtime for inference (lightweight, no PyTorch dependency)
To convert your trained model: python export_to_onnx.py
Run locally: streamlit run app.py
"""

import io
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image

# ── Try importing inference backend (ONNX preferred, PyTorch fallback) ──────
INFERENCE_MODE = None

try:
    import onnxruntime as ort
    INFERENCE_MODE = "onnx"
except ImportError:
    pass

if INFERENCE_MODE is None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.models as models
        INFERENCE_MODE = "torch"
    except ImportError:
        pass

if INFERENCE_MODE is None:
    st.error(
        "Neither `onnxruntime` nor `torch` is available. "
        "Please check your requirements.txt."
    )
    st.stop()

# ─────────────────────────────────────────────
# CONSTANTS & MPOB GRADING LOGIC
# ─────────────────────────────────────────────

CLASS_NAMES = ["ripe", "underripe", "unripe", "rotten", "empty"]

CLASS_INFO = {
    "ripe": {
        "label": "Ripe Bunch",
        "color": "#2e7d32",
        "emoji": "🟢",
        "description": (
            "Reddish orange colour. ≥10 fresh sockets. >50% fruits attached. "
            "Should reach mill within 24 hours of harvest."
        ),
        "penalty_table": "No penalty",
        "action": "Accept – pay at Basic Extraction Rate",
    },
    "underripe": {
        "label": "Underripe Bunch",
        "color": "#f57f17",
        "emoji": "🟡",
        "description": (
            "Reddish orange or purplish red. Mesocarp yellowish orange. "
            "<10 fresh sockets. Penalty per Table IV."
        ),
        "penalty_table": "Table IV",
        "action": "Accept with penalty – pay at Graded Extraction Rate",
    },
    "unripe": {
        "label": "Unripe Bunch",
        "color": "#b71c1c",
        "emoji": "🔴",
        "description": (
            "Black or purplish black fruits. Mesocarp yellowish. "
            "No fresh sockets. Heavy penalty per Table III."
        ),
        "penalty_table": "Table III",
        "action": "Accept with heavy penalty – consider rejection",
    },
    "rotten": {
        "label": "Rotten Bunch",
        "color": "#4a148c",
        "emoji": "⛔",
        "description": (
            "Partly or wholly blackish, rotten and mouldy. "
            "Penalty per Table VI."
        ),
        "penalty_table": "Table VI",
        "action": "Accept with penalty – pay at Graded Extraction Rate",
    },
    "empty": {
        "label": "Empty Bunch",
        "color": "#37474f",
        "emoji": "❌",
        "description": (
            ">90% fruitlets detached at mill inspection. "
            "Reject entire load if >20% of consignment."
        ),
        "penalty_table": "Table V",
        "action": "Reject load if consignment >20% empty bunches",
    },
}

# Per-1% penalty deduction from Basic Extraction Rate (MPOB Tables III–VI)
PENALTY_PER_PCT = {
    "ripe":      {"oil": 0.000, "kernel": 0.000},
    "underripe": {"oil": 0.030, "kernel": 0.010},
    "unripe":    {"oil": 0.120, "kernel": 0.040},
    "rotten":    {"oil": 0.120, "kernel": 0.040},
    "empty":     {"oil": 0.100, "kernel": 0.000},
}

BASIC_OER = 20.5   # % — Peninsular, Tenera DxP, age 4–18 yrs (Table I)
BASIC_KER = 5.5    # %

QUALITY_LIMITS = {
    "ripe":      "≥90%",
    "underripe": "≤10%",
    "unripe":    "0%",
    "rotten":    "0%",
    "empty":     "0%",
}

MODEL_PATH_ONNX  = "ffb_model.onnx"
MODEL_PATH_TORCH = "output/ffb_model_best.pth"
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING  (pure numpy — no torchvision needed)
# ─────────────────────────────────────────────

def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Return float32 NCHW array ready for inference."""
    img  = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr  = arr.transpose(2, 0, 1)          # HWC → CHW
    return np.expand_dims(arr, 0)           # (1,3,224,224)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Priority:
    1. ONNX model          — fast, light, works on Streamlit Cloud
    2. PyTorch .pth        — local fallback
    3. Demo / random weights — UI testing only
    """
    if INFERENCE_MODE == "onnx" and Path(MODEL_PATH_ONNX).exists():
        sess = ort.InferenceSession(
            MODEL_PATH_ONNX,
            providers=["CPUExecutionProvider"]
        )
        st.success("✅ ONNX model loaded.")
        return ("onnx", sess)

    if INFERENCE_MODE == "torch":
        model = _build_torch_model()
        if Path(MODEL_PATH_TORCH).exists():
            state = torch.load(MODEL_PATH_TORCH, map_location="cpu")
            model.load_state_dict(state)
            st.success("✅ PyTorch model loaded.")
        else:
            st.warning(
                "⚠️ No trained model found — running in **DEMO mode**. "
                "Predictions are random until you upload a trained model."
            )
        model.eval()
        return ("torch", model)

    st.warning("⚠️ Running in DEMO mode (no model file found).")
    return ("demo", None)


def _build_torch_model():
    model = models.efficientnet_b0(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_feat, len(CLASS_NAMES))
    )
    return model


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(model_tuple, img: Image.Image):
    mode, model = model_tuple
    arr = preprocess_pil(img)

    if mode == "onnx":
        input_name = model.get_inputs()[0].name
        logits = model.run(None, {input_name: arr})[0][0]
        probs  = softmax(logits)

    elif mode == "torch":
        with torch.no_grad():
            logits = model(torch.from_numpy(arr))[0]
        probs = F.softmax(logits, dim=0).numpy()

    else:
        probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)))

    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs


# ─────────────────────────────────────────────
# GRADING CALCULATIONS
# ─────────────────────────────────────────────

def compute_extraction(class_name: str, confidence: float):
    pct     = confidence * 100
    oil_pen = round(PENALTY_PER_PCT[class_name]["oil"]    * pct, 3)
    ker_pen = round(PENALTY_PER_PCT[class_name]["kernel"] * pct, 3)
    return (
        round(max(0.0, BASIC_OER - oil_pen), 2),
        round(max(0.0, BASIC_KER - ker_pen), 2),
        oil_pen, ker_pen,
    )


def quality_status(class_name: str, confidence: float):
    pct = confidence * 100
    if class_name == "ripe":
        return ("✅ GOOD QUALITY",  "#2e7d32") if pct >= 50 else ("✅ ACCEPTABLE", "#388e3c")
    if class_name == "underripe":
        return ("⚠️ SUBSTANDARD",   "#f57f17")
    if class_name == "empty" and pct > 20:
        return ("🚫 REJECT LOAD",   "#b71c1c")
    return ("❌ POOR QUALITY",      "#c62828")


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def build_report(class_name, confidence, all_probs,
                 graded_oer, graded_ker, oil_pen, ker_pen,
                 status, img_name):
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = CLASS_INFO[class_name]
    sep  = "─" * 58
    lines = [
        "=" * 58,
        "  MPOB OIL PALM FFB GRADING REPORT",
        "  MPOB Grading Manual, 2nd Edition (2003)",
        "=" * 58,
        f"Date/Time      : {now}",
        f"Image          : {img_name}",
        f"Inference mode : {INFERENCE_MODE}",
        sep,
        "GRADING RESULT",
        sep,
        f"Classification : {info['label']}",
        f"Confidence     : {confidence*100:.1f}%",
        f"Status         : {status}",
        f"Action         : {info['action']}",
        sep,
        "CLASS PROBABILITIES",
        sep,
    ]
    for n, p in zip(CLASS_NAMES, all_probs):
        bar = "█" * int(p * 28)
        lines.append(f"  {CLASS_INFO[n]['label']:20s}: {p*100:5.1f}%  {bar}")
    lines += [
        sep,
        "EXTRACTION RATE ESTIMATE",
        sep,
        f"Basic OER      : {BASIC_OER:.1f}%  (Peninsular, DxP, 4–18 yrs)",
        f"Basic KER      : {BASIC_KER:.1f}%",
        f"Oil penalty    : -{oil_pen:.3f}%",
        f"Kernel penalty : -{ker_pen:.3f}%",
        f"Graded OER     : {graded_oer:.2f}%",
        f"Graded KER     : {graded_ker:.2f}%",
        sep,
        "QUALITY LIMITS  (MPOB Sec. 5.2.3)",
        sep,
    ]
    for cat, lim in QUALITY_LIMITS.items():
        tick = "✓" if cat == "ripe" else "✗"
        lines.append(f"  {tick} {CLASS_INFO[cat]['label']:20s}: {lim}")
    lines += [
        sep,
        "DISCLAIMER",
        sep,
        "Prototype AI — results must be verified by a qualified",
        "MPOB-licensed grading officer (MPOB Manual, Sec. 3.2).",
        "=" * 58,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="MPOB FFB Grading",
        page_icon="🌴",
        layout="wide",
    )

    st.markdown("""
    <div style='background:linear-gradient(90deg,#1b5e20,#2e7d32);
                padding:1.2rem 1.5rem;border-radius:8px;margin-bottom:1rem;'>
        <h1 style='color:white;margin:0;font-size:1.7rem;'>
            🌴 Oil Palm FFB Grading System
        </h1>
        <p style='color:#a5d6a7;margin:0.3rem 0 0;font-size:0.9rem;'>
            Based on MPOB Oil Palm Fruit Grading Manual · 2nd Edition (2003)
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")
        conf_thresh = st.slider("Min. Confidence", 0.0, 1.0, 0.50, 0.05)
        st.divider()
        st.caption(f"**Inference:** `{INFERENCE_MODE}`")
        st.caption(
            "Place `ffb_model.onnx` (recommended) or "
            "`output/ffb_model_best.pth` in the repo root to use a trained model."
        )
        st.divider()
        st.subheader("Quality Limits")
        for cat, lim in QUALITY_LIMITS.items():
            st.caption(f"**{CLASS_INFO[cat]['label']}**: {lim}")

    tab_grade, tab_std, tab_about = st.tabs(
        ["📸 Grade FFB", "📋 MPOB Standards", "ℹ️ About"]
    )

    # ── Grade FFB tab ─────────────────────────────────────────────
    with tab_grade:
        model_tuple = load_model()
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("Image Input")
            src = st.radio("Source", ["Upload File", "Camera"], horizontal=True)
            img, img_name = None, "unknown"

            if src == "Upload File":
                up = st.file_uploader(
                    "Upload FFB image",
                    type=["jpg", "jpeg", "png", "bmp", "webp"]
                )
                if up:
                    img = Image.open(up).convert("RGB")
                    img_name = up.name
            else:
                cam = st.camera_input("Capture FFB")
                if cam:
                    img = Image.open(cam).convert("RGB")
                    img_name = "camera_capture.jpg"

            if img:
                st.image(img, use_container_width=True, caption=img_name)

        with col_res:
            if img:
                st.subheader("Grading Result")
                with st.spinner("Analysing…"):
                    cls, conf, probs = predict(model_tuple, img)

                info = CLASS_INFO[cls]
                oer, ker, oil_pen, ker_pen = compute_extraction(cls, conf)
                status, s_color = quality_status(cls, conf)

                st.markdown(f"""
                <div style='background:{info["color"]}18;
                            border-left:5px solid {info["color"]};
                            padding:1rem;border-radius:6px;margin-bottom:0.8rem;'>
                    <h2 style='color:{info["color"]};margin:0;'>
                        {info["emoji"]} {info["label"]}
                    </h2>
                    <p style='margin:0.4rem 0 0;'>
                        Confidence: <strong>{conf*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"<span style='background:{s_color};color:white;"
                    f"padding:0.3rem 0.9rem;border-radius:4px;"
                    f"font-weight:bold;'>{status}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("")

                if conf < conf_thresh:
                    st.warning(
                        f"Confidence {conf*100:.1f}% is below threshold "
                        f"{conf_thresh*100:.0f}% — manual inspection advised."
                    )

                st.caption(info["description"])
                st.caption(f"**Action:** {info['action']}")

                st.subheader("Class Probabilities")
                prob_dict = {
                    CLASS_INFO[n]["label"]: float(p)
                    for n, p in zip(CLASS_NAMES, probs)
                }
                st.bar_chart(prob_dict, height=220)

                st.subheader("Extraction Rate Estimate")
                m1, m2 = st.columns(2)
                m1.metric(
                    "Graded OER", f"{oer}%",
                    delta=f"-{oil_pen}%" if oil_pen > 0 else "No penalty",
                )
                m2.metric(
                    "Graded KER", f"{ker}%",
                    delta=f"-{ker_pen}%" if ker_pen > 0 else "No penalty",
                )
                st.caption(
                    f"Basic OER {BASIC_OER}% / KER {BASIC_KER}% "
                    "(Peninsular, DxP, age 4–18 yrs, Table I)"
                )

                report = build_report(
                    cls, conf, probs,
                    oer, ker, oil_pen, ker_pen,
                    status, img_name,
                )
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                with st.expander("📄 View Full Report"):
                    st.code(report, language=None)
                st.download_button(
                    "⬇️ Download Report (.txt)",
                    data=report.encode("utf-8"),
                    file_name=f"FFB_Grading_{ts}.txt",
                    mime="text/plain",
                )
            else:
                st.info("👆 Upload or capture a photo of an oil palm FFB to begin.")

    # ── MPOB Standards tab ────────────────────────────────────────
    with tab_std:
        st.subheader("Bunch Classifications (MPOB Sec. 4.3)")
        for key, info in CLASS_INFO.items():
            with st.expander(f"{info['emoji']}  {info['label']}"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Penalty table:** {info['penalty_table']}")
                st.markdown(f"**Mill action:** {info['action']}")

        st.divider()
        st.subheader("Ideal Consignment (MPOB Sec. 5.2.3)")
        st.table({
            "Category": [CLASS_INFO[c]["label"] for c in QUALITY_LIMITS],
            "Limit": list(QUALITY_LIMITS.values()),
        })

        st.subheader("Basic Extraction Rate — Table I (Age-Based, Tenera DxP)")
        st.table({
            "Palm Age (yrs)":        ["<3",    "3–4",   "4–18",  ">18"],
            "OER % – Peninsular":    ["16–18", "18–20", "20–21", "19–20"],
            "KER % – Peninsular":    ["4.0",   "5.0",   "5.5",   "5.5"],
            "OER % – Sabah/Sarawak": ["17–19", "19–21", "21–22", "20–21"],
            "KER % – Sabah/Sarawak": ["3.0",   "4.0",   "5.0",   "5.0"],
        })

        st.subheader("Rejection Thresholds (MPOB Sec. 5.2.2)")
        st.error("🚫 Reject entire load if Empty Bunches > **20%** of consignment")
        st.error("🚫 Reject entire load if Dirty Bunches > **30%** of consignment")

    # ── About tab ─────────────────────────────────────────────────
    with tab_about:
        st.subheader("About This System")
        st.markdown(f"""
        Prototype AI grading system based on the
        **MPOB Oil Palm Fruit Grading Manual, 2nd Edition (2003)**.

        | Item | Detail |
        |------|--------|
        | Model | EfficientNet-B0 (transfer learning) |
        | Inference backend | `{INFERENCE_MODE}` |
        | Classes | Ripe · Underripe · Unripe · Rotten · Empty |
        | Grading logic | MPOB Tables I–XI |

        **To deploy a trained model on Streamlit Cloud:**
        1. Train with `train_ffb_classifier.py`
        2. Export: `python export_to_onnx.py`
        3. Commit `ffb_model.onnx` to your repository

        **Limitations:** All results must be verified by a qualified
        MPOB-licensed grading officer (MPOB Manual, Section 3.2).
        """)


if __name__ == "__main__":
    main()
