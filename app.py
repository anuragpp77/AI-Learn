import tensorflow as tf
import pickle
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Learning Companion",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme Injection ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global Reset & Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Background ── */
.stApp {
    background-color: #FFFDFD;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 3rem 3rem !important;
    max-width: 1200px;
}

/* ── Top Nav Bar ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 2rem 0;
    border-bottom: 1px solid rgba(28, 20, 10, 0.12);
    margin-bottom: 2.5rem;
}
.nav-logo {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1C140A;
    letter-spacing: -0.5px;
}
.nav-pills {
    display: flex;
    gap: 8px;
    background: rgba(28, 20, 10, 0.07);
    border-radius: 100px;
    padding: 5px;
}
.nav-pill {
    font-size: 13px;
    font-weight: 600;
    padding: 6px 18px;
    border-radius: 100px;
    color: #6B5E4E;
    cursor: pointer;
    transition: all 0.2s;
}
.nav-pill.active {
    background: #1C140A;
    color: #F2EDE4;
}
.nav-profile {
    display: flex;
    align-items: center;
    gap: 8px;
    background: white;
    border: 1px solid rgba(28,20,10,0.12);
    border-radius: 100px;
    padding: 6px 14px 6px 8px;
    font-size: 13px;
    font-weight: 500;
    color: #1C140A;
    cursor: pointer;
}
.nav-avatar {
    width: 26px;
    height: 26px;
    background: #3D3626;
    border-radius: 50%;
    display: inline-block;
}

/* ── Hero heading ── */
.hero-heading {
    font-size: 3.5rem;
    font-weight: 900;
    color: #1C140A;
    line-height: 1.05;
    letter-spacing: 0px;
    margin: 0 0 0.5rem 0;
}
.hero-sub {
    font-size: 1rem;
    color: #7A6A57;
    font-weight: 400;
    margin-bottom: 2.5rem;
    letter-spacing: 0.02em;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1.5px dashed rgba(28, 20, 10, 0.2) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3D3626 !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
}

/* ── Text Input ── */
.stTextInput > div > div > input {
    background: #FFFFFF !important;
    border: 1.5px solid rgba(28, 20, 10, 0.15) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 15px !important;
    color: #1C140A !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #3D3626 !important;
    box-shadow: 0 0 0 3px rgba(61, 54, 38, 0.08) !important;
}
.stTextInput label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #7A6A57 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

/* ── Section Cards ── */
.result-card {
    background: #FFFFFF;
    border-radius: 20px;
    border: 1px solid rgba(28, 20, 10, 0.08);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.card-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9E8E7C;
    margin-bottom: 0.75rem;
}
.card-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1C140A;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}

/* ── Badge pills ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 100px;
}
.badge-dark {
    background: #1C140A;
    color: #F2EDE4;
}
.badge-olive {
    background: #3D3626;
    color: #F2EDE4;
}
.badge-muted {
    background: rgba(28,20,10,0.07);
    color: #3D3626;
}
.badge-success {
    background: #DFF5E4;
    color: #1A5C2C;
}
.badge-info {
    background: #E4EDF5;
    color: #1A3C5C;
}

/* ── Confidence bar ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.conf-label {
    font-size: 13px;
    font-weight: 500;
    color: #1C140A;
    width: 80px;
}
.conf-track {
    flex: 1;
    background: rgba(28,20,10,0.08);
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    background: #1C140A;
}
.conf-pct {
    font-size: 12px;
    font-weight: 600;
    color: #7A6A57;
    width: 42px;
    text-align: right;
}

/* ── AI response area ── */
.ai-response {
    background: #1C140A;
    border-radius: 20px;
    padding: 2rem;
    color: #F2EDE4;
    line-height: 1.7;
    font-size: 15px;
}
.ai-response-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7A6A57;
    margin-bottom: 0.75rem;
}
.ai-response-heading {
    font-size: 1.1rem;
    font-weight: 800;
    color: #F2EDE4;
    margin-bottom: 1rem;
    letter-spacing: -0.3px;
}

/* ── Detected object tags ── */
.obj-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}
.obj-tag {
    background: rgba(28,20,10,0.07);
    border-radius: 100px;
    padding: 5px 13px;
    font-size: 13px;
    font-weight: 500;
    color: #3D3626;
    border: 1px solid rgba(28,20,10,0.1);
}

/* ── Divider ── */
hr.styled {
    border: none;
    border-top: 1px solid rgba(28,20,10,0.1);
    margin: 2rem 0;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #1C140A !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(28,20,10,0.08) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(28,20,10,0.08) !important;
    border-radius: 16px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #1C140A !important;
    font-size: 14px !important;
}

/* ── Columns ── */
[data-testid="stHorizontalBlock"] {
    gap: 1.5rem;
}

/* ── File uploader label ── */
[data-testid="stFileUploaderDropzone"] p {
    color: #7A6A57 !important;
    font-size: 14px !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #FFFFFF !important;
    border: 1.5px solid rgba(28,20,10,0.12) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    color: #1C140A !important;
    caret-color: #1C140A !important;
    -webkit-text-fill-color: #1C140A !important;
    line-height: 1.7 !important;
}
.stTextArea textarea:focus {
    border-color: #3D3626 !important;
    box-shadow: 0 0 0 3px rgba(61,54,38,0.08) !important;
    outline: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Nav Bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <span class="nav-logo">AI Learning<br>Companion</span>
    <div class="nav-pills">
        <span class="nav-pill active">&#9783; Analyse</span>
        <span class="nav-pill">&#128200; History</span>
    </div>
    <div class="nav-profile">
        <span class="nav-avatar"></span>
        My Profile
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<p class="hero-heading">Your AI-Powered Learning Companion</p>
<p class="hero-sub">UPLOAD AN IMAGE &nbsp;·&nbsp; ASK ANYTHING &nbsp;·&nbsp; EXPLAIN · SUMMARISE · QUIZ</p>
""", unsafe_allow_html=True)


# ── Models ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    yolo = YOLO("yolov8m.pt")
    yolo.to("cpu")
    with tf.device('/CPU:0'):
        cnn  = load_model("models/cnn_best.h5")
        lstm = load_model("models/advanced_intent_lstm.h5")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("models/max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return yolo, cnn, lstm, tokenizer, max_len

yolo_model, cnn_model, intent_model, intent_tokenizer, MAX_LEN = load_all_models()

intent_map  = {0: "Explain", 1: "Summary", 2: "Quiz"}
content_map = {0: "Text",    1: "Diagram", 2: "Mixed"}

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ── Input Row ────────────────────────────────────────────────────────────────
col_upload, col_query = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<p class="card-label">&#128228; Upload image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

with col_query:
    st.markdown('<p class="card-label">&#128172; Your question</p>', unsafe_allow_html=True)
    user_query = st.text_input(
        "",
        value="explain this",
        placeholder="e.g. explain this, summarise, quiz me...",
        label_visibility="collapsed",
    )


# ── Main Pipeline ─────────────────────────────────────────────────────────────
if uploaded_file:
    raw_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    col_img, col_det = st.columns([1, 1], gap="large")

    # ── Preview ────────────────────────────────────────────────────────────
    with col_img:
        st.markdown('<div class="result-card"><p class="card-label">&#128247; Uploaded image</p>', unsafe_allow_html=True)
        st.image(image_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── YOLO ───────────────────────────────────────────────────────────────
    with col_det:
        st.markdown('<div class="result-card"><p class="card-label">&#128269; Object detection</p>', unsafe_allow_html=True)
        with st.spinner("Detecting objects..."):
            results       = yolo_model(image_bgr)
            annotated     = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

            detected_labels = (
                [results[0].names[int(c)] for c in results[0].boxes.cls]
                if results[0].boxes is not None else []
            )

        if detected_labels:
            tags = "".join(f'<span class="obj-tag">{lbl}</span>' for lbl in sorted(set(detected_labels)))
            st.markdown(f'<div class="obj-tags">{tags}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-muted">No objects detected</span>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── OCR ────────────────────────────────────────────────────────────────
    st.markdown('<div class="result-card"><p class="card-label">&#128196; Extracted text (OCR)</p>', unsafe_allow_html=True)
    with st.spinner("Running OCR..."):
        gray           = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        extracted_text = pytesseract.image_to_string(gray).strip()
        if not extracted_text:
            extracted_text = "No readable text found in the image."

    st.text_area("", extracted_text, height=140, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CNN + Intent side by side ──────────────────────────────────────────
    col_cnn, col_intent = st.columns(2, gap="large")

    with col_cnn:
        st.markdown('<div class="result-card"><p class="card-label">&#129504; Content type</p>', unsafe_allow_html=True)
        with st.spinner("Classifying..."):
            resized   = cv2.resize(image_rgb, (128, 128)).astype(np.float32) / 255.0
            batch_img = np.expand_dims(resized, axis=0)
            cnn_probs = cnn_model.predict(batch_img, verbose=0)[0]
            cnn_pred  = int(np.argmax(cnn_probs))

        content_type = content_map.get(cnn_pred, "Unknown")
        st.markdown(f'<p class="card-title">{content_type}</p>', unsafe_allow_html=True)

        for i, label in content_map.items():
            pct = float(cnn_probs[i]) * 100
            st.markdown(f"""
            <div class="conf-row">
                <span class="conf-label">{label}</span>
                <div class="conf-track"><div class="conf-fill" style="width:{pct:.0f}%"></div></div>
                <span class="conf-pct">{pct:.0f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_intent:
        st.markdown('<div class="result-card"><p class="card-label">&#127919; Intent detection</p>', unsafe_allow_html=True)
        with st.spinner("Analysing intent..."):
            cleaned_query = clean_text(user_query)

        if not cleaned_query.strip():
            st.warning("Please enter a query.")
            st.stop()

        seq = intent_tokenizer.texts_to_sequences([cleaned_query])
        if not seq[0]:
            st.warning("Input not understood — try words like 'explain', 'summarise', or 'quiz me'.")
            st.stop()

        padded      = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
        intent_probs = intent_model.predict(padded, verbose=0)[0]
        intent_pred  = int(np.argmax(intent_probs))
        intent       = intent_map.get(intent_pred, "Unknown")

        st.markdown(f'<p class="card-title">{intent}</p>', unsafe_allow_html=True)

        for i, label in intent_map.items():
            pct = float(intent_probs[i]) * 100
            st.markdown(f"""
            <div class="conf-row">
                <span class="conf-label">{label}</span>
                <div class="conf-track"><div class="conf-fill" style="width:{pct:.0f}%"></div></div>
                <span class="conf-pct">{pct:.0f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── AI Response ────────────────────────────────────────────────────────
    if intent == "Explain":
        response_title = "Detailed Explanation"
        response_body  = (
            f"The image is primarily <strong>{content_type.lower()}</strong> content.<br><br>"
            + extracted_text.replace("\n", "<br>")
        )
    elif intent == "Summary":
        snippet        = extracted_text[:300] + ("..." if len(extracted_text) > 300 else "")
        response_title = "Summary"
        response_body  = snippet.replace("\n", "<br>")
    elif intent == "Quiz":
        if extracted_text and "No readable" not in extracted_text:
            lines   = [l.strip() for l in extracted_text.split("\n") if l.strip()]
            q_count = min(3, len(lines))
            qs      = "<br><br>".join(
                f"<strong>Q{i+1}.</strong> What does the following mean: <em>\"{lines[i]}\"</em>?"
                for i in range(q_count)
            )
            response_title = "Practice Questions"
            response_body  = qs if qs else "Not enough text in the image to generate quiz questions."
        else:
            response_title = "Quiz"
            response_body  = "Not enough text in the image to generate quiz questions."
    else:
        response_title = "Response"
        response_body  = "Try asking me to 'explain', 'summarise', or 'quiz me'."

    st.markdown(f"""
    <div class="ai-response">
        <p class="ai-response-label">&#129302; AI Response</p>
        <p class="ai-response-heading">{response_title}</p>
        <div>{response_body}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer expander ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("&#128202; Model confidence breakdown"):
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Content type probabilities**")
            for i, label in content_map.items():
                st.markdown(f"`{label}` — {float(cnn_probs[i])*100:.1f}%")
        with fc2:
            st.markdown("**Intent probabilities**")
            for i, label in intent_map.items():
                st.markdown(f"`{label}` — {float(intent_probs[i])*100:.1f}%")
