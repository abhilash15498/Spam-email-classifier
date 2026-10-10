"""Interactive Streamlit application for SMS Spam Message Classifier.

Features:
- Full Dynamic Theme Engine: Flawless ☀️ Light Mode & 🌙 Dark Mode across all widgets.
- High-contrast sidebar info cards and test metric badges.
- Prominent blinking caret ('writing blip') inside the message input box.
- Real-time single message classification with probability breakdown.
- Pre-loaded interactive preset buttons with instant session-state synchronization.
- Token signal analysis (shows detected URL, currency, phone signals).
- CSV batch upload with automated batch classification and export.
"""

from pathlib import Path
import io
import json
import joblib
import pandas as pd
import streamlit as st

# Application Configuration
st.set_page_config(
    page_title="SMS Spam Message Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "spam_classifier_pipeline.joblib"
METADATA_FILE = PROJECT_ROOT / "models" / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFUSION_MATRIX_FILE = REPORTS_DIR / "confusion_matrix.png"


@st.cache_resource(show_spinner="Loading trained ML model...")
def load_trained_model():
    """Load the persisted joblib model pipeline once and cache in memory."""
    if not MODEL_FILE.exists():
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        return None


@st.cache_data
def load_model_metadata():
    """Load model performance metadata and training statistics."""
    if not METADATA_FILE.exists():
        return None
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Sidebar Appearance Toggle
with st.sidebar:
    st.header("🎨 Appearance")
    selected_theme = st.radio(
        "Choose Interface Theme:",
        options=["☀️ Light Mode", "🌙 Dark Mode"],
        index=0,
        horizontal=True,
    )
    is_dark = "Dark" in selected_theme

metadata = load_model_metadata()

# Inject Dynamic CSS First so entire page & sidebar inherit styles immediately
if is_dark:
    theme_css = """
    <style>
    /* Dark Theme Global Canvas */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* Headings & Typography */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #f8fafc !important;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8 !important;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] label,
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
    }

    /* Message Text Area & BLINKING CARET CURSOR */
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"] {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        caret-color: #ef4444 !important; /* Bold visible blinking red cursor */
        border: 1.5px solid #334155 !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
        line-height: 1.5 !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #ef4444 !important;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.3) !important;
        caret-color: #ef4444 !important;
    }

    /* Preset Secondary Buttons */
    .stButton > button,
    button[data-testid="baseButton-secondary"] {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1.5px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #334155 !important;
        border-color: #64748b !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }
    .stButton > button p,
    button[data-testid="baseButton-secondary"] p {
        color: #f1f5f9 !important;
    }

    /* Prominent Primary Classify Button */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.4) !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: #ffffff !important;
    }
    button[data-testid="baseButton-primary"] p {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar Custom Cards */
    .sidebar-info-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-info-row {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
    }
    .sidebar-info-row strong {
        color: #f8fafc;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
        margin-top: 0.5rem;
    }
    .metric-badge {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.6rem 0.5rem;
        text-align: center;
    }
    .metric-badge-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-badge-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.1rem;
    }

    /* Result Alert Boxes */
    .result-box-spam {
        background: linear-gradient(135deg, #450a0a 0%, #1f0505 100%) !important;
        border: 1.5px solid #ef4444 !important;
        border-left: 8px solid #ef4444 !important;
        border-radius: 10px !important;
        padding: 1.25rem !important;
        color: #fee2e2 !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 18px rgba(239, 68, 68, 0.3) !important;
    }
    .result-box-spam * {
        color: #fee2e2 !important;
    }

    .result-box-ham {
        background: linear-gradient(135deg, #052e16 0%, #021a0d 100%) !important;
        border: 1.5px solid #22c55e !important;
        border-left: 8px solid #22c55e !important;
        border-radius: 10px !important;
        padding: 1.25rem !important;
        color: #dcfce7 !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 18px rgba(34, 197, 94, 0.3) !important;
    }
    .result-box-ham * {
        color: #dcfce7 !important;
    }

    .signal-card {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 0.85rem 1.1rem !important;
        margin-top: 0.6rem !important;
        color: #cbd5e1 !important;
    }
    .signal-card * {
        color: #cbd5e1 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f87171 !important;
        border-bottom-color: #f87171 !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] p {
        color: inherit !important;
    }
    </style>
    """
else:
    theme_css = """
    <style>
    /* Light Theme Global Canvas */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1e293b !important;
    }

    /* Headings & Typography */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0f172a !important;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #475569 !important;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] label,
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }

    /* Message Text Area & BLINKING CARET CURSOR */
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        caret-color: #ef4444 !important; /* Bold visible blinking red cursor */
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        font-size: 1.05rem !important;
        line-height: 1.5 !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #ef4444 !important;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2) !important;
        caret-color: #ef4444 !important;
    }

    /* Preset Secondary Buttons */
    .stButton > button,
    button[data-testid="baseButton-secondary"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
        transform: translateY(-1px);
    }
    .stButton > button p,
    button[data-testid="baseButton-secondary"] p {
        color: #1e293b !important;
    }

    /* Prominent Primary Classify Button */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35) !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: #ffffff !important;
    }
    button[data-testid="baseButton-primary"] p {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar Custom Cards */
    .sidebar-info-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-info-row {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
    }
    .sidebar-info-row strong {
        color: #0f172a;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
        margin-top: 0.5rem;
    }
    .metric-badge {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 0.5rem;
        text-align: center;
    }
    .metric-badge-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0284c7;
    }
    .metric-badge-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.1rem;
    }

    /* Result Alert Boxes */
    .result-box-spam {
        background: linear-gradient(135deg, #fee2e2 0%, #fef2f2 100%) !important;
        border: 1.5px solid #f87171 !important;
        border-left: 8px solid #ef4444 !important;
        border-radius: 10px !important;
        padding: 1.25rem !important;
        color: #991b1b !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.12) !important;
    }
    .result-box-spam * {
        color: #991b1b !important;
    }

    .result-box-ham {
        background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%) !important;
        border: 1.5px solid #86efac !important;
        border-left: 8px solid #22c55e !important;
        border-radius: 10px !important;
        padding: 1.25rem !important;
        color: #166534 !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.12) !important;
    }
    .result-box-ham * {
        color: #166534 !important;
    }

    .signal-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.85rem 1.1rem !important;
        margin-top: 0.6rem !important;
        color: #334155 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    .signal-card * {
        color: #334155 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #64748b !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ef4444 !important;
        border-bottom-color: #ef4444 !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] p {
        color: inherit !important;
    }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)


# Sidebar Model Details Section (Rendered with high-contrast cards)
with st.sidebar:
    st.divider()
    st.header("⚙️ Model Details")

    if metadata:
        model_name = metadata.get("model_name", "Trained Classifier")
        vocab_size = f"{metadata.get('vocabulary_size', 0):,}"
        train_count = f"{metadata.get('training_samples', 0):,}"
        test_count = f"{metadata.get('test_samples', 0):,}"

        info_card_html = f"""
        <div class="sidebar-info-card">
            <div class="sidebar-info-row"><span>Architecture:</span> <strong>{model_name}</strong></div>
            <div class="sidebar-info-row"><span>Features:</span> <strong>TF-IDF (1-2 ngrams)</strong></div>
            <div class="sidebar-info-row"><span>Vocabulary:</span> <strong>{vocab_size} tokens</strong></div>
            <div class="sidebar-info-row"><span>Training Set:</span> <strong>{train_count} msgs</strong></div>
            <div class="sidebar-info-row"><span>Test Set:</span> <strong>{test_count} msgs</strong></div>
        </div>
        """
        st.markdown(info_card_html, unsafe_allow_html=True)

        st.subheader("📊 Verified Test Metrics")
        metrics = metadata.get("metrics_on_test_split", {})
        acc_val = f"{metrics.get('accuracy', 0)*100:.1f}%"
        prec_val = f"{metrics.get('precision_spam', 0)*100:.1f}%"
        f1_val = f"{metrics.get('f1_spam', 0)*100:.1f}%"
        rec_val = f"{metrics.get('recall_spam', 0)*100:.1f}%"

        metrics_html = f"""
        <div class="metrics-grid">
            <div class="metric-badge">
                <div class="metric-badge-value">{acc_val}</div>
                <div class="metric-badge-label">Accuracy</div>
            </div>
            <div class="metric-badge">
                <div class="metric-badge-value">{prec_val}</div>
                <div class="metric-badge-label">Spam Prec.</div>
            </div>
            <div class="metric-badge">
                <div class="metric-badge-value">{f1_val}</div>
                <div class="metric-badge-label">Spam F1</div>
            </div>
            <div class="metric-badge">
                <div class="metric-badge-value">{rec_val}</div>
                <div class="metric-badge-label">Spam Recall</div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

        if CONFUSION_MATRIX_FILE.exists():
            st.divider()
            with st.expander("🔍 View Test Confusion Matrix", expanded=False):
                st.image(str(CONFUSION_MATRIX_FILE), caption="Held-out Test Set (20%)", use_container_width=True)
    else:
        st.warning("Model metadata not found. Run `src.spam_classifier.train` to generate model artifacts.")

    st.divider()
    st.markdown(
        """
        ### 💡 Operational ML Principle
        In spam filtering, **Precision** is prioritized over Recall to prevent legitimate personal or critical messages from being falsely marked as spam.
        """
    )


# Main Application Header
st.markdown('<div class="main-header">🛡️ SMS Spam Message Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Natural Language Processing system combining TF-IDF feature extraction with probabilistic classification.</div>',
    unsafe_allow_html=True,
)

# Load Classifier Pipeline
pipeline = load_trained_model()
if pipeline is None:
    st.error(
        "⚠️ **Model Artifact Missing!**\n\n"
        "The model pipeline file `models/spam_classifier_pipeline.joblib` was not found. "
        "Please train the model first by running:\n\n"
        "```bash\npython -m src.spam_classifier.train\n```"
    )
    st.stop()


# Session State Management for Message Input
if "message_text" not in st.session_state:
    st.session_state["message_text"] = "WINNER!! As a valued customer you have won a £1000 cash prize! Call 09061701461 to claim now!"


def set_input_message(msg: str):
    """Callback to instantly update message_text in session state."""
    st.session_state["message_text"] = msg


# Tabs: Single Message vs Batch CSV
tab1, tab2 = st.tabs(["💬 Single Message Classifier", "📁 Batch CSV Processing"])

with tab1:
    st.markdown("### Test a Message")
    st.write("Click any quick-test preset or type/paste your own text below:")

    # Quick Presets with reliable on_click callback
    st.markdown("**Quick Demo Presets:**")
    col1, col2, col3, col4 = st.columns([1.2, 1.3, 1.3, 0.8])

    with col1:
        st.button(
            "📩 Legitimate Sample",
            on_click=set_input_message,
            args=("Hey, are you free this evening? Let's grab dinner at 7 PM.",),
            use_container_width=True,
        )
    with col2:
        st.button(
            "⚠️ Spam Sample 1 (Prize)",
            on_click=set_input_message,
            args=("WINNER!! As a valued customer you have won a £1000 cash prize! Call 09061701461 to claim now!",),
            use_container_width=True,
        )
    with col3:
        st.button(
            "⚠️ Spam Sample 2 (Urgent)",
            on_click=set_input_message,
            args=("URGENT! Your mobile account has £500 bonus points waiting. Click http://claim-bonus-uk.com now to redeem.",),
            use_container_width=True,
        )
    with col4:
        st.button(
            "🧹 Clear",
            on_click=set_input_message,
            args=("",),
            use_container_width=True,
        )

    # Message Input linked directly to session_state
    message_input = st.text_area(
        "Message Text:",
        key="message_text",
        placeholder="Type or paste your SMS text message here...",
        height=140,
    )

    # Classify Button (Full width, vibrant red button)
    classify_btn = st.button("🔍 Classify Message", type="primary", use_container_width=True)

    # Prediction Logic executes when button clicked
    if classify_btn:
        cleaned_text = message_input.strip()
        if not cleaned_text:
            st.warning("⚠️ Please enter a message or choose a demo preset before clicking Classify.")
        else:
            with st.spinner("Classifying message patterns..."):
                pred_label = int(pipeline.predict([cleaned_text])[0])

                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba([cleaned_text])[0]
                    prob_ham = float(probs[0])
                    prob_spam = float(probs[1])
                else:
                    prob_ham, prob_spam = (0.0, 1.0) if pred_label == 1 else (1.0, 0.0)

            # Display Prominent Color-Coded Result
            if pred_label == 1:
                st.markdown(
                    f"""
                    <div class="result-box-spam">
                        <h2 style="margin:0; font-size: 1.6rem;">🚨 SPAM DETECTED</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">
                            <strong>Classification:</strong> SPAM (Unsolicited, promotional, or deceptive pattern detected).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box-ham">
                        <h2 style="margin:0; font-size: 1.6rem;">✅ NOT SPAM (HAM)</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">
                            <strong>Classification:</strong> LEGITIMATE MESSAGE (Standard conversational pattern).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Probability Breakdown
            st.markdown("#### 📈 Predicted Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("Legitimate (Ham) Probability", f"{prob_ham * 100:.2f}%")
                st.progress(prob_ham)
            with prob_col2:
                st.metric("Spam Probability", f"{prob_spam * 100:.2f}%")
                st.progress(prob_spam)

            # Signal Inspection (Explainability for NLP Review)
            st.markdown("#### 🔍 Detected Feature Signals")
            has_currency = any(s in cleaned_text for s in ["£", "$", "€", "₹", "¥"])
            has_url = "http" in cleaned_text.lower() or "www." in cleaned_text.lower() or ".com" in cleaned_text.lower()
            has_digits = any(c.isdigit() for c in cleaned_text)
            has_urgent = any(w in cleaned_text.lower() for w in ["urgent", "claim", "winner", "prize", "won", "free", "win"])

            signals_detected = []
            if has_currency:
                signals_detected.append("💰 Currency Symbol detected (mapped to `currency_token`)")
            if has_url:
                signals_detected.append("🔗 Web Link / Domain detected (mapped to `url_token`)")
            if has_digits:
                signals_detected.append("🔢 Numbers / Phone digits detected (mapped to `phone_token`)")
            if has_urgent:
                signals_detected.append("⚡ Promotional / Urgency keywords detected")

            if signals_detected:
                signals_html = "<br>".join([f"• {s}" for s in signals_detected])
                st.markdown(
                    f'<div class="signal-card"><strong>NLP Preprocessing Signals:</strong><br>{signals_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="signal-card"><strong>NLP Preprocessing Signals:</strong><br>• Standard conversational text with no deceptive markers.</div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "ℹ️ **Disclaimer:** Predicted probabilities are statistical estimates learned from the training corpus "
                "and should not be taken as absolute guarantees."
            )


with tab2:
    st.markdown("### Bulk CSV Batch Classification")
    st.write("Classify hundreds of messages at once by uploading a CSV or using our pre-built test datasets.")

    SAMPLE_CSV_PATH = PROJECT_ROOT / "sample_test_messages.csv"
    TEST_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "test_split.csv"

    # Pre-built test options
    st.markdown("#### 🚀 Quick Test Without Uploading:")
    quick_col1, quick_col2 = st.columns(2)

    load_builtin_sample = False
    load_uci_test = False

    with quick_col1:
        if st.button("⚡ Test Built-In Sample Dataset (15 msgs)", use_container_width=True):
            load_builtin_sample = True
    with quick_col2:
        if st.button("🔬 Test Full UCI Held-Out Split (1,115 msgs)", use_container_width=True):
            load_uci_test = True

    # Download Template Button
    if SAMPLE_CSV_PATH.exists():
        with open(SAMPLE_CSV_PATH, "r", encoding="utf-8") as f:
            sample_csv_data = f.read()
        st.download_button(
            label="📥 Download Sample Test CSV File (`sample_test_messages.csv`)",
            data=sample_csv_data,
            file_name="sample_test_messages.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 📂 Or Upload Your Own CSV:")
    uploaded_file = st.file_uploader("Upload CSV file (must contain a `message` or `text` column)", type=["csv"])

    # Determine data source
    batch_df = None
    data_source_name = ""

    if load_builtin_sample and SAMPLE_CSV_PATH.exists():
        batch_df = pd.read_csv(SAMPLE_CSV_PATH)
        data_source_name = "Built-in Sample Dataset (15 messages)"
    elif load_uci_test and TEST_SPLIT_PATH.exists():
        batch_df = pd.read_csv(TEST_SPLIT_PATH)
        data_source_name = "UCI Held-Out Test Set (1,115 messages)"
    elif uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            data_source_name = f"Uploaded file: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if batch_df is not None:
        st.success(f"Loaded **{len(batch_df)}** records from *{data_source_name}*.")

        # Identify message column
        text_col = None
        for col in ["message", "text", "sms", "content", "Message", "Text"]:
            if col in batch_df.columns:
                text_col = col
                break

        if text_col is None:
            st.error("Could not find a text column. Please ensure your CSV has a column named `message` or `text`.")
        else:
            st.info(f"Using column **'{text_col}'** for text classification.")

            with st.spinner("Classifying messages..."):
                messages = batch_df[text_col].fillna("").astype(str)
                predictions = pipeline.predict(messages)
                batch_df["Predicted_Class"] = ["🚨 Spam" if p == 1 else "✅ Not Spam" for p in predictions]

                if hasattr(pipeline, "predict_proba"):
                    probas = pipeline.predict_proba(messages)
                    batch_df["Spam_Probability"] = (probas[:, 1] * 100).round(2).astype(str) + "%"
                    batch_df["Ham_Probability"] = (probas[:, 0] * 100).round(2).astype(str) + "%"

            # Batch Summary Metrics
            spam_count = sum(predictions == 1)
            ham_count = sum(predictions == 0)

            sum_c1, sum_c2, sum_c3 = st.columns(3)
            with sum_c1:
                st.metric("Total Messages Processed", len(batch_df))
            with sum_c2:
                st.metric("Legitimate (Ham)", f"{ham_count} ({(ham_count/len(batch_df)*100):.1f}%)")
            with sum_c3:
                st.metric("Spam Detected", f"{spam_count} ({(spam_count/len(batch_df)*100):.1f}%)")

            st.markdown("#### Classification Results Preview (Top 25 rows):")
            st.dataframe(batch_df.head(25), use_container_width=True)

            # Export options
            csv_buffer = io.StringIO()
            batch_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Full Classified CSV Results",
                data=csv_buffer.getvalue(),
                file_name="spam_classification_results.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

