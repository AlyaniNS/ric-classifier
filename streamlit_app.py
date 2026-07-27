import streamlit as st
import os
import base64
from PIL import Image
import localization
from model import RICClassifier


# ── helpers ──────────────────────────────────────────────────────────────────

def b64(path: str) -> str:
    """Return base64 of a local image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def html(markup: str):
    """Render HTML, collapsing indentation so Markdown never treats it as a code block."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RIC Classification",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global CSS ────────────────────────────────────────────────────────────────

html("""
<style>
/* ---- hide streamlit chrome ---- */
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stDeployButton, footer { display:none !important; }

/* ---- center spinner ---- */
[data-testid="stSpinner"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    width: 100% !important;
}
[data-testid="stSpinner"] > div {
    justify-content: center !important;
}

/* ---- page bg ---- */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #f5f7fa !important;
}
.block-container {
    max-width: 1080px !important;
    padding: 2.5rem 1.5rem 4rem !important;
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* ---- card mixin ---- */
.card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(2,6,23,.06);
    transition: transform .2s;
}
.card:hover { transform: translateY(-2px); }

/* ---- hero ---- */
.hero-title {
    font-weight: 800;
    font-size: clamp(24px,3.5vw,36px);
    letter-spacing: -.02em;
    color: #0b1324;
    margin: 0;
    text-align: center;
}
.hero-sub {
    color: #475569;
    font-size: 16px;
    margin: 4px 0 0;
    text-align: center;
}

/* ---- language selectbox ---- */
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
    border-radius: 8px !important;
    border-color: #e2e8f0 !important;
    background: #fff !important;
    color: #475569 !important;
    font-size: 14px !important;
}

/* ---- file uploader: style the Streamlit widget to look like our upload-area ---- */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] > section {
    border: 2px dashed #d1d5db !important;
    border-radius: 12px !important;
    background: #f8fafc !important;
    transition: border-color .3s, background .3s !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: #10b981 !important;
    background: #f0fdf4 !important;
}
/* hide the native label rendered above the widget */
[data-testid="stFileUploader"] label { display:none !important; }
/* browse files button */
[data-testid="stFileUploader"] button {
    border-radius: 999px !important;
    font-weight: 700 !important;
    color: #059669 !important;
    border: 2px solid #10b981 !important;
    background: #fff !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #f0fdf4 !important;
    border-color: #047857 !important;
    color: #047857 !important;
}

/* ---- progress bars ---- */
.pb-row {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    min-height: 22px;
}
.pb-label {
    width: 195px;
    flex-shrink: 0;
    font-size: 10px;
    font-weight: 700;
    color: #475569;
    margin-right: 10px;
    white-space: nowrap;
}
.pb-track {
    flex-grow: 1;
    background: #eef2f6;
    border-radius: 10px;
    height: 18px;
    overflow: hidden;
}
.pb-fill {
    height: 100%;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 700;
    color: #fff;
}
.pb-fill-top  { background: linear-gradient(90deg,#059669,#047857); }
.pb-fill-rest { background: linear-gradient(90deg,#10b981,#059669); }
.pb-val-outside {
    font-size: 10px; font-weight: 700; color: #475569;
    margin-left: 8px; min-width: 34px;
}

/* ---- result card layout ---- */
.result-card {
    padding: 1.25rem;
    height: 100%;
}
.result-img {
    max-width: 100%;
    max-height: 200px;
    object-fit: contain;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

/* ---- footer — theme consistent (white with top border) ---- */
.footer {
    padding: 22px 24px;
    margin-top: 3rem;
    font-size: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer a {
    color: #059669;
    text-decoration: none;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 8px;
    transition: all 0.2s;
}
.footer a:hover { color: #047857; background: #f0fdf4; }

/* ---- mobile responsive ---- */
@media (max-width: 768px) {
    .block-container { padding: 1.5rem 1rem 3rem !important; }

    /* Hero */
    .hero-title { font-size: clamp(20px,6vw,28px) !important; }
    .hero-sub { font-size: 13px !important; }

    /* Progress bars — stack label above bar on narrow screens */
    .pb-row { flex-wrap: wrap; margin-bottom: 12px; }
    .pb-label { width: 100% !important; margin-bottom: 4px; margin-right: 0; }
    .pb-val-outside { margin-left: 6px; }

    /* Result cards */
    .result-card { padding: 1rem !important; }

    /* Footer — stack vertically on mobile */
    .footer { flex-direction: column; gap: 12px; text-align: center; padding: 18px 16px; }
}
</style>
""")


# ── session state ─────────────────────────────────────────────────────────────

for key, val in [
    ("language", "id"),
    ("uploader_key", 0),
    ("active_image", None),
    ("active_image_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── load model (cached) ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_classifier():
    os.environ["MODEL_VARIANT"] = "finetuned"
    return RICClassifier()

with st.spinner("Memuat model..."):
    clf = load_classifier()

# ── header row: title + language ──────────────────────────────────────────────

col_title, col_lang = st.columns([9, 2])

with col_lang:
    lang_choice = st.selectbox(
        "lang", ["Bahasa Indonesia", "English"],
        index=0 if st.session_state.language == "id" else 1,
        label_visibility="collapsed",
    )
    new_lang = "id" if lang_choice == "Bahasa Indonesia" else "en"
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
        st.rerun()

lang = st.session_state.language
sess = {"language": lang}
T = localization.LANGUAGES[lang]

with col_title:
    html(f"""
    <div style="margin-bottom:28px;">
        <h2 class="hero-title">{T['title']}</h2>
        <p class="hero-sub">{T['subtitle']}</p>
    </div>
    """)
# STREAMLIT LIMITATION: file_uploader always shows "Drag and drop file here"
# and "Limit 200MB per file" — hardcoded in Streamlit's React component.
uploaded_file = st.file_uploader(
    T["choose_file"],
    type=["png", "jpg", "jpeg", "gif", "bmp"],
    key=f"up_{st.session_state.uploader_key}",
)
if uploaded_file and uploaded_file.name != st.session_state.active_image_name:
    st.session_state.active_image = uploaded_file
    st.session_state.active_image_name = uploaded_file.name
    st.rerun()

html('</div>')  # close upload card

# ── sample images ─────────────────────────────────────────────────────────────

html(f'<p style="color:#475569;margin:22px 0 8px;font-weight:600;text-align:center;font-size:13px;">{T["or_try_sample"]}</p>')

# Spacer columns to center the 3 sample thumbnails
_, c1, c2, c3, _ = st.columns([3, 1, 1, 1, 3])
for idx, (col, fname) in enumerate(zip([c1, c2, c3], ["dummy-1.png", "dummy-2.png", "dummy-3.png"])):
    path = os.path.join("static", "dummy", fname)
    with col:
        if os.path.exists(path):
            img_b64 = b64(path)
            is_active = (st.session_state.active_image_name == fname)
            border = "border:2px solid #10b981;" if is_active else "border:1px solid #d1d5db;"
            html(f'<div style="text-align:center;margin-bottom:6px;"><img src="data:image/png;base64,{img_b64}" style="width:52px;height:52px;border-radius:50%;object-fit:cover;{border}display:inline-block;transition:transform 0.2s;" /></div>')
            if st.button(T["sample_label"].format(idx + 1), key=f"dummy_{idx}", use_container_width=True):
                st.session_state.active_image = path
                st.session_state.active_image_name = fname
                st.session_state.uploader_key += 1
                st.rerun()

# ── result area ───────────────────────────────────────────────────────────────

if st.session_state.active_image is not None:
    src = st.session_state.active_image

    with st.spinner(T["processing"]):
        prediction, probs, confidence = clf.predict(src)

    pred_full = localization.get_plastic_type_name(sess, prediction)

    # get image bytes as base64
    if isinstance(src, str):
        img_data = b64(src)
    else:
        raw = src.read(); src.seek(0)
        img_data = base64.b64encode(raw).decode()

    st.markdown("---")
    col_img, col_pred = st.columns([4, 8], gap="large")

    # ── left: uploaded image card ─────────────────────────────────────────
    with col_img:
        html(f"""
        <div class="card result-card" style="text-align:center;">
            <h6 style="font-weight:600;color:#475569;font-size:13px;margin:0 0 12px;">{T['uploaded_image']}</h6>
            <div style="display:flex;justify-content:center;align-items:center;">
                <img class="result-img" src="data:image/jpeg;base64,{img_data}" />
            </div>
        </div>
        """)
        # native button for state reset — must be a real Streamlit widget
        if st.button(T["change_image"], use_container_width=True, key="change_img"):
            st.session_state.active_image = None
            st.session_state.active_image_name = None
            st.session_state.uploader_key += 1
            st.rerun()

    # ── right: prediction card ────────────────────────────────────────────
    with col_pred:
        # Build progress bar rows — threshold 5% matches original classification_result.html
        bars = ""
        for i, (label, val) in enumerate(probs.items()):
            name = localization.get_plastic_type_name(sess, label)
            cls = "pb-fill-top" if i == 0 else "pb-fill-rest"
            inner = f"{val:.1f}%" if val >= 5 else ""
            val_str = f"{val:.1f}%"
            outside = f'<span class="pb-val-outside">{val_str}</span>' if val < 5 else ""
            width = f"{val:.2f}%"
            bars += (
                f'<div class="pb-row">'
                f'<div class="pb-label">{name}</div>'
                f'<div class="pb-track">'
                f'<div class="pb-fill {cls}" style="width:{width};">{inner}</div>'
                f'</div>{outside}</div>'
            )

        html(f'<div class="card result-card"><h5 style="color:#047857;font-weight:700;font-size:17px;text-align:center;margin:0 0 18px;">{T["result_title"]}: <strong>{pred_full}</strong></h5><h6 style="font-weight:600;color:#475569;font-size:13px;margin:0 0 14px;">{T["confidence_percentage"]}:</h6>{bars}</div>')

else:
    html(f"<p style='text-align:center;color:#475569;margin-top:24px;'>{T['no_file_selected']}</p>")

footer_text = T.get("footer", "Klasifikasi Kode Resin Pada Kemasan Plastik")
html(f'<div class="footer"><span>{footer_text}</span><a href="https://github.com/AlyaniNS/" target="_blank" rel="noopener noreferrer"><svg viewBox="0 0 16 16" width="18" height="18" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.57-.18-3.21-.78-3.21-3.48 0-.78.28-1.41.72-1.92-.08-.18-.31-.9.07-1.89 0 0 .6-.19 1.9.72a6.47 6.47 0 0 1 3.5 0c1.3-.92 1.9-.72 1.9-.72.38.99.15 1.71.07 1.89.44.51.72 1.13.72 1.92 0 2.71-1.64 3.3-3.21 3.48.25.22.47.65.47 1.3 0 .93-.01 1.68-.01 1.91 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg> GitHub</a></div>')

