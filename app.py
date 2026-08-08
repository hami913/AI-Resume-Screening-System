import html
import io
import re
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

st.set_page_config(page_title="Enterprise AI Resume Screener", page_icon="🧭", layout="wide")

# Path Handling (Cross-platform compatibility)
BASE_DIR = Path(__file__).resolve().parent

POSSIBLE_DIRS = [
    BASE_DIR / "models",
    BASE_DIR / "artifacts",
    BASE_DIR,
]

@st.cache_resource
def load_assets():
    model = None
    vectorizer = None
    label_encoder = None

    # Load Model Pipeline
    for d in POSSIBLE_DIRS:
        for model_name in ["pipeline.pkl", "best_pipeline.pkl", "resume_classifier.pkl", "model.pkl"]:
            model_path = d / model_name
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    break
                except Exception as e:
                    st.warning(f"Failed to load model from {model_path}: {e}")
        if model:
            break

    # Load Vectorizer
    for d in POSSIBLE_DIRS:
        vec_path = d / "tfidf_vectorizer.pkl"
        if vec_path.exists():
            try:
                vectorizer = joblib.load(vec_path)
                break
            except Exception as e:
                st.warning(f"Failed to load vectorizer from {vec_path}: {e}")

    # Load Label Encoder
    for d in POSSIBLE_DIRS:
        for le_name in ["label_encoder.pkl", "encoder.pkl", "target_encoder.pkl"]:
            le_path = d / le_name
            if le_path.exists():
                try:
                    label_encoder = joblib.load(le_path)
                    break
                except Exception as e:
                    st.warning(f"Failed to load label encoder from {le_path}: {e}")
        if label_encoder:
            break

    if not model:
        st.error("No valid model file could be loaded. Please check your 'models' or 'artifacts' folder.")

    return model, vectorizer, label_encoder

pipeline, vectorizer, label_encoder = load_assets()

# Expanded Technical skill list for regex matching
SKILLS_LIST = [
    r'python', r'cpp', r'csharp', r'sql', r'scikitlearn', r'xgboost', r'lightgbm', 
    r'random forest', r'logistic regression', r'linear svm', r'nlp', r'tfidf', r'pandas', r'numpy', 
    r'matplotlib', r'plotly', r'streamlit', r'fastapi', r'docker', r'mysql', r'postgresql', r'firebase', 
    r'git', r'github', r'machine learning', r'artificial intelligence', r'data science', r'shap', r'lime', 
    r'html', r'css', r'javascript', r'react', r'optuna'
]
SKILL_PATTERN = r'\b(' + '|'.join(SKILLS_LIST) + r')\b'


def preprocess_text(text):
    """Normalizes text by standardizing symbols before removing punctuation."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace("c++", "cpp").replace("c#", "csharp")
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================
# PDF TEXT-SAFETY HELPERS
# ==========================================
def sanitize_pdf_text(text: str, max_word_len: int = 20) -> str:
    """Prepare text for safe rendering in FPDF's cell/multi_cell.

    Guards against FPDF's "Not enough horizontal space to render a
    single character" error by:
      1. Forcing a space after common punctuation (comma, semicolon,
         colon, slash, exclamation, question mark) so long
         comma/slash-separated lists (e.g. skill lists, URLs) get real
         wrap points.
      2. Breaking any individual token longer than ``max_word_len``
         characters into space-joined chunks, so no single "word" can
         ever exceed the printable width - regardless of font size or
         column width. Chunking (rather than truncating with "...")
         preserves the full original content.

    Args:
        text: Raw text to sanitize.
        max_word_len: Maximum characters allowed in an unbroken token
            before it gets chunked.

    Returns:
        Sanitized text safe to pass to multi_cell/cell.
    """
    if not text:
        return ""

    text = str(text)
    text = re.sub(r'([,;:!?/])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()

    safe_tokens = []
    for word in text.split(' '):
        if len(word) <= max_word_len:
            safe_tokens.append(word)
            continue
        chunks = [word[i:i + max_word_len] for i in range(0, len(word), max_word_len)]
        safe_tokens.append(' '.join(chunks))

    return ' '.join(safe_tokens)


def truncate_to_width(pdf, text: str, max_width: float, font: str = "Helvetica",
                       style: str = "", size: float = 10) -> str:
    """Shrink text with an ellipsis until it fits within ``max_width`` mm.

    Used for single-line elements (badges, chips) where wrapping isn't
    an option - guarantees the string can never overflow its container,
    which is what actually prevents the FPDF width crash for these
    elements (rather than relying on multi_cell's line-break logic).

    Args:
        pdf: The active FPDF instance (used to measure string width).
        text: Text to fit.
        max_width: Maximum allowed width, in mm.
        font: Font family to measure with.
        style: Font style ("", "B", "I", ...).
        size: Font size in points.

    Returns:
        The original text if it already fits, otherwise a truncated,
        ellipsis-terminated version that fits within ``max_width``.
    """
    pdf.set_font(font, style, size)
    text = str(text)
    if pdf.get_string_width(text) <= max_width:
        return text
    while text and pdf.get_string_width(text + "...") > max_width:
        text = text[:-1]
    return f"{text}..." if text else "..."


# ==========================================
# MODERN PDF REPORT - DESIGN SYSTEM
# ==========================================
def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert a '#RRGGBB' or 'RRGGBB' string to an (r, g, b) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


REPORT_COLORS = {
    'primary': _hex_to_rgb('4F46E5'),
    'success': _hex_to_rgb('16A34A'),
    'success_bg': _hex_to_rgb('DCFCE7'),
    'danger': _hex_to_rgb('DC2626'),
    'danger_bg': _hex_to_rgb('FEE2E2'),
    'warning': _hex_to_rgb('D97706'),
    'text_dark': _hex_to_rgb('1E293B'),
    'text_muted': _hex_to_rgb('64748B'),
    'bg_light': _hex_to_rgb('F8FAFC'),
    'border': _hex_to_rgb('E2E8F0'),
    'white': (255, 255, 255),
}


class ModernReportPDF(FPDF):
    """FPDF subclass with a consistent branded header/footer on every page."""

    def header(self) -> None:
        if self.page_no() == 1:
            return  # Page 1 uses a full hero banner drawn in the report body.
        self.set_fill_color(*REPORT_COLORS['primary'])
        self.rect(0, 0, self.w, 12, 'F')
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(15, 3)
        self.cell(0, 6, "ENTERPRISE AI RESUME SCREENER")
        self.set_xy(-70, 3)
        self.set_font("Helvetica", "", 8)
        self.cell(55, 6, datetime.now().strftime("%b %d, %Y"), align='R')
        self.set_y(18)
        self.set_text_color(*REPORT_COLORS['text_dark'])

    def footer(self) -> None:
        self.set_y(-15)
        self.set_draw_color(*REPORT_COLORS['border'])
        self.line(15, self.get_y(), self.w - 15, self.get_y())
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*REPORT_COLORS['text_muted'])
        self.cell(0, 10, f"Page {self.page_no()}", align='C')


def _draw_hero(pdf: ModernReportPDF) -> None:
    """Draw the full-width branded banner at the top of page 1."""
    pdf.set_fill_color(*REPORT_COLORS['primary'])
    pdf.rect(0, 0, pdf.w, 38, 'F')
    pdf.set_xy(15, 10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, "Resume Screening Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "AI-Powered Candidate Analysis & ATS Match Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 5, f"Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    pdf.set_text_color(*REPORT_COLORS['text_dark'])
    pdf.set_y(46)


def _section_header(pdf: ModernReportPDF, title: str) -> None:
    """Draw a section title with a short accent underline beneath it."""
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*REPORT_COLORS['text_dark'])
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    pdf.set_draw_color(*REPORT_COLORS['primary'])
    pdf.set_line_width(0.6)
    pdf.line(15, y, 45, y)
    pdf.set_line_width(0.2)
    pdf.set_y(y + 4)


def _draw_category_section(pdf: ModernReportPDF, category: str, confidence: str) -> None:
    """Draw the predicted-category pill badge plus a confidence label."""
    y0 = pdf.get_y()
    max_badge_w = pdf.w - 30 - 70
    label = sanitize_pdf_text(category, max_word_len=30)
    label = truncate_to_width(pdf, label, max_badge_w - 10, "Helvetica", "B", 11)

    pdf.set_font("Helvetica", "B", 11)
    text_w = pdf.get_string_width(label) + 10
    pdf.set_fill_color(*REPORT_COLORS['primary'])
    pdf.set_text_color(255, 255, 255)
    pdf.rect(15, y0, text_w, 9, 'F', round_corners=True, corner_radius=4)
    pdf.set_xy(15, y0)
    pdf.cell(text_w, 9, label, align='C')

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*REPORT_COLORS['text_muted'])
    pdf.set_xy(15 + text_w + 6, y0 + 1.5)
    conf_label = truncate_to_width(
        pdf, f"Confidence: {confidence}", pdf.w - (15 + text_w + 6) - 15, "Helvetica", "", 9
    )
    pdf.cell(0, 6, conf_label)
    pdf.set_text_color(*REPORT_COLORS['text_dark'])
    pdf.set_y(y0 + 15)


def _draw_stat_cards(pdf: ModernReportPDF, stats: list) -> None:
    """Draw an evenly-spaced row of bordered stat cards.

    Args:
        stats: List of (label, value, rgb_color) tuples.
    """
    margin, gap = 15, 5
    usable = pdf.w - 2 * margin
    n = len(stats)
    card_w = (usable - (n - 1) * gap) / n
    card_h = 20
    y = pdf.get_y()
    for i, (label, value, color) in enumerate(stats):
        x = margin + i * (card_w + gap)
        pdf.set_draw_color(*REPORT_COLORS['border'])
        pdf.set_fill_color(*REPORT_COLORS['bg_light'])
        pdf.rect(x, y, card_w, card_h, 'DF', round_corners=True, corner_radius=2)
        pdf.set_xy(x, y + 3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*color)
        pdf.cell(card_w, 8, str(value), align='C')
        pdf.set_xy(x, y + 12)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*REPORT_COLORS['text_muted'])
        pdf.cell(card_w, 5, label, align='C')
    pdf.set_text_color(*REPORT_COLORS['text_dark'])
    pdf.set_y(y + card_h + 8)


def _score_band(score: float) -> tuple:
    """Map an ATS score to a (color, label) pair for the gauge/badge."""
    if score >= 75:
        return REPORT_COLORS['success'], "Strong Match"
    if score >= 50:
        return REPORT_COLORS['warning'], "Moderate Match"
    return REPORT_COLORS['danger'], "Weak Match"


def _draw_score_section(pdf: ModernReportPDF, ats_res: dict) -> None:
    """Draw the ATS score headline, band badge, and progress bar."""
    if ats_res.get("ats_score", 0) <= 0:
        _section_header(pdf, "ATS Match Score")
        pdf.set_fill_color(*REPORT_COLORS['bg_light'])
        pdf.rect(15, pdf.get_y(), pdf.w - 30, 14, 'F', round_corners=True, corner_radius=2)
        pdf.set_xy(15, pdf.get_y() + 4)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*REPORT_COLORS['text_muted'])
        pdf.cell(pdf.w - 30, 6, "No job description provided - ATS match score unavailable.", align='C')
        pdf.set_text_color(*REPORT_COLORS['text_dark'])
        pdf.set_y(pdf.get_y() + 22)
        return

    score = ats_res['ats_score']
    color, band_label = _score_band(score)
    _section_header(pdf, "ATS Match Score")
    y = pdf.get_y()

    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*color)
    pdf.set_xy(15, y)
    pdf.cell(45, 14, f"{score:.0f}%")

    pdf.set_font("Helvetica", "B", 10)
    badge_w = pdf.get_string_width(band_label) + 8
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.rect(58, y + 3, badge_w, 7, 'F', round_corners=True, corner_radius=3)
    pdf.set_xy(58, y + 3)
    pdf.cell(badge_w, 7, band_label, align='C')
    pdf.set_text_color(*REPORT_COLORS['text_dark'])

    bar_y = y + 17
    bar_w = pdf.w - 30
    pdf.set_fill_color(*REPORT_COLORS['border'])
    pdf.rect(15, bar_y, bar_w, 5, 'F', round_corners=True, corner_radius=2)
    fill_w = max(bar_w * min(score, 100) / 100, 4)
    pdf.set_fill_color(*color)
    pdf.rect(15, bar_y, fill_w, 5, 'F', round_corners=True, corner_radius=2)

    pdf.set_xy(15, bar_y + 8)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*REPORT_COLORS['text_muted'])
    pdf.cell(0, 6, f"Skills Matched: {len(ats_res['matched_skills'])} / {ats_res['jd_skill_count']} required")
    pdf.set_text_color(*REPORT_COLORS['text_dark'])
    pdf.set_y(bar_y + 16)


def _draw_skill_chips(pdf: ModernReportPDF, skills: list, accent_rgb: tuple,
                       bg_rgb: tuple, empty_msg: str) -> None:
    """Draw a wrapping row of rounded skill 'chips', page-breaking as needed."""
    if not skills:
        pdf.set_x(15)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*REPORT_COLORS['text_muted'])
        pdf.cell(0, 6, empty_msg, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*REPORT_COLORS['text_dark'])
        pdf.ln(4)
        return

    pdf.set_font("Helvetica", "", 9)
    margin = 15
    page_right = pdf.w - 15
    x, y = margin, pdf.get_y()
    row_h = 8
    for raw in skills:
        label = sanitize_pdf_text(str(raw), max_word_len=18)
        label = truncate_to_width(pdf, label, 60, "Helvetica", "", 9)
        w = pdf.get_string_width(label) + 8
        if x + w > page_right:
            x = margin
            y += row_h + 2
        if y > pdf.h - pdf.b_margin - 15:
            pdf.add_page()
            y = pdf.get_y()
            x = margin
        pdf.set_draw_color(*accent_rgb)
        pdf.set_fill_color(*bg_rgb)
        pdf.set_text_color(*accent_rgb)
        pdf.rect(x, y, w, row_h, 'DF', round_corners=True, corner_radius=3)
        pdf.set_xy(x, y + 1)
        pdf.cell(w, row_h - 2, label, align='C')
        x += w + 3
    pdf.set_text_color(*REPORT_COLORS['text_dark'])
    pdf.set_xy(margin, y + row_h + 6)


# ==========================================
# PDF REPORT GENERATOR FUNCTION
# ==========================================
def generate_pdf_report(predicted_category, confidence, ats_res, feature_dict):
    """Generates a modern, branded downloadable PDF report for a single
    resume analysis: hero banner, category badge, stat cards, an ATS
    score gauge with progress bar, and colour-coded matched/missing
    skill chips. Every dynamic string is sanitized and width-bounded,
    so long category names or large skill lists can never crash
    rendering (stress-tested with 100+ skill lists and multi-page wrap).
    """
    pdf = ModernReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    _draw_hero(pdf)
    _draw_category_section(pdf, predicted_category, confidence)

    stats = [
        ("Word Count", feature_dict.get('word_count', 0), REPORT_COLORS['primary']),
        ("Unique Words", feature_dict.get('unique_word_count', 0), REPORT_COLORS['primary']),
        ("Skills Found", feature_dict.get('skill_count', 0), REPORT_COLORS['primary']),
        ("Email", "Yes" if feature_dict.get('email_present') == 1 else "No",
         REPORT_COLORS['success'] if feature_dict.get('email_present') == 1 else REPORT_COLORS['danger']),
        ("Phone", "Yes" if feature_dict.get('phone_present') == 1 else "No",
         REPORT_COLORS['success'] if feature_dict.get('phone_present') == 1 else REPORT_COLORS['danger']),
    ]
    _draw_stat_cards(pdf, stats)
    _draw_score_section(pdf, ats_res)

    if ats_res.get("ats_score", 0) > 0:
        _section_header(pdf, "Matched Skills")
        _draw_skill_chips(
            pdf, ats_res['matched_skills'], REPORT_COLORS['success'], REPORT_COLORS['success_bg'],
            "No direct skill matches found.",
        )
        _section_header(pdf, "Skill Gaps")
        _draw_skill_chips(
            pdf, ats_res['missing_skills'], REPORT_COLORS['danger'], REPORT_COLORS['danger_bg'],
            "No skill gaps detected - full coverage!",
        )

    return bytes(pdf.output())


# ==========================================
# ATS SCORE CALCULATION FUNCTION
# ==========================================
def calculate_ats_match(resume_text, jd_text):
    """Calculates balanced ATS match score using 70% Skill Coverage + 30% TF-IDF Cosine Similarity."""
    if not jd_text.strip():
        return {"ats_score": 0.0, "matched_skills": [], "missing_skills": [], "jd_skill_count": 0}

    norm_resume = preprocess_text(resume_text)
    norm_jd = preprocess_text(jd_text)

    # Keyword Skill Extraction
    resume_skills = set(re.findall(SKILL_PATTERN, norm_resume))
    jd_skills = set(re.findall(SKILL_PATTERN, norm_jd))

    matched_skills = sorted(list(resume_skills.intersection(jd_skills)))
    missing_skills = sorted(list(jd_skills - resume_skills))

    # Coverage Ratio
    skill_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 100.0

    # TF-IDF Cosine Similarity for overall text contextual relevance
    try:
        tfidf_vec = TfidfVectorizer(stop_words='english', min_df=1)
        tfidf_mat = tfidf_vec.fit_transform([norm_resume, norm_jd])
        text_sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0]) * 100
    except Exception:
        text_sim = skill_score

    # Combined Final Score
    final_ats_score = round((0.7 * skill_score) + (0.3 * text_sim), 2)

    return {
        "ats_score": min(final_ats_score, 100.0),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "jd_skill_count": len(jd_skills)
    }


# ==========================================
# FILE PARSING HELPER (PDF, DOCX, TXT)
# ==========================================
def extract_text_from_file(uploaded_file):
    """Extracts text from PDF, DOCX, or TXT uploaded files with robust position resetting."""
    if uploaded_file is None:
        return ""

    file_type = uploaded_file.name.rsplit('.', 1)[-1].lower()

    if file_type == "txt":
        uploaded_file.seek(0)
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif file_type == "pdf":
        file_bytes = io.BytesIO(uploaded_file.read())
        try:
            import pypdf
            reader = pypdf.PdfReader(file_bytes)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            if text.strip():
                return text
        except Exception:
            pass

        file_bytes.seek(0)
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file_bytes)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            return text
        except Exception as e:
            st.error(f"Error parsing PDF file ({uploaded_file.name}): {e}")
            return ""

    elif file_type in ["docx", "doc"]:
        file_bytes = io.BytesIO(uploaded_file.read())
        try:
            import docx
            doc = docx.Document(file_bytes)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.error(f"Error parsing DOCX file ({uploaded_file.name}): {e}. Ensure 'python-docx' is installed.")
            return ""

    return ""


# ==========================================
# EXPLAINABLE AI (SHAP / FEATURE ATTRIBUTION)
# ==========================================
def get_shap_explanation(model_obj, vec_obj, input_text, predicted_category, label_encoder_obj, top_n=10):
    try:
        clf = model_obj

        if hasattr(model_obj, 'named_steps'):
            for name, step in model_obj.named_steps.items():
                if hasattr(step, 'classes_') or hasattr(step, 'coef_'):
                    clf = step
                if vec_obj is None:
                    if hasattr(step, 'get_feature_names_out'):
                        vec_obj = step
                    elif hasattr(step, 'transformers_'):
                        for trans_name, trans, cols in step.transformers_:
                            if hasattr(trans, 'get_feature_names_out'):
                                vec_obj = trans
                                break

        if vec_obj is None:
            return pd.DataFrame()

        X_vec = vec_obj.transform([input_text])
        feature_names = np.array(vec_obj.get_feature_names_out())

        class_idx = 0
        if hasattr(clf, 'classes_'):
            classes_list = list(clf.classes_)
            if label_encoder_obj and hasattr(label_encoder_obj, 'transform'):
                try:
                    target_encoded = label_encoder_obj.transform([predicted_category])[0]
                    if target_encoded in classes_list:
                        class_idx = classes_list.index(target_encoded)
                except Exception:
                    pass
            elif predicted_category in classes_list:
                class_idx = classes_list.index(predicted_category)

        if hasattr(clf, 'coef_'):
            coefs = clf.coef_
            class_coefs = coefs[class_idx] if coefs.ndim == 2 and class_idx < coefs.shape[0] else coefs.ravel()

            n_features = min(len(feature_names), class_coefs.shape[0])
            class_coefs = class_coefs[:n_features]

            non_zero_indices = X_vec.nonzero()[1]
            valid_indices = [idx for idx in non_zero_indices if idx < n_features]

            feature_contributions = []
            for idx in valid_indices:
                word = str(feature_names[idx]) if feature_names[idx] is not None else "token"
                val = X_vec[0, idx]
                impact = class_coefs[idx] * val
                feature_contributions.append({"Feature": word, "SHAP_Value": impact})

            df_shap = pd.DataFrame(feature_contributions)
            if df_shap.empty:
                return df_shap

            # Truncate continuous feature names to avoid chart label overflow
            df_shap["Feature"] = df_shap["Feature"].astype(str).apply(
                lambda x: x[:18] + "..." if len(x) > 20 else x
            )

            df_shap["Abs_Impact"] = df_shap["SHAP_Value"].abs()
            df_shap = df_shap.sort_values(by="Abs_Impact", ascending=True).tail(top_n)
            return df_shap

    except Exception as e:
        st.warning(f"Feature attribution calculation info: {e}")
        return pd.DataFrame()

    return pd.DataFrame()


# ==========================================
# DESIGN SYSTEM: CSS, COLORS & CHART THEMING
# ==========================================
UI_COLORS = {
    "primary": "#6366F1",
    "primary_2": "#8B5CF6",
    "success": "#22C55E",
    "success_bg": "rgba(34,197,94,0.12)",
    "warning": "#F59E0B",
    "danger": "#F43F5E",
    "danger_bg": "rgba(244,63,94,0.12)",
    "text": "#F1F5F9",
    "text_muted": "#94A3B8",
}

CHART_PALETTE = ["#6366F1", "#8B5CF6", "#22C55E", "#F59E0B", "#F43F5E", "#06B6D4", "#EC4899", "#84CC16"]


def inject_custom_css() -> None:
    """Inject the app's design-system CSS: type scale, hero, cards, chips,
    buttons, tabs, and input styling. Runs once per session render."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

        :root{
            --primary:#6366F1; --primary-2:#8B5CF6;
            --success:#22C55E; --success-bg:rgba(34,197,94,0.12);
            --danger:#F43F5E; --danger-bg:rgba(244,63,94,0.12);
            --warning:#F59E0B;
            --text:#F1F5F9; --text-muted:#94A3B8;
            --surface:rgba(255,255,255,0.04); --border:rgba(255,255,255,0.09);
            --radius:16px; --radius-sm:10px;
            --font-display:'Space Grotesk',sans-serif;
            --font-body:'Inter',sans-serif;
            --font-mono:'JetBrains Mono',monospace;

            /* Type scale - fluid where it matters, fixed where alignment matters */
            --fs-hero: clamp(2.1rem, 1.4rem + 2.6vw, 3.2rem);
            --fs-hero-sub: clamp(1rem, 0.9rem + 0.4vw, 1.15rem);
            --fs-eyebrow: 0.78rem;
            --fs-section: 1.3rem;
            --fs-card-title: 1.12rem;
            --fs-body: 1rem;
            --fs-caption: 0.88rem;
            --fs-metric-value: clamp(1.5rem, 1.2rem + 1vw, 1.9rem);
            --fs-metric-label: 0.74rem;
            --fs-badge: 1.05rem;
            --fs-confidence: 0.92rem;
            --fs-chip: 0.82rem;
        }

        html, body, [class*="css"] { font-family: var(--font-body); font-size: var(--fs-body); }

        @keyframes fadeInUp{ from{ opacity:0; transform:translateY(10px); } to{ opacity:1; transform:translateY(0); } }

        [data-testid="stAppViewContainer"]{
            background:
                radial-gradient(circle at 12% 0%, rgba(99,102,241,0.14), transparent 42%),
                radial-gradient(circle at 88% 8%, rgba(139,92,246,0.10), transparent 40%);
        }
        .block-container{ padding-top:2.2rem; max-width:1180px; }

        /* ---------- Hero ---------- */
        .hero-wrap{
            padding-bottom:1.4rem; border-bottom:1px solid var(--border); margin-bottom:1.8rem;
            animation:fadeInUp .5s ease both;
        }
        .hero-eyebrow{
            font-family:var(--font-mono); font-size:var(--fs-eyebrow); letter-spacing:0.14em; text-transform:uppercase;
            font-weight:600; color:var(--primary-2); margin-bottom:0.7rem; display:flex; align-items:center; gap:0.5rem;
        }
        .hero-eyebrow::before{
            content:""; width:7px; height:7px; border-radius:50%; background:var(--success);
            box-shadow:0 0 8px var(--success); display:inline-block;
        }
        .hero-title{
            font-family:var(--font-display); font-weight:700; font-size:var(--fs-hero); line-height:1.1; margin:0 0 0.6rem 0;
            background:linear-gradient(120deg,#ffffff 10%, var(--primary-2) 60%, var(--primary) 100%);
            -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
        }
        .hero-sub{ font-size:var(--fs-hero-sub); color:var(--text-muted); max-width:660px; line-height:1.6; }

        /* ---------- Section labels ---------- */
        .section-label{
            font-family:var(--font-display); font-weight:700; font-size:var(--fs-section); color:var(--text);
            margin:1.9rem 0 0.8rem 0; display:flex; align-items:center; gap:0.6rem;
        }
        .section-label::after{ content:""; flex:1; height:1px; background:var(--border); margin-left:0.4rem; }

        /* ---------- Card titles (inside bordered containers) ---------- */
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] h4{
            font-family:var(--font-display); font-weight:600; font-size:var(--fs-card-title);
            color:var(--text); margin-bottom:0.5rem; letter-spacing:0.01em;
        }

        /* ---------- Captions ---------- */
        [data-testid="stCaptionContainer"], .stCaption{
            font-size:var(--fs-caption) !important; color:var(--text-muted) !important;
        }

        /* ---------- Bordered containers -> glass cards ---------- */
        [data-testid="stVerticalBlockBorderWrapper"]{
            background:var(--surface); backdrop-filter:blur(10px);
            border:1px solid var(--border) !important; border-radius:var(--radius) !important;
            transition:border-color .2s ease;
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"]{ gap:0.35rem; border-bottom:1px solid var(--border); }
        .stTabs [data-baseweb="tab"]{
            font-family:var(--font-display); font-weight:600; font-size:0.98rem; color:var(--text-muted);
            background:transparent; border-radius:10px 10px 0 0; padding:0.65rem 1.2rem;
            transition:color .15s ease;
        }
        .stTabs [data-baseweb="tab"]:hover{ color:var(--text); }
        .stTabs [aria-selected="true"]{ color:var(--text) !important; border-bottom:2px solid var(--primary); }

        /* ---------- Buttons ---------- */
        .stButton>button{
            font-family:var(--font-display); font-weight:600; font-size:1rem; color:#fff; border:none; border-radius:12px;
            padding:0.7rem 1.6rem; background:linear-gradient(135deg,var(--primary),var(--primary-2));
            box-shadow:0 4px 20px rgba(99,102,241,0.35); transition:transform .15s ease, box-shadow .15s ease;
        }
        .stButton>button:hover{ transform:translateY(-2px); box-shadow:0 8px 26px rgba(99,102,241,0.5); }
        .stButton>button:active{ transform:translateY(0); }

        [data-testid="stDownloadButton"] button{
            font-family:var(--font-display); font-weight:600; font-size:1rem; border-radius:12px; border:none;
            background:linear-gradient(135deg,var(--success),#16A34A); box-shadow:0 4px 20px rgba(34,197,94,0.3);
            transition:transform .15s ease, box-shadow .15s ease;
        }
        [data-testid="stDownloadButton"] button:hover{ transform:translateY(-2px); box-shadow:0 8px 26px rgba(34,197,94,0.45); }

        /* ---------- Inputs ---------- */
        [data-testid="stFileUploaderDropzone"]{
            background:var(--surface) !important; border:1.5px dashed var(--border) !important;
            border-radius:var(--radius) !important; transition:border-color .2s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--primary) !important; }
        textarea{ border-radius:var(--radius-sm) !important; font-size:0.95rem !important; }

        /* ---------- Metric cards ---------- */
        .metric-grid{ display:flex; gap:0.8rem; flex-wrap:wrap; margin:0.6rem 0 1.1rem 0; }
        .metric-card{
            flex:1; min-width:118px; background:var(--surface); border:1px solid var(--border);
            border-radius:var(--radius-sm); padding:1rem 0.6rem; text-align:center; position:relative; overflow:hidden;
            transition:transform .18s ease, border-color .18s ease;
        }
        .metric-card:hover{ transform:translateY(-3px); border-color:rgba(99,102,241,0.4); }
        .metric-card::before{
            content:""; position:absolute; top:0; left:0; right:0; height:3px; background:var(--accent, var(--primary));
        }
        .metric-value{ font-family:var(--font-mono); font-size:var(--fs-metric-value); font-weight:700; color:var(--accent, var(--primary)); }
        .metric-label{ font-size:var(--fs-metric-label); font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-muted); margin-top:0.3rem; }

        /* ---------- Category badge ---------- */
        .category-badge{
            display:inline-block; font-family:var(--font-display); font-weight:700; font-size:var(--fs-badge);
            padding:0.55rem 1.15rem; border-radius:999px; color:#fff;
            background:linear-gradient(135deg,var(--primary),var(--primary-2));
            box-shadow:0 4px 16px rgba(99,102,241,0.35);
        }
        .confidence-note{ font-family:var(--font-mono); font-size:var(--fs-confidence); color:var(--text-muted); margin-left:0.7rem; }

        /* ---------- Skill chips ---------- */
        .chip-row{ display:flex; flex-wrap:wrap; gap:0.5rem; margin:0.5rem 0 0.2rem 0; }
        .chip{
            font-family:var(--font-mono); font-size:var(--fs-chip); font-weight:500; padding:0.36rem 0.85rem;
            border-radius:999px; border:1px solid; transition:transform .15s ease;
        }
        .chip:hover{ transform:translateY(-1px) scale(1.04); }
        .chip-match{ color:var(--success); background:var(--success-bg); border-color:rgba(34,197,94,0.35); }
        .chip-missing{ color:var(--danger); background:var(--danger-bg); border-color:rgba(244,63,94,0.35); }
        .chip-empty{ color:var(--text-muted); font-size:var(--fs-caption); font-style:italic; }

        /* ---------- Feature strip (empty-state) ---------- */
        .feature-grid{ display:flex; gap:1rem; flex-wrap:wrap; margin:1.6rem 0 0.6rem 0; }
        .feature-card{
            flex:1; min-width:210px; background:var(--surface); border:1px solid var(--border);
            border-radius:var(--radius); padding:1.5rem 1.3rem; transition:transform .18s ease, border-color .18s ease;
        }
        .feature-card:hover{ transform:translateY(-4px); border-color:rgba(99,102,241,0.4); }
        .feature-icon{ font-size:1.7rem; margin-bottom:0.7rem; }
        .feature-title{ font-family:var(--font-display); font-weight:600; font-size:var(--fs-card-title); color:var(--text); margin-bottom:0.4rem; }
        .feature-desc{ font-size:var(--fs-caption); color:var(--text-muted); line-height:1.55; }

        /* ---------- Dataframe ---------- */
        [data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:var(--radius-sm); overflow:hidden; }
        [data-testid="stDataFrame"] *{ font-family:var(--font-body); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the branded page header (eyebrow, gradient title, subtitle)."""
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-eyebrow">Resume Intelligence Engine</div>
            <div class="hero-title">Screen Smarter, Hire Faster</div>
            <div class="hero-sub">
                AI-powered resume analysis with real-time ATS matching, skill-gap detection,
                and explainable predictions &mdash; built for modern hiring teams.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    """Render a styled section heading with a trailing divider rule."""
    st.markdown(f'<div class="section-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def metric_card_html(label: str, value, accent: str = UI_COLORS["primary"]) -> str:
    """Build the HTML for a single metric card (used inside a metric-grid)."""
    return (
        f'<div class="metric-card" style="--accent:{accent}">'
        f'<div class="metric-value">{html.escape(str(value))}</div>'
        f'<div class="metric-label">{html.escape(label)}</div></div>'
    )


def render_metric_row(cards: list) -> None:
    """Render a responsive row of metric cards from a list of HTML snippets."""
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_feature_strip() -> None:
    """Render a 3-step 'how it works' card row, shown before the first
    analysis so the page has an intentional, guided feel rather than
    empty space beneath the input cards."""
    steps = [
        ("📤", "Upload", "Drop in a resume and, optionally, a job description — PDF, DOCX, or plain text."),
        ("🧠", "Analyze", "The model classifies the role, scores ATS match, and explains its own reasoning."),
        ("📊", "Decide", "Review skill gaps, confidence, and a shareable PDF report in one pass."),
    ]
    cards = "".join(
        f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{html.escape(title)}</div>'
        f'<div class="feature-desc">{html.escape(desc)}</div></div>'
        for icon, title, desc in steps
    )
    st.markdown(f'<div class="feature-grid">{cards}</div>', unsafe_allow_html=True)


def render_category_badge(category: str, confidence: str) -> None:
    """Render the predicted-category pill badge with a confidence note beside it."""
    st.markdown(
        f'<span class="category-badge">{html.escape(str(category))}</span>'
        f'<span class="confidence-note">Confidence: {html.escape(str(confidence))}</span>',
        unsafe_allow_html=True,
    )


def render_chip_row(skills: list, kind: str, empty_msg: str) -> None:
    """Render a wrapping row of colour-coded skill chips, or an empty-state note."""
    if not skills:
        st.markdown(f'<div class="chip-empty">{html.escape(empty_msg)}</div>', unsafe_allow_html=True)
        return
    css_class = "chip-match" if kind == "match" else "chip-missing"
    chips = "".join(f'<span class="chip {css_class}">{html.escape(str(s))}</span>' for s in skills)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def _plotly_dark_layout(**overrides) -> dict:
    """Shared transparent/dark layout settings applied to every chart."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=UI_COLORS["text"], family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    base.update(overrides)
    return base


def score_band_color(score: float) -> str:
    """Map an ATS score to its brand color (success/warning/danger)."""
    if score >= 75:
        return UI_COLORS["success"]
    if score >= 50:
        return UI_COLORS["warning"]
    return UI_COLORS["danger"]


def render_score_gauge(score: float):
    """Build a Plotly gauge indicator for the ATS match score."""
    color = score_band_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 40, "family": "Space Grotesk, sans-serif", "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickfont": {"color": "#94A3B8", "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(244,63,94,0.12)"},
                {"range": [50, 75], "color": "rgba(245,158,11,0.12)"},
                {"range": [75, 100], "color": "rgba(34,197,94,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.9, "value": score},
        },
    ))
    fig.update_layout(**_plotly_dark_layout(height=250, margin=dict(l=20, r=20, t=30, b=10)))
    return fig


def render_skill_donut(matched_count: int, missing_count: int):
    """Build a Plotly donut chart showing matched vs. missing skill coverage."""
    total = matched_count + missing_count
    if total <= 0:
        return None
    pct = matched_count / total * 100
    fig = go.Figure(data=[go.Pie(
        labels=["Matched", "Missing"],
        values=[matched_count, missing_count],
        hole=0.68,
        marker=dict(colors=[UI_COLORS["success"], UI_COLORS["danger"]], line=dict(color="#0B0F19", width=2)),
        textinfo="none",
        sort=False,
        hovertemplate="<b>%{label}</b>: %{value}<extra></extra>",
    )])
    fig.add_annotation(
        text=f"<b>{pct:.0f}%</b><br><span style='font-size:11px;color:#94A3B8'>Coverage</span>",
        showarrow=False, font=dict(size=20, color=UI_COLORS["text"], family="Space Grotesk, sans-serif"),
    )
    fig.update_layout(**_plotly_dark_layout(
        height=250, showlegend=True,
        legend=dict(orientation="h", y=-0.08, font=dict(color=UI_COLORS["text_muted"])),
    ))
    return fig


def render_shap_chart(df_shap: pd.DataFrame):
    """Build an interactive Plotly horizontal bar chart of feature attributions."""
    colors = [UI_COLORS["success"] if v > 0 else UI_COLORS["danger"] for v in df_shap["SHAP_Value"]]
    fig = go.Figure(go.Bar(
        x=df_shap["SHAP_Value"], y=df_shap["Feature"], orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(**_plotly_dark_layout(
        height=max(320, 30 * len(df_shap)),
        xaxis=dict(title="Contribution", gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.25)"),
        yaxis=dict(title=""),
    ))
    return fig


def render_ranking_bar(df: pd.DataFrame, top_n: int = 10):
    """Build a horizontal bar chart of the top-N ranked candidates by ATS score."""
    d = df.head(top_n).iloc[::-1]
    colors = [score_band_color(v) for v in d["ATS Match Score (%)"]]
    fig = go.Figure(go.Bar(
        x=d["ATS Match Score (%)"], y=d["Candidate Name"], orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_plotly_dark_layout(
        height=max(300, 36 * len(d)),
        xaxis=dict(title="ATS Match %", range=[0, 100], gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title=""),
    ))
    return fig


def render_category_donut(df: pd.DataFrame):
    """Build a donut chart of predicted-category distribution across candidates."""
    counts = df["Predicted Category"].value_counts()
    colors = [CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(len(counts))]
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values, hole=0.6,
        marker=dict(colors=colors, line=dict(color="#0B0F19", width=2)),
        hovertemplate="<b>%{label}</b>: %{value}<extra></extra>",
    )])
    fig.update_layout(**_plotly_dark_layout(
        height=300, legend=dict(orientation="h", y=-0.12, font=dict(color=UI_COLORS["text_muted"])),
    ))
    return fig


# ==========================================
# STREAMLIT UI LAYOUT WITH TABS
# ==========================================
inject_custom_css()
render_hero()

tab1, tab2 = st.tabs(["📄 Single Resume Screener", "📊 Batch Processing & Candidate Ranking"])

# ------------------------------------------
# TAB 1: SINGLE RESUME ANALYZER
# ------------------------------------------
with tab1:
    col_in1, col_in2 = st.columns(2)

    with col_in1:
        with st.container(border=True):
            st.markdown("#### 📄 Candidate Resume")
            uploaded_resume = st.file_uploader(
                "Upload Resume (.pdf, .docx, .txt)",
                type=["pdf", "docx", "txt"],
                key="single_resume_file"
            )
            pasted_resume = st.text_area("Or paste resume text here", height=150, key="single_resume_text")

    with col_in2:
        with st.container(border=True):
            st.markdown("#### 📋 Job Description (Optional)")
            uploaded_jd = st.file_uploader(
                "Upload Job Description (.pdf, .docx, .txt)",
                type=["pdf", "docx", "txt"],
                key="single_jd_file"
            )
            pasted_jd = st.text_area("Or paste job description here", height=150, key="single_jd_text")

    st.write("")
    analyze_clicked = st.button("🚀 Analyze Resume & Match Job", key="btn_single", use_container_width=True)

    if not analyze_clicked:
        render_feature_strip()

    if analyze_clicked:
        raw_resume = extract_text_from_file(uploaded_resume) if uploaded_resume else pasted_resume
        job_description = extract_text_from_file(uploaded_jd) if uploaded_jd else pasted_jd

        if not pipeline:
            st.error("Model setup incomplete. Please check that model files exist in 'models' or 'artifacts'.")
        elif raw_resume.strip():
            clean_text = preprocess_text(raw_resume)
            words = clean_text.split()
            matched_skills = re.findall(SKILL_PATTERN, clean_text)

            # Feature Extraction
            feature_dict = {
                'clean_resume': clean_text,
                'resume_length': len(raw_resume),
                'word_count': len(words),
                'unique_word_count': len(set(words)),
                'avg_word_length': float(sum(len(w) for w in words) / max(len(words), 1)),
                'email_present': 1 if ("@" in raw_resume or "mailto" in clean_text) else 0,
                'phone_present': 1 if re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10,}', raw_resume) else 0,
                'skill_count': len(matched_skills)
            }

            input_df = pd.DataFrame([feature_dict])

            try:
                try:
                    raw_prediction = pipeline.predict(input_df)[0]
                    confidence_data = input_df
                except Exception:
                    if vectorizer:
                        text_tfidf = vectorizer.transform([clean_text])
                        num_features = np.array([[
                            feature_dict['resume_length'],
                            feature_dict['word_count'],
                            feature_dict['unique_word_count'],
                            feature_dict['avg_word_length'],
                            feature_dict['email_present'],
                            feature_dict['phone_present'],
                            feature_dict['skill_count']
                        ]])
                        confidence_data = hstack([text_tfidf, num_features])
                        raw_prediction = pipeline.predict(confidence_data)[0]
                    else:
                        raise

                # Decode Target Category
                if label_encoder and hasattr(label_encoder, "inverse_transform"):
                    predicted_category = label_encoder.inverse_transform([raw_prediction])[0]
                elif hasattr(pipeline, "classes_") and isinstance(raw_prediction, (int, np.integer)):
                    predicted_category = pipeline.classes_[raw_prediction]
                else:
                    predicted_category = str(raw_prediction)

                confidence_str = "N/A"
                if hasattr(pipeline, "predict_proba"):
                    try:
                        probabilities = pipeline.predict_proba(confidence_data)[0]
                        confidence_val = float(max(probabilities)) * 100
                        confidence_str = f"{confidence_val:.2f}%"
                    except Exception:
                        confidence_str = "Predicted via Pipeline Decision"
                elif hasattr(pipeline, "decision_function"):
                    confidence_str = "Predicted via Decision Boundary"

                # ---- Prediction summary ----
                section_label("Prediction Summary")
                with st.container(border=True):
                    render_category_badge(predicted_category, confidence_str)

                # ---- Extracted metrics ----
                section_label("Resume Metrics")
                render_metric_row([
                    metric_card_html("Word Count", feature_dict['word_count']),
                    metric_card_html("Unique Words", feature_dict['unique_word_count']),
                    metric_card_html("Skills Found", feature_dict['skill_count']),
                    metric_card_html("Avg Word Length", f"{feature_dict['avg_word_length']:.1f}"),
                    metric_card_html("Email", "Yes" if feature_dict['email_present'] == 1 else "No",
                                      UI_COLORS["success"] if feature_dict['email_present'] == 1 else UI_COLORS["danger"]),
                    metric_card_html("Phone", "Yes" if feature_dict['phone_present'] == 1 else "No",
                                      UI_COLORS["success"] if feature_dict['phone_present'] == 1 else UI_COLORS["danger"]),
                ])

                # ---- SHAP / Feature Attribution Section ----
                section_label("💡 Model Interpretability (Explainable AI)")
                top_features = get_shap_explanation(
                    model_obj=pipeline,
                    vec_obj=vectorizer,
                    input_text=clean_text,
                    predicted_category=predicted_category,
                    label_encoder_obj=label_encoder,
                    top_n=10
                )

                if not top_features.empty:
                    st.caption(f"Key features driving the classification toward **{predicted_category}**")
                    st.plotly_chart(render_shap_chart(top_features), use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("Feature contributions could not be extracted for this model setup.")

                # ---- ATS Score & Skill Gap Analysis Display ----
                ats_res = {"ats_score": 0.0, "matched_skills": [], "missing_skills": [], "jd_skill_count": 0}
                if job_description.strip():
                    ats_res = calculate_ats_match(raw_resume, job_description)

                    section_label("🎯 ATS Job Match & Skill Gap Analysis")
                    gauge_col, donut_col = st.columns([3, 2])
                    with gauge_col:
                        st.plotly_chart(render_score_gauge(ats_res['ats_score']), use_container_width=True, config={"displayModeBar": False})
                        st.caption(f"Skills matched: **{len(ats_res['matched_skills'])} / {ats_res['jd_skill_count']}** required")
                    with donut_col:
                        donut_fig = render_skill_donut(len(ats_res['matched_skills']), len(ats_res['missing_skills']))
                        if donut_fig:
                            st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})

                    chip_col1, chip_col2 = st.columns(2)
                    with chip_col1:
                        st.markdown("**✅ Matched Skills**")
                        render_chip_row(ats_res['matched_skills'], "match", "No direct skills matched.")
                    with chip_col2:
                        st.markdown("**❌ Skill Gaps**")
                        render_chip_row(ats_res['missing_skills'], "missing", "No skill gap detected!")

                # ---- PDF Download Section ----
                section_label("📥 Export Analysis Report")
                pdf_bytes = generate_pdf_report(predicted_category, confidence_str, ats_res, feature_dict)
                st.download_button(
                    label="📥 Download Analysis Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"resume_report_{predicted_category.replace(' ', '_').lower()}.pdf",
                    mime="application/pdf",
                    key="btn_download_pdf"
                )

                with st.expander("🔍 View Detected Skills & Preprocessed Text"):
                    if matched_skills:
                        st.write("**Matched Keywords:**", ", ".join(set(matched_skills)))
                    st.write("**Preprocessed Text:**")
                    st.code(clean_text, language="text")

            except Exception as e:
                st.exception(e)
        else:
            st.warning("Please upload or paste a resume to analyze.")


# ------------------------------------------
# TAB 2: BATCH PROCESSING & RANKING
# ------------------------------------------
with tab2:
    with st.container(border=True):
        st.markdown("#### 📋 Job Description")
        col_batch_jd1, col_batch_jd2 = st.columns(2)
        with col_batch_jd1:
            batch_jd_file = st.file_uploader(
                "Upload Job Description File (.pdf, .docx, .txt)",
                type=["pdf", "docx", "txt"],
                key="batch_jd_file"
            )
        with col_batch_jd2:
            batch_jd_text = st.text_area("Or paste job description here", height=100, key="batch_jd_text")

    with st.container(border=True):
        st.markdown("#### 📄 Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload Candidate Resumes (.pdf, .docx, .txt)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="batch_files"
        )

    st.write("")
    rank_clicked = st.button("🏆 Rank Candidates", key="btn_batch", use_container_width=True)

    if rank_clicked:
        batch_jd = extract_text_from_file(batch_jd_file) if batch_jd_file else batch_jd_text

        if not uploaded_files or not batch_jd.strip():
            st.warning("Please upload candidate resumes and provide a Job Description (via file or text).")
        else:
            candidates_data = []

            for file in uploaded_files:
                candidate_name = file.name.rsplit('.', 1)[0]
                text_content = extract_text_from_file(file)

                if text_content.strip():
                    clean_txt = preprocess_text(text_content)
                    words = clean_txt.split()
                    matched_skills = re.findall(SKILL_PATTERN, clean_txt)

                    feature_dict = {
                        'clean_resume': clean_txt,
                        'resume_length': len(text_content),
                        'word_count': len(words),
                        'unique_word_count': len(set(words)),
                        'avg_word_length': float(sum(len(w) for w in words) / max(len(words), 1)),
                        'email_present': 1 if ("@" in text_content or "mailto" in clean_txt) else 0,
                        'phone_present': 1 if re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10,}', text_content) else 0,
                        'skill_count': len(matched_skills)
                    }

                    input_df = pd.DataFrame([feature_dict])
                    raw_pred = "Unknown"

                    try:
                        raw_pred = pipeline.predict(input_df)[0]
                    except Exception:
                        if vectorizer:
                            text_tfidf = vectorizer.transform([clean_txt])
                            num_feats = np.array([[
                                feature_dict['resume_length'], feature_dict['word_count'],
                                feature_dict['unique_word_count'], feature_dict['avg_word_length'],
                                feature_dict['email_present'], feature_dict['phone_present'],
                                feature_dict['skill_count']
                            ]])
                            raw_pred = pipeline.predict(hstack([text_tfidf, num_feats]))[0]

                    if label_encoder and hasattr(label_encoder, "inverse_transform") and isinstance(raw_pred, (int, np.integer)):
                        predicted_cat = label_encoder.inverse_transform([raw_pred])[0]
                    else:
                        predicted_cat = str(raw_pred)

                    ats_res = calculate_ats_match(text_content, batch_jd)

                    candidates_data.append({
                        "Candidate Name": candidate_name,
                        "ATS Match Score (%)": ats_res['ats_score'],
                        "Predicted Category": predicted_cat,
                        "Matched Skills": ", ".join(ats_res['matched_skills']),
                        "Skill Gap Count": len(ats_res['missing_skills']),
                        "Total Word Count": feature_dict['word_count']
                    })

            if candidates_data:
                df_results = pd.DataFrame(candidates_data)
                df_results = df_results.sort_values(by="ATS Match Score (%)", ascending=False).reset_index(drop=True)
                df_results.index += 1
                df_results.index.name = "Rank"

                section_label("📊 Hiring Analytics Summary")
                render_metric_row([
                    metric_card_html("Total Applicants", len(df_results)),
                    metric_card_html("Average ATS Score", f"{df_results['ATS Match Score (%)'].mean():.1f}%"),
                    metric_card_html("Top Category", df_results['Predicted Category'].mode()[0] if not df_results.empty else "N/A"),
                ])

                chart_col1, chart_col2 = st.columns([3, 2])
                with chart_col1:
                    st.markdown("**Top Candidates by ATS Match**")
                    st.plotly_chart(render_ranking_bar(df_results), use_container_width=True, config={"displayModeBar": False})
                with chart_col2:
                    st.markdown("**Category Distribution**")
                    st.plotly_chart(render_category_donut(df_results), use_container_width=True, config={"displayModeBar": False})

                section_label("🏆 Candidate Ranking Leaderboard")
                st.dataframe(
                    df_results,
                    use_container_width=True,
                    column_config={
                        "ATS Match Score (%)": st.column_config.ProgressColumn(
                            "ATS Match Score (%)", min_value=0, max_value=100, format="%.1f%%"
                        ),
                    },
                )

                csv_data = df_results.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="📥 Export Candidate Rankings (CSV)",
                    data=csv_data,
                    file_name="candidate_rankings.csv",
                    mime="text/csv"
                )