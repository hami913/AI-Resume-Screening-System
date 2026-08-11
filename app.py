import os
import tempfile

import requests
import streamlit as st

from resume_analyzer import analyze_resume


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = """You are an AI Career Assistant.

Give practical, specific and structured career advice.

Help with:
- career planning
- skill gaps
- resumes
- projects
- job preparation
- interviews
- AI/ML careers
- learning roadmaps

Always answer the complete question.

Use headings, bullet points, and actionable steps."""


st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="🧭",
    layout="wide",
)


# ============================================================
# DESIGN SYSTEM — fonts, palette, component theming
# ============================================================
# Palette:
#   Ink        #101B2D  -> headings / hero
#   Paper      #F5F7FB  -> app background
#   Card       #FFFFFF  -> surfaces
#   Emerald    #12876F  -> growth / matched / positive
#   Amber      #E3A008  -> scores / achievement
#   Coral      #E5484D  -> gaps / critical / missing
#   Slate      #33415C  -> body text
#   Slate-Muted#7A8699  -> secondary text
# Type:
#   Display -> Sora (headings)
#   Body    -> Inter
#   Data    -> JetBrains Mono (scores, tags)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --ink: #14162B;
    --ink-2: #2B2F63;
    --paper: #F7F7FB;
    --card: #FFFFFF;
    --brand: #5B4EF0;
    --brand-2: #8B7CF6;
    --brand-bg: #EEECFE;
    --emerald: #12876F;
    --emerald-bg: #E4F3EF;
    --amber: #B9790A;
    --amber-bg: #FCF1D6;
    --coral: #D8434B;
    --coral-bg: #FCE8E8;
    --slate: #262A3D;
    --slate-muted: #6E7290;
    --border: #E7E7F1;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: var(--paper); }

/* ---- Headings ---- */
h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}
h1 { font-weight: 800 !important; }
h2 { font-weight: 700 !important; }
h3 { font-weight: 700 !important; }

p, li, span, label, div { color: var(--slate); }

hr { border-color: var(--border) !important; }

/* ---- Hero banner ---- */
.hero {
    background: linear-gradient(125deg, var(--ink) 0%, var(--ink-2) 50%, var(--brand) 130%);
    border-radius: 20px;
    padding: 34px 40px;
    margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(20, 22, 43, 0.18);
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #C4BBFF;
    margin-bottom: 8px;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 32px;
    color: #FFFFFF;
    margin: 0 0 6px 0;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    color: #C7D0E0;
    margin: 0;
    max-width: 720px;
}

/* ---- Section eyebrow / label above headers ---- */
.section-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--brand);
    background: var(--brand-bg);
    border-radius: 999px;
    padding: 4px 12px;
    margin-bottom: 6px;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: var(--card);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 10px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    color: var(--slate-muted);
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: var(--brand) !important;
    color: #FFFFFF !important;
}

/* ---- Buttons ---- */
.stButton > button {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    border-radius: 10px;
    border: 1px solid var(--brand);
    background: var(--brand);
    color: #FFFFFF;
    padding: 0.55em 1.1em;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: var(--brand-2);
    border-color: var(--brand-2);
    color: #FFFFFF;
}

/* ---- Cards (bordered containers) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: var(--card);
    border-radius: 16px;
    border: 1px solid var(--border);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 2px 10px rgba(16, 27, 45, 0.04);
    border-radius: 16px;
    margin-bottom: 4px;
}

/* ---- Metrics ---- */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px 10px 16px;
}
div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px !important;
    color: var(--slate-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif;
    color: var(--ink) !important;
    font-weight: 800 !important;
}

/* ---- Progress bars ---- */
div[data-testid="stProgress"] div[role="progressbar"] > div {
    background: linear-gradient(90deg, var(--brand), var(--brand-2)) !important;
}
div[data-testid="stProgress"] {
    background: transparent;
}

/* ---- Expander ---- */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--card);
    overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    color: var(--ink);
}

/* ---- Alerts (success / warning / error / info) ---- */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif;
}

/* ---- File uploader ---- */
div[data-testid="stFileUploaderDropzone"] {
    background: var(--card);
    border: 1.5px dashed var(--border);
    border-radius: 14px;
}

/* ---- Skill / tag pills ---- */
.pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    font-weight: 500;
    padding: 4px 11px;
    margin: 3px 5px 3px 0;
    border-radius: 999px;
    border: 1px solid transparent;
    white-space: nowrap;
}
.pill-slate { background: var(--brand-bg); color: var(--ink-2); border-color: transparent; }
.pill-emerald { background: var(--emerald-bg); color: var(--emerald); }
.pill-amber { background: var(--amber-bg); color: var(--amber); }
.pill-coral { background: var(--coral-bg); color: var(--coral); }

/* ---- Chat bubbles ---- */
div[data-testid="stChatMessage"] {
    border-radius: 16px;
    border: 1px solid var(--border);
    background: var(--card);
}

/* ---- Sub-headers with icon chip ---- */
.subhead {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    color: var(--ink);
    font-size: 17px;
    margin-bottom: 2px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def pills_html(items, variant="slate"):
    """Render a list of strings as rounded pill badges (HTML)."""

    if not items:
        return ""

    return "".join(
        f'<span class="pill pill-{variant}">{item}</span>' for item in items
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analyzed_filename" not in st.session_state:
    st.session_state.analyzed_filename = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_file(uploaded_file):
    """Save Streamlit uploaded file temporarily."""

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def analyze_uploaded_resume(uploaded_file):
    """Run the existing resume analyzer safely."""

    temp_path = None

    try:

        temp_path = save_uploaded_file(uploaded_file)

        result = analyze_resume(temp_path)

        return result, None

    except Exception as e:

        return None, str(e)

    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except OSError:
                pass


def score_color(score):
    """Return Streamlit metric delta color."""

    try:
        score = float(score)
    except (TypeError, ValueError):
        return "off"

    if score >= 75:
        return "normal"

    if score >= 50:
        return "off"

    return "inverse"


def display_score(label, score):

    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    score = max(0.0, min(100.0, score))

    st.metric(
        label,
        f"{score:.2f}%",
    )

    st.progress(score / 100)


def call_llama_api(messages):

    try:

        response = requests.post(
            API_URL,
            json={
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 512,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        answer = data["choices"][0]["message"]["content"]

        return answer, None

    except requests.exceptions.ConnectionError:

        return (
            None,
            "Could not connect to the local Llama server. "
            "Make sure llama-server is running on port 8080.",
        )

    except requests.exceptions.Timeout:

        return (
            None,
            "The Llama request timed out. Please try again.",
        )

    except Exception as e:

        return None, f"Llama API error: {e}"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">🧭 CAREER INTELLIGENCE SUITE</div>
        <p class="hero-title">AI Career Assistant</p>
        <p class="hero-subtitle">
            AI Resume Intelligence · Job Matching · Skill Gap Analysis ·
            Llama 3.1 Career Assistant — everything you need to plan your
            next career move, in one place.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

resume_tab, chat_tab = st.tabs(
    [
        "📄  Resume Analyzer",
        "💬  Career Assistant",
    ]
)


# ============================================================
# RESUME ANALYZER
# ============================================================

with resume_tab:

    st.markdown('<div class="section-tag">📄 Step 1</div>', unsafe_allow_html=True)
    st.header("AI Resume Analyzer")

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx"],
        help="Upload a PDF or DOCX resume.",
    )

    analyze_clicked = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True,
    )

    if uploaded_file is not None:

        if (
            analyze_clicked
            or st.session_state.analyzed_filename
            != uploaded_file.name
        ):

            with st.spinner(
                "Analyzing resume, extracting skills, "
                "matching jobs, and calculating ATS score..."
            ):

                result, error = analyze_uploaded_resume(
                    uploaded_file
                )

            if error:

                st.error(
                    f"Resume analysis failed: {error}"
                )

                st.session_state.analysis_result = None

            else:

                st.session_state.analysis_result = result

                st.session_state.analyzed_filename = (
                    uploaded_file.name
                )

                st.success(
                    "Resume analyzed successfully."
                )


    result = st.session_state.analysis_result


    # ========================================================
    # NO RESULT
    # ========================================================

    if result is None:

        st.info(
            "Upload a PDF/DOCX resume and click "
            "**Analyze Resume** to begin."
        )


    # ========================================================
    # RESULTS
    # ========================================================

    else:

        # ----------------------------------------------------
        # CAREER PREDICTION
        # ----------------------------------------------------

        st.divider()

        with st.container(border=True):

            st.markdown('<div class="section-tag">🎯 Prediction</div>', unsafe_allow_html=True)
            st.header("Career Prediction")

            predicted_career = result.get(
                "predicted_career",
                "N/A",
            )

            st.success(
                f"Predicted Career: **{predicted_career}**"
            )


            # ------------------------------------------------
            # TOP CAREERS
            # ------------------------------------------------

            top_careers = result.get(
                "top_careers",
                [],
            )

            if top_careers:

                with st.expander(
                    "View Top 5 Career Predictions"
                ):

                    for index, career in enumerate(
                        top_careers,
                        start=1,
                    ):

                        if (
                            isinstance(career, tuple)
                            and len(career) >= 2
                        ):

                            career_name = career[0]
                            career_score = career[1]

                            st.markdown(
                                f'<div class="subhead">{index}. {career_name}</div>',
                                unsafe_allow_html=True,
                            )

                            st.caption(
                                f"Model score: "
                                f"{career_score:.4f}"
                            )

                        else:

                            st.markdown(
                                f'<div class="subhead">{index}. {career}</div>',
                                unsafe_allow_html=True,
                            )


        # ----------------------------------------------------
        # EXTRACTED SKILLS
        # ----------------------------------------------------

        st.divider()

        with st.container(border=True):

            st.markdown('<div class="section-tag">🧠 Skills</div>', unsafe_allow_html=True)
            st.header("Extracted Skills")

            skills = result.get(
                "skills",
                [],
            )

            if skills:

                st.write(
                    f"**Total Skills:** {len(skills)}"
                )

                skill_columns = st.columns(4)

                for index, skill in enumerate(
                    sorted(skills)
                ):

                    with skill_columns[
                        index % 4
                    ]:

                        st.markdown(
                            pills_html([skill], "slate"),
                            unsafe_allow_html=True,
                        )

            else:

                st.info(
                    "No skills were detected."
                )


        # ----------------------------------------------------
        # ATS SCORE
        # ----------------------------------------------------

        st.divider()

        with st.container(border=True):

            st.markdown('<div class="section-tag">📊 ATS</div>', unsafe_allow_html=True)
            st.header("ATS Score")

            ats = result.get(
                "ats_score",
                {},
            )

            overall_ats = ats.get(
                "ats_score",
                0,
            )

            st.metric(
                "Overall ATS Score",
                f"{float(overall_ats):.2f}%",
                delta_color=score_color(
                    overall_ats
                ),
            )

            st.progress(
                max(
                    0.0,
                    min(
                        1.0,
                        float(overall_ats) / 100,
                    ),
                )
            )


            # ------------------------------------------------
            # ATS FACTORS
            # ------------------------------------------------

            st.subheader(
                "ATS Score Breakdown"
            )

            ats_col1, ats_col2, ats_col3 = (
                st.columns(3)
            )

            with ats_col1:

                display_score(
                    "Skill Match (50%)",
                    ats.get(
                        "skill_coverage",
                        0,
                    ),
                )

            with ats_col2:

                display_score(
                    "Contact Information (10%)",
                    ats.get(
                        "contact_score",
                        0,
                    ),
                )

            with ats_col3:

                display_score(
                    "Resume Sections (15%)",
                    ats.get(
                        "section_score",
                        0,
                    ),
                )


            ats_col4, ats_col5 = st.columns(2)

            with ats_col4:

                display_score(
                    "Resume Length (10%)",
                    ats.get(
                        "length_score",
                        0,
                    ),
                )

            with ats_col5:

                display_score(
                    "Keyword Quality (15%)",
                    ats.get(
                        "uniqueness_score",
                        0,
                    ),
                )


            # ------------------------------------------------
            # RESUME SECTIONS
            # ------------------------------------------------

            st.subheader(
                "📋 Resume Sections"
            )

            sections = ats.get(
                "sections",
                {},
            )

            section_map = {
                "experience": "Experience",
                "education": "Education",
                "projects": "Projects",
                "certifications": "Certifications",
                "summary": "Summary / Profile",
            }

            section_columns = st.columns(5)

            for index, (
                key,
                label,
            ) in enumerate(
                section_map.items()
            ):

                with section_columns[index]:

                    if sections.get(key):

                        st.success(
                            f"✅ {label}"
                        )

                    else:

                        st.warning(
                            f"⚠️ {label}"
                        )


        # ----------------------------------------------------
        # JOB RECOMMENDATIONS
        # ----------------------------------------------------

        st.divider()

        st.markdown('<div class="section-tag">💼 Matches</div>', unsafe_allow_html=True)
        st.header(
            "Top 5 Job Recommendations"
        )

        job_matches = result.get(
            "job_matches",
            [],
        )

        if job_matches:

            for index, job in enumerate(
                job_matches[:5],
                start=1,
            ):

                job_title = job.get(
                    "job_title",
                    "Unknown Job",
                )

                category = job.get(
                    "category",
                    "Unknown",
                )

                match_score = job.get(
                    "match_score",
                    0,
                )

                with st.container(border=True):

                    st.subheader(
                        f"{index}. {job_title}"
                    )

                    st.caption(
                        f"Category: {category}"
                    )

                    display_score(
                        "Job Match Score",
                        match_score,
                    )

                    matched_skills = job.get(
                        "matched_skills",
                        [],
                    )

                    missing_skills = job.get(
                        "missing_skills",
                        [],
                    )

                    job_col1, job_col2 = (
                        st.columns(2)
                    )

                    with job_col1:

                        st.markdown(
                            "**✅ Matched Skills**"
                        )

                        if matched_skills:

                            st.markdown(
                                pills_html(matched_skills, "emerald"),
                                unsafe_allow_html=True,
                            )

                        else:

                            st.caption(
                                "No matched skills."
                            )

                    with job_col2:

                        st.markdown(
                            "**⚠️ Missing Skills**"
                        )

                        if missing_skills:

                            st.markdown(
                                pills_html(missing_skills, "coral"),
                                unsafe_allow_html=True,
                            )

                        else:

                            st.caption(
                                "No missing skills."
                            )

        else:

            st.info(
                "No job recommendations found."
            )


        # ----------------------------------------------------
        # SKILL GAP
        # ----------------------------------------------------

        st.divider()

        with st.container(border=True):

            st.markdown('<div class="section-tag">🧩 Gap Analysis</div>', unsafe_allow_html=True)
            st.header(
                "Skill Gap Analysis"
            )

            skill_gap = result.get(
                "skill_gap",
                {},
            )

            total_missing = skill_gap.get(
                "total_missing_skills",
                0,
            )

            st.metric(
                "Total Missing Skills",
                total_missing,
            )


            critical_skills = skill_gap.get(
                "critical_skills",
                [],
            )

            important_skills = skill_gap.get(
                "important_skills",
                [],
            )

            other_skills = skill_gap.get(
                "other_skills",
                [],
            )


            gap_col1, gap_col2 = (
                st.columns(2)
            )

            with gap_col1:

                st.subheader(
                    "🔴 Critical Skills"
                )

                if critical_skills:

                    for skill in critical_skills:

                        st.error(
                            skill
                        )

                else:

                    st.success(
                        "No critical skill gaps."
                    )


            with gap_col2:

                st.subheader(
                    "🟠 Important Skills"
                )

                if important_skills:

                    for skill in important_skills:

                        st.warning(
                            skill
                        )

                else:

                    st.success(
                        "No important skill gaps."
                    )


            if other_skills:

                with st.expander(
                    f"View Other Missing Skills "
                    f"({len(other_skills)})"
                ):

                    st.markdown(
                        pills_html(other_skills, "slate"),
                        unsafe_allow_html=True,
                    )


            # ------------------------------------------------
            # ATS MATCHED / MISSING
            # ------------------------------------------------

            ats_matched = ats.get(
                "matched_skills",
                [],
            )

            ats_missing = ats.get(
                "missing_skills",
                [],
            )

            if ats_matched or ats_missing:

                with st.expander(
                    "View ATS Matched / Missing Skills"
                ):

                    match_col1, match_col2 = (
                        st.columns(2)
                    )

                    with match_col1:

                        st.markdown(
                            "**✅ ATS Matched Skills**"
                        )

                        if ats_matched:

                            st.markdown(
                                pills_html(ats_matched, "emerald"),
                                unsafe_allow_html=True,
                            )

                        else:

                            st.caption(
                                "None"
                            )

                    with match_col2:

                        st.markdown(
                            "**⚠️ ATS Missing Skills**"
                        )

                        if ats_missing:

                            st.markdown(
                                pills_html(ats_missing, "coral"),
                                unsafe_allow_html=True,
                            )

                        else:

                            st.caption(
                                "None"
                            )


# ============================================================
# LLAMA CAREER ASSISTANT
# ============================================================

with chat_tab:

    st.markdown('<div class="section-tag">💬 Mentor</div>', unsafe_allow_html=True)
    st.header(
        "Llama 3.1 Career Assistant"
    )

    st.caption(
        "Powered by your fine-tuned Career LoRA"
    )


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    prompt = st.chat_input(
        "Ask me about careers, skills, projects, interviews..."
    )


    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):

            st.markdown(prompt)


        api_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        api_messages.extend(
            st.session_state.messages
        )


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                answer, error = call_llama_api(
                    api_messages
                )


            if error:

                st.error(error)

            else:

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.session_state.messages:

        if st.button(
            "🗑️ Clear Chat History"
        ):

            st.session_state.messages = []

            st.rerun() 