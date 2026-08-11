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
    page_icon="🤖",
    layout="wide",
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

st.title("🤖 AI Career Assistant")

st.caption(
    "AI Resume Intelligence + Job Matching + "
    "Skill Gap Analysis + Llama 3.1 Career Assistant"
)


# ============================================================
# TABS
# ============================================================

resume_tab, chat_tab = st.tabs(
    [
        "📄 Resume Analyzer",
        "💬 Career Assistant",
    ]
)


# ============================================================
# RESUME ANALYZER
# ============================================================

with resume_tab:

    st.header("📄 AI Resume Analyzer")

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

        st.header("🎯 Career Prediction")

        predicted_career = result.get(
            "predicted_career",
            "N/A",
        )

        st.success(
            f"Predicted Career: **{predicted_career}**"
        )


        # ----------------------------------------------------
        # TOP CAREERS
        # ----------------------------------------------------

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

                        st.write(
                            f"**{index}. "
                            f"{career_name}**"
                        )

                        st.caption(
                            f"Model score: "
                            f"{career_score:.4f}"
                        )

                    else:

                        st.write(
                            f"**{index}. {career}**"
                        )


        # ----------------------------------------------------
        # EXTRACTED SKILLS
        # ----------------------------------------------------

        st.divider()

        st.header("🧠 Extracted Skills")

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

                    st.write(
                        f"• `{skill}`"
                    )

        else:

            st.info(
                "No skills were detected."
            )


        # ----------------------------------------------------
        # ATS SCORE
        # ----------------------------------------------------

        st.divider()

        st.header("📊 ATS Score")

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


        # ----------------------------------------------------
        # ATS FACTORS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # RESUME SECTIONS
        # ----------------------------------------------------

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

        st.header(
            "💼 Top 5 Job Recommendations"
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

                        st.write(
                            ", ".join(
                                f"`{skill}`"
                                for skill
                                in matched_skills
                            )
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

                        st.write(
                            ", ".join(
                                f"`{skill}`"
                                for skill
                                in missing_skills
                            )
                        )

                    else:

                        st.caption(
                            "No missing skills."
                        )

                st.divider()

        else:

            st.info(
                "No job recommendations found."
            )


        # ----------------------------------------------------
        # SKILL GAP
        # ----------------------------------------------------

        st.header(
            "🧩 Skill Gap Analysis"
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

                st.write(
                    ", ".join(
                        f"`{skill}`"
                        for skill in other_skills
                    )
                )


        # ----------------------------------------------------
        # ATS MATCHED / MISSING
        # ----------------------------------------------------

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

                        st.write(
                            ", ".join(
                                f"`{skill}`"
                                for skill
                                in ats_matched
                            )
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

                        st.write(
                            ", ".join(
                                f"`{skill}`"
                                for skill
                                in ats_missing
                            )
                        )

                    else:

                        st.caption(
                            "None"
                        )


# ============================================================
# LLAMA CAREER ASSISTANT
# ============================================================

with chat_tab:

    st.header(
        "💬 Llama 3.1 Career Assistant"
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
