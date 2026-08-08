"""
Drop-in fix for the FPDF crash:
    "Not enough horizontal space to render a single character"

WHAT TO DO
----------
In app.py, replace your existing `sanitize_pdf_text` function AND your
existing `generate_pdf_report` function with the two/three definitions
below. Nothing else in app.py needs to change - `generate_pdf_report`
keeps the exact same signature and return value
(bytes(pdf.output())), so the call site in your Streamlit code is
untouched.

WHY IT WAS CRASHING
--------------------
FPDF's multi_cell() can only break a line at whitespace. If a single
unbroken "word" (a long skill name, a squashed category name, a URL,
etc.) is wider than the printable width, FPDF has no wrap point and
raises the exception. The old sanitize_pdf_text() only ran on some of
the dynamic strings, and its truncation ("...") could still leave an
oversized token in edge cases. safe_multi_cell() below is now the only
way generate_pdf_report touches multi_cell, so every dynamic string is
guaranteed safe.
"""

import re


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
    # Force a wrap point after punctuation that commonly glues list
    # items or path segments together without a following space.
    text = re.sub(r'([,;:!?/])([^\s])', r'\1 \2', text)
    # Collapse any double spaces introduced above.
    text = re.sub(r'\s+', ' ', text).strip()

    safe_tokens = []
    for word in text.split(' '):
        if len(word) <= max_word_len:
            safe_tokens.append(word)
            continue
        # Oversized token (e.g. a long URL or squashed category name):
        # split into fixed-size chunks instead of truncating, so every
        # chunk is guaranteed to fit within max_word_len characters.
        chunks = [word[i:i + max_word_len] for i in range(0, len(word), max_word_len)]
        safe_tokens.append(' '.join(chunks))

    return ' '.join(safe_tokens)


def safe_multi_cell(
    pdf,
    w: float,
    h: float,
    text: str,
    max_word_len: int = 20,
    min_effective_width_mm: float = 15.0,
    **kwargs,
) -> None:
    """Defensive wrapper around ``FPDF.multi_cell`` that cannot crash.

    Sanitizes the text, encodes it safely for FPDF's latin-1 core
    fonts, makes sure there is enough room on the current line before
    printing, and - as a last resort - hard-breaks the text and
    shrinks the font if FPDF still refuses to render it. This is the
    single choke point every dynamic string in the report should pass
    through.

    Args:
        pdf: The active FPDF instance.
        w: Cell width (0 = remaining width to the right margin, same
            convention as FPDF.multi_cell).
        h: Line height.
        text: Text to render.
        max_word_len: Max unbroken token length before chunking.
        min_effective_width_mm: If the current line has less printable
            width than this, start a fresh line at the left margin
            first rather than risk a too-narrow render.
        **kwargs: Passed through to pdf.multi_cell (e.g. align, border).
    """
    safe_text = sanitize_pdf_text(text, max_word_len=max_word_len)
    safe_text = safe_text.encode('latin-1', 'replace').decode('latin-1')

    if not safe_text:
        return

    effective_width = w if w > 0 else (pdf.w - pdf.r_margin - pdf.x)
    if effective_width < min_effective_width_mm:
        pdf.ln(h)
        pdf.set_x(pdf.l_margin)
        w = 0

    try:
        pdf.multi_cell(w, h, safe_text, **kwargs)
    except Exception:
        # Last-resort fallback: hard-break every few characters
        # regardless of word boundaries and shrink the font slightly,
        # so report generation can never crash on rendering.
        current_size = pdf.font_size_pt
        pdf.set_font_size(max(current_size - 2, 6))
        forced_break = ' '.join(safe_text[i:i + 8] for i in range(0, len(safe_text), 8))
        pdf.multi_cell(w, h, forced_break, **kwargs)
        pdf.set_font_size(current_size)


def generate_pdf_report(predicted_category, confidence, ats_res, feature_dict):
    """Generate a downloadable PDF report for a single resume analysis.

    Behaviourally identical to the original function - same signature,
    same sections, same return type - but every piece of dynamic text
    now flows through ``safe_multi_cell`` so long/unbroken strings
    (skill lists, category names, etc.) can never crash rendering.

    Args:
        predicted_category: The model's predicted job category.
        confidence: Confidence string/label to display.
        ats_res: Dict with keys 'ats_score', 'matched_skills',
            'missing_skills', 'jd_skill_count'.
        feature_dict: Dict of extracted resume features.

    Returns:
        The generated PDF as bytes.
    """
    from fpdf import FPDF  # local import to keep this file standalone

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Enterprise AI Resume Screener Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Section: Prediction Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "1. Categorization Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    safe_multi_cell(pdf, 0, 6, f"Predicted Job Category: {predicted_category}")
    safe_multi_cell(pdf, 0, 6, f"Model Confidence: {confidence}")
    pdf.ln(4)

    # Section: ATS Match Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "2. ATS Job Match & Skill Gap Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    if ats_res.get("ats_score", 0) > 0:
        safe_multi_cell(pdf, 0, 6, f"ATS Match Score: {ats_res['ats_score']}%")
        safe_multi_cell(
            pdf, 0, 6,
            f"Matched Skills Count: {len(ats_res['matched_skills'])} / {ats_res['jd_skill_count']}",
        )

        matched_str = ", ".join(ats_res['matched_skills']) if ats_res['matched_skills'] else "None"
        missing_str = ", ".join(ats_res['missing_skills']) if ats_res['missing_skills'] else "None"

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Matched Skills:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        safe_multi_cell(pdf, 0, 5, matched_str)

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Missing Skills (Gap Analysis):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        safe_multi_cell(pdf, 0, 5, missing_str)
    else:
        safe_multi_cell(pdf, 0, 6, "No Job Description provided for ATS matching.")

    pdf.ln(4)

    # Section: Extracted Metrics
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "3. Resume Structural Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    safe_multi_cell(pdf, 0, 6, f"Word Count: {feature_dict.get('word_count', 0)}")
    safe_multi_cell(pdf, 0, 6, f"Unique Word Count: {feature_dict.get('unique_word_count', 0)}")
    safe_multi_cell(pdf, 0, 6, f"Skills Detected: {feature_dict.get('skill_count', 0)}")
    safe_multi_cell(pdf, 0, 6, f"Email Detected: {'Yes' if feature_dict.get('email_present') == 1 else 'No'}")
    safe_multi_cell(pdf, 0, 6, f"Phone Detected: {'Yes' if feature_dict.get('phone_present') == 1 else 'No'}")

    return bytes(pdf.output())