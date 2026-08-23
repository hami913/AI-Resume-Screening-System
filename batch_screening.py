from __future__ import annotations

import csv
import io
import os
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import streamlit as st


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            value = value.strip().replace("%", "")
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_name(filename: str) -> str:
    name = os.path.splitext(filename or "Candidate")[0]
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "Candidate"


def _fit_status(score: float) -> str:
    if score >= 80:
        return "Strong Match"
    if score >= 65:
        return "Competitive"
    if score >= 50:
        return "Needs Improvement"
    return "Weak Match"


def _list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [
            item.strip()
            for item in re.split(r"[,;\n|]", value)
            if item.strip()
        ]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()
    return [text] if text else []


def _display_skills(
    skills: Sequence[str],
    display_skill: Optional[Callable[[str], str]],
) -> str:

    output = []

    for skill in skills:
        try:
            output.append(
                display_skill(skill)
                if display_skill
                else str(skill)
            )
        except Exception:
            output.append(str(skill))

    return ", ".join(output)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""

    buffer = io.StringIO()
    fieldnames = list(rows[0].keys())

    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in rows:
        writer.writerow({
            key: row.get(key, "")
            for key in fieldnames
        })

    return buffer.getvalue().encode("utf-8-sig")


def _init_state() -> None:
    defaults = {
        "batch_screen_results": [],
        "batch_screen_failures": [],
        "batch_target_title": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = (
                list(value)
                if isinstance(value, list)
                else value
            )


def render_batch_resume_screening(
    *,
    analyze_uploaded_resume,
    compute_jd_match,
    get_predicted_career=None,
    display_skill=None,
    max_upload_mb: int = 12,
) -> None:

    _init_state()

    st.markdown("## Batch Resume Screening")

    st.caption(
        "Compare multiple candidates against one job description "
        "using the same ATS engine as your individual resume analysis."
    )

    left, right = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    # ========================================================
    # JOB DESCRIPTION
    # ========================================================

    with left:

        job_description = st.text_area(
            "Target Job Description",
            key="batch_job_description",
            height=280,
            placeholder=(
                "Paste the complete job description here.\n\n"
                "For best results include:\n"
                "• Job title\n"
                "• Required skills\n"
                "• Preferred skills\n"
                "• Responsibilities\n"
                "• Qualifications"
            ),
        )

    # ========================================================
    # MULTIPLE RESUMES
    # ========================================================

    with right:

        uploaded_files = st.file_uploader(
            "Upload Multiple Resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="batch_resume_uploader",
            help=(
                "Upload multiple PDF or DOCX resumes. "
                f"Recommended maximum size: {max_upload_mb} MB per resume."
            ),
        )

        if uploaded_files:
            st.success(
                f"{len(uploaded_files)} resume(s) selected."
            )

    # ========================================================
    # RUN SCREENING
    # ========================================================

    run = st.button(
        "Screen & Rank Candidates",
        key="batch_run_screening",
        type="primary",
        use_container_width=True,
    )

    if run:

        if not str(job_description or "").strip():

            st.warning(
                "Please paste the target job description first."
            )

        elif not uploaded_files:

            st.warning(
                "Please upload at least one resume."
            )

        else:

            results: List[Dict[str, Any]] = []
            failures: List[Dict[str, str]] = []

            total = len(uploaded_files)

            progress = st.progress(
                0.0,
                text="Preparing batch screening..."
            )

            for index, uploaded_file in enumerate(
                uploaded_files,
                start=1
            ):

                progress.progress(
                    (index - 1) / total,
                    text=(
                        f"Analyzing {index}/{total}: "
                        f"{uploaded_file.name}"
                    ),
                )

                # --------------------------------------------
                # Analyze Resume
                # --------------------------------------------

                try:
                    try:
                        uploaded_file.seek(0)
                    except Exception:
                        pass

                    resume_result, error = analyze_uploaded_resume(
                        uploaded_file
                    )

                except Exception as exc:

                    failures.append({
                        "File": uploaded_file.name,
                        "Stage": "Resume analysis",
                        "Error": f"{type(exc).__name__}: {exc}",
                    })

                    progress.progress(index / total)
                    continue

                if error or not resume_result:

                    failures.append({
                        "File": uploaded_file.name,
                        "Stage": "Resume analysis",
                        "Error": error or "Resume analysis returned no result",
                    })

                    progress.progress(index / total)
                    continue

                # --------------------------------------------
                # Same ATS Engine Used By Main App
                # --------------------------------------------

                try:

                    match = compute_jd_match(
                        resume_result,
                        job_description,
                    )

                except Exception as exc:

                    failures.append({
                        "File": uploaded_file.name,
                        "Stage": "ATS matching",
                        "Error": f"{type(exc).__name__}: {exc}",
                    })

                    progress.progress(index / total)
                    continue

                if not match:

                    failures.append({
                        "File": uploaded_file.name,
                        "Stage": "ATS matching",
                        "Error": "ATS engine returned no result",
                    })

                    progress.progress(index / total)
                    continue

                ats_score = round(
                    _safe_float(
                        match.get("ats_score")
                    ),
                    1
                )

                matched = _list(
                    match.get("matched_skills")
                )

                missing = _list(
                    match.get("missing_skills")
                )

                critical = _list(
                    match.get("critical_gaps")
                )

                # --------------------------------------------
                # Predicted Career
                # --------------------------------------------

                predicted_career = "Unavailable"

                if get_predicted_career:

                    try:
                        predicted_career = (
                            get_predicted_career(
                                resume_result
                            )
                        )
                    except Exception:
                        pass

                # --------------------------------------------
                # Store Candidate Result
                # --------------------------------------------

                results.append({

                    "Rank": 0,

                    "Candidate":
                        _candidate_name(
                            uploaded_file.name
                        ),

                    "Match %":
                        ats_score,

                    "Fit":
                        _fit_status(ats_score),

                    "Predicted Career":
                        predicted_career,

                    "Skill Coverage %":
                        round(
                            _safe_float(
                                match.get(
                                    "skill_coverage"
                                )
                            ),
                            1
                        ),

                    "Keyword Context %":
                        round(
                            _safe_float(
                                match.get(
                                    "uniqueness_score"
                                )
                            ),
                            1
                        ),

                    "Resume Structure %":
                        round(
                            _safe_float(
                                match.get(
                                    "section_score"
                                )
                            ),
                            1
                        ),

                    "Experience Evidence %":
                        round(
                            _safe_float(
                                match.get(
                                    "experience_score"
                                )
                            ),
                            1
                        ),

                    "Matched Skills":
                        _display_skills(
                            matched,
                            display_skill
                        ),

                    "Missing Skills":
                        _display_skills(
                            missing,
                            display_skill
                        ),

                    "Critical Gaps":
                        _display_skills(
                            critical,
                            display_skill
                        ),

                    "File":
                        uploaded_file.name,

                    "_match":
                        match,
                })

                progress.progress(index / total)

            # =================================================
            # SORT BY BEST ATS MATCH
            # =================================================

            results.sort(
                key=lambda row: row["Match %"],
                reverse=True
            )

            for rank, row in enumerate(
                results,
                start=1
            ):
                row["Rank"] = rank

            st.session_state.batch_screen_results = results
            st.session_state.batch_screen_failures = failures

            if results:

                st.session_state.batch_target_title = (
                    results[0]["_match"].get(
                        "target_title"
                    )
                )

            progress.empty()

            if results:
                st.success(
                    f"Screening complete — "
                    f"{len(results)} candidate(s) ranked."
                )

            if failures:
                st.warning(
                    f"{len(failures)} resume(s) "
                    "could not be processed."
                )

    # ========================================================
    # SHOW RESULTS
    # ========================================================

    results = st.session_state.get(
        "batch_screen_results",
        []
    )

    failures = st.session_state.get(
        "batch_screen_failures",
        []
    )

    if not results:

        st.info(
            "Ranked candidates will appear here "
            "after you run batch screening."
        )

        return

    st.divider()

    st.markdown("## Candidate Ranking")

    target = st.session_state.get(
        "batch_target_title"
    )

    if target:
        st.caption(
            f"Target role: {target}"
        )

    # ========================================================
    # SUMMARY KPIs
    # ========================================================

    strong_count = sum(
        1
        for row in results
        if row["Match %"] >= 80
    )

    competitive_count = sum(
        1
        for row in results
        if 65 <= row["Match %"] < 80
    )

    cols = st.columns(4)

    cols[0].metric(
        "Candidates",
        len(results)
    )

    cols[1].metric(
        "Top Candidate",
        results[0]["Candidate"]
    )

    cols[2].metric(
        "Top Match",
        f"{results[0]['Match %']:.1f}%"
    )

    cols[3].metric(
        "Strong Matches",
        strong_count
    )

    # ========================================================
    # SCORE FILTER
    # ========================================================

    minimum_score = st.slider(
        "Minimum candidate match",
        0,
        100,
        0,
        5,
        key="batch_minimum_score",
    )

    filtered = [
        row
        for row in results
        if row["Match %"] >= minimum_score
    ]

    public_columns = [
        "Rank",
        "Candidate",
        "Match %",
        "Fit",
        "Predicted Career",
        "Skill Coverage %",
        "Keyword Context %",
        "Resume Structure %",
        "Experience Evidence %",
        "Matched Skills",
        "Missing Skills",
        "Critical Gaps",
        "File",
    ]

    public_rows = [
        {
            column: row.get(column, "")
            for column in public_columns
        }
        for row in filtered
    ]

    st.dataframe(
        public_rows,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # CSV EXPORT
    # ========================================================

    all_public_rows = [
        {
            column: row.get(column, "")
            for column in public_columns
        }
        for row in results
    ]

    st.download_button(
        "Download Ranked Candidate List",
        data=_csv_bytes(all_public_rows),
        file_name="batch_candidate_ranking.csv",
        mime="text/csv",
        use_container_width=True,
        key="batch_csv_download",
    )

    # ========================================================
    # TOP 5 DETAILS
    # ========================================================

    st.markdown("## Top Candidate Analysis")

    for row in filtered[:5]:

        match = row.get("_match", {})

        with st.expander(
            (
                f"#{row['Rank']} — "
                f"{row['Candidate']} — "
                f"{row['Match %']:.1f}% "
                f"{row['Fit']}"
            ),
            expanded=row["Rank"] == 1,
        ):

            metrics = st.columns(4)

            metrics[0].metric(
                "Overall Match",
                f"{row['Match %']:.1f}%"
            )

            metrics[1].metric(
                "Skill Coverage",
                f"{row['Skill Coverage %']:.1f}%"
            )

            metrics[2].metric(
                "Keyword Context",
                f"{row['Keyword Context %']:.1f}%"
            )

            metrics[3].metric(
                "Evidence",
                f"{row['Experience Evidence %']:.1f}%"
            )

            st.markdown("**Matched Skills**")
            st.write(
                row["Matched Skills"]
                or "No direct matches detected."
            )

            st.markdown("**Missing Skills**")
            st.write(
                row["Missing Skills"]
                or "No catalogued skills missing."
            )

            if row["Critical Gaps"]:

                st.markdown("**Critical Gaps**")
                st.warning(
                    row["Critical Gaps"]
                )

            fit_summary = str(
                match.get("fit_summary") or ""
            )

            if fit_summary:

                st.markdown("**Fit Summary**")
                st.write(fit_summary)

    st.caption(
        f"Strong Match candidates: {strong_count} · "
        f"Competitive candidates: {competitive_count}. "
        "The ranking measures alignment with the supplied "
        "job description and is not an automatic hiring decision."
    )

    # ========================================================
    # FAILED FILES
    # ========================================================

    if failures:

        with st.expander(
            f"Files That Could Not Be Processed "
            f"({len(failures)})"
        ):

            st.dataframe(
                failures,
                use_container_width=True,
                hide_index=True,
            )
