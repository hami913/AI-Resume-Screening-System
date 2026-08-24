import os
import re
import joblib
import numpy as np
import pandas as pd

from pypdf import PdfReader
from docx import Document


# =========================
# Paths
# =========================

MODEL_PATH = "models/best_pipeline.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# =========================
# Load trained model
# =========================

pipeline = joblib.load(MODEL_PATH)


# =========================
# Text Extraction
# =========================

def extract_text_from_pdf(file_path):
    """Extract text from a PDF resume."""
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_text_from_docx(file_path):
    """Extract text from a DOCX resume."""
    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return text.strip()


def extract_resume_text(file_path):
    """Automatically extract resume text based on file type."""

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    elif extension == ".docx":
        return extract_text_from_docx(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Please provide a PDF or DOCX resume."
        )


# =========================
# Basic Feature Extraction
# =========================

def clean_resume(text):
    """Basic resume text cleaning."""

    text = text.lower()

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(
        r"\+?\d[\d\s().-]{7,}\d",
        " ",
        text
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_skills(text):
    """Extract recognized skills from resume text."""

    skill_keywords = [
        "python",
        "java",
        "c++",
        "javascript",
        "typescript",
        "sql",
        "html",
        "css",
        "react",
        "angular",
        "node",
        "django",
        "flask",
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "data science",
        "data analysis",
        "nlp",
        "computer vision",
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "git",
        "github",
        "firebase",
        "flutter",
        "dart",
        "mongodb",
        "mysql",
        "postgresql",
        "power bi",
        "tableau",
        "excel",
        "statistics",
    ]

    text_lower = text.lower()

    return [
        skill
        for skill in skill_keywords
        if skill in text_lower
    ]


def extract_features(text):
    """Extract the numerical features used by the trained pipeline."""

    cleaned = clean_resume(text)

    words = cleaned.split()

    word_count = len(words)

    unique_word_count = len(set(words))

    resume_length = len(cleaned)

    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
    else:
        avg_word_length = 0

    email_present = int(
        bool(re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I))
    )

    phone_present = int(
        bool(re.search(r"\+?\d[\d\s().-]{7,}\d", text))
    )

    skill_count = len(extract_skills(text))

    return {
        "clean_resume": cleaned,
        "resume_length": resume_length,
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "avg_word_length": avg_word_length,
        "email_present": email_present,
        "phone_present": phone_present,
        "skill_count": skill_count,
    }



# =========================
# ATS Feature Analysis
# =========================

def analyze_ats_features(text, features):
    """Extract resume-quality features for ATS scoring."""

    text_lower = text.lower()

    section_keywords = {
        "experience": [
            "experience",
            "work experience",
            "professional experience",
            "employment history",
        ],
        "education": [
            "education",
            "academic background",
            "qualifications",
        ],
        "projects": [
            "projects",
            "personal projects",
            "academic projects",
        ],
        "certifications": [
            "certifications",
            "certificates",
            "licenses",
        ],
        "summary": [
            "summary",
            "professional summary",
            "profile",
            "objective",
        ],
    }

    sections = {}

    for section, keywords in section_keywords.items():
        sections[section] = int(
            any(keyword in text_lower for keyword in keywords)
        )

    # Resume section quality
    section_score = (
        sections["experience"] * 25
        + sections["education"] * 20
        + sections["projects"] * 20
        + sections["certifications"] * 15
        + sections["summary"] * 20
    )

    # Resume length quality.
    # Around 350-500 words is considered a strong range.
    word_count = features["word_count"]

    if 350 <= word_count <= 500:
        length_score = 100
    elif 250 <= word_count < 350:
        length_score = 80 + ((word_count - 250) / 100) * 20
    elif 500 < word_count <= 700:
        length_score = 100 - ((word_count - 500) / 200) * 20
    elif word_count < 250:
        length_score = max((word_count / 250) * 80, 0)
    else:
        length_score = max(80 - ((word_count - 700) / 500) * 80, 0)

    # Vocabulary uniqueness.
    # Used as a quality signal, but kept bounded to avoid
    # disproportionately affecting the final ATS score.
    uniqueness_ratio = (
        features["unique_word_count"] / word_count
        if word_count > 0
        else 0
    )

    uniqueness_score = min(max(uniqueness_ratio * 100, 0), 100)

    return {
        "section_score": round(section_score, 2),
        "length_score": round(length_score, 2),
        "uniqueness_score": round(uniqueness_score, 2),
        "sections": sections,
    }


# =========================
# Skill Normalization
# =========================

SKILL_ALIASES = {
    "apache spark": "spark",
    "spark framework": "spark",
    "scikit learn": "scikit-learn",
    "scikit learn library": "scikit-learn",
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "artificial-intelligence": "artificial intelligence",
    "natural language processing": "nlp",
    "postgres": "postgresql",
    "ms sql": "sql",
    "microsoft sql server": "sql",
    "github": "github",
    "git": "git",
    "sql server": "sql",
    "powerbi": "power bi",
    "problem-solving": "problem solving",
}

# Conservative related-skill relationships.
# Related matches receive less weight than exact matches.
SKILL_FAMILIES = {
    "machine learning": {"deep learning", "artificial intelligence", "tensorflow", "pytorch"},
    "deep learning": {"machine learning", "artificial intelligence", "tensorflow", "pytorch"},
    "artificial intelligence": {"machine learning", "deep learning", "tensorflow", "pytorch"},
    "data science": {"data analysis", "statistical analysis", "statistics"},
    "data analysis": {"data science", "statistical analysis", "statistics"},
    "sql": {"sql databases", "sql knowledge", "sql queries", "sql scripting"},
    "python": {"scripting languages", "programming languages"},
    "spark": {"stream analytics"},
    "cloud computing": {"cloud technologies"},
    "cloud technologies": {"cloud computing"},
    "data modeling": {"data management", "data warehousing"},
    "data warehousing": {"data management", "data modeling", "etl"},
    "git": {"version control systems"},
    "github": {"version control systems"},
}

RELATED_SKILL_WEIGHT = 0.50

def normalize_skill(skill):
    """Normalize skill names for reliable matching."""

    skill = str(skill).strip().lower()
    return SKILL_ALIASES.get(skill, skill)


def normalize_skills(skills):
    """Normalize and deduplicate a collection of skills."""

    return {
        normalize_skill(skill)
        for skill in skills
        if str(skill).strip()
    }



# ============================================================
# CLOSED-SET SAFETY / UNKNOWN CAREER DETECTION
# ============================================================

CAREER_RESUME_EVIDENCE = {
    "ADVOCATE": [
        "advocate", "attorney", "lawyer", "legal counsel",
        "litigation", "legal drafting", "corporate law",
        "criminal defense", "intellectual property",
        "bar council", "high court", "llb", "ll.b",
    ],

    "ARTS": [
        "fine arts", "artist", "graphic design",
        "illustration", "illustrator", "photoshop",
        "creative design", "typography",
    ],

    "AUTOMATION TESTING": [
        "automation testing", "test automation",
        "selenium", "cypress", "playwright",
        "testng", "junit",
    ],

    "BLOCKCHAIN": [
        "blockchain", "solidity", "ethereum",
        "smart contract", "web3", "hyperledger",
    ],

    "BUSINESS ANALYST": [
        "business analyst", "business analysis",
        "requirements gathering", "stakeholder management",
        "process mapping", "user stories",
    ],

    "CIVIL ENGINEER": [
        "civil engineer", "civil engineering",
        "site engineer", "construction",
        "structural analysis", "quantity surveying",
        "boq", "surveying", "primavera",
    ],

    "DATA SCIENCE": [
        "data scientist", "data science",
        "machine learning", "deep learning",
        "artificial intelligence", "tensorflow",
        "pytorch", "scikit-learn", "pandas",
        "natural language processing", "nlp",
    ],

    "DATABASE": [
        "database administrator", "dba",
        "database administration", "database design",
        "sql server", "oracle database",
        "mysql", "postgresql",
    ],

    "DEVOPS ENGINEER": [
        "devops", "docker", "kubernetes",
        "jenkins", "terraform", "ansible",
        "ci/cd", "continuous integration",
    ],

    "DOTNET DEVELOPER": [
        ".net developer", "dotnet developer",
        ".net", "asp.net", "c#",
    ],

    "ETL DEVELOPER": [
        "etl developer", "etl",
        "informatica", "talend", "ssis",
        "data warehousing",
    ],

    "ELECTRICAL ENGINEERING": [
        "electrical engineer", "electrical engineering",
        "plc", "scada", "power systems",
        "circuit design", "electrical design",
    ],

    "HR": [
        "human resources", "hr manager",
        "hr executive", "recruitment",
        "recruiter", "talent acquisition",
        "payroll", "employee relations",
    ],

    "HADOOP": [
        "hadoop", "apache spark",
        "hive", "hbase", "mapreduce",
    ],

    "HEALTH AND FITNESS": [
        "personal trainer", "fitness trainer",
        "fitness training", "nutrition",
        "strength training", "exercise physiology",
        "wellness",
    ],

    "JAVA DEVELOPER": [
        "java developer", "java",
        "spring boot", "hibernate",
    ],

    "MECHANICAL ENGINEER": [
        "mechanical engineer", "mechanical engineering",
        "solidworks", "catia", "creo",
        "ansys", "hvac", "mechanical design",
    ],

    "NETWORK SECURITY ENGINEER": [
        "network security", "cybersecurity",
        "cyber security", "penetration testing",
        "vulnerability assessment", "firewall",
        "siem", "wireshark", "splunk",
    ],

    "OPERATIONS MANAGER": [
        "operations manager", "operations management",
        "supply chain", "logistics",
        "inventory management", "process improvement",
        "six sigma",
    ],

    "PMO": [
        "pmo", "project management office",
        "project management", "project planning",
        "project governance", "risk management",
    ],

    "PYTHON DEVELOPER": [
        "python developer", "python",
        "django", "flask", "fastapi",
    ],

    "SAP DEVELOPER": [
        "sap developer", "sap",
        "abap", "sap hana", "s/4hana",
        "sap fico", "sap mm", "sap sd",
    ],

    "SALES": [
        "sales manager", "sales executive",
        "sales", "business development",
        "lead generation", "account management",
        "cold calling",
    ],

    "TESTING": [
        "software testing", "manual testing",
        "quality assurance", "qa engineer",
        "regression testing", "test cases",
    ],

    "WEB DESIGNING": [
        "web designer", "web design",
        "web designing", "responsive design",
        "wordpress", "html", "css",
    ],
}


def evaluate_career_prediction_reliability(
    text,
    predicted_career,
    rankings
):
    """
    Reject weak closed-set predictions.

    LinearSVC must choose one of its trained classes even when
    the resume belongs to an unseen profession. This function
    prevents that forced prediction from being presented as fact.
    """

    normalized_career = normalize_career_name(
        predicted_career
    )

    resume_text = " " + re.sub(
        r"\s+",
        " ",
        str(text).lower()
    ) + " "

    evidence_terms = CAREER_RESUME_EVIDENCE.get(
        normalized_career,
        []
    )

    matched_evidence = []

    for term in evidence_terms:

        term_lower = term.lower().strip()

        if term_lower in resume_text:
            matched_evidence.append(term)

    # Remove duplicates while preserving order.
    matched_evidence = list(
        dict.fromkeys(matched_evidence)
    )

    evidence_count = len(matched_evidence)

    if rankings:
        top_score = float(rankings[0][1])
    else:
        top_score = 0.0

    if len(rankings) >= 2:
        second_score = float(rankings[1][1])
        decision_margin = top_score - second_score
    else:
        decision_margin = 0.0

    # Strong direct profession/title evidence.
    career_title_variants = {
        "ADVOCATE": [
            "advocate", "attorney", "lawyer",
        ],
        "DATA SCIENCE": [
            "data scientist", "data science",
            "machine learning engineer",
        ],
        "CIVIL ENGINEER": [
            "civil engineer", "civil engineering",
        ],
        "MECHANICAL ENGINEER": [
            "mechanical engineer",
            "mechanical engineering",
        ],
        "ELECTRICAL ENGINEERING": [
            "electrical engineer",
            "electrical engineering",
        ],
        "DEVOPS ENGINEER": [
            "devops engineer", "devops",
        ],
        "BUSINESS ANALYST": [
            "business analyst",
        ],
        "PYTHON DEVELOPER": [
            "python developer",
        ],
        "JAVA DEVELOPER": [
            "java developer",
        ],
        "DOTNET DEVELOPER": [
            ".net developer", "dotnet developer",
        ],
        "NETWORK SECURITY ENGINEER": [
            "network security engineer",
            "cybersecurity engineer",
        ],
        "OPERATIONS MANAGER": [
            "operations manager",
        ],
        "HR": [
            "human resources", "hr manager",
        ],
        "SALES": [
            "sales manager", "sales executive",
        ],
        "WEB DESIGNING": [
            "web designer", "web designing",
        ],
    }

    direct_title_match = any(
        term in resume_text
        for term in career_title_variants.get(
            normalized_career,
            []
        )
    )

    # --------------------------------------------------------
    # RELIABILITY RULES
    # --------------------------------------------------------

    reliable = False

    # Explicit profession title is very strong evidence.
    if direct_title_match:
        reliable = True

    # Multiple domain signals plus some model separation.
    elif evidence_count >= 3:
        reliable = True

    elif (
        evidence_count >= 2
        and decision_margin >= 0.02
    ):
        reliable = True

    # One strong domain clue is accepted only when the model
    # has a clearly stronger decision margin.
    elif (
        evidence_count >= 1
        and decision_margin >= 0.12
    ):
        reliable = True

    return {
        "reliable": reliable,
        "decision_margin": round(
            decision_margin,
            6
        ),
        "top_decision_score": round(
            top_score,
            6
        ),
        "evidence_count": evidence_count,
        "matched_evidence": matched_evidence,
        "direct_title_match": direct_title_match,
    }


# =========================
# Career Prediction
# =========================

def predict_careers(text, top_n=5):
    """
    Predict career category with closed-set safety protection.

    If the trained classifier chooses a category but the resume
    does not contain enough supporting domain evidence, return
    Other / Unknown instead of forcing a wrong profession.
    """

    features = extract_features(text)
    dataframe = pd.DataFrame([features])

    raw_prediction_id = pipeline.predict(
        dataframe
    )[0]

    raw_prediction = label_encoder.inverse_transform(
        [int(raw_prediction_id)]
    )[0]

    decision_scores = pipeline.decision_function(
        dataframe
    )[0]

    class_ids = pipeline.classes_

    rankings = sorted(
        [
            (
                label_encoder.inverse_transform(
                    [int(class_id)]
                )[0],
                float(score),
            )
            for class_id, score
            in zip(class_ids, decision_scores)
        ],
        key=lambda x: x[1],
        reverse=True
    )

    reliability = (
        evaluate_career_prediction_reliability(
            text,
            raw_prediction,
            rankings,
        )
    )

    if reliability["reliable"]:
        prediction = raw_prediction
        prediction_status = "recognized"
    else:
        prediction = "Other / Unknown"
        prediction_status = "unsupported_or_uncertain"

    # Use broad skills for user-facing/job-matching logic when
    # available. Do NOT change extract_features(), because the
    # trained classifier depends on its original feature design.
    if "extract_resume_skills" in globals():
        skills = sorted(
            extract_resume_skills(text)
        )
    else:
        skills = extract_skills(text)

    if prediction == "Other / Unknown":
        job_matches = []
    else:
        job_matches = match_jobs(
            skills,
            prediction,
            top_n=top_n
        )

    ats_score = (
        calculate_ats_score(
            text,
            features,
            job_matches[0]
        )
        if job_matches
        else None
    )

    skill_gap = analyze_skill_gap(
        job_matches,
        skills
    )

    return {
        "predicted_career": prediction,

        # Keep raw model output for debugging/explainability.
        "raw_predicted_career": raw_prediction,

        "prediction_status": prediction_status,
        "prediction_reliable": reliability["reliable"],
        "decision_margin": reliability["decision_margin"],
        "top_decision_score": reliability["top_decision_score"],
        "career_evidence_count": reliability["evidence_count"],
        "career_evidence": reliability["matched_evidence"],

        "top_careers": rankings[:top_n],
        "skills": skills,
        "job_matches": job_matches,
        "ats_score": ats_score,
        "skill_gap": skill_gap,
    }

def match_jobs(resume_skills, top_n=5):
    """Match resume skills against jobs in the job dataset."""

    if not os.path.exists(JOB_DATA_PATH):
        raise FileNotFoundError(
            f"Job dataset not found: {JOB_DATA_PATH}"
        )

    jobs = pd.read_csv(JOB_DATA_PATH)

    resume_skills = normalize_skills(resume_skills)

    if not resume_skills:
        return []

    results = []

    for _, job in jobs.iterrows():

        job_skills = _parse_job_skills(job["job_skill_set"])

        if not job_skills:
            continue

        matched = set()
        missing = set()
        weighted_score = 0.0

        for job_skill in job_skills:
            if job_skill in resume_skills:
                matched.add(job_skill)
                weighted_score += 1.0
                continue

            related_resume_skill = None

            for resume_skill in resume_skills:
                if job_skill in SKILL_FAMILIES.get(resume_skill, set()):
                    related_resume_skill = resume_skill
                    break

            if related_resume_skill:
                matched.add(f"{job_skill} (~{related_resume_skill})")
                weighted_score += RELATED_SKILL_WEIGHT
            else:
                missing.add(job_skill)

        match_score = (
            weighted_score / len(job_skills) * 100
            if job_skills
            else 0
        )

        results.append({
            "job_id": job["job_id"],
            "category": job["category"],
            "job_title": job["job_title"],
            "match_score": round(match_score, 2),
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
        })

    results.sort(
        key=lambda item: (
            item["match_score"],
            len(item["matched_skills"])
        ),
        reverse=True
    )

    return results[:top_n]


# =========================
# ATS Scoring
# =========================

def calculate_ats_score(text, features, job_match):
    """Calculate a multi-factor ATS score for the best matching job."""

    skill_score = job_match["match_score"]

    contact_score = (
        features["email_present"] * 50
        + features["phone_present"] * 50
    )

    ats_features = analyze_ats_features(text, features)

    section_score = ats_features["section_score"]
    length_score = ats_features["length_score"]
    uniqueness_score = ats_features["uniqueness_score"]

    ats_score = (
        skill_score * 0.50
        + contact_score * 0.10
        + section_score * 0.15
        + length_score * 0.10
        + uniqueness_score * 0.15
    )

    return {
        "ats_score": round(min(max(ats_score, 0), 100), 2),
        "skill_coverage": round(skill_score, 2),
        "contact_score": round(contact_score, 2),
        "section_score": round(section_score, 2),
        "length_score": round(length_score, 2),
        "uniqueness_score": round(uniqueness_score, 2),
        "sections": ats_features["sections"],
        "matched_skills": job_match["matched_skills"],
        "missing_skills": job_match["missing_skills"],
    }


# =========================
# Skill Gap Analysis
# =========================

def analyze_skill_gap(job_matches, resume_skills):
    """Analyze missing skills across the user's best job matches."""

    resume_skills = normalize_skills(resume_skills)

    if not job_matches:
        return {
            "total_missing_skills": 0,
            "critical_skills": [],
            "important_skills": [],
            "other_skills": [],
        }

    skill_frequency = {}

    for job in job_matches:
        for skill in job.get("missing_skills", []):
            normalized = normalize_skill(skill)

            if normalized not in resume_skills:
                skill_frequency[normalized] = (
                    skill_frequency.get(normalized, 0) + 1
                )

    total_jobs = len(job_matches)

    critical_skills = sorted(
        [
            skill for skill, count in skill_frequency.items()
            if count >= max(1, total_jobs * 0.6)
        ],
        key=lambda skill: (-skill_frequency[skill], skill)
    )

    important_skills = sorted(
        [
            skill for skill, count in skill_frequency.items()
            if skill not in critical_skills
            and count >= max(1, total_jobs * 0.3)
        ],
        key=lambda skill: (-skill_frequency[skill], skill)
    )

    other_skills = sorted(
        [
            skill for skill in skill_frequency
            if skill not in critical_skills
            and skill not in important_skills
        ],
        key=lambda skill: (-skill_frequency[skill], skill)
    )

    return {
        "total_missing_skills": len(skill_frequency),
        "critical_skills": critical_skills,
        "important_skills": important_skills,
        "other_skills": other_skills,
    }


# =========================
# Resume Analyzer
# =========================

def analyze_resume(file_path, top_n=5):
    """Complete resume analysis pipeline."""

    print("\n" + "=" * 60)
    print("AI RESUME ANALYZER")
    print("=" * 60)

    print(f"\nFile: {file_path}")

    print("\n[1/3] Extracting resume text...")

    text = extract_resume_text(file_path)

    if not text:
        raise ValueError(
            "No text could be extracted from this resume."
        )

    print(f"Extracted characters: {len(text):,}")

    print("\n[2/3] Running career prediction...")

    result = predict_careers(text, top_n=top_n)

    print("\n[3/3] Career analysis complete.")

    print("\n" + "-" * 60)
    print("PREDICTED CAREER")
    print("-" * 60)

    print(result["predicted_career"])

    print("\n" + "-" * 60)
    print(f"TOP {top_n} CAREER MATCHES")
    print("-" * 60)

    for rank, (career, score) in enumerate(
        result["top_careers"],
        start=1
    ):
        print(f"{rank}. {career:<30} Score: {score:.4f}")

    print("\n" + "-" * 60)
    print("EXTRACTED SKILLS")
    print("-" * 60)

    print(f"Total Skills: {len(result['skills'])}")
    print(", ".join(sorted(result["skills"])))

    print("\n" + "-" * 60)
    print("ATS SCORE")
    print("-" * 60)

    if result["ats_score"]:
        ats = result["ats_score"]
        print(f"ATS Score:       {ats['ats_score']:.2f}%")
        print(f"Skill Coverage:  {ats['skill_coverage']:.2f}%")
        print(f"Contact Score:   {ats['contact_score']:.2f}%")
    else:
        print("ATS Score: N/A - no job matches found")

    print("\n" + "-" * 60)
    print("BEST JOB MATCH")
    print("-" * 60)

    if result["job_matches"]:
        best_job = result["job_matches"][0]

        print(f"Job Title:       {best_job['job_title']}")
        print(f"Category:        {best_job['category']}")
        print(f"Match Score:     {best_job['match_score']:.2f}%")

        print("\nMatched Skills:")
        print(", ".join(best_job["matched_skills"]) or "None")

        print("\nMissing Skills:")
        print(", ".join(best_job["missing_skills"]) or "None")
    else:
        print("No suitable job matches found.")

    print("\n" + "-" * 60)
    print("SKILL GAP ANALYSIS")
    print("-" * 60)

    gap = result["skill_gap"]

    print(f"Total Missing Skills: {gap['total_missing_skills']}")

    print("\nCritical Skills:")
    print(", ".join(gap["critical_skills"]) or "None")

    print("\nImportant Skills:")
    print(", ".join(gap["important_skills"]) or "None")

    print("\nOther Skills:")
    print(", ".join(gap["other_skills"]) or "None")

    print("\n" + "=" * 60)

    return result




# =========================
# Career-Aware Job Matching
# =========================

JOB_DATA_PATH = "data/all_job_post.csv"

RELATED_SKILL_WEIGHT = 0.50

GENERIC_SKILLS = {
    "python",
    "sql",
    "communication",
    "teamwork",
    "problem solving",
    "problem-solving",
    "git",
    "github",
    "excel",
}

CAREER_JOB_PROFILES = {
    "DATA SCIENCE": {
        "title": {
            "data scientist",
            "data science",
            "machine learning engineer",
            "ml engineer",
            "ai engineer",
            "data analyst",
            "data engineer",
            "analytics",
            "business intelligence",
            "nlp engineer",
            "research scientist",
            "machine learning",
            "deep learning",
        },
        "description": {
            "data science",
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "data analysis",
            "predictive modeling",
            "statistical analysis",
            "natural language processing",
            "nlp",
            "computer vision",
            "neural networks",
            "data engineering",
            "business intelligence",
            "analytics",
        },
        "categories": {
            "DATA SCIENCE",
            "DATA-SCIENCE",
            "ANALYTICS",
            "ARTIFICIAL-INTELLIGENCE",
        },
    },

    "INFORMATION TECHNOLOGY": {
        "title": {
            "software engineer",
            "software developer",
            "web developer",
            "full stack",
            "frontend developer",
            "backend developer",
            "devops",
            "cloud engineer",
            "systems administrator",
            "network engineer",
            "it specialist",
            "it support",
            "information technology",
        },
        "description": {
            "software development",
            "web development",
            "cloud computing",
            "devops",
            "networking",
            "system administration",
            "information technology",
            "cybersecurity",
            "database administration",
        },
        "categories": {
            "INFORMATION-TECHNOLOGY",
            "IT",
            "TECHNOLOGY",
        },
    },

    "FINANCE": {
        "title": {
            "financial analyst",
            "finance analyst",
            "accountant",
            "accounting",
            "financial manager",
            "finance manager",
            "investment analyst",
            "financial advisor",
            "banking",
            "auditor",
            "tax analyst",
        },
        "description": {
            "financial analysis",
            "accounting",
            "investment",
            "banking",
            "financial planning",
            "budgeting",
            "auditing",
            "taxation",
            "financial reporting",
        },
        "categories": {
            "FINANCE",
            "ACCOUNTING",
            "BANKING",
        },
    },

    "HR": {
        "title": {
            "hr",
            "human resources",
            "hr manager",
            "hr specialist",
            "recruiter",
            "recruitment",
            "talent acquisition",
            "people operations",
            "hr coordinator",
        },
        "description": {
            "human resources",
            "recruitment",
            "talent acquisition",
            "employee relations",
            "performance management",
            "payroll",
            "people operations",
            "hr management",
        },
        "categories": {
            "HR",
            "HUMAN-RESOURCES",
        },
    },

    "SALES": {
        "title": {
            "sales representative",
            "sales executive",
            "sales manager",
            "account executive",
            "sales specialist",
            "sales development representative",
            "sales consultant",
        },
        "description": {
            "sales",
            "lead generation",
            "customer acquisition",
            "sales strategy",
            "account management",
            "client relationship",
            "business development",
        },
        "categories": {
            "SALES",
        },
    },

    "BUSINESS DEVELOPMENT": {
        "title": {
            "business development",
            "business development manager",
            "business development executive",
            "business development representative",
            "account manager",
            "partnerships",
            "growth manager",
            "growth specialist",
        },
        "description": {
            "business development",
            "partnerships",
            "market expansion",
            "lead generation",
            "client acquisition",
            "business growth",
            "strategic partnerships",
        },
        "categories": {
            "BUSINESS-DEVELOPMENT",
            "BUSINESS DEVELOPMENT",
        },
    },
}


def normalize_career_name(career):
    if not career:
        return ""

    career = str(career).strip().upper()

    aliases = {
        "DATA SCIENTIST": "DATA SCIENCE",
        "DATA ANALYST": "DATA SCIENCE",
        "MACHINE LEARNING": "DATA SCIENCE",
        "MACHINE LEARNING ENGINEER": "DATA SCIENCE",
        "ML ENGINEER": "DATA SCIENCE",
        "AI ENGINEER": "DATA SCIENCE",
        "ARTIFICIAL INTELLIGENCE": "DATA SCIENCE",
        "IT": "INFORMATION TECHNOLOGY",
        "INFORMATION-TECHNOLOGY": "INFORMATION TECHNOLOGY",
        "HUMAN RESOURCES": "HR",
        "BUSINESS-DEVELOPMENT": "BUSINESS DEVELOPMENT",
    }

    return aliases.get(career, career)


def extract_resume_skills(text):
    """
    Broad multi-career skill extractor used for job matching.

    NOTE:
    extract_skills() is intentionally NOT changed because the
    trained ML classifier was trained with that original feature.
    """

    skill_groups = {

        # ---------------- LEGAL / ADVOCATE ----------------
        "litigation": [
            "litigation",
            "civil litigation",
            "corporate litigation",
            "commercial litigation",
        ],
        "legal drafting": [
            "legal drafting",
            "drafting legal",
        ],
        "corporate law": [
            "corporate law",
            "corporate legal",
        ],
        "commercial law": ["commercial law"],
        "criminal law": ["criminal law"],
        "criminal defense": [
            "criminal defense",
            "criminal defence",
        ],
        "intellectual property": [
            "intellectual property",
            "intellectual property law",
            "ip law",
        ],
        "legal research": ["legal research"],
        "statutory interpretation": [
            "statutory interpretation"
        ],
        "contract drafting": [
            "contract drafting",
            "drafting contracts",
        ],
        "contract law": ["contract law"],
        "court representation": [
            "court representation",
            "representing clients before courts",
            "court appearances",
        ],
        "arbitration": ["arbitration"],
        "mediation": ["mediation"],
        "regulatory compliance": [
            "regulatory compliance",
            "legal compliance",
        ],
        "due diligence": ["due diligence"],
        "legal counsel": ["legal counsel"],

        # ---------------- DATA / AI ----------------
        "python": ["python"],
        "sql": ["sql"],
        "machine learning": ["machine learning"],
        "deep learning": ["deep learning"],
        "artificial intelligence": [
            "artificial intelligence"
        ],
        "data science": ["data science"],
        "data analysis": [
            "data analysis",
            "data analytics",
        ],
        "nlp": [
            "nlp",
            "natural language processing",
        ],
        "computer vision": ["computer vision"],
        "statistics": [
            "statistics",
            "statistical analysis",
        ],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch"],
        "keras": ["keras"],
        "scikit-learn": [
            "scikit-learn",
            "sklearn",
        ],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "power bi": ["power bi"],
        "tableau": ["tableau"],

        # ---------------- SOFTWARE ----------------
        "java": ["java"],
        "c++": ["c++"],
        "c#": ["c#"],
        ".net": [".net", "dotnet"],
        "javascript": ["javascript"],
        "typescript": ["typescript"],
        "html": ["html"],
        "css": ["css"],
        "react": ["react"],
        "angular": ["angular"],
        "node.js": ["node.js", "nodejs"],
        "django": ["django"],
        "flask": ["flask"],
        "fastapi": ["fastapi"],
        "spring boot": ["spring boot"],
        "rest api": [
            "rest api",
            "restful api",
        ],

        # ---------------- TESTING ----------------
        "automation testing": [
            "automation testing",
            "test automation",
        ],
        "selenium": ["selenium"],
        "cypress": ["cypress"],
        "playwright": ["playwright"],
        "pytest": ["pytest"],
        "junit": ["junit"],
        "testng": ["testng"],
        "api testing": ["api testing"],
        "manual testing": ["manual testing"],
        "software testing": ["software testing"],
        "quality assurance": ["quality assurance"],
        "regression testing": ["regression testing"],

        # ---------------- DEVOPS ----------------
        "aws": ["aws", "amazon web services"],
        "azure": ["azure"],
        "gcp": ["gcp", "google cloud"],
        "docker": ["docker"],
        "kubernetes": ["kubernetes"],
        "jenkins": ["jenkins"],
        "terraform": ["terraform"],
        "ansible": ["ansible"],
        "ci/cd": ["ci/cd"],
        "linux": ["linux"],
        "git": ["git"],
        "github": ["github"],

        # ---------------- DATABASE / ETL ----------------
        "mysql": ["mysql"],
        "postgresql": ["postgresql", "postgres"],
        "mongodb": ["mongodb"],
        "oracle": ["oracle"],
        "sql server": ["sql server"],
        "database administration": [
            "database administration",
            "database administrator",
        ],
        "database design": ["database design"],
        "data warehousing": [
            "data warehousing",
            "data warehouse",
        ],
        "etl": ["etl"],
        "ssis": ["ssis"],
        "informatica": ["informatica"],
        "talend": ["talend"],

        # ---------------- HADOOP ----------------
        "hadoop": ["hadoop"],
        "spark": ["apache spark", "spark"],
        "hive": ["hive"],
        "hbase": ["hbase"],
        "mapreduce": ["mapreduce"],
        "kafka": ["kafka"],

        # ---------------- BLOCKCHAIN ----------------
        "blockchain": ["blockchain"],
        "solidity": ["solidity"],
        "ethereum": ["ethereum"],
        "smart contracts": [
            "smart contracts",
            "smart contract",
        ],
        "web3": ["web3"],
        "hyperledger": ["hyperledger"],

        # ---------------- BUSINESS ANALYST ----------------
        "business analysis": ["business analysis"],
        "requirements gathering": [
            "requirements gathering",
            "requirement gathering",
        ],
        "stakeholder management": [
            "stakeholder management"
        ],
        "process mapping": ["process mapping"],
        "gap analysis": ["gap analysis"],
        "user stories": ["user stories"],
        "jira": ["jira"],
        "confluence": ["confluence"],

        # ---------------- PMO ----------------
        "project management": ["project management"],
        "project planning": ["project planning"],
        "risk management": ["risk management"],
        "governance": ["governance"],
        "agile": ["agile"],
        "scrum": ["scrum"],

        # ---------------- OPERATIONS ----------------
        "operations management": [
            "operations management"
        ],
        "supply chain": ["supply chain"],
        "logistics": ["logistics"],
        "inventory management": [
            "inventory management"
        ],
        "process improvement": ["process improvement"],
        "six sigma": ["six sigma"],
        "vendor management": ["vendor management"],

        # ---------------- CIVIL ENGINEERING ----------------
        "autocad": ["autocad"],
        "civil 3d": ["civil 3d"],
        "primavera p6": [
            "primavera p6",
            "primavera",
        ],
        "staad pro": ["staad pro", "staad.pro"],
        "etabs": ["etabs"],
        "structural analysis": [
            "structural analysis"
        ],
        "quantity surveying": [
            "quantity surveying"
        ],
        "boq": [
            "boq",
            "bill of quantities",
        ],
        "site supervision": ["site supervision"],
        "construction management": [
            "construction management"
        ],
        "surveying": ["surveying"],
        "cost estimation": [
            "cost estimation",
            "project estimation",
        ],

        # ---------------- MECHANICAL ----------------
        "solidworks": ["solidworks"],
        "catia": ["catia"],
        "creo": ["creo"],
        "ansys": ["ansys"],
        "mechanical design": ["mechanical design"],
        "hvac": ["hvac"],
        "manufacturing": ["manufacturing"],
        "thermodynamics": ["thermodynamics"],

        # ---------------- ELECTRICAL ----------------
        "matlab": ["matlab"],
        "simulink": ["simulink"],
        "plc": ["plc"],
        "scada": ["scada"],
        "circuit design": ["circuit design"],
        "power systems": [
            "power systems",
            "power system",
        ],
        "electrical design": ["electrical design"],

        # ---------------- NETWORK SECURITY ----------------
        "cybersecurity": [
            "cybersecurity",
            "cyber security",
        ],
        "network security": ["network security"],
        "penetration testing": [
            "penetration testing",
            "pentesting",
        ],
        "vulnerability assessment": [
            "vulnerability assessment"
        ],
        "firewall": ["firewall", "firewalls"],
        "siem": ["siem"],
        "wireshark": ["wireshark"],
        "splunk": ["splunk"],
        "incident response": ["incident response"],

        # ---------------- HR ----------------
        "recruitment": [
            "recruitment",
            "recruiting",
        ],
        "talent acquisition": ["talent acquisition"],
        "onboarding": ["onboarding"],
        "payroll": ["payroll"],
        "performance management": [
            "performance management"
        ],
        "employee relations": ["employee relations"],
        "hr policies": ["hr policies"],
        "hris": ["hris"],
        "hrms": ["hrms"],

        # ---------------- SALES ----------------
        "sales": ["sales"],
        "business development": [
            "business development"
        ],
        "lead generation": ["lead generation"],
        "negotiation": ["negotiation"],
        "account management": ["account management"],
        "crm": ["crm"],
        "salesforce": ["salesforce"],
        "cold calling": ["cold calling"],

        # ---------------- SAP ----------------
        "sap": ["sap"],
        "sap hana": ["sap hana", "s/4hana"],
        "abap": ["abap"],
        "sap fico": ["sap fico"],
        "sap mm": ["sap mm"],
        "sap sd": ["sap sd"],
        "sap basis": ["sap basis"],

        # ---------------- ARTS / WEB DESIGN ----------------
        "photoshop": [
            "photoshop",
            "adobe photoshop",
        ],
        "illustrator": [
            "illustrator",
            "adobe illustrator",
        ],
        "graphic design": ["graphic design"],
        "illustration": ["illustration"],
        "typography": ["typography"],
        "photography": ["photography"],
        "video editing": ["video editing"],
        "figma": ["figma"],
        "web design": [
            "web design",
            "web designing",
        ],
        "responsive design": ["responsive design"],
        "wordpress": ["wordpress"],

        # ---------------- HEALTH / FITNESS ----------------
        "personal training": ["personal training"],
        "fitness training": ["fitness training"],
        "nutrition": ["nutrition"],
        "strength training": ["strength training"],
        "exercise physiology": [
            "exercise physiology"
        ],
        "wellness": ["wellness"],

        # ---------------- GENERAL ----------------
        "excel": [
            "excel",
            "microsoft excel",
            "ms excel",
        ],
    }

    normalized_text = " " + re.sub(
        r"\s+",
        " ",
        str(text).lower()
    ) + " "

    found = set()

    for canonical_skill, aliases in skill_groups.items():

        for alias in aliases:

            alias = alias.lower().strip()

            # Symbol-heavy skills.
            if any(
                symbol in alias
                for symbol in ["+", "#", ".", "/"]
            ):
                if alias in normalized_text:
                    found.add(canonical_skill)
                    break

                continue

            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(alias).replace(
                    r"\ ",
                    r"\s+"
                )
                + r"(?![a-z0-9])"
            )

            if re.search(pattern, normalized_text):
                found.add(canonical_skill)
                break

    return found

def parse_job_skills(value):
    """
    Safely parse job_skill_set from CSV.
    Supports common formats:
    - Python, SQL, TensorFlow
    - ['python', 'sql']
    - JSON-like lists
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    value = value.strip("[](){}")

    parts = re.split(r"[,;|]", value)

    skills = set()

    for part in parts:
        skill = part.strip().strip("'\"").lower()

        if skill:
            skills.add(skill)

    return skills


def calculate_career_relevance(
    career,
    job_title,
    job_description,
    job_category
):
    """
    Strict career relevance score from 0-100.

    Job title is the strongest signal.

    For Data Science:
        Primary role       = 100
        Secondary role     = 80
        Cross-domain role  = penalized

    This prevents titles such as:
        Finance Data Scientist
        HR Data Engineer
        IT Data Analyst

    from beating pure Data Science roles.
    """

    career = normalize_career_name(career)

    title = str(job_title or "").lower().strip()
    description = str(job_description or "").lower()
    category = str(job_category or "").upper().strip()

    if career == "DATA SCIENCE":

        # -----------------------------------------------------
        # PURE / PRIMARY DATA SCIENCE ROLES
        # -----------------------------------------------------

        primary_titles = [
            "data scientist",
            "machine learning engineer",
            "ml engineer",
            "ai engineer",
            "artificial intelligence engineer",
            "nlp engineer",
            "research scientist",
            "machine learning scientist",
            "deep learning engineer",
        ]

        # -----------------------------------------------------
        # RELATED DATA SCIENCE ROLES
        # -----------------------------------------------------

        secondary_titles = [
            "data analyst",
            "data engineer",
            "data science",
            "analytics",
            "business intelligence",
            "bi analyst",
            "machine learning",
        ]

        # -----------------------------------------------------
        # OTHER BUSINESS DOMAINS
        # -----------------------------------------------------

        cross_domain_terms = [
            "finance",
            "financial",
            "banking",
            "accounting",
            "hr",
            "human resources",
            "recruitment",
            "sales",
            "business development",
            "marketing",
            "insurance",
            "healthcare finance",
            "information technology",
            "information-technology",
            "it ",
        ]

        # -----------------------------------------------------
        # PRIMARY TITLE
        # -----------------------------------------------------

        for keyword in primary_titles:

            if keyword in title:

                # Pure Data Scientist / ML Engineer / AI Engineer.
                score = 100.0

                # Penalize cross-domain variants.
                if any(
                    term in title
                    for term in cross_domain_terms
                ):
                    score = 65.0

                return round(score, 2)

        # -----------------------------------------------------
        # SECONDARY TITLE
        # -----------------------------------------------------

        for keyword in secondary_titles:

            if keyword in title:

                # Normal Data Analyst/Data Engineer.
                score = 80.0

                # Cross-domain Data Analyst/Data Engineer
                # should be noticeably lower.
                if any(
                    term in title
                    for term in cross_domain_terms
                ):
                    score = 50.0

                return round(score, 2)

        # -----------------------------------------------------
        # DESCRIPTION-ONLY RELEVANCE
        # -----------------------------------------------------

        description_keywords = [
            "data science",
            "machine learning",
            "deep learning",
            "predictive modeling",
            "natural language processing",
            "computer vision",
            "artificial intelligence",
            "statistical modeling",
            "data analysis",
            "neural networks",
        ]

        description_matches = sum(
            1
            for keyword in description_keywords
            if keyword in description
        )

        if description_matches >= 5:
            score = 65.0
        elif description_matches >= 3:
            score = 55.0
        elif description_matches >= 2:
            score = 45.0
        elif description_matches == 1:
            score = 30.0
        else:
            score = 0.0

        # Small category support.
        if category in {
            "DATA SCIENCE",
            "DATA-SCIENCE",
            "ANALYTICS",
            "ARTIFICIAL-INTELLIGENCE",
        }:
            score += 10.0

        return round(min(score, 75.0), 2)

    # ---------------------------------------------------------
    # ADVOCATE / LEGAL
    # ---------------------------------------------------------

    if career == "ADVOCATE":

        primary_titles = [
            "advocate",
            "attorney",
            "lawyer",
            "legal counsel",
            "legal associate",
            "associate attorney",
            "senior associate attorney",
            "litigation associate",
            "litigation attorney",
            "corporate lawyer",
            "corporate attorney",
            "corporate counsel",
            "commercial lawyer",
            "criminal defense attorney",
            "criminal defence attorney",
            "criminal lawyer",
            "legal advisor",
            "legal adviser",
            "legal officer",
            "legal consultant",
            "general counsel",
            "in-house counsel",
            "ip attorney",
            "intellectual property attorney",
            "compliance counsel",
        ]

        secondary_titles = [
            "paralegal",
            "legal specialist",
            "legal analyst",
            "contract specialist",
            "contracts specialist",
            "document review",
            "compliance specialist",
        ]

        legal_description_terms = [
            "litigation",
            "legal drafting",
            "legal research",
            "corporate law",
            "commercial law",
            "criminal law",
            "criminal defense",
            "criminal defence",
            "intellectual property",
            "contract law",
            "contract drafting",
            "court proceedings",
            "court representation",
            "legal counsel",
            "legal advice",
            "statutory",
            "arbitration",
            "legal compliance",
        ]

        unrelated_title_terms = [
            "human resource",
            "human resources",
            "hr generalist",
            "recruiter",
            "recruitment",
            "business development coordinator",
            "sales manager",
            "sales executive",
            "software engineer",
            "developer",
            "data scientist",
            "data analyst",
            "machine learning",
            "civil engineer",
            "mechanical engineer",
            "electrical engineer",
        ]

        # Clearly unrelated role -> reject completely.
        if any(
            term in title
            for term in unrelated_title_terms
        ):
            return 0.0

        # Strong legal job title.
        for keyword in primary_titles:
            if keyword in title:

                score = 100.0

                # Specialized domain attorney roles remain legal,
                # but rank below a pure legal/advocate role.
                specialization_terms = [
                    "finance",
                    "financial",
                    "banking",
                    "tax",
                    "real estate",
                    "insurance",
                ]

                if any(
                    term in title
                    for term in specialization_terms
                ):
                    score = 85.0

                return score

        # Supporting legal roles.
        for keyword in secondary_titles:
            if keyword in title:

                description_matches = sum(
                    1
                    for term in legal_description_terms
                    if term in description
                )

                if description_matches >= 2:
                    return 75.0

                return 55.0

        # Description-based legal relevance.
        description_matches = sum(
            1
            for term in legal_description_terms
            if term in description
        )

        if description_matches >= 5:
            return 70.0

        if description_matches >= 3:
            return 55.0

        if description_matches >= 2:
            return 40.0

        # Category support only if category explicitly looks legal.
        if any(
            term in category
            for term in [
                "LEGAL",
                "LAW",
                "ADVOCATE",
                "ATTORNEY",
            ]
        ):
            return 45.0

        return 0.0

    # ---------------------------------------------------------
    # OTHER CAREERS
    # ---------------------------------------------------------

    profile = CAREER_JOB_PROFILES.get(career)

    if not profile:
        # Unknown/unconfigured career must not make unrelated
        # jobs look moderately relevant.
        return 0.0

    title_matches = sum(
        1
        for keyword in profile["title"]
        if keyword.lower() in title
    )

    description_matches = sum(
        1
        for keyword in profile["description"]
        if keyword.lower() in description
    )

    category_match = (
        category in profile["categories"]
    )

    if title_matches > 0:
        score = 85.0

        if description_matches >= 3:
            score += 10.0

        if category_match:
            score += 5.0

        return min(round(score, 2), 100.0)

    if description_matches >= 3:
        return 60.0

    if description_matches >= 1:
        return 40.0

    if category_match:
        return 35.0

    return 0.0

def match_jobs(
    resume_skills,
    predicted_career,
    top_n=5
):
    """
    Resume-aware and career-aware job matching.

    Matching uses:
    1. Predicted career
    2. Job title
    3. Job description
    4. Job skill-set column
    5. Skills extracted from the full job text

    Completely unrelated jobs are rejected.
    """

    if not os.path.exists(JOB_DATA_PATH):
        raise FileNotFoundError(
            f"Job dataset not found: {JOB_DATA_PATH}"
        )

    jobs = pd.read_csv(JOB_DATA_PATH)

    resume_skills = normalize_skills(
        {
            str(skill).lower().strip()
            for skill in resume_skills
            if str(skill).strip()
        }
    )

    generic_skills = (
        GENERIC_SKILLS
        if "GENERIC_SKILLS" in globals()
        else set()
    )

    skill_families = (
        SKILL_FAMILIES
        if "SKILL_FAMILIES" in globals()
        else {}
    )

    results = []

    for _, job in jobs.iterrows():

        job_title = str(
            job.get("job_title", "") or ""
        )

        job_description = str(
            job.get("job_description", "") or ""
        )

        job_category = str(
            job.get("category", "") or ""
        )

        # ====================================================
        # 1. CAREER RELEVANCE
        # ====================================================

        career_score = calculate_career_relevance(
            predicted_career,
            job_title,
            job_description,
            job_category,
        )

        # ----------------------------------------------------
        # CAREER QUALITY GATE
        # ----------------------------------------------------
        # Do not force weak jobs into Top N just to fill slots.
        #
        # Advocate/legal resumes require strong title/domain
        # evidence. It is better to return 2-3 strong legal jobs
        # than 5 jobs containing unrelated/weak recommendations.
        normalized_career = normalize_career_name(
            predicted_career
        )

        if normalized_career == "ADVOCATE":
            minimum_career_score = 70.0
        else:
            minimum_career_score = 40.0

        if career_score < minimum_career_score:
            continue

        # ====================================================
        # 2. BUILD BETTER JOB SKILL PROFILE
        # ====================================================

        csv_job_skills = normalize_skills(
            parse_job_skills(
                job.get("job_skill_set", "")
            )
        )

        # IMPORTANT:
        # Detect skills from actual title + description as well.
        contextual_job_skills = normalize_skills(
            extract_resume_skills(
                job_title
                + "\n"
                + job_description
            )
        )

        job_skills = (
            csv_job_skills
            | contextual_job_skills
        )

        matched = set()
        missing = set()

        weighted_score = 0.0

        # ====================================================
        # 3. SKILL MATCH
        # ====================================================

        for job_skill in job_skills:

            # Exact skill match
            if job_skill in resume_skills:

                matched.add(job_skill)

                if job_skill in generic_skills:
                    weighted_score += 0.25
                else:
                    weighted_score += 1.0

                continue

            # Related skill match
            related_resume_skill = None

            for resume_skill in resume_skills:

                related = skill_families.get(
                    resume_skill,
                    set()
                )

                if job_skill in related:
                    related_resume_skill = resume_skill
                    break

            if related_resume_skill:

                matched.add(
                    f"{job_skill} (~{related_resume_skill})"
                )

                weighted_score += RELATED_SKILL_WEIGHT

            else:
                missing.add(job_skill)

        skill_score = (
            weighted_score / len(job_skills) * 100
            if job_skills
            else 0.0
        )

        # ====================================================
        # 4. RESUME-SPECIFIC BONUS
        # ====================================================

        # Reward jobs where several resume skills actually
        # appear in the job text.
        job_full_text = (
            job_title
            + " "
            + job_description
        ).lower()

        resume_evidence = sum(
            1
            for skill in resume_skills
            if skill in job_full_text
        )

        evidence_score = min(
            resume_evidence * 10.0,
            100.0
        )

        # ====================================================
        # 5. FINAL SCORE
        # ====================================================

        # Career must remain strongest.
        final_score = (
            career_score * 0.60
            + skill_score * 0.25
            + evidence_score * 0.15
        )

        results.append({
            "job_id": job.get("job_id"),
            "category": job.get("category"),
            "job_title": job.get("job_title"),
            "match_score": round(final_score, 2),
            "career_score": round(career_score, 2),
            "skill_score": round(skill_score, 2),
            "evidence_score": round(evidence_score, 2),
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
        })

    results.sort(
        key=lambda item: (
            item["match_score"],
            item["career_score"],
            item["evidence_score"],
            len(item["matched_skills"]),
        ),
        reverse=True
    )

    return results[:top_n]


# Keep the original analyzer and add job recommendations
# without changing the trained classifier or ATS logic.
_original_analyze_resume = analyze_resume


def analyze_resume(file_path, top_n=5):

    result = _original_analyze_resume(
        file_path,
        top_n=top_n
    )

    resume_text = extract_resume_text(file_path)

    resume_skills = extract_resume_skills(
        resume_text
    )

    result["skills"] = sorted(resume_skills)

    result["job_matches"] = match_jobs(
        resume_skills,
        result["predicted_career"],
        top_n=5
    )

    print("\n" + "=" * 60)
    print("TOP 5 CAREER-AWARE JOB MATCHES")
    print("=" * 60)

    for rank, job in enumerate(
        result["job_matches"],
        start=1
    ):
        print("\n" + "-" * 60)
        print(f"#{rank}")
        print("Category:", job["category"])
        print("Title:", job["job_title"])
        print("Match Score:", job["match_score"])
        print("Matched Skills:", job["matched_skills"])
        print("Missing Skills:", job["missing_skills"])

    return result


# =========================
# Command Line Interface
# =========================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python resume_analyzer.py <resume.pdf>")
        print("python resume_analyzer.py <resume.docx>")
        sys.exit(1)

    resume_path = sys.argv[1]

    if not os.path.exists(resume_path):
        print(f"\nError: File not found: {resume_path}")
        sys.exit(1)

    analyze_resume(resume_path)
