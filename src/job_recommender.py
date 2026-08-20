
import os
import ast
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.category_mapping import map_resume_category



BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


JOB_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "all_job_post.csv"
)



def load_jobs():

    return pd.read_csv(
        JOB_DATA_PATH
    )



def parse_job_skills(value):
    try:
        skills = ast.literal_eval(str(value))
        if isinstance(skills, list):
            return [str(skill).strip() for skill in skills if str(skill).strip()]
    except (ValueError, SyntaxError):
        pass
    return []


def find_matched_skills(resume_text, job_skills):
    resume_lower = str(resume_text).lower()
    return [skill for skill in job_skills if skill.lower() in resume_lower]


def prepare_job_text(row):

    return (
        str(row["job_title"])
        + " "
        + str(row["job_description"])
        + " "
        + str(row["job_skill_set"])
    )



def recommend_jobs(
    resume_text,
    resume_category,
    top_n=5
):

    jobs = load_jobs()


    job_category = map_resume_category(
        resume_category
    )


    filtered_jobs = jobs[
        jobs["category"] == job_category
    ]


    if len(filtered_jobs) == 0:
        filtered_jobs = jobs



    filtered_jobs = filtered_jobs.copy()

    filtered_jobs["parsed_skills"] = filtered_jobs["job_skill_set"].apply(
        parse_job_skills
    )

    filtered_jobs["matched_skills"] = filtered_jobs["parsed_skills"].apply(
        lambda skills: find_matched_skills(resume_text, skills)
    )


    filtered_jobs["combined_text"] = (
        filtered_jobs.apply(
            prepare_job_text,
            axis=1
        )
    )


    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )


    job_vectors = vectorizer.fit_transform(
        filtered_jobs["combined_text"]
    )


    resume_vector = vectorizer.transform(
        [resume_text]
    )


    scores = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]


    filtered_jobs["similarity_score"] = scores



    return filtered_jobs.sort_values(
        "similarity_score",
        ascending=False
    ).head(top_n)[
        [
            "job_id",
            "category",
            "job_title",
            "similarity_score",
            "matched_skills"
        ]
    ]
