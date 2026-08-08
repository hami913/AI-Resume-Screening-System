
import re
import numpy as np
import pandas as pd

from src.preprocessing import clean_resume


def create_resume_features(text):
    """
    Generate features required by resume classifier.
    """

    features = {}

    # Clean text
    cleaned = clean_resume(text)

    features["clean_resume"] = cleaned


    # Length features
    features["resume_length"] = len(text)

    words = cleaned.split()

    features["word_count"] = len(words)

    features["unique_word_count"] = len(
        set(words)
    )


    if len(words) > 0:
        features["avg_word_length"] = np.mean(
            [
                len(word)
                for word in words
            ]
        )
    else:
        features["avg_word_length"] = 0



    # Original text based features

    features["email_present"] = int(
        bool(
            re.search(
                r"\S+@\S+",
                text
            )
        )
    )


    features["phone_present"] = int(
        bool(
            re.search(
                r"\b\d{10}\b",
                text
            )
        )
    )



    # Basic skill extraction

    skills = [
        "python",
        "sql",
        "machine learning",
        "java",
        "tensorflow",
        "pytorch",
        "aws",
        "docker"
    ]


    lower_text = text.lower()

    features["skill_count"] = sum(
        skill in lower_text
        for skill in skills
    )


    return pd.DataFrame(
        [features]
    )
