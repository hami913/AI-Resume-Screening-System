import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources once at setup/initialization
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_resume(text):
    """
    Clean resume text for NLP preprocessing.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

    # Remove common phone number patterns (including +/()- formatting)
    text = re.sub(r"\(?\+\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}", " ", text)

    # Safe HTML tag removal
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode()

    # Preserve programming terms like c++, c#, .net before removing other punctuation
    text = re.sub(r"\bc\+\+", "cpp", text)
    text = re.sub(r"\bc\#", "csharp", text)
    text = re.sub(r"\b\.net\b", "dotnet", text)

    # Remove remaining punctuation and digits
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word) 
        for word in words 
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(cleaned_words)