# 🧭 AI Career Intelligence Platform

> **Turn your resume into a measurable career strategy.**

An AI-powered career intelligence platform that analyzes resumes, evaluates job fit, identifies skill gaps, and provides personalized career guidance through a locally running **Llama 3.1 8B** career mentor.

The platform combines **NLP, machine learning, explainable AI, generative AI, and interactive analytics** into a single Streamlit application.

---

## 🚀 What It Does

Traditional resume screeners often reduce a resume to a single ATS score.

This platform goes further.

It analyzes **what is actually present in a resume, what is missing for a target role, why a job match received its score, and what the candidate should improve next.**

### Core capabilities

* 📄 **Resume Intelligence**

  * Automated resume parsing
  * NLP-based skill extraction
  * Resume structure and quality analysis
  * ATS-oriented scoring

* 🎯 **Explainable Job Matching**

  * Match resumes against target job requirements
  * Identify matched and missing skills
  * Highlight high-priority skill gaps
  * Provide evidence-based matching explanations

* 📊 **Career Analytics**

  * Resume scoring
  * Skill coverage analysis
  * Job-fit analysis
  * Evidence mapping
  * Action-oriented recommendations

* 🤖 **AI Career Mentor**

  * Locally running Llama 3.1 8B
  * Context-aware career conversations
  * Guidance grounded in the user's resume and target role
  * Personalized recommendations instead of generic career advice

* 🐳 **Docker Support**

  * Containerized Streamlit application
  * Deployment-ready configuration
  * `.dockerignore` included to keep large model artifacts outside the image

---

# 🧠 System Architecture

```text
                    ┌──────────────────────┐
                    │      User Resume     │
                    │       PDF / DOCX     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Resume Processing  │
                    │   Parsing + NLP       │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │ ATS Analysis│  │ Skill       │  │ Structure   │
       │             │  │ Extraction  │  │ Analysis    │
       └──────┬──────┘  └──────┬──────┘  └─────────────┘
              │                │
              └────────┬───────┘
                       ▼
              ┌───────────────────┐
              │ Job Fit Analysis  │
              │                   │
              │ Match + Gaps +    │
              │ Evidence          │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Career Strategy   │
              │ Recommendations   │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Llama 3.1 8B      │
              │ Career Mentor     │
              └───────────────────┘
```

---

# 🛠️ Technology Stack

### Frontend

* **Streamlit**
* Custom CSS
* Interactive visualizations

### Machine Learning / NLP

* Python
* Scikit-learn
* TF-IDF
* Linear SVM / LinearSVC
* NLP-based skill matching
* LIME explainability

### Generative AI

* **Llama 3.1 8B Instruct**
* QLoRA fine-tuning
* 4-bit NF4 quantization
* PEFT / LoRA
* Local inference through `llama.cpp`

### Deployment

* Docker
* GitHub
* Local inference architecture

---

# 🤖 Llama 3.1 Career Mentor

The platform includes a locally running **Llama 3.1 8B Instruct** model customized for career assistance.

The model was fine-tuned using **QLoRA** to make the assistant more suitable for career-related conversations.

### Fine-tuning configuration

| Parameter       | Value                      |
| --------------- | -------------------------- |
| Base Model      | Llama 3.1 8B Instruct      |
| Fine-tuning     | QLoRA                      |
| Quantization    | 4-bit NF4                  |
| LoRA Rank       | 16                         |
| LoRA Alpha      | 32                         |
| LoRA Dropout    | 0.05                       |
| Training Epochs | 3                          |
| Hardware        | NVIDIA Tesla T4            |
| Dataset         | Career instruction dataset |

The goal is to make the model behave like a **context-aware career mentor**, rather than a generic chatbot.

---

# 📊 Resume Classification & Explainability

The resume classification pipeline uses a traditional machine learning approach alongside generative AI.

The production classifier is based on a **LinearSVC pipeline**, with TF-IDF feature extraction.

### Model evaluation

One of the final evaluation runs produced:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8253 |
| Precision | 0.8833 |
| Recall    | 0.7993 |
| Macro F1  | 0.7906 |

The project also uses **LIME** to provide explainability for model predictions.

---

# 🎯 Explainable Job Fit

Instead of simply saying:

> "Your resume matches this job."

the system attempts to answer:

* Which skills match?
* Which skills are missing?
* Which missing skills have the highest priority?
* Where is evidence of a skill found in the resume?
* Why did the candidate receive the current match score?
* What should the candidate improve?

This makes the system more useful for **career decision-making and resume optimization**.

---

# 📄 Resume Analysis Pipeline

```text
Resume Upload
     │
     ▼
Document Parsing
     │
     ▼
Text Extraction
     │
     ▼
NLP Processing
     │
     ├── Skill Extraction
     │
     ├── Resume Classification
     │
     ├── ATS Analysis
     │
     └── Structural Analysis
     │
     ▼
Job Matching
     │
     ├── Matched Skills
     ├── Missing Skills
     └── Evidence
     │
     ▼
Career Recommendations
     │
     ▼
Llama Career Mentor
```

---

# 🖥️ Application

The application is built with **Streamlit** and provides an interactive career analysis experience.

Main capabilities include:

* Resume upload
* ATS scoring
* Skill analysis
* Job matching
* Skill gap analysis
* Explainable evidence
* Career recommendations
* AI career mentor
* PDF report generation

---

# 🐳 Running with Docker

Build the Docker image:

```bash
docker build -t ai-career-intelligence .
```

Run the application:

```bash
docker run -p 8501:8501 ai-career-intelligence
```

Then open:

```text
http://localhost:8501
```

> The large local Llama model is intentionally excluded from the Docker image through `.dockerignore`.

---

# ⚙️ Local Installation

Clone the repository:

```bash
git clone https://github.com/hami913/AI-Resume-Screening-System.git
cd AI-Resume-Screening-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
AI-Career-Assistant/
│
├── app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
│
├── models/
│   ├── pipeline.pkl
│   ├── resume_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── final_adapter/
│
├── src/
│   ├── preprocessing.py
│   ├── ...
│
└── ...
```

> Large model files and sensitive/local artifacts should not be committed directly to GitHub.

---

# 🔬 Machine Learning Workflow

The resume intelligence component follows a conventional supervised ML pipeline:

```text
Resume Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Evaluation
      │
      ▼
Production LinearSVC
      │
      ▼
LIME Explainability
```

The project also explored techniques including:

* GridSearchCV
* Hyperparameter tuning
* Precision / Recall / F1 evaluation
* LIME
* NLP feature extraction

---

# 🎓 Why I Built This

Resume screening and job applications often feel like a black box.

A candidate can spend hours tailoring a resume without knowing:

* Why the resume is underperforming
* Which skills are missing
* Which requirements matter most
* Whether the resume actually demonstrates a skill
* What to improve next

This project explores how **AI + NLP + explainable machine learning + local LLMs** can make that process more transparent and actionable.

---

# 🔮 Future Improvements

The platform is still actively under development.

Planned improvements include:

* 🔹 Better semantic job matching
* 🔹 More advanced resume section analysis
* 🔹 Improved evidence extraction
* 🔹 Job-description intelligence
* 🔹 Career roadmap generation
* 🔹 Resume rewriting recommendations
* 🔹 More robust LLM evaluation
* 🔹 Production cloud deployment
* 🔹 Improved model serving and inference optimization
* 🔹 Automated resume-to-job optimization

---

# 🚧 Project Status

**Active Development**

This project is continuously evolving as I experiment with:

* Machine learning
* NLP
* Explainable AI
* Generative AI
* Llama fine-tuning
* Local LLM inference
* AI-powered career intelligence

The current version is functional, but the goal is to continue turning it into a more complete **AI Career Intelligence Platform**.

---

# 👨‍💻 Author

**Hamza Ahmad**

BS Artificial Intelligence Student

Interested in:

* Artificial Intelligence
* Machine Learning
* Generative AI
* NLP
* LLM Fine-tuning
* Explainable AI
* AI Product Development

---

# ⭐ If You Find This Interesting

Feel free to explore the project, experiment with the architecture, and share feedback.

If you find the project useful or interesting, consider giving the repository a ⭐.

---

## 📜 License

This project is intended primarily for educational, research, and portfolio purposes.
