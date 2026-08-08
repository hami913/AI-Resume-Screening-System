

def map_resume_category(resume_category):

    category = resume_category.lower()


    if any(
        keyword in category
        for keyword in [
            "data",
            "python",
            "machine",
            "ai",
            "software",
            "developer",
            "web",
            "java",
            "testing"
        ]
    ):
        return "INFORMATION-TECHNOLOGY"


    elif any(
        keyword in category
        for keyword in [
            "hr",
            "human",
            "recruit"
        ]
    ):
        return "HR"



    elif any(
        keyword in category
        for keyword in [
            "finance",
            "account",
            "bank"
        ]
    ):
        return "FINANCE"



    elif any(
        keyword in category
        for keyword in [
            "sales",
            "marketing",
            "business"
        ]
    ):
        return "BUSINESS-DEVELOPMENT"



    else:
        return "INFORMATION-TECHNOLOGY"
