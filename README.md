
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumes = [
    "Python machine learning deep learning data science SQL NLP",
    "Java spring boot backend developer microservices SQL",
    "Python computer vision OpenCV machine learning AI",
    "Frontend HTML CSS JavaScript React"
]

candidate_names = ["Sai", "Rahul", "Anjali", "Kiran"]

job_description = """
Looking for Python developer with machine learning,
deep learning, NLP, SQL and AI skills
"""

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(resumes + [job_description])

similarity = cosine_similarity(vectors[-1], vectors[:-1])

scores = similarity[0]

results = pd.DataFrame({
    'Candidate': candidate_names,
    'Match Score': scores
})

results = results.sort_values(by='Match Score', ascending=False)

print(results)
