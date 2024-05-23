import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reading resume_0.txt
with open('resumes/resume_0.txt', 'r') as f:
    resume_content = f.read()

# Reading jd_1.txt 
with open('job_descriptions/jd_1.txt', 'r') as f:
    jd_content = f.read()

# reading the skills
with open('meta/skills.txt', 'r') as f:
    skills = f.read().split('\n')


def get_cosine_similarity(resume_content, jd_content):
    # Create the Document Term Matrix
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform([resume_content, jd_content])

    # Getting the cosine similarity
    return cosine_similarity(sparse_matrix, dense_output=False)[0, 1]


nlp = spacy.load('en_core_web_sm')


def extract_skills(resume_content):
    resume = resume_content.lower()
    skills = [skill.lower() for skill in skills]

    resume = set(resume.split())
    skills = set(skills)

    return resume.intersection(skills)

# printing the extracted skills
print(extract_skills(resume_content))