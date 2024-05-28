from extractor import ResumeParser
from text_cleaner import TextCleaner

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

jd_path = "job_descriptions/jd_2.txt"
with open(jd_path, 'r') as file:
    jd = file.read()

parser = ResumeParser()
parser.load_resume('resumes/resume_0.txt')
parser.load_job_description('job_descriptions/jd_2.txt')
parser.load_skills('meta/skills.txt')

experience = parser.extract_experience()
cleaned_experience = TextCleaner()
cleaned_experience = cleaned_experience.clean_text(experience)
# print(cleaned_experience)

# calculating the cosine similarity 
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([cleaned_experience, jd])

# Now you can access the TF-IDF matrix as a sparse matrix
resume_tfidf = tfidf_matrix[0]
jd_tfidf = tfidf_matrix[1]

# Calculate the cosine similarity
overall_similarity = cosine_similarity(resume_tfidf, jd_tfidf)
print(overall_similarity)