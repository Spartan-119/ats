from extractor import ResumeParser
from text_cleaner import TextCleaner

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

parser = ResumeParser()
parser.load_resume('resumes/resume_0.txt')
parser.load_job_description('job_descriptions/jd_2.txt')
parser.load_skills('meta/skills.txt')

experience = parser.extract_experience()
cleaned_experience = TextCleaner(experience)
cleaned_experience = cleaned_experience.clean_text()