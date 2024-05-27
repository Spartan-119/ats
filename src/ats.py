import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reading resume_0.txt
with open('resumes/resume_0.txt', 'r') as f:
    resume_content = f.read()

# Reading jd_1.txt 
with open('job_descriptions/jd_2.txt', 'r') as f:
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

def extract_skills(resume_text):
    # Find the "Skills" heading
    skills_pattern = re.compile(r'Skills\s*[:\n]', re.IGNORECASE)
    skills_match = skills_pattern.search(resume_text)

    if skills_match:
        # Extract the skills section
        skills_start = skills_match.end()
        skills_end = resume_text.find('\n\n', skills_start)
        skills_section = resume_text[skills_start:skills_end].strip()

        # Split the skills section into lines
        skills_lines = skills_section.split('\n')

        # Extract skills from each line
        skills = []
        for line in skills_lines:
            line_skills = re.split(r'[:,-]', line)
            skills.extend([skill.strip() for skill in line_skills if skill.strip()])

        return list(set(skills))
    else:
        return []

# method to get the common skills from the resume and the JD
def get_common_skills(jd, resume):
    jd = set(jd.split())
    resume = set(resume)
    return jd.intersection(resume)

# printing the extracted skills
# print(extract_skills(resume_content))

# # printing the commong skills
# print(get_common_skills(jd_content, extract_skills(resume_content)))

###########################################

# method to extract names
def extract_names(resume_content):
    names = [ent.text for ent in resume_content if resume_content.label_ == "PERSON"]
    return names

# method to extract emails using regex
def extract_emails(resume_content):
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    emails = re.findall(email_pattern, resume_content)
    return emails

# method to extract phone numbers using regex
def extract_phone_numbers(resume_content):
    pattern = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    phone_numbers = re.findall(pattern, resume_content)
    return phone_numbers

# method to extract work experience
def extract_experience(resume_content):
     # Define the pattern to match the experience section
    pattern = r'Experience\s*(?:\r?\n\s*)*(.+?)(?:\r?\n\s*\r?\n|$)'
    
    # Search for the experience section
    match = re.search(pattern, resume_content, re.DOTALL)
    
    if match:
        # Return the matched experience section
        return match.group(1).strip()
    else:
        # Return an empty string if experience section not found
        return ''

print(extract_experience(resume_content))