import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeParser:
    """
    A class to parse resumes and job descriptions, extract relevant information,
    and compute similarities between resumes and job descriptions.
    """
    
    RESUME_SECTIONS = [
        "Contact Information", "Objective", "Summary", "Education", "Experience", 
        "Skills", "Projects", "Certifications", "Licenses", "Awards", "Honors", 
        "Publications", "References", "Technical Skills", "Computer Skills", 
        "Programming Languages", "Software Skills", "Soft Skills", "Language Skills", 
        "Professional Skills", "Transferable Skills", "Work Experience", 
        "Professional Experience", "Employment History", "Internship Experience", 
        "Volunteer Experience", "Leadership Experience", "Research Experience", 
        "Teaching Experience",
    ]

    def __init__(self):
        """
        Initializes the ResumeParser.
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.resume_content = None
        self.jd_content = None
        self.skills = None

    def load_resume(self, resume_path):
        """
        Loads the resume content from a file.

        :param resume_path: Path to the resume text file
        """
        self.resume_content = self._read_file(resume_path)

    def load_job_description(self, jd_path):
        """
        Loads the job description content from a file.

        :param jd_path: Path to the job description text file
        """
        self.jd_content = self._read_file(jd_path)

    def load_skills(self, skills_path):
        """
        Loads the skills from a file.

        :param skills_path: Path to the skills text file
        """
        self.skills = self._read_skills(skills_path)


    @staticmethod
    def _read_file(file_path):
        """
        Reads the content of a text file.

        :param file_path: Path to the file
        :return: Content of the file as a string
        """
        with open(file_path, 'r') as file:
            return file.read()

    @staticmethod
    def _read_skills(skills_path):
        """
        Reads the skills from a text file, each skill on a new line.

        :param skills_path: Path to the skills file
        :return: List of skills
        """
        with open(skills_path, 'r') as file:
            return file.read().split('\n')

    def get_cosine_similarity(self):
        """
        Computes the cosine similarity between the resume content and the job description content.

        :return: Cosine similarity score
        """
        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform([self.resume_content, self.jd_content])
        return cosine_similarity(sparse_matrix, dense_output=False)[0, 1]

    def extract_skills(self):
        """
        Extracts skills from the resume content.

        :return: List of extracted skills
        """
        skills_pattern = re.compile(r'Skills\s*[:\n]', re.IGNORECASE)
        skills_match = skills_pattern.search(self.resume_content)

        if skills_match:
            skills_start = skills_match.end()
            skills_end = self.resume_content.find('\n\n', skills_start)
            skills_section = self.resume_content[skills_start:skills_end].strip()
            skills_lines = skills_section.split('\n')

            extracted_skills = []
            for line in skills_lines:
                line_skills = re.split(r'[:,-]', line)
                extracted_skills.extend([skill.strip() for skill in line_skills if skill.strip()])

            return list(set(extracted_skills))
        else:
            return []

    def get_common_skills(self):
        """
        Finds the common skills between the job description and the resume.

        :return: Set of common skills
        """
        jd_skills = set(self.jd_content.split())
        resume_skills = set(self.extract_skills())
        return jd_skills.intersection(resume_skills)

    def extract_names(self):
        """
        Extracts names from the resume content using named entity recognition.

        :return: List of names found in the resume
        """
        doc = self.nlp(self.resume_content)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        return names

    def extract_emails(self):
        """
        Extracts email addresses from the resume content using regex.

        :return: List of email addresses
        """
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        emails = re.findall(email_pattern, self.resume_content)
        return emails

    def extract_phone_numbers(self):
        """
        Extracts phone numbers from the resume content using regex.

        :return: List of phone numbers
        """
        pattern = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
        phone_numbers = re.findall(pattern, self.resume_content)
        return phone_numbers

    def extract_experience(self):
        """
        Extracts the work experience section from the resume content.

        :return: Experience section as a string
        """
        pattern = r'Experience\s*(?:\r?\n\s*)*(.+?)(?=\r?\n\s*(?:{}))'.format('|'.join(self.RESUME_SECTIONS))
        match = re.search(pattern, self.resume_content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            return ''