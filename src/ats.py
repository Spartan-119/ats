# shit module I suppose?
import re
import spacy
from sentence_transformers import SentenceTransformer
from text_cleaner import TextCleaner

class ATS:
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
        Initializes the ATS.
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.resume_content = None
        self.jd_content = None
        self.cleaned_experience = None
        self.cleaned_skills = None

    def load_resume(self, resume_content):
        """
        Loads the resume content from a string.

        :param resume_content: Resume content as a string
        """
        self.resume_content = resume_content

    def load_job_description(self, jd_content):
        """
        Loads the job description content from a string.

        :param jd_content: Job description content as a string
        """
        self.jd_content = jd_content

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

    def clean_experience(self, experience):
        """
        Cleans the extracted experience text from the resume.
        
        Parameters:
        -----------
        experience : str
            The raw experience text extracted from the resume.
        """
        cleaner = TextCleaner()
        self.cleaned_experience = cleaner.clean_text(experience)

    def clean_skills(self, skills):
        """
        Cleans the extracted skills text from the resume.
        
        Parameters:
        -----------
        skills : str
            The raw skills text extracted from the resume.
        """
        cleaner = TextCleaner()
        self.cleaned_skills = cleaner.clean_text(skills)

    def clean_jd(self):
        """
        Cleans the job description text by applying text cleaning techniques.

        Returns:
            str: The cleaned job description text.
        """
        cleaner = TextCleaner()
        cleaned_jd = cleaner.clean_text(self.jd_content)
        return cleaned_jd

    def compute_similarity(self):
        """
        Computes the similarity score between the cleaned resume and cleaned job description text using the SentenceTransformer model.

        Returns:
            float: The similarity score between the cleaned resume and cleaned job description text.
        """
        model = SentenceTransformer('all-MiniLM-L6-v2')
        cleaned_resume = self.cleaned_experience + self.cleaned_skills
        cleaned_jd_text = self.clean_jd()
        sentences = [cleaned_resume, cleaned_jd_text]
        embeddings1 = model.encode(sentences[0])
        embeddings2 = model.encode(sentences[1])
        
        similarity_score = model.similarity(embeddings1, embeddings2)

        return similarity_score

if __name__ == "__main__":
    # Get user input for resume and job description
    resume_content = input("Please enter the resume content: ")
    jd_content = input("Please enter the job description content: ")
    
    # Create an instance of ATS
    ats = ATS()
    
    # Load and process data
    ats.load_resume(resume_content)
    ats.load_job_description(jd_content)
    
    # Extract and clean experience
    experience = ats.extract_experience()
    ats.clean_experience(experience)
    
    # Extract and clean skills
    skills = " ".join(ats.extract_skills())
    ats.clean_skills(skills)
    
    # Compute and print the similarity score
    similarity_score = ats.compute_similarity()
    print(f"The similarity score between the resume and job description is: {round(similarity_score.item() * 100, 2)}%")