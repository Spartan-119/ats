# shit module?
import re
from extractor import ResumeParser
from text_cleaner import TextCleaner
from sentence_transformers import SentenceTransformer

class ATS:
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
        self.parser = ResumeParser()
        self.cleaned_experience = None
        self.cleaned_skills = None
        self.jd = None

    def extract_skills(self, resume_content):
        """
        Extracts skills from the resume content.

        :return: List of extracted skills
        """
        skills_pattern = re.compile(r'Skills\s*[:\n]', re.IGNORECASE)
        skills_match = skills_pattern.search(resume_content)

        if skills_match:
            skills_start = skills_match.end()
            skills_end = resume_content.find('\n\n', skills_start)
            skills_section = resume_content[skills_start:skills_end].strip()
            skills_lines = skills_section.split('\n')

            extracted_skills = []
            for line in skills_lines:
                line_skills = re.split(r'[:,-]', line)
                extracted_skills.extend([skill.strip() for skill in line_skills if skill.strip()])

            return list(set(extracted_skills))
        else:
            return []
        
    def extract_experience(self, resume_content):
        """
        Extracts the work experience section from the resume content.

        :return: Experience section as a string
        """
        pattern = r'Experience\s*(?:\r?\n\s*)*(.+?)(?=\r?\n\s*(?:{}))'.format('|'.join(self.RESUME_SECTIONS))
        match = re.search(pattern, resume_content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            return ''

    def load_data(self, resume_text, jd_text, skills_path):
        """
        Loads and processes the resume, job description, and skills data.
        
        Parameters:
        -----------
        resume_text : str
            The resume text.
        jd_text : str
            The job description text.
        skills_path : str
            The path to the skills file.
        """
        # Load job description
        self.jd = jd_text
        resume = resume_text

        self.parser.load_skills(skills_path)

        # Extract experience section from the resume
        experience = self.extract_experience(resume)        
        # Clean the extracted experience text
        self.clean_experience(experience)

        # Extract skills from the resume section
        skills = " ".join(self.extract_skills(resume))
        # clean the extracted skills text
        self.clean_skills(skills)

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
    
    def clean_jd(self):
        """
        Cleans the job description text by applying text cleaning techniques.

        Returns:
            str: The cleaned job description text.
        """
        cleaner = TextCleaner()
        cleaned_jd = cleaner.clean_text(self.jd)
        return cleaned_jd


# Example usage:
if __name__ == "__main__":
    # Get resume text from the user
    resume_text = input("Enter your resume text: ")

    # Get job description text from the user
    jd_text = input("Enter the job description text: ")

    # Define the skills file path
    skills_path = 'meta/skills.txt'
    
    # Create an instance of ATS
    ats = ATS()
    
    # Load and process data
    ats.load_data(resume_text, jd_text, skills_path)
    
    # Compute and print the similarity score
    similarity_score = ats.compute_similarity()
    print(f"The similarity score between the resume and job description is: {round(similarity_score.item() * 100, 2)}%")