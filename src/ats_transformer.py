from extractor import ResumeParser
from text_cleaner import TextCleaner

from sentence_transformers import SentenceTransformer

class ATSTransformer:
    def __init__(self):
        self.parser = ResumeParser()
        self.cleaned_experience = None
        self.cleaned_skills = None
        self.jd = None

    def load_data(self, resume_path, jd_path, skills_path):
        """
        Loads and processes the resume, job description, and skills data.
        
        Parameters:
        -----------
        resume_path : str
            The path to the resume file.
        jd_path : str
            The path to the job description file.
        skills_path : str
            The path to the skills file.
        """
        # Load job description
        with open(jd_path, 'r') as file:
            self.jd = file.read()

        # Load resume and skills into the parser
        self.parser.load_resume(resume_path)
        self.parser.load_job_description(jd_path)
        self.parser.load_skills(skills_path)

        # Extract experience section from the resume
        experience = self.parser.extract_experience()        
        # Clean the extracted experience text
        self.clean_experience(experience)

        # Extract skills from the resume section
        skills = " ".join(self.parser.extract_skills())
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
    # Define the file paths
    resume_path = 'resumes/output.txt'
    jd_path = 'job_descriptions/jd_1.txt'
    skills_path = 'meta/skills.txt'
    
    # Create an instance of ATS
    ats = ATSTransformer()
    
    # Load and process data
    ats.load_data(resume_path, jd_path, skills_path)
    
    # Compute and print the similarity score
    similarity_score = ats.compute_similarity()
    print(f"The similarity score between the resume and job description is: {round(similarity_score.item() * 100, 2)}%")