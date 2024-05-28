from extractor import ResumeParser
from text_cleaner import TextCleaner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ATS:
    """
    A class to match a resume with a job description using TF-IDF vectorization and cosine similarity.
    
    Attributes:
    -----------
    parser : ResumeParser
        An instance of the ResumeParser to handle resume and job description parsing.
    cleaned_experience : str
        The cleaned experience text extracted from the resume.
    jd : str
        The job description text.
        
    Methods:
    --------
    load_data(self, resume_path, jd_path, skills_path):
        Loads and processes the resume, job description, and skills data.
        
    clean_experience(self, experience):
        Cleans the extracted experience text from the resume.
        
    compute_similarity(self):
        Computes the cosine similarity between the cleaned resume experience and the job description.
    """
    
    def __init__(self):
        """
        Initializes the ATS with necessary attributes.
        """
        self.parser = ResumeParser()
        self.cleaned_experience = None
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

    def compute_similarity(self):
        """
        Computes the cosine similarity between the cleaned resume experience and the job description.
        
        Returns:
        --------
        float
            The cosine similarity score between the resume experience and the job description.
        """
        # Initialize the TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        
        # Fit and transform the cleaned experience and job description texts
        tfidf_matrix = vectorizer.fit_transform([self.cleaned_experience, self.jd])
        
        # Extract the TF-IDF vectors for the resume experience and job description
        resume_tfidf = tfidf_matrix[0]
        jd_tfidf = tfidf_matrix[1]
        
        # Calculate the cosine similarity between the resume experience and job description
        similarity_score = cosine_similarity(resume_tfidf, jd_tfidf)[0][0]
        
        return similarity_score


# Example usage:
if __name__ == "__main__":
    # Define the file paths
    resume_path = 'resumes/resume_0.txt'
    jd_path = 'job_descriptions/jd_2.txt'
    skills_path = 'meta/skills.txt'
    
    # Create an instance of ATS
    ats = ATS()
    
    # Load and process data
    ats.load_data(resume_path, jd_path, skills_path)
    
    # Compute and print the similarity score
    similarity_score = ats.compute_similarity()
    print(f"The similarity score between the resume and job description is: {similarity_score}")