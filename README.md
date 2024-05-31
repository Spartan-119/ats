## ATS (Applicant Tracking System)

This is a simple Applicant Tracking System (ATS) project built with Python. The ATS is designed to help streamline the recruitment process by matching job descriptions with candidate resumes.

![ATS_Thumbnail](meta/This-is-my-resume-gif.gif)

### Project Structure

```
ATS
├── src
│   ├── __init__.py
│   ├── ats_tfidf.py
│   ├── ats_transformer.py
|   ├── extractor.py
|   ├── pdf2txt.py
│   └── text_cleaner.py
├── job_description
│   ├── contains job descriptions # .gitignore
├── resumes
│   └── contains resumes # .gitignore
└── README.md
```

- `src`: Contains the main src code.
- `job_description/`: A directory containing job description text files.
- `resumes/`: A directory containing candidate resume text files.
- `README.md`: This file, providing an overview of the project.

### Usage

1. Make sure you have Python installed on your system.
2. Run the `ats_transformer.py` (for higher precision) script using the following command on Windows:

   ```
   git clone https://github.com/Spartan-119/ats.git
   cd ats
   python -m venv ats_venv
   ats_venv\Scripts\activate
   pip install -r requirements.txt
   python src/ats_transformer.py
   ```
or on Linux or MacOS:

   ```
      git clone https://github.com/Spartan-119/ats.git
      cd ats
      python -m venv ats_venv
      source ats_venv/bin/activate
      pip install -r requirements.txt
      python src/ats_transformer.py
   ```
5. The script will read the job descriptions and candidate resumes from the respective directories and perform the matching process.
6. The results of the matching process will be displayed in the terminal or command prompt.

### Adding Job Descriptions and Resumes

To add new job descriptions or candidate resumes, simply create new text files with the appropriate content and place them in the `job_description/` or `resumes/` directories, respectively.

### Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
