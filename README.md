## ATS (Applicant Tracking System)

This is a simple Applicant Tracking System (ATS) project built with Python. The ATS is designed to help streamline the recruitment process by matching job descriptions with candidate resumes.

### Project Structure

```
ATS
├── ats.py
├── job_description
│   ├── jd_1.txt
│   └── jd_2.txt
├── resumes
│   └── resume_0.txt
└── README.md
```

- `ats.py`: The main Python script that runs the ATS.
- `jd/`: A directory containing job description text files.
- `resumes/`: A directory containing candidate resume text files.
- `README.md`: This file, providing an overview of the project.

### Usage

1. Make sure you have Python installed on your system.
2. Clone or download this repository to your local machine.
3. Navigate to the project directory in your terminal or command prompt.
4. Run the `ats.py` script using the following command:

   ```
   python ats.py
   ```

5. The script will read the job descriptions and candidate resumes from the respective directories and perform the matching process.
6. The results of the matching process will be displayed in the terminal or command prompt.

### Adding Job Descriptions and Resumes

To add new job descriptions or candidate resumes, simply create new text files with the appropriate content and place them in the `jd/` or `resumes/` directories, respectively.

### Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License

This project is licensed under the [MIT License](LICENSE).