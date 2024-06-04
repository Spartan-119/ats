"""
Module to convert PDF --> TXT while preserving only lowercased alphanumeric characters.
"""

import PyPDF2
import re
import string

# Open the PDF file
pdf_file = open('resumes/mradul_cv.pdf', 'rb')

# Create a PDF reader object
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Get the number of pages in the PDF
num_pages = len(pdf_reader.pages)

# Initialize an empty string to store the text
text = ''

# Loop through each page and extract the text
for page_num in range(num_pages):
    page = pdf_reader.pages[page_num]
    text += page.extract_text()

# Close the PDF file
pdf_file.close()

# Remove non-alphanumeric characters and convert to lowercase
cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

# Split text into lines and remove empty lines
lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]

# Join lines back into paragraphs
paragraphs = []
current_paragraph = []
for line in lines:
    if line:
        current_paragraph.append(line)
    else:
        paragraphs.append(' '.join(current_paragraph))
        current_paragraph = []

# Join remaining lines in the last paragraph
if current_paragraph:
    paragraphs.append(' '.join(current_paragraph))

# Write the cleaned text to a text file
with open('resumes/mradul_text.txt', 'w', encoding='utf-8') as text_file:
    text_file.write('\n\n'.join(paragraphs))

print('PDF converted to text file successfully!')