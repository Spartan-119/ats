import PyPDF2
import re

# Open the PDF file
pdf_file = open('resumes/Abin Varghese Resume.pdf', 'rb')

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

# Remove special characters
cleaned_text = re.sub(r'[^\w\s]', '', text)

# Close the PDF file
pdf_file.close()

# Save the cleaned text to a file
with open('resumes/output.txt', 'w', encoding='utf-8') as text_file:
    text_file.write(cleaned_text)

print('PDF converted to text and saved as output.txt')