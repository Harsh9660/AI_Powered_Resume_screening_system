import requests
import os
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="John Doe - Resume", ln=1, align='C')
pdf.cell(200, 10, txt="Skills: Python, Machine Learning, Docker, SQL", ln=2, align='L')
pdf.cell(200, 10, txt="Experience: 5 years as a Data Scientist at TechCorp.", ln=3, align='L')
pdf.output("data/raw/test_resume.pdf")
print("Sample PDF created at data/raw/test_resume.pdf")
print("To test the API manually:")
print("1. Run: export PYTHONPATH=$PYTHONPATH:. && python3 src/api/main.py")
print("2. Run: curl -X POST 'http://localhost:8000/rank-resumes' -F 'job_description=Looking for a Data Scientist with Python and Docker' -F 'resumes=@data/raw/test_resume.pdf'")
