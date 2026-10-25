from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import os

# IT Security Policy PDF
doc = SimpleDocTemplate("data/raw/it_security_policy.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []
story.append(Paragraph("INFORMATION TECHNOLOGY SECURITY POLICY", styles['Title']))
story.append(Paragraph("HCLTech Career Shaper Programme", styles['Heading2']))
story.append(Spacer(1, 12))
story.append(Paragraph("Effective Date: January 2024", styles['Normal']))
story.append(Paragraph("Version: 1.0", styles['Normal']))
story.append(Spacer(1, 24))

with open("data/raw/it_security_policy.txt", "r") as f:
    content = f.read()

for line in content.split("\n"):
    if line.strip():
        story.append(Paragraph(line.strip(), styles['Normal']))
    else:
        story.append(Spacer(1, 6))

doc.build(story)
print("Created it_security_policy.pdf")

# Data Privacy Policy PDF
doc = SimpleDocTemplate("data/raw/data_privacy_policy.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []
story.append(Paragraph("DATA PRIVACY / DPDP POLICY", styles['Title']))
story.append(Paragraph("HCLTech Career Shaper Programme", styles['Heading2']))
story.append(Spacer(1, 12))
story.append(Paragraph("Effective Date: January 2024", styles['Normal']))
story.append(Paragraph("Version: 1.0", styles['Normal']))
story.append(Spacer(1, 24))

with open("data/raw/data_privacy_policy.txt", "r") as f:
    content = f.read()

for line in content.split("\n"):
    if line.strip():
        story.append(Paragraph(line.strip(), styles['Normal']))
    else:
        story.append(Spacer(1, 6))

doc.build(story)
print("Created data_privacy_policy.pdf")

# WFH Policy PDF
doc = SimpleDocTemplate("data/raw/wfh_policy.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []
story.append(Paragraph("WORK FROM HOME POLICY", styles['Title']))
story.append(Paragraph("HCLTech Career Shaper Programme", styles['Heading2']))
story.append(Spacer(1, 12))
story.append(Paragraph("Effective Date: January 2024", styles['Normal']))
story.append(Paragraph("Version: 1.0", styles['Normal']))
story.append(Spacer(1, 24))

with open("data/raw/wfh_policy.txt", "r") as f:
    content = f.read()

for line in content.split("\n"):
    if line.strip():
        story.append(Paragraph(line.strip(), styles['Normal']))
    else:
        story.append(Spacer(1, 6))

doc.build(story)
print("Created wfh_policy.pdf")

print("All PDFs created successfully!")
