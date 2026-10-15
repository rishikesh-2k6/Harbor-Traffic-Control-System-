"""
Script to generate the complete, enhanced, and professionally formatted
Revised Project Report for 22AIE441 Dynamics and Control of Robotics.

Generates:
1. results/REVISED_PROJECT_REPORT.md (Markdown)
2. results/REVISED_PROJECT_REPORT.docx (Word document with styled tables and embedded figures)
3. results/REVISED_PROJECT_REPORT.pdf (PDF via ReportLab)
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

OUTPUT_MD = os.path.join(BASE_DIR, "REVISED_PROJECT_REPORT.md")
OUTPUT_DOCX = os.path.join(BASE_DIR, "REVISED_PROJECT_REPORT.docx")
OUTPUT_PDF = os.path.join(BASE_DIR, "REVISED_PROJECT_REPORT.pdf")

# Also save copies to PROJECT_DIR for convenience
ROOT_MD = os.path.join(PROJECT_DIR, "REVISED_PROJECT_REPORT.md")
ROOT_DOCX = os.path.join(PROJECT_DIR, "REVISED_PROJECT_REPORT.docx")

print(f"Target Base Directory: {BASE_DIR}")
