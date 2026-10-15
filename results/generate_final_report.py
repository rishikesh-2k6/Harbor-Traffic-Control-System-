"""
Generate the complete revised project report in:
1. results/REVISED_PROJECT_REPORT.md
2. results/REVISED_PROJECT_REPORT.docx (Word Document)
3. results/REVISED_PROJECT_REPORT.pdf (PDF via Microsoft Word dispatch)
"""

import os
import sys
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

OUTPUT_MD = os.path.join(BASE_DIR, "REVISED_PROJECT_REPORT.md")
OUTPUT_DOCX = os.path.join(BASE_DIR, "REVISED_PROJECT_REPORT.docx")
OUTPUT_PDF = os.path.join(BASE_DIR, "REVISED_PROJECT_REPORT.pdf")

ROOT_MD = os.path.join(PROJECT_DIR, "REVISED_PROJECT_REPORT.md")
ROOT_DOCX = os.path.join(PROJECT_DIR, "REVISED_PROJECT_REPORT.docx")
ROOT_PDF = os.path.join(PROJECT_DIR, "REVISED_PROJECT_REPORT.pdf")

# Styling Colors (Matching Amrita & Project Palette)
COLOR_MAROON = RGBColor(160, 28, 28)
COLOR_NAVY = RGBColor(15, 32, 67)
COLOR_TEXT_DARK = RGBColor(34, 34, 34)
COLOR_MUTED = RGBColor(85, 85, 85)
COLOR_BLUE = RGBColor(37, 99, 235)
HEX_HEADER_BG = "0F2043"
HEX_ROW_ALT = "F8FAFC"
HEX_BORDER = "CBD5E1"


def set_cell_background(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def set_table_borders(table):
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(f'''
            <w:tblBorders {nsdecls("w")}>
                <w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>
                <w:bottom w:val="single" w:sz="6" w:space="0" w:color="{HEX_BORDER}"/>
                <w:left w:val="none"/>
                <w:right w:val="none"/>
                <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>
                <w:insideV w:val="none"/>
            </w:tblBorders>
        ''')
        tblPr[0].append(borders)


def format_table_header(row, col_widths=None):
    for idx, cell in enumerate(row.cells):
        set_cell_background(cell, HEX_HEADER_BG)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(9.5)
                run.font.name = "Arial"
        if col_widths and idx < len(col_widths):
            cell.width = Inches(col_widths[idx])


def format_table_rows(table, col_widths=None):
    for r_idx, row in enumerate(table.rows[1:]):
        bg = HEX_ROW_ALT if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, cell in enumerate(row.cells):
            set_cell_background(cell, bg)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = COLOR_TEXT_DARK
                    run.font.name = "Arial"
            if col_widths and c_idx < len(col_widths):
                cell.width = Inches(col_widths[c_idx])


def add_figure_with_caption(doc, img_rel_path, caption_text, width_inches=5.8):
    full_path = os.path.join(BASE_DIR, img_rel_path)
    if not os.path.exists(full_path):
        full_path = os.path.join(PROJECT_DIR, img_rel_path)

    if os.path.exists(full_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(full_path, width=Inches(width_inches))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        r_cap = p_cap.add_run(caption_text)
        r_cap.font.bold = True
        r_cap.font.size = Pt(9.5)
        r_cap.font.color.rgb = COLOR_NAVY
        r_cap.font.name = "Arial"
    else:
        print(f"Warning: Image not found at {full_path}")


def build_docx_report():
    print("Building REVISED_PROJECT_REPORT.docx...")
    doc = docx.Document()

    # Page Margins: Standard 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Normal Style
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = COLOR_TEXT_DARK
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # PAGE 1: TITLE PAGE
    # -------------------------------------------------------------
    p_univ = doc.add_paragraph()
    p_univ.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_univ.paragraph_format.space_before = Pt(30)
    p_univ.paragraph_format.space_after = Pt(4)
    r = p_univ.add_run("AMRITA VISHWA VIDYAPEETHAM")
    r.font.bold = True
    r.font.size = Pt(18)
    r.font.color.rgb = COLOR_MAROON
    r.font.name = "Times New Roman"

    p_school = doc.add_paragraph()
    p_school.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_school.paragraph_format.space_after = Pt(45)
    r = p_school.add_run("SCHOOL OF COMPUTING, AMARAVATI – 522503")
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = COLOR_TEXT_DARK

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(12)
    r = p_title.add_run("UNDERWATER SENSOR NETWORKING\nNetworked Maritime Monitoring, Governance and Simulation")
    r.font.bold = True
    r.font.size = Pt(17)
    r.font.color.rgb = COLOR_NAVY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(40)
    r = p_sub.add_run("Adaptive Multi-Robot Acoustic Telemetry, Neural Classification, and Swarm Governance in UWSNs")
    r.font.italic = True
    r.font.size = Pt(11)
    r.font.color.rgb = COLOR_MUTED

    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_after = Pt(8)
    r = p_by.add_run("Submitted by")
    r.font.size = Pt(11)
    r.font.italic = True

    p_team = doc.add_paragraph()
    p_team.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_team.paragraph_format.space_after = Pt(36)
    r = p_team.add_run(
        "Ch. Dhanush Sai — AV.SC.U4AIE24109\n"
        "Ch. Poorna Sindhuja — AV.SC.U4AIE24110\n"
        "Ch. Rishikesh — AV.SC.U4AIE24111"
    )
    r.font.bold = True
    r.font.size = Pt(11.5)

    p_guide_lbl = doc.add_paragraph()
    p_guide_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_guide_lbl.paragraph_format.space_after = Pt(4)
    r = p_guide_lbl.add_run("Under the guidance of")
    r.font.italic = True
    r.font.size = Pt(11)

    p_guide = doc.add_paragraph()
    p_guide.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_guide.paragraph_format.space_after = Pt(36)
    r = p_guide.add_run("Dr. Ravishankar Prakash Desai")
    r.font.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = COLOR_NAVY

    p_course = doc.add_paragraph()
    p_course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_course.paragraph_format.space_after = Pt(10)
    r = p_course.add_run(
        "In partial accomplishment of the laboratory work for the subject\n"
        "22AIE441 Dynamics and Control of Robotics"
    )
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = COLOR_TEXT_DARK

    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 2: ABSTRACT
    # -------------------------------------------------------------
    h_abs = doc.add_heading(level=1)
    h_abs.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = h_abs.add_run("ABSTRACT")
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = COLOR_NAVY

    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.space_after = Pt(12)
    p_abs.paragraph_format.line_spacing = 1.25
    p_abs.add_run(
        "The Harbor Traffic Control System is a comprehensive cyber-physical simulation designed to monitor, "
        "classify, and govern surface and subsurface maritime vessel traffic using an Underwater Wireless Sensor Network (UWSN), "
        "mobile Autonomous Underwater Vehicles (AUVs), and acoustic telemetry. Operating across a 10 km × 10 km shallow-water harbor "
        "basin, the system coordinates 22 surface vessels belonging to six distinct commercial and recreational classes, 35 moored seabed "
        "hydrophones, 8 autonomous mobile AUV relays, and 3 surface gateway buoys. High-fidelity acoustic models—including the Mackenzie "
        "sound speed profile, Ross propeller noise radiation, Thorp chemical absorption, and Wenz ocean ambient noise—synthesize realistic "
        "21-dimensional physical telemetry vectors. Telemetry is routed upward through a dynamic Depth-Based Routing (DBR) acoustic graph "
        "gated by Signal-to-Noise Ratio (SNR ≥ 10 dB) and node availability under low-power listening (LPL) duty cycles. A multi-task deep "
        "neural network (VesselMLP) simultaneously performs discrete vessel classification and continuous displacement tonnage estimation."
    )

    p_abs2 = doc.add_paragraph()
    p_abs2.paragraph_format.space_after = Pt(12)
    p_abs2.paragraph_format.line_spacing = 1.25
    p_abs2.add_run(
        "The system incorporates automated maritime traffic governance, enforcing speed limits (10 kn in Outer Yard, 6 kn in Inner Zone), "
        "fairway screening at the Line of Control (LOC), and safe dock berth allocation. Mobile AUVs execute autonomous waypoint patrols, "
        "intercept overspeed violators, maintain multi-hop connectivity during storm severances, and autonomously return to surface buoys "
        "when battery levels drop below 15%. To evaluate environmental resilience, a stochastic Monsoon scenario simulates gale winds (36–52 kn), "
        "torrential rain noise (+15 dB), and 2.6× tidal currents, driving a +25.8 dB ambient noise surge (66.1 dB → 91.9 dB) and hydrodynamic "
        "mooring failure (MOORED → STRAINED → SNAPPED). In empirical evaluation, vessel classification achieves 100.0% accuracy across validated "
        "samples, while displacement regression is diagnosed with an objective-scaling defect (0.6% gradient signal ratio). Binary telemetry "
        "compression achieves a 93.75% reduction in latency (2.048s → 0.128s) and modem energy (20.48J → 1.28J). The entire framework is "
        "verified across 80 automated unit and integration tests, demonstrating a fully connected cyber-physical robotics pipeline."
    )

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(16)
    r_kw = p_kw.add_run("Keywords: ")
    r_kw.font.bold = True
    p_kw.add_run("Underwater Wireless Sensor Networks (UWSN), Autonomous Underwater Vehicles (AUVs), Swarm Robotics, Depth-Based Routing (DBR), Passive Acoustic Telemetry, Multi-Task Learning, Maritime Traffic Control.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TABLE OF CONTENTS & LISTS
    # -------------------------------------------------------------
    h_toc = doc.add_heading(level=1)
    r = h_toc.add_run("CONTENTS")
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = COLOR_NAVY

    toc_items = [
        ("1", "INTRODUCTION", "5"),
        ("1.1", "Introduction (Background)", "5"),
        ("1.2", "Problem Statement", "6"),
        ("1.3", "Objectives", "6"),
        ("1.4", "Contributions", "7"),
        ("1.5", "Scope, Academic Integrity and Non-Claims Delineation", "7"),
        ("2", "RELATED WORK", "8"),
        ("2.1", "Comprehensive Survey", "8"),
        ("2.2", "Summary and Findings of Survey", "10"),
        ("2.3", "Challenges in Underwater Robotics", "11"),
        ("2.4", "Motivation of the Project Work", "12"),
        ("3", "MATERIALS AND METHODS", "13"),
        ("3.1", "Mathematical Modelling and Datasets Description", "13"),
        ("3.1.1", "Acoustic Mathematical Models", "13"),
        ("3.1.2", "Underwater Network Model", "14"),
        ("3.1.3", "Machine-Learning Dataset and VesselMLP Model", "15"),
        ("3.2", "Method(s) of Mathematical Modelling and Data Processing", "16"),
        ("3.3", "Programming Tools and Environment Details", "17"),
        ("4", "METHODOLOGY", "18"),
        ("4.1", "Controller and Algorithm Design", "18"),
        ("4.1.1", "Acoustic Sensing Pipeline", "18"),
        ("4.1.2", "Underwater Network Routing and Health Scoring", "18"),
        ("4.1.3", "Vessel Classification and Traffic Governance", "19"),
        ("4.1.4", "AUV Motion Dynamics and Proportional Control", "20"),
        ("4.1.5", "Sensor Mooring Hydrodynamic Failure Mechanics", "21"),
        ("4.2", "Controller and Algorithm Design Steps", "21"),
        ("4.3", "Project Architecture", "22"),
        ("4.4", "Stability and Algorithmic Boundedness Analysis", "23"),
        ("4.5", "Project Complete Flowchart", "24"),
        ("5", "RESULTS AND DISCUSSION", "25"),
        ("5.1", "Simulation Setup and Environmental Scenarios", "25"),
        ("5.2", "Qualitative Results", "26"),
        ("5.3", "Quantitative Results", "27"),
        ("5.3.1", "Machine Learning Performance and Defect Diagnosis", "27"),
        ("5.3.2", "Underwater Networking and Compression Gains", "29"),
        ("5.3.3", "Live Measured Link Counts and Packet Delivery Ratio", "30"),
        ("5.3.4", "Robotics Dynamics and Berth Allocation Bug Fix", "31"),
        ("5.3.5", "Physical Plausibility Audit (Groq LPU Sanity Check)", "32"),
        ("5.3.6", "Software Engineering Verification and Automated Testing", "32"),
        ("5.4", "Discussion of Findings and Research Implications", "33"),
        ("6", "CONCLUSION AND FUTURE DIRECTIONS", "35"),
        ("6.1", "Conclusion", "35"),
        ("6.2", "Future Directions", "36"),
        ("7", "REFERENCES", "37"),
        ("—", "IMPLEMENTATION SUMMARY & DEMONSTRATION GUIDE", "38")
    ]

    t_toc = doc.add_table(rows=1, cols=3)
    t_toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_toc)
    hdr = t_toc.rows[0]
    hdr.cells[0].paragraphs[0].add_run("CHAPTER")
    hdr.cells[1].paragraphs[0].add_run("TITLE")
    hdr.cells[2].paragraphs[0].add_run("PAGE NOs")
    format_table_header(hdr, [1.0, 4.4, 1.1])

    for ch, title, pg in toc_items:
        row = t_toc.add_row()
        row.cells[0].paragraphs[0].add_run(ch)
        row.cells[1].paragraphs[0].add_run(title)
        row.cells[2].paragraphs[0].add_run(pg)
    format_table_rows(t_toc, [1.0, 4.4, 1.1])

    doc.add_page_break()

    # -------------------------------------------------------------
    # LIST OF TABLES & FIGURES & ABBREVIATIONS
    # -------------------------------------------------------------
    doc.add_heading("LIST OF TABLES", level=2)
    tables_list = [
        ("Table 1", "Project objectives, module implementation, and verification status"),
        ("Table 2", "Summary and findings of the state-of-the-art related work survey"),
        ("Table 3", "21-dimensional acoustic, propagation, environmental, and engineered feature groups"),
        ("Table 4", "Underwater acoustic communication network parameters and link budget constants"),
        ("Table 5", "VesselMLP multi-task neural network architecture and training configuration"),
        ("Table 6", "Software programming tools, scientific libraries, and runtime environment"),
        ("Table 7", "Network health state classification thresholds and multi-factor evaluation weights"),
        ("Table 8", "Maritime traffic governance zones, speed limits, and Line-of-Control rules"),
        ("Table 9", "Normal Season vs. Monsoon Season environmental parameters comparison"),
        ("Table 10", "Quantitative evaluation results: ML, acoustics, networking, and governance"),
        ("Table 11", "Automated test suite coverage (80 tests across 3 independent runners)"),
        ("Table 12", "Module-to-function implementation summary and execution guide")
    ]
    for num, desc in tables_list:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.add_run(f"{num}. ").bold = True
        p.add_run(desc)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    doc.add_heading("LIST OF FIGURES", level=2)
    figures_list = [
        ("Figure 1", "VesselMLP multi-task neural network trunk and dual-head architecture"),
        ("Figure 2", "Network health state progression timeline across operational regimes"),
        ("Figure 3", "AUV behavioural state machine and battery discharge/recharge profiles"),
        ("Figure 4", "Sensor mooring hydrodynamic failure progression and squall rupture count"),
        ("Figure 5", "Underwater acoustic communication network topology (35 hydrophones, 8 AUVs, 3 buoys)"),
        ("Figure 6", "Complete cyber-physical simulation and autonomous traffic control execution flowchart"),
        ("Figure 7", "Ocean ambient acoustic noise surge under Monsoon conditions (+25.8 dB Wenz surge)"),
        ("Figure 8", "Validated vessel classification confusion matrix (Normal vs. Monsoon, 100% accuracy)"),
        ("Figure 9", "Displacement mass predicted vs. actual scatter plot (flatline defect)"),
        ("Figure 10", "Displacement head defect diagnosis: gradient norm ratio (0.6%) and loss-scale imbalance"),
        ("Figure 11", "Scenario-dependent hydrophone availability under low-power listening duty cycling"),
        ("Figure 12", "Binary telemetry frame compression performance gains (93.75% latency & energy saved)"),
        ("Figure 13", "Live measured active link count and Packet Delivery Ratio (Normal vs. Monsoon)"),
        ("Figure 14", "Berth allocation double-booking defect and queuing bug fix (Before vs. After)"),
        ("Figure 15", "Software engineering quality and automated test suite verification summary card")
    ]
    for num, desc in figures_list:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.add_run(f"{num}. ").bold = True
        p.add_run(desc)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    doc.add_heading("LIST OF ABBREVIATIONS", level=2)
    abbr_list = [
        ("AUV", "Autonomous Underwater Vehicle"),
        ("UWSN", "Underwater Wireless Sensor Network"),
        ("DBR", "Depth-Based Routing"),
        ("SNR", "Signal-to-Noise Ratio"),
        ("PDR", "Packet Delivery Ratio"),
        ("LOC", "Line of Control"),
        ("ML", "Machine Learning"),
        ("MLP", "Multi-Layer Perceptron"),
        ("MAE", "Mean Absolute Error"),
        ("RMSE", "Root Mean Square Error"),
        ("MAPE", "Mean Absolute Percentage Error"),
        ("TL", "Transmission Loss"),
        ("SL", "Source Level"),
        ("NL", "Noise Level"),
        ("LPL", "Low-Power Listening"),
        ("GUI", "Graphical User Interface"),
        ("HTTP", "Hypertext Transfer Protocol"),
        ("CSV", "Comma-Separated Values")
    ]
    t_ab = doc.add_table(rows=1, cols=2)
    t_ab.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_ab)
    format_table_header(t_ab.rows[0], [1.5, 5.0])
    t_ab.rows[0].cells[0].paragraphs[0].add_run("Abbreviation")
    t_ab.rows[0].cells[1].paragraphs[0].add_run("Meaning")
    for a, m in abbr_list:
        row = t_ab.add_row()
        row.cells[0].paragraphs[0].add_run(a).bold = True
        row.cells[1].paragraphs[0].add_run(m)
    format_table_rows(t_ab, [1.5, 5.0])

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 1: INTRODUCTION
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. INTRODUCTION", level=1)
    h1.paragraph_format.space_before = Pt(12)

    doc.add_heading("1.1 Introduction (Background)", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "Commercial maritime transportation forms the backbone of global supply chains, accounting for over 80% of world trade volume. "
        "Consequently, high-density harbor fairways, littoral straits, and port access channels present critical operational bottlenecks, "
        "heightening the risk of catastrophic vessel collisions, grounding in shallow waters, and unauthorized incursions into restricted zones. "
        "Conventional harbor traffic governance relies predominantly on surface Vessel Traffic Services (VTS) radar and Automatic Identification "
        "System (AIS) VHF transponders. While effective under nominal atmospheric conditions, surface radar is constrained by line-of-sight limitations "
        "and is incapable of penetrating the water column. Furthermore, AIS transponders are vulnerable to spoofing, intentional deactivation by rogue craft, "
        "and signal occlusion behind land masses or severe squalls."
    )
    p2 = doc.add_paragraph()
    p2.add_run(
        "To establish an un-spoofable, all-weather maritime surveillance capability, this project develops an integrated cyber-physical monitoring architecture "
        "combining Underwater Wireless Sensor Networks (UWSNs) with a collaborative swarm of Autonomous Underwater Vehicles (AUVs). Submerged hydrophones "
        "passively capture acoustic radiated noise emitted by ship machinery, propellers, and hydrodynamic cavitation. Because sound is the only physical wave "
        "capable of propagating over kilometer scales in ocean water, acoustic sensing provides robust, continuous vessel detection. However, static seabed grids "
        "suffer from severe acoustic absorption, multipath fading, shadow zones, and battery exhaustion under continuous transmission. By integrating mobile AUVs "
        "as dynamic communication relays, the network autonomously bridges topological partitions, intercepts speed violators, and preserves end-to-end telemetry "
        "delivery to surface gateway buoys even under severe oceanic storms."
    )

    doc.add_heading("1.2 Problem Statement", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "Traditional coastal radar and transponder systems fail to provide submerged awareness and cannot reliably govern non-cooperative vessels "
        "under adverse weather. Static underwater acoustic sensor networks deployed for harbor listening face severe challenges: high frequency-dependent "
        "attenuation, low acoustic propagation speed (~1500 m/s), time-varying ambient noise, battery depletion, and complete network partitioning during "
        "storm surges when seabed moorings fail. Therefore, the core research problem is: "
    )
    p_box = doc.add_paragraph()
    p_box.paragraph_format.left_indent = Inches(0.4)
    p_box.paragraph_format.right_indent = Inches(0.4)
    r_box = p_box.add_run(
        "\"How can a collaborative swarm of mobile AUVs dynamically establish and maintain multi-hop acoustic communication links with static seabed hydrophones "
        "to ensure resilient telemetry delivery, real-time machine-learning vessel classification, and autonomous traffic governance within a dynamic, "
        "disturbance-prone 10 km × 10 km harbor domain?\""
    )
    r_box.font.italic = True
    r_box.font.bold = True

    doc.add_heading("1.3 Objectives", level=2)
    p = doc.add_paragraph()
    p.add_run("The engineering objectives of the project and their module-level verification status are summarized in Table 1.")

    t_obj = doc.add_table(rows=1, cols=4)
    t_obj.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_obj)
    format_table_header(t_obj.rows[0], [0.6, 2.6, 1.4, 1.9])
    t_obj.rows[0].cells[0].paragraphs[0].add_run("No.")
    t_obj.rows[0].cells[1].paragraphs[0].add_run("Objective")
    t_obj.rows[0].cells[2].paragraphs[0].add_run("Module")
    t_obj.rows[0].cells[3].paragraphs[0].add_run("Verification Status")

    objs = [
        ("1", "Moored seabed hydrophone grid & acoustic physics modeling", "simulator.py", "Verified working via environment audit"),
        ("2", "Multi-task deep learning vessel classification (6 classes)", "mlmodel.py", "Verified 100% accuracy on 23 samples"),
        ("3", "Continuous vessel displacement tonnage estimation", "mlmodel.py", "Evaluated; diagnosed defect documented"),
        ("4", "Multi-hop Depth-Based Routing (DBR) acoustic networking", "network.py", "Live PDR & health scoring benchmarked"),
        ("5", "AUV swarm kinematics, patrol, dispatch & recharging", "auv.py", "Finite state machine verified in UI & tests"),
        ("6", "Harbor traffic governance (speed, LOC fairway, berths)", "traffic_algo.py", "Violations detected; berth collision fixed"),
        ("7", "Normal vs. Monsoon single-engine scenario simulation", "simulator.py", "Live runtime switch without restart"),
        ("8", "Persistent telemetry logging vs. ground-truth state", "prediction_logger.py", "23 unit tests pass; queryable CSV log"),
        ("9", "Reproducible model evaluation & benchmarking report", "evaluate_predictions.py", "Reproducible markdown & metrics pipeline"),
        ("10", "Maritime control-station 3D GUI & web dashboards", "main.py, *.html", "Grid responsive layout, WCAG AA contrast")
    ]
    for o_num, o_text, o_mod, o_ver in objs:
        row = t_obj.add_row()
        row.cells[0].paragraphs[0].add_run(o_num)
        row.cells[1].paragraphs[0].add_run(o_text)
        row.cells[2].paragraphs[0].add_run(o_mod)
        row.cells[3].paragraphs[0].add_run(o_ver)
    format_table_rows(t_obj, [0.6, 2.6, 1.4, 1.9])

    doc.add_heading("1.4 Contributions", level=2)
    contribs = [
        "Architected and simulated a unified 10 km × 10 km cyber-physical harbor basin integrating 22 surface vessels, 35 seabed hydrophones, 8 mobile AUV relays, and 3 surface gateway buoys into a single coupled execution loop.",
        "Formulated and implemented an empirical ocean acoustic pipeline incorporating Mackenzie sound speed profiles, Ross radiated noise, Thorp attenuation, Wenz ambient noise, and Doppler kinematic shift into a 21-dimensional telemetry vector.",
        "Implemented Depth-Based Routing (DBR) with binary frame compression, achieving a 93.75% reduction in acoustic transmission delay and modem battery consumption.",
        "Engineered an autonomous AUV swarm state machine (PATROLLING, DISPATCHED, CHARGING) capable of shadowing speeding vessels and preserving network connectivity during storm-induced sensor blackouts.",
        "Created dual-regime Normal and Monsoon environmental scenarios to analyze acoustic SNR degradation, sensor duty-cycle sleep masking, and hydrodynamic mooring rupture.",
        "Established an immutable ground-truth prediction logging architecture that eliminated silent metric invention, verified 100% classification accuracy, and isolated the mathematical cause of the displacement regression defect.",
        "Resolved a critical pre-existing berth-allocation double-booking defect where up to 13 speedboats collided on a single dock, replacing it with a physically safe queuing architecture."
    ]
    for c in contribs:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.add_run(c)

    doc.add_heading("1.5 Scope, Academic Integrity and Non-Claims Delineation", level=2)
    p_nc = doc.add_paragraph()
    p_nc.add_run(
        "To ensure absolute academic transparency and scientific rigor, the project establishes explicit boundaries between what was "
        "designed and implemented by the student authors versus what is prior art, empirical formulation, or explicitly NOT claimed:"
    )

    t_nc = doc.add_table(rows=1, cols=3)
    t_nc.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_nc)
    format_table_header(t_nc.rows[0], [1.8, 2.2, 2.5])
    t_nc.rows[0].cells[0].paragraphs[0].add_run("Capability / Area")
    t_nc.rows[0].cells[1].paragraphs[0].add_run("What is NOT Claimed")
    t_nc.rows[0].cells[2].paragraphs[0].add_run("Ground Truth / Actual Project Implementation")

    nc_data = [
        ("Vessel Speed Prediction", "We DO NOT claim an ML model predicts vessel speed.", "Speed is an internal kinematic state of the simulator (Actual_Speed_knots). Logged as MEASURED and NOT_PREDICTED."),
        ("Vessel Heading / Direction", "We DO NOT claim an ML model predicts heading or direction.", "Direction is determined by a geometric trigonometric threshold on the vessel angle in simulator.get_vessel_direction()."),
        ("Working Displacement Regression", "We DO NOT claim working vessel weight prediction.", "Displacement regression exhibits a severe flatline defect (MAE 22,713 kg, MAPE 97.7%) due to a 0.6% gradient starvation defect."),
        ("Physical Hardware Sea-Trials", "We DO NOT claim deployment on physical marine hardware.", "No physical hydrophones were submerged in an ocean. The dataset (collected_data.csv) is 100% synthetically generated from physics equations."),
        ("Invention of Acoustic Physics", "We DO NOT claim to have invented the underlying ocean models.", "Equations are classical published empirical models from Mackenzie (1981), Ross (1976), Thorp (1967), and Wenz (1962)."),
        ("Real-World Port Deployment", "We DO NOT claim commercial port navigation readiness.", "This is an academic cyber-physical simulation study. Real oceans include complex bottom multi-path and internal waves."),
        ("Automated Browser E2E Tests", "We DO NOT claim automated browser UI testing.", "The 80 automated tests verify the Python backend, routing, FSM, and ML. Desktop Tkinter and HTML were verified visually and via contrast audits."),
        ("AI Sanity Check vs. Certification", "We DO NOT claim Groq LPU certifies naval safety.", "The 100% score from openai/gpt-oss-120b is an automated software sanity check on parameter bounds, not experimental sea-trial validation.")
    ]
    for nc_a, nc_b, nc_c in nc_data:
        row = t_nc.add_row()
        row.cells[0].paragraphs[0].add_run(nc_a).bold = True
        row.cells[1].paragraphs[0].add_run(nc_b)
        row.cells[2].paragraphs[0].add_run(nc_c)
    format_table_rows(t_nc, [1.8, 2.2, 2.5])

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 2: RELATED WORK
    # -------------------------------------------------------------
    doc.add_heading("2. RELATED WORK", level=1)
    doc.add_heading("2.1 Comprehensive Survey", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "The development of intelligent maritime monitoring systems bridges four foundational domains: underwater acoustic wave propagation, "
        "underwater wireless sensor network protocols, autonomous multi-robot swarm dynamics, and machine-learning-based acoustic target classification. "
        "In the domain of underwater acoustics, empirical formulations developed by Mackenzie (1981) established the nine-term polynomial characterizing "
        "sound speed as a function of temperature, salinity, and depth. Donald Ross (1976) formulated fundamental mechanics of ship-radiated underwater noise, "
        "linking propeller tip cavitation and engine block vibrations to vessel speed and displacement tonnage. Thorp (1967) provided analytical expressions "
        "for acoustic absorption in seawater across low-frequency bands (1–50 kHz), while Wenz (1962) established standard spectral noise models capturing surface "
        "wind waves, turbulent agitation, and commercial shipping density."
    )
    p2 = doc.add_paragraph()
    p2.add_run(
        "In underwater networking, Akyildiz et al. (2005) identified the severe channel limitations that distinguish acoustic networks from terrestrial RF grids: "
        "extremely low bandwidth (~1–10 kbps), high latency (~1500 m/s propagation speed), and high bit error rates. Traditional terrestrial routing protocols "
        "(such as AODV and DSR) fail underwater due to the massive overhead of flooding route discovery packets. To resolve this, Yan, Shi, and Cui (2008) "
        "introduced Depth-Based Routing (DBR), a localization-free routing protocol that utilizes cheap onboard pressure sensors to greedily forward data packets "
        "to shallower neighbors toward surface gateways. While DBR handles static grids, multi-agent robotic swarms introduced by modern marine robotics "
        "allow Autonomous Underwater Vehicles to act as dynamic data mules, repositioning to bridge communication voids."
    )

    doc.add_heading("2.2 Summary and Findings of Survey", level=2)
    t_rw = doc.add_table(rows=1, cols=3)
    t_rw.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_rw)
    format_table_header(t_rw.rows[0], [1.4, 2.5, 2.6])
    t_rw.rows[0].cells[0].paragraphs[0].add_run("Domain / Area")
    t_rw.rows[0].cells[1].paragraphs[0].add_run("Approach in Present Project")
    t_rw.rows[0].cells[2].paragraphs[0].add_run("Key Role / Finding")

    rw_data = [
        ("Acoustic Sensing", "Empirical Mackenzie, Ross, Thorp & Wenz formulations", "Generates physically plausible 21-variable telemetry vectors from vessel kinematic states"),
        ("Underwater Networking", "SNR-gated dynamic NetworkX graph + multi-hop DBR", "Models communication range (1800m), acoustic packet progression, and link budget limits"),
        ("Swarm Robotics", "NumPy kinematics + AUV finite state machine", "Adds mobile relays, localized fairway patrol, overspeed interception, and buoy recharging"),
        ("Machine Learning", "PyTorch VesselMLP with 3 shared layers + dual heads", "Achieves 100% vessel classification; isolates regression objective loss-scale defect"),
        ("Traffic Control", "Deterministic speed limits, LOC screening, dock allocation", "Enforces harbor governance; eliminates multi-vessel berth collision defect via queuing"),
        ("Empirical Validation", "Permanent ground-truth logging & independent scenarios", "Separates ground truth from predictions, preventing artificial metric inflation")
    ]
    for d_a, d_b, d_c in rw_data:
        row = t_rw.add_row()
        row.cells[0].paragraphs[0].add_run(d_a)
        row.cells[1].paragraphs[0].add_run(d_b)
        row.cells[2].paragraphs[0].add_run(d_c)
    format_table_rows(t_rw, [1.4, 2.5, 2.6])

    doc.add_heading("2.3 Challenges in Underwater Robotics", level=2)
    challs = [
        "Acoustic Channel Hostility: Underwater acoustic propagation suffers from severe multipath fading, frequency-dependent attenuation (~1.4 dB/km at 12 kHz), and low propagation speed (1500 m/s), creating severe propagation delays.",
        "Dynamic Environmental Disturbances: Storm surges and seasonal monsoons increase ambient acoustic noise by over +25 dB, masking weak wake-up signals and drastically shrinking modem communication envelopes.",
        "Hydrodynamic Drag and Mooring Failures: High tidal currents (up to 5.5 m/s in squalls) induce severe mechanical strain on moored hydrophones, leading to tether snaps and sensor drift.",
        "Energy Asymmetry and Battery Depletion: Subsea modems consume substantial power during transmission (10 W), demanding aggressive duty-cycling and binary packet compression to prevent premature battery death.",
        "Multi-Task Loss Scale Balancing: Jointly training discrete classification (CrossEntropy ~5.6) alongside continuous mass regression (raw MSE ~6.3 × 10⁹) causes severe gradient imbalance, starving the regression head.",
        "Real-Time Cyber-Physical Integration: Coupling 60 FPS numerical physics with graph pathfinding, neural inference, and 3D visualization without UI freezing requires careful architectural decoupling."
    ]
    for ch in challs:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.add_run(ch)

    doc.add_heading("2.4 Motivation of the Project Work", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "The primary motivation of this project is to bridge the gap between static sensory observation and active robotic response within "
        "the academic framework of 22AIE441 Dynamics and Control of Robotics. A standard sensor network merely answers the question: "
        "'Was an acoustic signature detected?' In contrast, an autonomous cyber-physical robotic system must answer: "
        "'What vessel class is generating this noise? Is it violating harbor speed limits? Can our acoustic telemetry reach a surface buoy? "
        "Which AUV should be dispatched to intercept it, and how does ocean current drift alter the robot's required thrust?' "
        "By modeling AUV kinematics, proportional steering forces, drag damping, ocean current disturbances, and finite state transitions, "
        "this project provides an authentic laboratory study in robotic dynamics and networked control."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 3: MATERIALS AND METHODS
    # -------------------------------------------------------------
    doc.add_heading("3. MATERIALS AND METHODS", level=1)
    doc.add_heading("3.1 Mathematical Modelling and Datasets Description", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "The project is driven by continuous physical simulation across a 10,000 m × 10,000 m harbor basin. Telemetry is logged to "
        "collected_data.csv and evaluated against ground truth in logs/prediction_log.csv. The fleet comprises 22 surface vessels belonging to "
        "six maritime classes: Cargo Ship, Tanker, Cruiser, Ferry, Speedboat, and Fishing Vessel. Sensing is performed by 35 moored hydrophones, "
        "supplemented by 8 mobile AUV relays and 3 surface gateway buoys. For every vessel-sensor detection event within the 2000m detection radius, "
        "the simulator compiles a 21-dimensional physical feature vector structured into four distinct groups (Table 3)."
    )

    t_f = doc.add_table(rows=1, cols=4)
    t_f.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_f)
    format_table_header(t_f.rows[0], [1.3, 0.6, 2.6, 2.0])
    t_f.rows[0].cells[0].paragraphs[0].add_run("Feature Group")
    t_f.rows[0].cells[1].paragraphs[0].add_run("Count")
    t_f.rows[0].cells[2].paragraphs[0].add_run("Feature Names")
    t_f.rows[0].cells[3].paragraphs[0].add_run("Physical Unit / Description")

    feat_data = [
        ("Acoustic", "7", "Received_Level_dB, Source_Level_dB, Transmission_Loss_dB, Frequency_Hz, Doppler_Shift_Hz, SNR_dB, Ambient_Noise_dB", "dB re 1 µPa, Hz, acoustic pressure and noise levels"),
        ("Propagation", "3", "Sound_Speed_mps, Time_of_Arrival_s, Bearing_deg", "m/s, seconds, degrees azimuth wavefront"),
        ("Environmental", "5", "Water_Temp_C, Salinity_ppt, Sensor_Depth_m, Sea_State, Wind_Speed_knots", "°C, ppt, meters depth, Douglas scale, knots wind"),
        ("Engineered", "6", "Acoustic_Energy, SNR_Freq_interaction, Received_Level_squared, Doppler_Speed_proxy, Transmission_Efficiency, Env_Composite", "Converted linear power, interaction terms, efficiency margin")
    ]
    for fg_a, fg_b, fg_c, fg_d in feat_data:
        row = t_f.add_row()
        row.cells[0].paragraphs[0].add_run(fg_a).bold = True
        row.cells[1].paragraphs[0].add_run(fg_b)
        row.cells[2].paragraphs[0].add_run(fg_c)
        row.cells[3].paragraphs[0].add_run(fg_d)
    format_table_rows(t_f, [1.3, 0.6, 2.6, 2.0])

    doc.add_heading("3.1.1 Acoustic Mathematical Models", level=3)
    p = doc.add_paragraph()
    p.add_run("The acoustic simulation implements the following empirical oceanographic equations:\n")

    eqs = [
        ("Mackenzie (1981) Ocean Sound Speed Equation:",
         "c(T, S, D) = 1448.96 + 4.591T - 5.304×10⁻² T² + 2.374×10⁻⁴ T³ + 1.340(S - 35) + 1.630×10⁻² D + 1.675×10⁻⁷ D² - 1.025×10⁻² T(S - 35) - 7.139×10⁻¹³ T D³\n"
         "where T is temperature (°C), S is salinity (ppt), and D is depth (m)."),
        ("Ross (1976) Vessel Radiated Source Noise Model:",
         "SL = SL_base + 60 log₁₀(v / v_ref) + 20 log₁₀(W / W_ref) + 10 log₁₀(D_prop / D_ref) + N(0, σ²)\n"
         "where v is vessel speed, W is displacement tonnage, and D_prop is propeller diameter."),
        ("Thorp (1967) Chemical Relaxation Absorption Equation:",
         "α(f) = [0.11 f² / (1 + f²)] + [44 f² / (4100 + f²)] + 2.75×10⁻⁴ f² + 0.003  (dB/km)\n"
         "where f is frequency in kHz. Evaluates frequency-dependent attenuation."),
        ("Spherical Geometric Spreading & Transmission Loss:",
         "TL = 20 log₁₀(R) + α(f) · (R / 1000)  (dB)\n"
         "where R is slant range distance in meters."),
        ("Wenz (1962) Multi-Component Ocean Ambient Noise Model:",
         "NL_total = 10 log₁₀(10^(NL_turb/10) + 10^(NL_ship/10) + 10^(NL_waves/10) + 10^(NL_therm/10)) + NL_rain  (dB)\n"
         "combining low-frequency turbulence, shipping traffic density, wind wave action, and droplet noise."),
        ("Acoustic Kinematic Doppler Frequency Shift:",
         "Δf = f₀ [ c / (c - v_radial) - 1 ] ≈ f₀ (v cos θ / c)\n"
         "where θ is the vessel bearing angle relative to the receiving hydrophone.")
    ]
    for eq_title, eq_body in eqs:
        p_eq = doc.add_paragraph()
        p_eq.paragraph_format.left_indent = Inches(0.2)
        p_eq.paragraph_format.space_after = Pt(4)
        p_eq.add_run(eq_title + "\n").bold = True
        r_body = p_eq.add_run(eq_body)
        r_body.font.name = "Courier New"
        r_body.font.size = Pt(9.5)

    doc.add_heading("3.1.2 Underwater Network Model", level=3)
    p = doc.add_paragraph()
    p.add_run(
        "Underwater wireless communication is parameterized according to modern commercial subsea acoustic modems "
        "(e.g., EvoLogics S2C / Subnero M25). Table 4 lists the physical layer link budget constants."
    )

    t_net = doc.add_table(rows=1, cols=3)
    t_net.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_net)
    format_table_header(t_net.rows[0], [2.2, 2.0, 2.3])
    t_net.rows[0].cells[0].paragraphs[0].add_run("Parameter")
    t_net.rows[0].cells[1].paragraphs[0].add_run("Value")
    t_net.rows[0].cells[2].paragraphs[0].add_run("Physical Role / Basis")

    net_params = [
        ("Carrier Frequency (f_comm)", "12 kHz", "Optimal low-frequency coastal acoustic propagation"),
        ("Modem Source Level (SL_comm)", "178 dB re 1 µPa @ 1m", "Standard subsea transducer acoustic transmit power"),
        ("Communication Radius (R_comm)", "1800 m", "Maximum acoustic modem transmission range"),
        ("Detection Radius (R_detect)", "2000 m", "Hydrophone passive acoustic listening horizon"),
        ("Demodulation SNR Threshold", "10.0 dB", "Minimum SNR required for zero packet bit error rate"),
        ("Acoustic Bitrate", "1000 bps (1 kbps)", "Standard commercial acoustic FSK/OFDM data rate"),
        ("Acoustic Propagation Speed", "1500 m/s", "Speed of sound in seawater used for latency calculations"),
        ("Modem Transmit Power", "10.0 Watts", "Electrical power consumption during packet transmission")
    ]
    for p_a, p_b, p_c in net_params:
        row = t_net.add_row()
        row.cells[0].paragraphs[0].add_run(p_a).bold = True
        row.cells[1].paragraphs[0].add_run(p_b)
        row.cells[2].paragraphs[0].add_run(p_c)
    format_table_rows(t_net, [2.2, 2.0, 2.3])

    doc.add_heading("3.1.3 Machine-Learning Dataset and VesselMLP Model", level=3)
    p = doc.add_paragraph()
    p.add_run(
        "VesselMLP is implemented in PyTorch using a multi-task learning architecture. Instead of running separate neural networks, "
        "a single unified network processes the 21 physical features through a shared trunk before splitting into task-specific heads (Table 5)."
    )

    t_mlp = doc.add_table(rows=1, cols=3)
    t_mlp.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_mlp)
    format_table_header(t_mlp.rows[0], [2.0, 2.2, 2.3])
    t_mlp.rows[0].cells[0].paragraphs[0].add_run("Property")
    t_mlp.rows[0].cells[1].paragraphs[0].add_run("Configuration")
    t_mlp.rows[0].cells[2].paragraphs[0].add_run("Notes")

    mlp_params = [
        ("Input Dimension", "21 Features", "StandardScaler normalized (train split only)"),
        ("Shared Trunk Architecture", "Linear(21→128)→ReLU→Linear(128→64)→ReLU→Linear(64→32)→ReLU", "Extracts shared latent acoustic representations"),
        ("Classification Head", "Linear(32 → 6 Classes)", "Outputs logits for CrossEntropyLoss"),
        ("Regression Head", "Linear(32 → 1 Output)", "Outputs displacement mass (kg) for MSELoss"),
        ("Joint Loss Formulation", "loss = loss_type + (loss_weight * 1e-8)", "Diagnosed cause of displacement defect"),
        ("Optimization Algorithm", "Adam (learning rate = 0.005)", "Backpropagation with gradient clipping"),
        ("Training Dataset Split", "80% Train / 20% Test (random_state=42)", "Trained across 388,178 collected telemetry rows"),
        ("Target Vessel Classes", "Cargo, Cruiser, Ferry, Fishing, Speedboat, Tanker", "Encoded numerically via sklearn LabelEncoder")
    ]
    for p_a, p_b, p_c in mlp_params:
        row = t_mlp.add_row()
        row.cells[0].paragraphs[0].add_run(p_a).bold = True
        row.cells[1].paragraphs[0].add_run(p_b)
        row.cells[2].paragraphs[0].add_run(p_c)
    format_table_rows(t_mlp, [2.0, 2.2, 2.3])

    p_mlp_fig = doc.add_paragraph()
    p_mlp_fig.add_run(
        "Figure 1 illustrates the neural network architecture of VesselMLP, detailing the 21-dimensional input feature "
        "processing through the 128-64-32 shared latent representation layers and subsequent split into the discrete "
        "classification head (6 classes) and continuous displacement regression head."
    )
    add_figure_with_caption(doc, "results/assets/fig_ml_model_architecture.png",
                           "Figure 1. VesselMLP multi-task neural network trunk and dual-head architecture.", 5.5)

    doc.add_heading("3.2 Method(s) of Mathematical Modelling and Data Processing", level=2)
    p = doc.add_paragraph()
    p.add_run("The end-to-end execution methodology follows eleven sequential phases:\n")
    proc_steps = [
        "Initialize the harbor domain: 22 surface vessels, 35 seabed hydrophones, 8 AUVs, and 3 surface gateway buoys.",
        "Select environmental scenario (Normal or Monsoon), initializing oceanographic parameters and stochastic current seeds.",
        "Simulate vessel kinematic updates and evaluate radiated acoustic source level using the Ross formulation.",
        "Compute acoustic propagation: evaluate Mackenzie sound speed, Thorp transmission loss, and Wenz ambient noise at each hydrophone.",
        "Construct 21-dimensional telemetry feature vector for each detected vessel within 2000m range.",
        "Evaluate modem-to-modem link SNR and reconstruct the dynamic acoustic communication graph in NetworkX.",
        "Execute Depth-Based Routing (DBR) to greedily forward telemetry packets along negative depth gradients toward surface buoys.",
        "Standardize features using fitted StandardScaler and pass them to VesselMLP for multi-task inference.",
        "Execute harbor traffic governance: evaluate zone speed limits, Line-of-Control fairway screening, and berth assignments.",
        "Dispatch nearest AUVs to shadow overspeed violators and enforce boundary compliance.",
        "Write predictions and matching ground truth to prediction_log.csv and refresh desktop 3D station and web dashboards."
    ]
    for idx, s in enumerate(proc_steps, 1):
        p_s = doc.add_paragraph()
        p_s.paragraph_format.left_indent = Inches(0.2)
        p_s.paragraph_format.space_after = Pt(2)
        p_s.add_run(f"Step {idx}: ").bold = True
        p_s.add_run(s)

    doc.add_heading("3.3 Programming Tools and Environment Details", level=2)
    p = doc.add_paragraph()
    p.add_run("Table 6 details the programming technologies, scientific computing libraries, and hardware interfaces used in the project.")

    t_tools = doc.add_table(rows=1, cols=3)
    t_tools.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_tools)
    format_table_header(t_tools.rows[0], [1.8, 1.8, 2.9])
    t_tools.rows[0].cells[0].paragraphs[0].add_run("Software Layer")
    t_tools.rows[0].cells[1].paragraphs[0].add_run("Technology / Library")
    t_tools.rows[0].cells[2].paragraphs[0].add_run("Functional Role")

    tools_data = [
        ("Core Language", "Python 3.13 / 3.9+", "Common runtime across simulation, ML, networking, and UI"),
        ("Numerical Computing", "NumPy & SciPy", "Vectorized kinematics, acoustic formulas, and ocean current fields"),
        ("Data Handling", "pandas", "Telemetry aggregation, feature engineering, and CSV validation logging"),
        ("Deep Learning", "PyTorch (CUDA 12.8)", "VesselMLP architecture, training, and sub-millisecond inference"),
        ("Hardware Acceleration", "NVIDIA RTX 3050 (6GB)", "CUDA-accelerated tensor computations and graphics loop"),
        ("Graph Algorithms", "NetworkX", "Dynamic UWSN topology graph construction, BFS reachability, DBR paths"),
        ("Desktop Control Station", "Tkinter + Matplotlib 3D", "Grid-based responsive 3D visualization and notifications sidebar"),
        ("Companion Web UI", "HTML5 / Vanilla CSS / JS", "Real-time dual-theme dashboards served over Python http.server (Port 8000)"),
        ("Physics Plausibility Audit", "Groq Cloud LPU", "Hardware reasoning audit (openai/gpt-oss-120b) over 104 parameters")
    ]
    for t_a, t_b, t_c in tools_data:
        row = t_tools.add_row()
        row.cells[0].paragraphs[0].add_run(t_a).bold = True
        row.cells[1].paragraphs[0].add_run(t_b)
        row.cells[2].paragraphs[0].add_run(t_c)
    format_table_rows(t_tools, [1.8, 1.8, 2.9])

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 4: METHODOLOGY
    # -------------------------------------------------------------
    doc.add_heading("4. METHODOLOGY", level=1)
    doc.add_heading("4.1 Controller and Algorithm Design", level=2)

    doc.add_heading("4.1.1 Acoustic Sensing Pipeline", level=3)
    p = doc.add_paragraph()
    p.add_run(
        "For every vessel within the 2000m detection range of a hydrophone, received sound pressure level is computed from source level "
        "and transmission loss. SNR is then evaluated against the ambient ocean noise: "
    )
    p_eq = doc.add_paragraph()
    p_eq.paragraph_format.left_indent = Inches(0.4)
    r = p_eq.add_run("SNR = Received_Level_dB - Ambient_Noise_dB\nTransmission_Efficiency = Source_Level_dB - Transmission_Loss_dB")
    r.font.name = "Courier New"
    r.font.bold = True

    doc.add_heading("4.1.2 Underwater Network Routing and Health Scoring", level=3)
    p = doc.add_paragraph()
    p.add_run(
        "The communication network treats hydrophones, AUVs, and surface buoys as nodes in a graph. An edge is created between nodes A and B "
        "if and only if: distance(A, B) ≤ 1800 m AND SNR(A, B) ≥ 10.0 dB. "
        "Depth-Based Routing forwards packets along negative depth gradients toward surface gateways (buoys at Z = 0m). "
        "Network health is scored continuously via a multi-factor equation (Connectivity 35%, Delivery 35%, Mooring 20%, Latency 10%) "
        "and categorized into four discrete operational states (Table 7)."
    )

    t_h = doc.add_table(rows=1, cols=3)
    t_h.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_h)
    format_table_header(t_h.rows[0], [1.5, 1.8, 3.2])
    t_h.rows[0].cells[0].paragraphs[0].add_run("Health State")
    t_h.rows[0].cells[1].paragraphs[0].add_run("Score Threshold")
    t_h.rows[0].cells[2].paragraphs[0].add_run("Physical Operational Condition")

    health_states = [
        ("STABLE", "≥ 75.0 %", "High gateway reachability (>85%), PDR >95%, zero mooring failures"),
        ("DEGRADED", "50.0 – 74.9 %", "Elevated ambient noise, hop count increases (1.4 → 2.3), intermittent drops"),
        ("CRITICAL", "25.0 – 49.9 %", "Multiple moorings strained/snapped, PDR drops below 70%, AUV relays essential"),
        ("COLLAPSED", "< 25.0 % (or mass fail)", "Severe partition (>70% buoys unreachable) or mass mooring failure (≥65% snapped)")
    ]
    for h_a, h_b, h_c in health_states:
        row = t_h.add_row()
        row.cells[0].paragraphs[0].add_run(h_a).bold = True
        row.cells[1].paragraphs[0].add_run(h_b)
        row.cells[2].paragraphs[0].add_run(h_c)
    format_table_rows(t_h, [1.5, 1.8, 3.2])

    p_hlth_fig = doc.add_paragraph()
    p_hlth_fig.add_run(
        "Figure 2 illustrates the dynamic progression of composite network health scores across the four operational "
        "states: STABLE (nominal conditions), DEGRADED (early storm onset), CRITICAL (extensive mooring strain), "
        "and COLLAPSED (mass mooring snap and gateway partition)."
    )
    add_figure_with_caption(doc, "results/assets/fig_network_health_timeline.png",
                           "Figure 2. Network health state progression timeline across operational regimes.", 5.5)

    doc.add_heading("4.1.3 Vessel Classification and Traffic Governance", level=3)
    p = doc.add_paragraph()
    p.add_run(
        "VesselMLP predicts the vessel category, which feeds directly into the traffic management module (traffic_algo.py). "
        "The harbor domain is partitioned into three functional zones (Table 8)."
    )

    t_gov = doc.add_table(rows=1, cols=4)
    t_gov.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_gov)
    format_table_header(t_gov.rows[0], [1.4, 1.4, 1.2, 2.5])
    t_gov.rows[0].cells[0].paragraphs[0].add_run("Zone")
    t_gov.rows[0].cells[1].paragraphs[0].add_run("X Boundary")
    t_gov.rows[0].cells[2].paragraphs[0].add_run("Speed Limit")
    t_gov.rows[0].cells[3].paragraphs[0].add_run("Traffic Screening & Governance Action")

    gov_data = [
        ("Outer Yard", "X > 5000 m", "10.0 knots", "Line-of-Control Screening: Cargo/Tanker → HALT; Cruiser/Ferry → SLOW DOWN (50%); Speedboat/Fishing → CONTINUE"),
        ("Inner Zone", "700 m < X ≤ 5000 m", "6.0 knots", "Channel Navigation: Assigned specific dock slot; speed monitored"),
        ("Harbor Basin", "X ≤ 700 m", "0.0 knots", "Docked State: Moored at assigned berth; zero maneuvering speed")
    ]
    for g_a, g_b, g_c, g_d in gov_data:
        row = t_gov.add_row()
        row.cells[0].paragraphs[0].add_run(g_a).bold = True
        row.cells[1].paragraphs[0].add_run(g_b)
        row.cells[2].paragraphs[0].add_run(g_c)
        row.cells[3].paragraphs[0].add_run(g_d)
    format_table_rows(t_gov, [1.4, 1.4, 1.2, 2.5])

    doc.add_heading("4.1.4 AUV Motion Dynamics and Proportional Control", level=3)
    p = doc.add_paragraph()
    p.add_run(
        "Each AUV is modeled as a 6-DOF dynamic body with localized waypoint guidance. The net force governing AUV motion is:\n"
    )
    p_dyn = doc.add_paragraph()
    p_dyn.paragraph_format.left_indent = Inches(0.4)
    r_dyn = p_dyn.add_run(
        "F_net = F_control + F_drag + F_disturbance\n"
        "F_control = K_p · (x_target - x_current) - K_d · v_current    (||F_control|| ≤ 900 N)\n"
        "F_drag = - 0.5 · ρ · C_d · A · ||v|| · v ≈ - c_drag · v\n"
        "F_disturbance = v_current · mass · 0.6\n"
        "m · (dv/dt) = F_net    (||v|| ≤ 10.0 m/s)"
    )
    r_dyn.font.name = "Courier New"
    r_dyn.font.bold = True

    p_dyn2 = doc.add_paragraph()
    p_dyn2.add_run(
        "AUV operational state follows a finite state machine: PATROLLING (executing sector waypoints), DISPATCHED (intercepting speeding craft), "
        "and CHARGING (docked at buoy_beta). Battery drains proportionally to speed: drain = 0.015 · (1 + ||v|| / v_max). "
        "When battery drops below 15.0%, the AUV automatically aborts its mission and navigates to the surface buoy, recharging at 0.2%/frame until reaching 95.0%."
    )

    p_auv_fig = doc.add_paragraph()
    p_auv_fig.add_run(
        "Figure 3 illustrates the AUV behavioural finite state machine and the corresponding battery discharge during "
        "patrol missions followed by autonomous homing and surface buoy recharging."
    )
    add_figure_with_caption(doc, "results/assets/fig_robotics_auv_state_battery.png",
                           "Figure 3. AUV behavioural state machine and battery discharge/recharge profiles.", 5.5)

    doc.add_heading("4.1.5 Sensor Mooring Hydrodynamic Failure Mechanics", level=3)
    p = doc.add_paragraph()
    p.add_run(
        "Moored hydrophones are tethered to the seabed. When ocean current speed is nominal (≤ 1.3 m/s), moorings remain MOORED with gentle orbital sway (6m). "
        "When currents exceed 1.3 m/s, moorings transition to STRAINED, accumulating mechanical fatigue: Δstrain = dt · (v_current / 1.2). "
        "When current velocity exceeds the sensor's individual breaking threshold (2.2–3.4 m/s) or accumulated strain exceeds 12.0, the mooring snaps (SNAPPED). "
        "Snapped sensors drift with the current (14× drift multiplier) and drop out of the communication network."
    )

    p_moor_fig = doc.add_paragraph()
    p_moor_fig.add_run(
        "Figure 4 charts the hydrodynamic strain accumulation on seabed hydrophones under accelerating ocean currents, "
        "alongside the resulting squall-induced mooring rupture timeline."
    )
    add_figure_with_caption(doc, "results/assets/fig_robotics_mooring_progression.png",
                           "Figure 4. Sensor mooring hydrodynamic failure progression and squall rupture count.", 5.5)

    doc.add_heading("4.2 Controller and Algorithm Design Steps", level=2)
    p = doc.add_paragraph()
    p.add_run("The runtime algorithmic loop of the controller executes the following eleven steps:\n")

    steps_42 = [
        "Initialize the harbor domain, loading 22 fleet vessels, 35 hydrophones, 8 AUVs, and 3 gateway buoys.",
        "Generate environmental parameters for the active scenario (Normal vs. Monsoon) and evaluate Wenz ambient noise.",
        "Advance vessel dynamics, updating 3D coordinates and computing Ross propeller radiated noise levels.",
        "Compute acoustic propagation loss, Mackenzie sound speed, and Doppler frequency shifts for all active sensors.",
        "Compile the 21-dimensional acoustic and kinematic feature vector for detected vessels.",
        "Evaluate modem link budget SNR and reconstruct the dynamic acoustic communication graph in NetworkX.",
        "Execute Depth-Based Routing (DBR) to forward telemetry packets toward the nearest reachable surface buoy.",
        "Pass normalized features to VesselMLP to classify vessel type and estimate displacement mass.",
        "Apply maritime traffic rules: check zone speed limits, execute Line-of-Control screening, and route to berths.",
        "Update AUV swarm state machine: dispatch robots to intercept violators and home low-battery units to charging buoys.",
        "Evaluate composite network health, log validation records, and refresh the 3D control station and web dashboards."
    ]
    for idx, s in enumerate(steps_42, 1):
        p_s = doc.add_paragraph()
        p_s.paragraph_format.left_indent = Inches(0.2)
        p_s.paragraph_format.space_after = Pt(2)
        p_s.add_run(f"Step {idx}: ").bold = True
        p_s.add_run(s)

    doc.add_heading("4.3 Project Architecture", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "Figure 5 illustrates the cyber-physical network topology across the 10 km × 10 km harbor basin, detailing the spatial distribution "
        "of the 35 moored seabed hydrophones, 8 mobile AUV relays, and 3 surface gateway buoys along with active acoustic communication links."
    )
    add_figure_with_caption(doc, "results/assets/fig_network_topology.png",
                           "Figure 5. Underwater acoustic communication network topology (35 hydrophones, 8 AUVs, 3 buoys).", 5.6)

    doc.add_heading("4.4 Stability and Algorithmic Boundedness Analysis", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "Because the system is a cyber-physical simulation, stability is analyzed in the sense of algorithmic boundedness (BIBO stability). "
        "Several mathematical constraints ensure the simulation cannot enter unbounded or physically impossible states:\n"
        "1. Velocity and Force Saturation: AUV propulsive forces are hard-clamped to ||F_control|| ≤ 900 N, and maximum velocity is saturated at 10.0 m/s. "
        "Hydrodynamic drag F_drag = -c_drag · v provides strictly dissipative damping, guaranteeing asymptotic velocity stability.\n"
        "2. Spatial Domain Clamping: AUV and vessel coordinates are bounded within the 10 km × 10 km domain, and vertical coordinates are strictly constrained "
        "between seabed bathymetry and -2.0 m water surface clearance, preventing unphysical grounding or airborne flight.\n"
        "3. Finite State Machine Determinism: AUV battery levels are strictly bounded in [0%, 100%]. State transitions between PATROLLING, DISPATCHED, "
        "and CHARGING are hysteresis-gated (15% low threshold vs. 95% full threshold), preventing chattering between states.\n"
        "4. Network Graph Guardrails: DBR routing hops are capped at a maximum of 12 hops, preventing routing loops or infinite recursion."
    )

    doc.add_heading("4.5 Project Complete Flowchart", level=2)
    p = doc.add_paragraph()
    p.add_run("Figure 6 outlines the complete sequential execution flow of the Harbor Traffic Control System.")
    add_figure_with_caption(doc, "presentation_assets/flowchart.png",
                           "Figure 6. Complete cyber-physical simulation and autonomous traffic control execution flowchart.", 4.5)

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 5: RESULTS AND DISCUSSION
    # -------------------------------------------------------------
    doc.add_heading("5. RESULTS AND DISCUSSION", level=1)
    doc.add_heading("5.1 Simulation Setup and Environmental Scenarios", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "The cyber-physical simulation evaluates system performance across two comprehensive environmental scenarios (Table 9). "
        "Switching scenarios is executed live during runtime without resetting vessel positions or interrupting telemetry."
    )

    t_scen = doc.add_table(rows=1, cols=4)
    t_scen.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_scen)
    format_table_header(t_scen.rows[0], [1.8, 1.4, 1.4, 1.9])
    t_scen.rows[0].cells[0].paragraphs[0].add_run("Parameter")
    t_scen.rows[0].cells[1].paragraphs[0].add_run("Normal Season")
    t_scen.rows[0].cells[2].paragraphs[0].add_run("Monsoon Season")
    t_scen.rows[0].cells[3].paragraphs[0].add_run("Physical Impact")

    scen_data = [
        ("Douglas Sea State", "2 – 3 (Moderate)", "6 – 8 (Very Rough / High)", "Wave height surges from 0.8m to 6.5m"),
        ("Surface Wind Speed", "8 – 16 knots", "36 – 52 knots (Gale)", "Surges ocean surface turbulence noise"),
        ("Water Temperature", "18.0 – 22.0 °C", "12.0 – 15.0 °C", "Cold runoff reduces sound velocity"),
        ("Water Salinity", "34.5 – 35.5 ppt", "30.0 – 32.5 ppt", "Freshwater dilution lowers sound speed"),
        ("Rain Impact Noise", "0.0 dB", "+15.0 dB", "Acoustic masking across high frequencies"),
        ("Tidal Current Multiplier", "1.0× (~0.35 m/s)", "2.6× (0.9–5.5 m/s)", "Induces sensor mooring strain and drift"),
        ("Measured Ambient Noise", "66.1 dB re 1 µPa", "91.9 dB re 1 µPa", "+25.8 dB degradation in modem SNR")
    ]
    for s_a, s_b, s_c, s_d in scen_data:
        row = t_scen.add_row()
        row.cells[0].paragraphs[0].add_run(s_a).bold = True
        row.cells[1].paragraphs[0].add_run(s_b)
        row.cells[2].paragraphs[0].add_run(s_c)
        row.cells[3].paragraphs[0].add_run(s_d)
    format_table_rows(t_scen, [1.8, 1.4, 1.4, 1.9])

    p_scen_fig = doc.add_paragraph()
    p_scen_fig.add_run(
        "Figure 7 displays the ocean ambient acoustic noise surge driven by simulated Monsoon conditions, "
        "showing the +25.8 dB increase across turbulence, wind wave, shipping, and rain noise components."
    )
    add_figure_with_caption(doc, "results/assets/fig_physics_ambient_noise.png",
                           "Figure 7. Ocean ambient acoustic noise surge under Monsoon conditions (+25.8 dB Wenz surge).", 5.5)

    doc.add_heading("5.2 Qualitative Results", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "Under Normal conditions, the system demonstrates exceptional stability: all 35 seabed hydrophones maintain stable moorings, "
        "ambient noise remains at a quiet 66.1 dB, and over 91 communication links form a resilient acoustic mesh. AUVs smoothly execute sector patrols, "
        "and telemetry packets reach surface gateways in an average of 1.4 hops. In contrast, when Monsoon conditions are triggered, "
        "ambient noise surges by +25.8 dB to 91.9 dB, collapsing acoustic modem SNR across peripheral links. High current velocities (spiking to 5.5 m/s "
        "in squalls) exceed mooring breaking thresholds, causing seabed sensors to transition from MOORED to STRAINED and ultimately SNAPPED. "
        "As snapped sensors drift away, the network dynamically adapts: DBR reroutes packets through mobile AUV relays, increasing average hop count "
        "to 2.3 hops and preserving packet delivery."
    )

    doc.add_heading("5.3 Quantitative Results", level=2)

    doc.add_heading("5.3.1 Machine Learning Performance and Defect Diagnosis", level=3)
    p = doc.add_paragraph()
    p.add_run(
        "Vessel classification performance was validated against the persistent ground-truth telemetry log. As shown in Figure 8, "
        "VesselMLP achieves 100.0% exact-match classification accuracy (23/23 samples) across all six classes under both Normal and Monsoon conditions. "
        "The distinct acoustic source levels and propeller Doppler signatures allow robust discrimination."
    )
    add_figure_with_caption(doc, "results/assets/fig_ml_confusion_matrix.png",
                           "Figure 8. Validated vessel classification confusion matrix (Normal vs. Monsoon, 100% accuracy).", 5.5)

    p_reg = doc.add_paragraph()
    p_reg.add_run(
        "In contrast, the displacement regression head exhibits severe failure, with a Mean Absolute Error (MAE) of 22,713 kg, "
        "Root Mean Squared Error (RMSE) of 36,121 kg, and Mean Absolute Percentage Error (MAPE) of 97.7%. As depicted in Figure 9, "
        "predicted displacement produces a flat output (~400–1,700 kg) completely uncorrelated with actual vessel mass (1,000 to 385,000 kg)."
    )
    add_figure_with_caption(doc, "results/assets/fig_ml_displacement_scatter.png",
                           "Figure 9. Displacement mass predicted vs. actual scatter plot (flatline defect).", 5.5)

    p_diag = doc.add_paragraph()
    p_diag.add_run(
        "A diagnostic audit of mlmodel._retrain() isolated the mathematical root cause: raw MSE on kilogram targets measures ~6.26 × 10⁹ "
        "against CrossEntropy ~5.56. While an arbitrary 10⁻⁸ scaling factor was applied to MSE, the weight loss still measures 62.6 (11× larger than CE). "
        "Crucially, backpropagation gradient norms reaching the shared trunk reveal that the weight head supplies only 0.817 vs. 135.0 from the "
        "classification head (Figure 10). Thus, the regression head receives only 0.60% of the gradient signal, starving its optimization."
    )
    add_figure_with_caption(doc, "results/assets/fig_ml_gradient_defect.png",
                           "Figure 10. Displacement head defect diagnosis: gradient norm ratio (0.6%) and loss-scale imbalance.", 5.5)

    doc.add_heading("5.3.2 Underwater Networking and Compression Gains", level=3)
    p_net = doc.add_paragraph()
    p_net.add_run(
        "Under low-power listening duty cycling, sensor availability drops from 91.4% (32/35) in Normal seas to 71.4% (25/35) in Monsoon conditions "
        "due to acoustic preamble masking (Figure 11). Implementing 16-byte binary telemetry compression delivers transformative benefits: "
        "transmission latency drops from 2.048 s to 0.128 s per hop (93.75% faster), and modem energy consumption drops from 20.48 J to 1.28 J "
        "(93.75% saved), as illustrated in Figure 12."
    )
    add_figure_with_caption(doc, "results/assets/fig_network_duty_cycling.png",
                           "Figure 11. Scenario-dependent hydrophone availability under low-power listening duty cycling.", 4.8)
    add_figure_with_caption(doc, "results/assets/fig_network_packet_compression.png",
                           "Figure 12. Binary telemetry frame compression performance gains (93.75% latency & energy saved).", 5.5)

    doc.add_heading("5.3.3 Live Measured Link Counts and Packet Delivery Ratio", level=3)
    p_live = doc.add_paragraph()
    p_live.add_run(
        "Re-running the simulation pipeline over 150 frames reveals the quantitative impact of environmental degradation (Figure 13). "
        "In Normal conditions, the network maintains an average of 91.2 active links and 99.8% Packet Delivery Ratio (PDR). In Monsoon conditions, "
        "increased noise severs 21.7 links (dropping to 69.5 links), but mobile AUV swarm relays preserve a 92.4% PDR by dynamically expanding "
        "average routing hop count from 1.40 to 2.30 hops."
    )
    add_figure_with_caption(doc, "results/assets/fig_network_links_pdr.png",
                           "Figure 13. Live measured active link count and Packet Delivery Ratio (Normal vs. Monsoon).", 5.5)

    doc.add_heading("5.3.4 Robotics Dynamics and Berth Allocation Bug Fix", level=3)
    p_rob = doc.add_paragraph()
    p_rob.add_run(
        "Figure 14 demonstrates the resolution of the pre-existing berth-allocation bug: previously, when all designated "
        "class berths were booked, fallback logic stacked up to 13 speedboats onto Dock-C simultaneously, leading to severe "
        "physical collisions. The re-engineered governance architecture queues excess arriving vessels in a safe HOLDING_FOR_BERTH "
        "holding area outside the fairway at zero speed, dynamically allocating berths as departing vessels clear the docks."
    )
    add_figure_with_caption(doc, "results/assets/fig_robotics_berth_fix.png",
                           "Figure 14. Berth allocation double-booking defect and queuing bug fix (Before vs. After).", 5.5)

    doc.add_heading("5.3.5 Physical Plausibility Audit (Groq LPU Sanity Check)", level=3)
    p_san = doc.add_paragraph()
    p_san.add_run(
        "To verify that synthetic simulation telemetry conforms to real-world physics, an AI data validation audit was executed via Groq Cloud LPU "
        "running openai/gpt-oss-120b. The audit evaluated 104 physical parameters across ocean currents (0.1–1.5 m/s), Mackenzie sound speed (1440–1545 m/s), "
        "Thorp transmission loss, and vessel draft constraints. The model reported 100.0% physical plausibility (zero physical law violations), "
        "serving as an essential physics sanity check (not academic validation)."
    )

    doc.add_heading("5.3.6 Software Engineering Verification and Automated Testing", level=3)
    p_test = doc.add_paragraph()
    p_test.add_run(
        "The complete software repository was subjected to rigorous test automation across 80 tests in three independent suites (Figure 15). "
        "All 80 tests pass with 100% success rate (Table 11)."
    )
    add_figure_with_caption(doc, "results/assets/fig_verification_summary.png",
                           "Figure 15. Software engineering quality and automated test suite verification summary card.", 5.2)

    t_test = doc.add_table(rows=1, cols=4)
    t_test.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_test)
    format_table_header(t_test.rows[0], [2.2, 1.0, 1.0, 2.3])
    t_test.rows[0].cells[0].paragraphs[0].add_run("Test Suite")
    t_test.rows[0].cells[1].paragraphs[0].add_run("Count")
    t_test.rows[0].cells[2].paragraphs[0].add_run("Result")
    t_test.rows[0].cells[3].paragraphs[0].add_run("Scope Covered")

    t_data = [
        ("Prediction Validation (test_prediction_validation.py)", "23 Tests", "PASS", "Exact-match classification, circular error wraparound, MAE/RMSE/MAPE"),
        ("Groq AI Validation (test_ai_data_validator.py)", "5 Tests", "PASS", "104 physical parameter checks, Groq API integration, fallback physics validator"),
        ("Integration & Regression (test_integration.py)", "52 Tests", "PASS", "Full simulation loop, DBR routing, berth collision regression, CSS tokens audit"),
        ("Total Automated Tests", "80 Tests", "ALL PASS", "100% Pass Rate across all unit, integration, and regression suites")
    ]
    for td_a, td_b, td_c, td_d in t_data:
        row = t_test.add_row()
        row.cells[0].paragraphs[0].add_run(td_a).bold = True
        row.cells[1].paragraphs[0].add_run(td_b)
        row.cells[2].paragraphs[0].add_run(td_c)
        row.cells[3].paragraphs[0].add_run(td_d)
    format_table_rows(t_test, [2.2, 1.0, 1.0, 2.3])

    doc.add_heading("5.4 Discussion of Findings and Research Implications", level=2)
    p_disc = doc.add_paragraph()
    p_disc.add_run(
        "The empirical findings demonstrate both the profound strengths and specific limitations of the current architecture. "
        "On the networking and robotics side, the AUV swarm proves indispensable: static UWSNs suffer catastrophic topological collapse "
        "when monsoon currents sever moorings and ambient noise surges by +25 dB. Dynamic AUV relays preserve >92% packet delivery ratio. "
        "Furthermore, 16-byte binary telemetry compression provides an enormous 93.75% efficiency gain in acoustic channel utilization.\n\n"
        "On the machine-learning side, the findings provide a vital lesson in multi-task loss balancing: achieving 100% classification accuracy "
        "coincided with a 97.7% MAPE regression failure because raw kilogram MSE overwhelmed CrossEntropy, forcing the shared trunk to ignore "
        "the regression gradient (0.6% share). By transparently documenting this defect rather than concealing it, the project establishes a rigorous "
        "foundation for future multi-task optimization."
    )

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 6: CONCLUSION AND FUTURE DIRECTIONS
    # -------------------------------------------------------------
    doc.add_heading("6. CONCLUSION AND FUTURE DIRECTIONS", level=1)
    doc.add_heading("6.1 Conclusion", level=2)
    p_concl = doc.add_paragraph()
    p_concl.add_run(
        "This project successfully designed, implemented, and benchmarked an autonomous cyber-physical Harbor Traffic Control System "
        "integrating empirical underwater acoustic physics, deep neural multi-task inference, Depth-Based Routing, and a collaborative AUV swarm. "
        "By coupling passive seabed hydrophones with mobile robotic relays, the system overcomes the line-of-sight and deactivation vulnerabilities "
        "of conventional surface radar and AIS transponders. Key project outcomes include:\n"
        "• Validated 100.0% vessel classification accuracy across six ship classes using the multi-task VesselMLP model.\n"
        "• Maintained 92.4% Packet Delivery Ratio under severe Monsoon weather (+25.8 dB ambient noise surge) via autonomous AUV swarm relaying.\n"
        "• Achieved 93.75% savings in acoustic transmission delay and modem battery consumption through 16-byte binary telemetry compression.\n"
        "• Automated harbor traffic rules, enforcing speed limits, Line-of-Control screening, and collision-free berth queuing.\n"
        "• Verified system robustness across 80 automated tests, establishing an honest, transparent baseline of cyber-physical capabilities."
    )

    doc.add_heading("6.2 Future Directions", level=2)
    p_fd = doc.add_paragraph()
    p_fd.add_run("The following nine prioritized engineering initiatives are recommended for future research:\n")

    future_dirs = [
        "Rebalance Displacement Loss Objective: Standardize displacement target values to zero-mean unit-variance (or apply logarithmic scaling), balance the multi-task loss terms, and retrain VesselMLP to resolve the regression defect.",
        "Implement Multi-Agent Reinforcement Learning (MARL): Replace rule-based waypoint patrol with cooperative multi-agent deep RL (MAPPO) for energy-optimal coverage and adaptive swarm shadowing.",
        "Physical Marine Hardware Validation: Deploy algorithms onto physical acoustic modems (EvoLogics S2C / Subnero) in a coastal testbed to validate multipath and Doppler compensation.",
        "Trained Speed & Heading Models: Develop dedicated neural prediction heads for vessel speed and heading angle with strict feature-leakage controls.",
        "Cross-Domain Heterogeneous Swarms: Integrate Autonomous Surface Vessels (ASVs) and aerial patrol drones (UAVs) to create a synchronized sea-air-subsea traffic governance grid.",
        "Dynamic OFDM Modulation: Implement physical-layer adaptive modulation, dynamically switching bitrates (500 bps to 8 kbps) based on real-time channel SNR feedback.",
        "Background Asynchronous Logging: Offload telemetry CSV writing to a dedicated worker thread to eliminate main-thread I/O blocking during high-density simulations.",
        "Automated GUI & Browser Testing: Implement automated headless browser and GUI testing using Playwright to validate real-time dashboard updates.",
        "Harbor Berth Capacity Expansion: Expand modeled berth infrastructure from 13 to 22 docks to allow full simultaneous docking without queuing."
    ]
    for idx, fd in enumerate(future_dirs, 1):
        p_fd_item = doc.add_paragraph()
        p_fd_item.paragraph_format.left_indent = Inches(0.2)
        p_fd_item.paragraph_format.space_after = Pt(3)
        p_fd_item.add_run(f"Item {idx}: ").bold = True
        p_fd_item.add_run(fd)

    doc.add_page_break()

    # -------------------------------------------------------------
    # CHAPTER 7: REFERENCES
    # -------------------------------------------------------------
    doc.add_heading("7. REFERENCES", level=1)
    refs = [
        "[1] Mackenzie, K. V. (1981). Nine-term equation for sound speed in the oceans. The Journal of the Acoustical Society of America, 70(3), 807–812.",
        "[2] Ross, D. (1976). Mechanics of Underwater Noise. Pergamon Press, New York.",
        "[3] Thorp, W. H. (1967). Analytic description of the low-frequency attenuation coefficient. The Journal of the Acoustical Society of America, 42(1), 270–270.",
        "[4] Wenz, G. M. (1962). Acoustic ambient noise in the ocean: Spectra and sources. The Journal of the Acoustical Society of America, 34(12), 1936–1956.",
        "[5] Yan, H., Shi, Z. J., & Cui, J. H. (2008). DBR: Depth-based routing for underwater sensor networks. In NETWORKING 2008: Ad Hoc and Sensor Networks, Wireless Networks, Next Generation Internet, 283–294. Springer.",
        "[6] Akyildiz, I. F., Pompili, D., & Melodia, T. (2005). Underwater acoustic sensor networks: research challenges. Ad Hoc Networks, 3(3), 257–279.",
        "[7] Fossen, T. I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control. John Wiley & Sons.",
        "[8] Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32.",
        "[9] Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. In Proceedings of the 7th Python in Science Conference, 11–15.",
        "[10] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830."
    ]
    for r in refs:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.left_indent = Inches(0.3)
        p_r.paragraph_format.first_line_indent = Inches(-0.3)
        p_r.paragraph_format.space_after = Pt(4)
        p_r.add_run(r)

    doc.add_page_break()

    # -------------------------------------------------------------
    # IMPLEMENTATION SUMMARY & EXECUTION GUIDE
    # -------------------------------------------------------------
    doc.add_heading("IMPLEMENTATION SUMMARY & DEMONSTRATION GUIDE", level=1)
    p_sum = doc.add_paragraph()
    p_sum.add_run("Table 12 provides a complete architectural module-to-function mapping for faculty evaluation and live demonstration.")

    t_mod = doc.add_table(rows=1, cols=3)
    t_mod.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_mod)
    format_table_header(t_mod.rows[0], [1.8, 2.5, 2.2])
    t_mod.rows[0].cells[0].paragraphs[0].add_run("Module File")
    t_mod.rows[0].cells[1].paragraphs[0].add_run("Architectural Role")
    t_mod.rows[0].cells[2].paragraphs[0].add_run("Execution Command")

    mod_data = [
        ("main.py", "Tkinter 3D control station, master loop, sidebar notifications, HTTP telemetry server", "python main.py"),
        ("simulator.py", "Ocean acoustics physics engine, 22 vessels, 35 hydrophones, Normal/Monsoon scenarios", "Imported by main.py"),
        ("network.py", "Dynamic acoustic graph, Depth-Based Routing, packet progress, health evaluation", "Imported by main.py"),
        ("mlmodel.py", "VesselMLP PyTorch architecture, feature scaling, model loading & inference", "python diagnose_weight_head.py"),
        ("auv.py", "Autonomous underwater vehicle kinematics, waypoint patrol, dispatch, buoy charging", "Imported by main.py"),
        ("traffic_algo.py", "Speed limits, Line-of-Control screening, collision-free berth queuing", "Imported by main.py"),
        ("prediction_logger.py", "Persistent ground-truth vs. prediction telemetry logger (logs/prediction_log.csv)", "Imported by mlmodel.py"),
        ("evaluate_predictions.py", "Independent evaluation report generator across Normal and Monsoon scenarios", "python evaluate_predictions.py"),
        ("ai_data_validator.py", "Groq LPU physical plausibility audit across 104 hydrodynamic & acoustic parameters", "python ai_data_validator.py"),
        ("run_all_tests.py", "Master automated test runner executing all 80 unit, integration and regression tests", "python run_all_tests.py"),
        ("dashboard.html", "Real-time maritime traffic web dashboard with light/dark theme toggle (Port 8000)", "http://127.0.0.1:8000/dashboard.html"),
        ("networking_dashboard.html", "Real-time UWSN acoustic networking and health dashboard (Port 8000)", "http://127.0.0.1:8000/networking_dashboard.html")
    ]
    for m_a, m_b, m_c in mod_data:
        row = t_mod.add_row()
        row.cells[0].paragraphs[0].add_run(m_a).bold = True
        row.cells[1].paragraphs[0].add_run(m_b)
        row.cells[2].paragraphs[0].add_run(m_c)
    format_table_rows(t_mod, [1.8, 2.5, 2.2])

    p_end = doc.add_paragraph()
    p_end.paragraph_format.space_before = Pt(16)
    p_end.add_run(
        "Demonstration Controls in 3D Station: Space (Pause/Resume), Ctrl+S (Start), Ctrl+R (Reset), "
        "⇄ Season (Toggle Normal/Monsoon), 🌪 Squall (Inject Squall Surge), ⚓ Re-Anchor (Restore Snapped Hydrophones)."
    ).italic = True

    doc.save(OUTPUT_DOCX)
    print(f"Successfully saved Word document: {OUTPUT_DOCX}")
    doc.save(ROOT_DOCX)
    print(f"Saved root copy: {ROOT_DOCX}")


def convert_docx_to_pdf():
    print("Converting Word document to PDF using Microsoft Word Application...")
    import win32com.client
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(OUTPUT_DOCX))
        doc.SaveAs(os.path.abspath(OUTPUT_PDF), FileFormat=17) # 17 = wdFormatPDF
        doc.Close()
        word.Quit()
        print(f"Successfully generated PDF: {OUTPUT_PDF}")
        import shutil
        shutil.copyfile(OUTPUT_PDF, ROOT_PDF)
        print(f"Saved root PDF copy: {ROOT_PDF}")
        return True
    except Exception as e:
        print("Word PDF conversion failed, fallback needed:", e)
        return False


def build_markdown_report():
    print("Generating REVISED_PROJECT_REPORT.md...")
    doc = docx.Document(OUTPUT_DOCX)
    
    fig_map = {
        "Figure 1.": "results/assets/fig_ml_model_architecture.png",
        "Figure 2.": "results/assets/fig_network_health_timeline.png",
        "Figure 3.": "results/assets/fig_robotics_auv_state_battery.png",
        "Figure 4.": "results/assets/fig_robotics_mooring_progression.png",
        "Figure 5.": "results/assets/fig_network_topology.png",
        "Figure 6.": "presentation_assets/flowchart.png",
        "Figure 7.": "results/assets/fig_physics_ambient_noise.png",
        "Figure 8.": "results/assets/fig_ml_confusion_matrix.png",
        "Figure 9.": "results/assets/fig_ml_displacement_scatter.png",
        "Figure 10.": "results/assets/fig_ml_gradient_defect.png",
        "Figure 11.": "results/assets/fig_network_duty_cycling.png",
        "Figure 12.": "results/assets/fig_network_packet_compression.png",
        "Figure 13.": "results/assets/fig_network_links_pdr.png",
        "Figure 14.": "results/assets/fig_robotics_berth_fix.png",
        "Figure 15.": "results/assets/fig_verification_summary.png"
    }

    lines = []
    lines.append("# UNDERWATER SENSOR NETWORKING: Networked Maritime Monitoring, Governance and Simulation\n")
    lines.append("> **Amrita Vishwa Vidyapeetham**  \n> **School of Computing, Amaravati – 522503**  \n> **Course:** 22AIE441 Dynamics and Control of Robotics  \n> **Under the Guidance of:** Dr. Ravishankar Prakash Desai  \n>  \n> **Submitted by:**  \n> • Ch. Dhanush Sai — AV.SC.U4AIE24109  \n> • Ch. Poorna Sindhuja — AV.SC.U4AIE24110  \n> • Ch. Rishikesh — AV.SC.U4AIE24111  \n\n---\n")
    
    skip_titles = [
        "AMRITA VISHWA VIDYAPEETHAM", "SCHOOL OF COMPUTING", "UNDERWATER SENSOR NETWORKING",
        "Adaptive Multi-Robot Acoustic Telemetry", "Submitted by", "Ch. Dhanush Sai",
        "Under the guidance of", "Dr. Ravishankar Prakash Desai", "In partial accomplishment"
    ]

    in_body = False
    for child in doc.element.body:
        if child.tag.endswith('p'):
            p = docx.text.paragraph.Paragraph(child, doc)
            t = p.text.strip()
            if not t:
                continue
            if any(t.startswith(s) for s in skip_titles):
                continue
            if "1. INTRODUCTION" in t or "1 INTRODUCTION" in t:
                in_body = True

            st = p.style.name.lower()
            if 'heading 1' in st:
                lines.append(f"\n# {t}\n")
            elif 'heading 2' in st:
                lines.append(f"\n## {t}\n")
            elif 'heading 3' in st:
                lines.append(f"\n### {t}\n")
            elif 'bullet' in st:
                lines.append(f"- {t}")
            elif re.match(r'^Figure \d+\.', t):
                if in_body:
                    parts = t.split()
                    fig_key = parts[0] + " " + parts[1] if len(parts) >= 2 else t
                    img_path = fig_map.get(fig_key, "")
                    if img_path:
                        lines.append(f"\n![{t}]({img_path})\n*{t}*\n")
                    else:
                        lines.append(f"\n*{t}*\n")
                else:
                    lines.append(f"- **{t}**")
            elif re.match(r'^Table \d+\.', t):
                if in_body:
                    lines.append(f"\n**{t}**\n")
                else:
                    lines.append(f"- **{t}**")
            elif t.startswith('F_') or t.startswith('m ·') or t.startswith('SNR =') or t.startswith('SL =') or t.startswith('TL =') or t.startswith('c ='):
                lines.append(f"```text\n{t}\n```\n")
            else:
                lines.append(f"{t}\n")
        elif child.tag.endswith('tbl'):
            tbl = docx.table.Table(child, doc)
            if len(tbl.rows) > 0:
                header_cells = [c.text.strip().replace('\n', ' ') for c in tbl.rows[0].cells]
                lines.append("\n| " + " | ".join(header_cells) + " |")
                lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
                for row in tbl.rows[1:]:
                    row_cells = [c.text.strip().replace('\n', '<br>') for c in row.cells]
                    lines.append("| " + " | ".join(row_cells) + " |")
                lines.append("")

    full_md = "\n".join(lines)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(full_md)
    with open(ROOT_MD, "w", encoding="utf-8") as f:
        f.write(full_md)
    print(f"Successfully saved Markdown report: {OUTPUT_MD}")
    print(f"Saved root Markdown copy: {ROOT_MD}")


if __name__ == "__main__":
    build_docx_report()
    convert_docx_to_pdf()
    build_markdown_report()
    print("All report regenerations completed successfully!")
