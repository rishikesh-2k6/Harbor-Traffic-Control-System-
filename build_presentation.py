import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "presentation_assets")
OUTPUT_PPTX = os.path.join(SCRIPT_DIR, "Harbor_Traffic_Control_Presentation.pptx")

# -------------------------------------------------------------
# Color Palette (Matching Amrita & Attached PPT Template)
# -------------------------------------------------------------
C_MAROON      = RGBColor(160, 28, 28)     # Primary Amrita Header
C_ORANGE      = RGBColor(224, 112, 16)    # Primary Theme Orange / Accent
C_DARK_ORANGE = RGBColor(195, 82, 0)      # Footer bar orange
C_CARD_BG     = RGBColor(254, 240, 222)   # Light peach card background
C_CARD_BORDER = RGBColor(240, 167, 99)    # Card border
C_TEXT_DARK   = RGBColor(34, 34, 34)      # Near black text
C_TEXT_MUTED  = RGBColor(85, 85, 85)      # Muted grey text
C_WHITE       = RGBColor(255, 255, 255)
C_LIGHT_GRAY  = RGBColor(245, 246, 248)
C_LINE_GRAY   = RGBColor(210, 210, 210)
C_NAVY        = RGBColor(15, 32, 67)

FONT_SERIF = "Times New Roman"
FONT_SANS  = "Arial"

def add_footer(slide, slide_num, total_slides=10):
    # Bottom orange footer bar
    footer_height = Inches(0.4)
    footer_top = Inches(7.5 - 0.4)
    footer_bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), footer_top, Inches(13.333), footer_height
    )
    footer_bar.fill.solid()
    footer_bar.fill.fore_color.rgb = C_DARK_ORANGE
    footer_bar.line.color.rgb = C_DARK_ORANGE

    # Footer text (Left: Date, Center: Course, Right: Slide Num)
    txBox = slide.shapes.add_textbox(Inches(0.4), footer_top, Inches(12.533), footer_height)
    tf = txBox.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = f"04-10-2026                 INTRODUCTION TO AI ROBOTICS (22AIE214)                 {slide_num}"
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_WHITE
    p.font.name = FONT_SANS
    p.font.bold = True

def add_header(slide, title_text):
    # Top slide title
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(28)
    p.font.name = FONT_SERIF
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK

    # Subtle separator line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.22), Inches(11.733), Inches(0.02)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = C_LINE_GRAY
    line.line.color.rgb = C_LINE_GRAY

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank slide

    # =========================================================
    # SLIDE 1: Title Slide
    # =========================================================
    s1 = prs.slides.add_slide(blank_layout)

    # University Name
    t_box = s1.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.733), Inches(0.6))
    p = t_box.text_frame.paragraphs[0]
    p.text = "Amrita Vishwa Vidyapeetham, Amaravati Campus"
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = C_MAROON
    p.font.name = FONT_SERIF

    # Thin underline
    line1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(1.15), Inches(10.333), Inches(0.015))
    line1.fill.solid()
    line1.fill.fore_color.rgb = C_LINE_GRAY
    line1.line.color.rgb = C_LINE_GRAY

    # Lab-Based Project Title
    t_box2 = s1.shapes.add_textbox(Inches(0.8), Inches(1.25), Inches(11.733), Inches(0.45))
    p = t_box2.text_frame.paragraphs[0]
    p.text = "Lab-Based Project Title"
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_DARK
    p.font.name = FONT_SERIF

    # Main Project Title
    t_box3 = s1.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.733), Inches(0.9))
    p = t_box3.text_frame.paragraphs[0]
    p.text = "Adaptive Multi-Robot Networking & Acoustic Telemetry in UWSN for Harbor Monitoring"
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(21)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.font.name = FONT_SERIF

    # Student Table
    rows, cols = 4, 2
    left, top, width, height = Inches(2.2), Inches(3.0), Inches(8.933), Inches(1.8)
    table_shape = s1.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    table.columns[0].width = Inches(5.1)
    table.columns[1].width = Inches(3.833)

    data = [
        ("Name of the Student", "Roll No."),
        ("J Kiran Sri Datta Sai", "AV.SC.U4AIE24121"),
        ("Mogalapalli H N V Satya Sai Ganesh", "AV.SC.U4AIE24139"),
        ("Penigonda Rahul", "AV.SC.U4AIE24151")
    ]

    for r_idx, (name, roll) in enumerate(data):
        cell_a = table.cell(r_idx, 0)
        cell_b = table.cell(r_idx, 1)
        cell_a.text = name
        cell_b.text = roll

        bg_col = C_ORANGE if r_idx == 0 else (C_CARD_BG if r_idx % 2 == 1 else RGBColor(255, 248, 240))
        fg_col = C_WHITE if r_idx == 0 else C_TEXT_DARK
        bold_flag = True if r_idx == 0 else False

        for cell in (cell_a, cell_b):
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_col
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(13 if r_idx == 0 else 12)
            p.font.bold = bold_flag
            p.font.color.rgb = fg_col
            p.font.name = FONT_SANS
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    # Guidance
    g_box = s1.shapes.add_textbox(Inches(0.8), Inches(5.35), Inches(11.733), Inches(0.9))
    tf = g_box.text_frame
    p1 = tf.paragraphs[0]
    p1.text = "Under the Guidance of"
    p1.alignment = PP_ALIGN.CENTER
    p1.font.italic = True
    p1.font.size = Pt(14)
    p1.font.name = FONT_SERIF
    p1.font.color.rgb = C_TEXT_DARK

    p2 = tf.add_paragraph()
    p2.text = "Dr Ravishankar Prakash Desai"
    p2.alignment = PP_ALIGN.CENTER
    p2.font.bold = True
    p2.font.size = Pt(16)
    p2.font.name = FONT_SERIF
    p2.font.color.rgb = C_TEXT_DARK

    add_footer(s1, 1)

    # =========================================================
    # SLIDE 2: Introduction
    # =========================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Introduction")

    # Left text box
    tb = s2.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    bullets_s2 = [
        "Maritime commerce accounts for over 80% of global freight transport; harbor traffic density poses extreme navigational safety and coastal defense challenges.",
        "Conventional monitoring depends heavily on surface radar (VTS) and transponders (AIS), which cannot penetrate water columns and can be deactivated by rogue vessels.",
        "Underwater Wireless Sensor Networks (UWSNs) provide submerged monitoring, but static seabed sensors suffer from high acoustic attenuation, multi-path fading, and battery depletion.",
        "Harsh ocean dynamics, thermocline gradients, and seasonal storms (monsoons) severely degrade acoustic modem Signal-to-Noise Ratio (SNR), severing static links.",
        "Integrating Autonomous Underwater Vehicles (AUVs) as dynamic swarm relays bridges communication dead zones and enables proactive autonomous maritime traffic governance."
    ]

    for i, b_text in enumerate(bullets_s2):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + b_text
        p.font.size = Pt(13)
        p.font.name = FONT_SANS
        p.font.color.rgb = C_TEXT_DARK
        p.space_after = Pt(12)

    # Right side image (harbor architecture overview)
    arch_img = os.path.join(ASSETS_DIR, "harbor_architecture.png")
    if os.path.exists(arch_img):
        s2.shapes.add_picture(arch_img, Inches(7.8), Inches(1.6), width=Inches(4.8))

    add_footer(s2, 2)

    # =========================================================
    # SLIDE 3: Problem Statement, Objectives and Tools
    # =========================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Problem Statement, Objectives and Programming Language")

    # Left column: Problem & Objectives
    tb = s3.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(7.3), Inches(5.4))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Problem Statement:"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = C_MAROON
    p.font.name = FONT_SANS

    p = tf.add_paragraph()
    p.text = "Traditional coastal monitoring cannot detect non-cooperative underwater and surface vessels in all-weather conditions. Static underwater sensor networks suffer from limited communication range, high ambient noise, and topology fragmentation during storm surges."
    p.font.size = Pt(12)
    p.font.color.rgb = C_TEXT_DARK
    p.font.name = FONT_SANS
    p.space_after = Pt(10)

    p = tf.add_paragraph()
    p.text = "Objectives:"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = C_MAROON
    p.font.name = FONT_SANS

    objs = [
        "Simulate a 10 km × 10 km harbor domain with 35 moored hydrophones, 8 AUVs, and 3 surface gateway buoys.",
        "Model empirical underwater acoustics: Mackenzie sound speed, Ross propeller noise, Thorp attenuation, and Wenz ambient noise.",
        "Extract 23 acoustic features and train a multi-task deep neural network (VesselMLP) to classify 6 ship types and predict displacement weight.",
        "Implement Depth-Based Routing (DBR) with binary packet compression to achieve low latency and conserve battery energy.",
        "Automate harbor traffic governance: speed violation detection, Line of Control (LOC) fairway screening, and dock berth allocation."
    ]
    for obj in objs:
        p = tf.add_paragraph()
        p.text = "• " + obj
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT_DARK
        p.font.name = FONT_SANS
        p.space_after = Pt(6)

    # Right Card: Tools Used
    card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.4), Inches(1.5), Inches(4.2), Inches(5.2))
    card.fill.solid()
    card.fill.fore_color.rgb = C_CARD_BG
    card.line.color.rgb = C_CARD_BORDER
    card.line.width = Pt(1.5)

    tb_card = s3.shapes.add_textbox(Inches(8.55), Inches(1.6), Inches(3.9), Inches(5.0))
    tfc = tb_card.text_frame
    tfc.word_wrap = True

    p = tfc.paragraphs[0]
    p.text = "Tools & Environment Used"
    p.alignment = PP_ALIGN.CENTER
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = C_DARK_ORANGE
    p.font.name = FONT_SANS
    p.space_after = Pt(12)

    tools_items = [
        ("Language", "Python 3.9+"),
        ("AI / ML", "PyTorch (VesselMLP), Scikit-Learn"),
        ("Networking", "NetworkX (Dynamic DBR Graphs)"),
        ("Physics & Math", "NumPy, SciPy (Acoustic Equations)"),
        ("GUI & Dashboards", "Matplotlib 3D, Tkinter, HTML5/JS"),
        ("Dataset", "132,000+ Acoustic Telemetry Records"),
        ("Deployment", "Decoupled Web Server (Port 8000)")
    ]

    for label, val in tools_items:
        p = tfc.add_paragraph()
        p.text = f"• {label}: "
        p.font.bold = True
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT_DARK
        p.font.name = FONT_SANS
        
        # Add value part
        run = p.add_run()
        run.text = val
        run.font.bold = False
        run.font.color.rgb = RGBColor(60, 60, 60)
        p.space_after = Pt(6)

    add_footer(s3, 3)

    # =========================================================
    # SLIDE 4: Methodology
    # =========================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Methodology")

    tb = s4.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.733), Inches(5.6))
    tf = tb.text_frame
    tf.word_wrap = True

    sections_s4 = [
        ("Robotic System Architecture:",
         "Multi-tier cyber-physical network in a 10 km × 10 km harbor domain: 35 moored seabed hydrophones (depth -50m to -65m), 8 autonomous mobile AUVs (depth -10m to -25m), and 3 surface gateway buoys (Z = 0m)."),
        ("Acoustic Physics & Sensing Pipeline:",
         "Vessels radiate propeller and machinery sound (Ross model). Hydrophones sample received acoustic pressure, calculate Doppler shift and transmission loss (Mackenzie sound speed + Thorp absorption), and compile a 23-feature telemetry vector."),
        ("Multi-Task Deep Learning Inference (VesselMLP):",
         "A unified PyTorch neural network with 3 shared latent layers (128 → 64 → 32) feeds dual heads simultaneously: 6-class ship classification (Cargo, Tanker, Cruiser, Ferry, Speedboat, Fishing Vessel) via Cross-Entropy and displacement weight regression via MSE."),
        ("Adaptive Depth-Based Routing (DBR) & Swarm Relaying:",
         "Packets route greedily along negative depth gradients towards surface buoys. Under 70/30 sensor sleep duty cycles or monsoon link severance, mobile AUVs autonomously act as data mules, dynamic relays, and overspeed interceptors."),
        ("Environmental Simulation & Protocol Optimization:",
         "Unified single-engine simulation comparing Normal Season (65.8 dB ambient noise) vs. Monsoon Season (91.9 dB noise, 2.6× storm currents). 16-byte packet compression slashes transmission delay and modem battery consumption by 93.75%.")
    ]

    for idx, (title, desc) in enumerate(sections_s4):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = title
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = C_DARK_ORANGE
        p.font.name = FONT_SANS

        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.size = Pt(11.5)
        p_desc.font.color.rgb = C_TEXT_DARK
        p_desc.font.name = FONT_SANS
        p_desc.space_after = Pt(8)

    add_footer(s4, 4)

    # =========================================================
    # SLIDE 5: Flow-chart
    # =========================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Flow-chart")

    flowchart_img = os.path.join(ASSETS_DIR, "flowchart.png")
    if os.path.exists(flowchart_img):
        s5.shapes.add_picture(flowchart_img, Inches(2.2), Inches(1.4), width=Inches(8.933))

    add_footer(s5, 5)

    # =========================================================
    # SLIDE 6: Simulation Result (Model & Classification)
    # =========================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Simulation Result: Machine Learning & Classification")

    # Left side: 4-panel image
    ml_img = os.path.join(ASSETS_DIR, "ml_results.png")
    if os.path.exists(ml_img):
        s6.shapes.add_picture(ml_img, Inches(0.8), Inches(1.4), width=Inches(7.2))

    # Right side: Bullet summary
    tb_res = s6.shapes.add_textbox(Inches(8.2), Inches(1.4), Inches(4.5), Inches(5.4))
    tf = tb_res.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Key ML Performance Metrics"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = C_NAVY
    p.font.name = FONT_SANS
    p.space_after = Pt(10)

    res_bullets = [
        "Overall Test Accuracy: 95.80% across 6 maritime vessel categories.",
        "Macro Average AUC: 0.989, demonstrating near-perfect discriminative capability across noisy acoustic regimes.",
        "Class Sensitivity: Speedboat (98.0%), Tanker (98.0%), Fishing Vessel (97.0%), Cargo Ship (96.0%).",
        "Loss Convergence: Joint Cross-Entropy + MSE loss dropped monotonically from 1.020 → 0.048.",
        "Continuous Weight Estimation: High fidelity regression from 1 kTon to 400 kTon displacement weight.",
        "Real-Time Execution: Sub-millisecond inference (<0.8 ms) enables instantaneous traffic decision making."
    ]

    for b in res_bullets:
        p = tf.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = C_TEXT_DARK
        p.font.name = FONT_SANS
        p.space_after = Pt(8)

    add_footer(s6, 6)

    # =========================================================
    # SLIDE 7: Simulation Result (Networking & Scenarios)
    # =========================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Simulation Result: Multi-Robot Networking & Scenarios")

    net_img = os.path.join(ASSETS_DIR, "networking_results.png")
    if os.path.exists(net_img):
        s7.shapes.add_picture(net_img, Inches(0.8), Inches(1.4), width=Inches(7.2))

    tb_net = s7.shapes.add_textbox(Inches(8.2), Inches(1.4), Inches(4.5), Inches(5.4))
    tf = tb_net.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Networking & Swarm Resilience"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = C_NAVY
    p.font.name = FONT_SANS
    p.space_after = Pt(10)

    net_bullets = [
        "Monsoon Acoustic Degradation: Torrential rain and storm gale noise surge ambient noise by +26.1 dB (65.8 dB → 91.9 dB), cutting 1km modem SNR from 50.5 dB to 24.5 dB.",
        "Swarm Relay Resilience: Despite broken peripheral links and drifting sensors, mobile AUV relays maintain 92.4% Packet Delivery Ratio (vs 96.8% in calm seas).",
        "Adaptive Multi-Hop Routing: DBR automatically elevates average hop count from 1.4 → 2.3 hops to bypass acoustic partitions.",
        "Frame Compression Impact: Reducing packet size from 256B to 16B slashes transmission latency from 2.048s to 0.128s per hop (93.75% reduction).",
        "Energy Conservation: Modem energy consumption drops from 20.48 J → 1.28 J per hop, drastically extending subsea battery operational lifespan."
    ]

    for b in net_bullets:
        p = tf.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT_DARK
        p.font.name = FONT_SANS
        p.space_after = Pt(8)

    add_footer(s7, 7)

    # =========================================================
    # SLIDE 8: Conclusion & Future Scope
    # =========================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Conclusion")

    tb = s8.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Conclusion:"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = C_MAROON
    p.font.name = FONT_SANS

    c_points = [
        "Developed a unified cyber-physical harbor traffic simulation integrating ocean acoustics, multi-task deep learning, and multi-robot networking across a 10 km × 10 km domain.",
        "Multi-task PyTorch VesselMLP successfully classifies 6 ship classes (95.8% accuracy) and predicts displacement weight from 23 passive acoustic telemetry features.",
        "Autonomous AUV swarm dynamically preserves network connectivity, achieving >92% PDR even under severe monsoon noise (+26 dB) and 70/30 sensor sleep duty cycles.",
        "Binary telemetry frame compression achieves a 93.75% reduction in acoustic transmission delay and modem battery consumption.",
        "Autonomous traffic governance enforces harbor speed limits, screens vessels at the Line of Control, and automatically allocates designated dock berths."
    ]
    for pt in c_points:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT_DARK
        p.font.name = FONT_SANS
        p.space_after = Pt(4)

    p = tf.add_paragraph()
    p.text = "Future Scope:"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = C_MAROON
    p.font.name = FONT_SANS
    p.space_before = Pt(8)

    fs_points = [
        "Physical Hardware Validation: Deploy on physical marine testbeds with subsea acoustic modems (EvoLogics / Subnero).",
        "Multi-Agent Reinforcement Learning (MARL): Implement cooperative swarm pathfinding, collision avoidance, and energy-optimal shadowing.",
        "Heterogeneous Cross-Domain Swarms: Integrate Autonomous Surface Vessels (ASVs) and aerial patrol drones (UAVs) for hybrid air-sea tracking.",
        "Adaptive Acoustic Modulation: Dynamic physical-layer OFDM bitrate switching based on instantaneous SNR and Doppler feedback.",
        "Edge Deployment: Port lightweight quantized VesselMLP models onto embedded AUV companion computers (NVIDIA Jetson Orin Nano)."
    ]
    for pt in fs_points:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT_DARK
        p.font.name = FONT_SANS
        p.space_after = Pt(4)

    add_footer(s8, 8)

    # =========================================================
    # SLIDE 9: References
    # =========================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "References")

    tb = s9.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.4))
    tf = tb.text_frame
    tf.word_wrap = True

    refs = [
        "[1] Mackenzie, K. V. (1981). Nine-term equation for sound speed in the oceans. The Journal of the Acoustical Society of America, 70(3), 807–812.",
        "[2] Ross, D. (1976). Mechanics of Underwater Noise. Pergamon Press, New York.",
        "[3] Wenz, G. M. (1962). Acoustic ambient noise in the ocean: Spectra and sources. The Journal of the Acoustical Society of America, 34(12), 1936–1956.",
        "[4] Yan, H., Shi, Z. J., & Cui, J. H. (2008). DBR: Depth-based routing for underwater sensor networks. In NETWORKING 2008: Ad Hoc and Sensor Networks, Wireless Networks, Next Generation Internet, 283–294. Springer.",
        "[5] Akyildiz, I. F., Pompili, D., & Melodia, T. (2005). Underwater acoustic sensor networks: research challenges. Ad Hoc Networks, 3(3), 257–279.",
        "[6] Thorp, W. H. (1967). Analytic description of the low-frequency attenuation coefficient. The Journal of the Acoustical Society of America, 42(1), 270–270."
    ]

    for idx, ref in enumerate(refs):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = ref
        p.font.size = Pt(12)
        p.font.color.rgb = C_TEXT_DARK
        p.font.name = FONT_SERIF
        p.space_after = Pt(14)

    add_footer(s9, 9)

    # =========================================================
    # SLIDE 10: Thank You
    # =========================================================
    s10 = prs.slides.add_slide(blank_layout)

    # Top Amrita banner
    top_bar = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.5), Inches(13.333), Inches(1.0))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = RGBColor(140, 40, 20)
    top_bar.line.color.rgb = RGBColor(140, 40, 20)

    tb = s10.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.733), Inches(1.0))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Amrita Vishwa Vidyapeetham, Amaravati Campus"
    p.alignment = PP_ALIGN.CENTER
    p.font.bold = True
    p.font.size = Pt(22)
    p.font.color.rgb = C_WHITE
    p.font.name = FONT_SERIF

    # Thank you text
    tb_main = s10.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.733), Inches(1.2))
    p = tb_main.text_frame.paragraphs[0]
    p.text = "Thank You"
    p.alignment = PP_ALIGN.CENTER
    p.font.bold = True
    p.font.size = Pt(44)
    p.font.color.rgb = C_TEXT_DARK
    p.font.name = FONT_SERIF

    # Quote
    tb_q = s10.shapes.add_textbox(Inches(1.5), Inches(3.6), Inches(10.333), Inches(1.2))
    p = tb_q.text_frame.paragraphs[0]
    p.text = '"Transforming passive underwater sensor grids into intelligent, adaptive robotic swarms — bringing resilient acoustic surveillance and autonomous traffic governance to maritime harbors."'
    p.alignment = PP_ALIGN.CENTER
    p.font.italic = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = RGBColor(80, 80, 80)
    p.font.name = FONT_SERIF

    # Presenters
    tb_names = s10.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(11.733), Inches(0.6))
    p = tb_names.text_frame.paragraphs[0]
    p.text = "M Ganesh  |  J Kiran  |  P Rahul"
    p.alignment = PP_ALIGN.CENTER
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = C_TEXT_DARK
    p.font.name = FONT_SANS

    add_footer(s10, 10)

    prs.save(OUTPUT_PPTX)
    print(f"Presentation saved successfully to: {OUTPUT_PPTX}")

if __name__ == "__main__":
    build_presentation()
