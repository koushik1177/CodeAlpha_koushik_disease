"""
Medical Diagnostic Report Generator Service.

Generates PDF, CSV, and TXT medical diagnostic reports
for clinical records and patient downloads.
"""

from datetime import datetime
from typing import Dict, Any

from fpdf import FPDF


# ============================================================
# TEXT REPORT
# ============================================================

def generate_txt_report(
    result: Dict[str, Any],
    patient_name: str = "Anonymous Patient"
) -> str:
    """
    Generates a plain-text clinical diagnostic summary report.
    """

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return f"""
==================================================
DIAGNOSTIC CLINICAL SUMMARY REPORT
==================================================

Patient Name: {patient_name}
Date & Time: {timestamp}
Disease Assessed: {result['disease']}

DIAGNOSTIC ASSESSMENT
--------------------------------------------------
Classification Status: {result['status']}
Calculated Risk Probability: {result['risk_percentage']}%
Severity Category: {result['risk_category']}

CLINICAL RECOMMENDATIONS
--------------------------------------------------
- Consult a certified medical specialist for confirmation.
- Review diagnostic indicators and laboratory panels.
- Re-evaluate diagnostic metrics periodically.

DISCLAIMER:
This AI diagnostic report is designed as an educational
decision-support tool. All predictions must be confirmed
by a licensed medical practitioner.

==================================================
""".strip()


# ============================================================
# CSV REPORT
# ============================================================

def generate_csv_report(
    result: Dict[str, Any],
    patient_name: str = "Anonymous Patient"
) -> str:
    """
    Generates a CSV string containing diagnostic metrics.
    """

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    header = (
        "Patient Name,"
        "Disease,"
        "Status,"
        "Risk Percentage,"
        "Risk Category,"
        "Timestamp"
    )

    row = (
        f'"{patient_name}",'
        f'"{result["disease"]}",'
        f'"{result["status"]}",'
        f'{result["risk_percentage"]},'
        f'"{result["risk_category"]}",'
        f'"{timestamp}"'
    )

    return f"{header}\n{row}"


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf_report(
    result: Dict[str, Any],
    patient_name: str = "Anonymous Patient"
) -> bytes:
    """
    Generates a PDF clinical diagnostic report.

    Returns:
        bytes: PDF document bytes.
    """

    pdf = FPDF()

    pdf.set_auto_page_break(
        auto=True,
        margin=15
    )

    pdf.add_page()


    # ========================================================
    # HEADER
    # ========================================================

    pdf.set_font(
        "Helvetica",
        "B",
        16
    )

    pdf.set_text_color(
        30,
        58,
        138
    )

    pdf.cell(
        0,
        10,
        "CLINICAL DIAGNOSTIC REPORT",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C"
    )


    pdf.set_font(
        "Helvetica",
        "",
        10
    )

    pdf.set_text_color(
        100,
        116,
        139
    )

    pdf.cell(
        0,
        6,
        "AI-Powered Risk Assessment System",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C"
    )

    pdf.ln(8)


    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    pdf.set_fill_color(
        241,
        245,
        249
    )

    start_y = pdf.get_y()

    pdf.rect(
        10,
        start_y,
        190,
        28,
        "F"
    )


    pdf.set_xy(
        14,
        start_y + 4
    )

    pdf.set_font(
        "Helvetica",
        "B",
        10
    )

    pdf.set_text_color(
        15,
        23,
        42
    )


    pdf.cell(
        90,
        6,
        f"Patient Name: {patient_name}"
    )

    pdf.cell(
        90,
        6,
        (
            "Date: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ),
        new_x="LMARGIN",
        new_y="NEXT"
    )


    pdf.set_x(14)

    pdf.cell(
        90,
        6,
        f"Disease Evaluated: {result['disease']}"
    )

    pdf.cell(
        90,
        6,
        (
            "Report ID: "
            f"MED-{int(datetime.now().timestamp())}"
        ),
        new_x="LMARGIN",
        new_y="NEXT"
    )


    pdf.ln(12)


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    pdf.set_font(
        "Helvetica",
        "B",
        12
    )

    pdf.set_text_color(
        30,
        58,
        138
    )

    pdf.cell(
        0,
        8,
        "Diagnostic Result Summary",
        new_x="LMARGIN",
        new_y="NEXT"
    )


    pdf.line(
        10,
        pdf.get_y(),
        200,
        pdf.get_y()
    )

    pdf.ln(4)


    # Diagnostic Status

    pdf.set_font(
        "Helvetica",
        "",
        11
    )

    pdf.set_text_color(
        30,
        41,
        59
    )

    pdf.cell(
        60,
        8,
        "Diagnostic Status:"
    )


    pdf.set_font(
        "Helvetica",
        "B",
        11
    )

    pdf.cell(
        0,
        8,
        result["status"],
        new_x="LMARGIN",
        new_y="NEXT"
    )


    # Risk Probability

    pdf.set_font(
        "Helvetica",
        "",
        11
    )

    pdf.cell(
        60,
        8,
        "Risk Probability Score:"
    )


    pdf.set_font(
        "Helvetica",
        "B",
        11
    )

    pdf.cell(
        0,
        8,
        f"{result['risk_percentage']}%",
        new_x="LMARGIN",
        new_y="NEXT"
    )


    # Severity

    pdf.set_font(
        "Helvetica",
        "",
        11
    )

    pdf.cell(
        60,
        8,
        "Severity Category:"
    )


    pdf.set_font(
        "Helvetica",
        "B",
        11
    )

    pdf.cell(
        0,
        8,
        result["risk_category"],
        new_x="LMARGIN",
        new_y="NEXT"
    )


    pdf.ln(8)


    # ========================================================
    # CLINICAL GUIDELINES
    # ========================================================

    pdf.set_font(
        "Helvetica",
        "B",
        12
    )

    pdf.set_text_color(
        30,
        58,
        138
    )

    pdf.cell(
        0,
        8,
        "Clinical Guidelines & Next Steps",
        new_x="LMARGIN",
        new_y="NEXT"
    )


    pdf.line(
        10,
        pdf.get_y(),
        200,
        pdf.get_y()
    )

    pdf.ln(4)


    pdf.set_font(
        "Helvetica",
        "",
        10
    )

    pdf.set_text_color(
        51,
        65,
        85
    )


    recommendations = (
        "1. This assessment is generated using trained "
        "supervised machine learning classifiers.\n"
        "2. Schedule follow-up laboratory panels and "
        "diagnostic imaging with a specialist.\n"
        "3. Maintain routine physiological monitoring "
        "and lifestyle recommendations."
    )


    pdf.multi_cell(
        0,
        6,
        recommendations
    )


    pdf.ln(10)


    # ========================================================
    # DISCLAIMER
    # ========================================================

    pdf.set_font(
        "Helvetica",
        "I",
        8
    )

    pdf.set_text_color(
        148,
        163,
        184
    )


    disclaimer = (
        "DISCLAIMER: This diagnostic report is intended "
        "exclusively for educational and clinical "
        "decision-support assistance. All AI predictions "
        "must be verified by a licensed medical physician."
    )


    pdf.multi_cell(
        0,
        5,
        disclaimer
    )


    # ========================================================
    # RETURN PDF
    # ========================================================

    return bytes(
        pdf.output()
    )