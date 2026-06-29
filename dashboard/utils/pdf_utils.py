from io import BytesIO
from datetime import datetime

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)
from reportlab.lib import colors

def add_page_number(canvas, doc):

    page_num = canvas.getPageNumber()

    canvas.setFont("Helvetica", 9)

    canvas.drawRightString(
        560,
        30,
        f"Page {page_num}"
    )

def generate_pdf_report(
    sample,
    probability,
    risk_level,
    top_features,
    ai_report
):
    """
    Generates a professional credit risk report as a PDF.
    Returns the PDF as a BytesIO object.
    """

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    elements = []

    # -------------------------------------------------
    # Title
    # -------------------------------------------------

    elements.append(
        Paragraph(
            "<b>Explainable Credit Risk Assessment Report</b>",
            styles["Heading1"]
        )
    )

    elements.append(
        Paragraph(
            "AI-Powered Explainable Credit Risk Assessment",
            styles["Italic"]
        )
    )

    elements.append(
        Paragraph(
            "<font size=10 color='grey'>"
            "Model: XGBoost + SHAP + Gemini RAG"
            "</font>",
            styles["BodyText"]
        )
    )

    elements.append(Spacer(1, 10))

    elements.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey
        )
    )

    elements.append(Spacer(1, 15))

    # -------------------------------------------------
    # Applicant Summary
    # -------------------------------------------------

    elements.append(
        Paragraph("<b>Applicant Summary</b>", styles["Heading2"])
    )

    applicant_data = [
        [Paragraph("<b>Age</b>", styles["BodyText"]), f"{sample['AGE_YEARS'].iloc[0]:.1f} years"],
        [Paragraph("<b>Employment</b>", styles["BodyText"]), f"{sample['EMPLOYMENT_YEARS'].iloc[0]:.1f} years"],
        [Paragraph("<b>Annual Income</b>", styles["BodyText"]), f"INR  {sample['AMT_INCOME_TOTAL'].iloc[0]:,.0f}"],
        [Paragraph("<b>Credit Amount</b>", styles["BodyText"]), f"INR {sample['AMT_CREDIT'].iloc[0]:,.0f}"],
        [Paragraph("<b>Loan Annuity</b>", styles["BodyText"]), f"INR {sample['AMT_ANNUITY'].iloc[0]:,.0f}"]
    ]

    applicant_table = Table(
        applicant_data,
        colWidths=[170, 220]
    )

    applicant_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8F0FE")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (-1,-1), "CENTER")
    ]))

    elements.append(applicant_table)

    elements.append(Spacer(1, 10))

    elements.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey
        )
    )

    elements.append(Spacer(1, 15))

    # -------------------------------------------------
    # Prediction Summary
    # -------------------------------------------------

    elements.append(
        Paragraph("<b>Prediction Summary</b>", styles["Heading2"])
    )

    prediction_data = [
        [Paragraph("<b>Default Probability</b>", styles["BodyText"]), f"{probability*100:.2f}%"],
        [Paragraph("<b>Risk Level</b>", styles["BodyText"]), risk_level]
    ]

    prediction_table = Table(
        prediction_data, 
        colWidths=[170, 220]
    )

    prediction_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8F0FE")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (-1,-1), "CENTER")
    ]))

    elements.append(prediction_table)

    elements.append(Spacer(1, 10))

    elements.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey
        )
    )

    elements.append(Spacer(1, 15))

    # -------------------------------------------------
    # Top SHAP Features
    # -------------------------------------------------

    elements.append(
        Paragraph("<b>Top SHAP Features Contributions</b>", styles["Heading2"])
    )

    shap_rows = [["Feature", "SHAP Value", "Impact"]]

    for _, row in top_features.iterrows():

        impact = (
            "↑ Increases Risk"
            if row["SHAP Value"] > 0
            else "↓ Decreases Risk"
        )

        shap_rows.append([
            Paragraph(
                row["Feature"],
                styles["BodyText"]
            ),
            f"{row['SHAP Value']:.3f}",
            impact
        ])

    shap_table = Table(
        shap_rows,
        colWidths=[250,70,130]
    )

    shap_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#D9EAF7")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,1), (-1,-1), 10),
        ("TOPPADDING", (0,1), (-1,-1), 10),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))

    elements.append(shap_table)

    elements.append(Spacer(1, 10))

    elements.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey
        )
    )

    elements.append(Spacer(1, 15))

    # -------------------------------------------------
    # AI Report
    # -------------------------------------------------

    elements.append(
        Paragraph("<b>AI Credit Risk Assessment</b>", styles["Heading2"])
    )

    for line in ai_report.split("\n"):

        line = line.strip()
        line = line.replace("**", "")
        line = line.replace("`", "")

        if not line:
            continue

        # Markdown Heading
        if line.startswith("## "):

            elements.append(
                Paragraph(
                    f"<b>{line.replace('## ', '')}</b>",
                    styles["Heading2"]
                )
            )
            continue

        # Bullet Point
        if line.startswith("* "):

            elements.append(
                Paragraph(
                    f"• {line[2:]}",
                    styles["BodyText"]
                )
            )

            continue

        elements.append(
            Paragraph(
                line,
                styles["BodyText"]
            )
        )

    elements.append(Spacer(1, 10))

    elements.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey
        )
    )

    elements.append(Spacer(1, 15))

    # -------------------------------------------------
    # Footer
    # -------------------------------------------------

    elements.append(
        Paragraph(
            "<b>Generated by:</b> Explainable Credit Risk System<br/>"
            f"<b>Date:</b> {datetime.now().strftime('%d %B %Y %H:%M')}",
            styles["Italic"]
        )
    )

    doc.build(
        elements,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    buffer.seek(0)

    return buffer