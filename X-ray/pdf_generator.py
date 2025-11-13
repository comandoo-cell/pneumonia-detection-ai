from datetime import datetime
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


_FONT_NAME = "Helvetica"
_FONT_CANDIDATES = []
if os.name == "nt":
    windows_dir = os.environ.get("WINDIR", "C:\\Windows")
    _FONT_CANDIDATES.extend(
        [
            os.path.join(windows_dir, "Fonts", "arial.ttf"),
            os.path.join(windows_dir, "Fonts", "arialuni.ttf"),
        ]
    )

for font_path in _FONT_CANDIDATES:
    if font_path and os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("CustomArial", font_path))
            _FONT_NAME = "CustomArial"
            break
        except Exception:
            continue


def generate_pdf_report(scan_data, output_path):
    try:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
            leftMargin=0.65 * inch,
            rightMargin=0.65 * inch,
        )

        story = []
        styles = getSampleStyleSheet()
        for base_style in ("Normal", "BodyText", "Heading1", "Heading2", "Heading3"):
            if base_style in styles:
                styles[base_style].fontName = _FONT_NAME

        accent_color = colors.HexColor("#1f4e79")
        soft_gray = colors.HexColor("#f4f7fb")
        light_border = colors.HexColor("#c6d6e5")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_logo_path = os.path.join(base_dir, "static", "photo", "logo.png")

        hospital_name = scan_data.get("hospital_name") or "Manisa Celal Bayar Üniversitesi Hastanesi"
        department_name = scan_data.get("department_name") or "Göğüs Radyolojisi Departmanı"
        doctor_name = scan_data.get("doctor_name") or "Muhammed Muhammed"
        scan_id = scan_data.get("scan_id", "N/A")
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        logo_path = scan_data.get("logo_path") or default_logo_path
        logo_img = None

        if logo_path and os.path.exists(logo_path):
            try:
                logo_img = Image(logo_path, width=1.0 * inch, height=1.0 * inch)
            except Exception:
                logo_img = None

        header_title_style = ParagraphStyle(
            "HeaderTitle",
            parent=styles["Heading1"],
            alignment=0,
            fontSize=16,
            textColor=colors.white,
            spaceAfter=2,
            fontName=_FONT_NAME,
        )
        header_sub_style = ParagraphStyle(
            "HeaderSub",
            parent=styles["Normal"],
            alignment=0,
            fontSize=10,
            leading=12,
            textColor=colors.white,
            fontName=_FONT_NAME,
        )
        header_meta_style = ParagraphStyle(
            "HeaderMeta",
            parent=styles["Normal"],
            alignment=2,
            fontSize=10,
            leading=12,
            textColor=colors.white,
            fontName=_FONT_NAME,
        )

        header_rows = [
            [
                logo_img if logo_img else "",
                Paragraph(f"<b>{hospital_name}</b>", header_title_style),
                Paragraph(
                    f"<b>Rapor Tarihi:</b> {report_date}<br/><b>Tarama No:</b> #{scan_id}",
                    header_meta_style,
                ),
            ],
            [
                "",
                Paragraph(department_name, header_sub_style),
                Paragraph(f"<b>Doktor:</b> Dr. {doctor_name}", header_meta_style),
            ],
        ]

        header_table = Table(header_rows, colWidths=[1.1 * inch, 3.7 * inch, 1.5 * inch])
        header_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), accent_color),
                    ("ROWSPAN", (0, 0), (0, 1)),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )

        story.append(header_table)
        story.append(Spacer(1, 0.15 * inch))

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading2"],
            alignment=1,
            textColor=accent_color,
            fontSize=18,
            leading=22,
            spaceAfter=8,
            fontName=_FONT_NAME,
        )
        story.append(Paragraph("Akciğer Röntgen Tarama Raporu", title_style))

        section_label_style = ParagraphStyle(
            "SectionLabel",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            textColor=accent_color,
            spaceAfter=4,
            fontName=_FONT_NAME,
        )
        info_value_style = ParagraphStyle(
            "InfoValue",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#23303f"),
            fontName=_FONT_NAME,
        )

        patient_pairs = [
            ("Ad Soyad", scan_data.get("patient_name", "Belirtilmemiş") or "Belirtilmemiş"),
            ("Yaş", str(scan_data.get("age", "Belirtilmemiş") or "Belirtilmemiş")),
            ("Cinsiyet", scan_data.get("gender", "Belirtilmemiş") or "Belirtilmemiş"),
            ("TC Kimlik No", scan_data.get("tc_kimlik", "Belirtilmemiş") or "Belirtilmemiş"),
            ("Telefon", scan_data.get("phone", "Belirtilmemiş") or "Belirtilmemiş"),
            ("Tarama Tarihi", scan_data.get("scan_date", datetime.now().strftime("%Y-%m-%d %H:%M"))),
        ]

        patient_rows = []
        for idx in range(0, len(patient_pairs), 2):
            first_label, first_value = patient_pairs[idx]
            if idx + 1 < len(patient_pairs):
                second_label, second_value = patient_pairs[idx + 1]
            else:
                second_label, second_value = "", ""

            patient_rows.append(
                [
                    Paragraph(f"<b>{first_label}:</b>", section_label_style),
                    Paragraph(str(first_value), info_value_style),
                    Paragraph(f"<b>{second_label}:</b>", section_label_style) if second_label else Paragraph("", section_label_style),
                    Paragraph(str(second_value), info_value_style) if second_label else Paragraph("", info_value_style),
                ]
            )

        patient_table = Table(patient_rows, colWidths=[1.2 * inch, 1.8 * inch, 1.2 * inch, 1.8 * inch])
        patient_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), soft_gray),
                    ("BOX", (0, 0), (-1, -1), 0.75, light_border),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, light_border),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        story.append(patient_table)
        story.append(Spacer(1, 0.18 * inch))

        prediction = scan_data.get("prediction", "UNKNOWN")
        confidence = scan_data.get("confidence", 0)
        result_color = colors.HexColor("#0f9d58") if prediction == "NORMAL" else colors.HexColor("#d93025")
        prediction_label = "NORMAL" if prediction == "NORMAL" else "ZATORRE (Pnömoni)"
        result_row_style = ParagraphStyle(
            "ResultRow",
            parent=info_value_style,
            fontSize=12,
            leading=14,
            textColor=colors.white,
            fontName=_FONT_NAME,
        )
        summary_header_style = ParagraphStyle(
            "SummaryHeader",
            parent=section_label_style,
            textColor=colors.white,
        )

        summary_table = Table(
            [
                [
                    Paragraph("<b>Tarama Sonucu</b>", summary_header_style),
                    Paragraph("<b>Güven Oranı</b>", summary_header_style),
                    Paragraph("<b>Notlar</b>", summary_header_style),
                ],
                [
                    Paragraph(prediction_label, result_row_style),
                    Paragraph(f"{confidence}%", result_row_style),
                    Paragraph(scan_data.get("notes", "Belirtilmemiş") or "Belirtilmemiş", result_row_style),
                ],
            ],
            colWidths=[2.0 * inch, 1.5 * inch, 2.3 * inch],
        )
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), accent_color),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, 1), accent_color),
                    ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
                    ("LINEBEFORE", (0, 1), (0, 1), 4, result_color),
                    ("BOX", (0, 0), (-1, -1), 0.75, light_border),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, light_border),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(summary_table)
        story.append(Spacer(1, 0.18 * inch))

        images = []
        if scan_data.get("image_path") and os.path.exists(scan_data["image_path"]):
            original_img = Image(scan_data["image_path"], width=2.45 * inch, height=2.45 * inch)
            images.append((Paragraph("Orijinal Görüntü", section_label_style), original_img))

        if scan_data.get("heatmap_path") and os.path.exists(scan_data["heatmap_path"]):
            heatmap_img = Image(scan_data["heatmap_path"], width=2.45 * inch, height=2.45 * inch)
            images.append((Paragraph("Isı Haritası (Grad-CAM)", section_label_style), heatmap_img))

        if images:
            image_row = []
            for label, img in images:
                image_block = Table(
                    [[label], [Spacer(1, 0.05 * inch)], [img]],
                    colWidths=[2.55 * inch],
                )
                image_block.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                            ("BOX", (0, 0), (-1, -1), 0.75, light_border),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                image_row.append(image_block)

            story.append(Table([image_row], colWidths=[2.7 * inch] * len(image_row)))
            story.append(Spacer(1, 0.18 * inch))

        story.append(Paragraph("<b>Tıbbi Tavsiyeler:</b>", section_label_style))

        if prediction == "PNEUMONIA":
            recommendations = [
                "En kısa sürede bir uzman hekime başvurunuz.",
                "Kesin tanı için ek test gerekebilir.",
                "Belirtileri (öksürük, ateş, nefes darlığı) göz ardı etmeyin.",
                "Tedavi ve ilaç talimatlarını dikkatle uygulayın.",
            ]
        else:
            recommendations = [
                "Görüntü normal; düzenli kontrolleri sürdürün.",
                "Sağlıklı yaşam tarzı ve sigaradan uzak durmak önerilir.",
                "Yeni belirtiler ortaya çıkarsa hekiminize danışın.",
                "Doktorunuzun tavsiyelerine göre takip planlayın.",
            ]

        bullet_style = ParagraphStyle(
            "BulletStyle",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#23303f"),
            fontName=_FONT_NAME,
        )
        bullet_list = ListFlowable(
            [ListItem(Paragraph(item, bullet_style), bulletColor=accent_color) for item in recommendations],
            bulletType="bullet",
            start="•",
            leftPadding=12,
        )
        story.append(bullet_list)
        story.append(Spacer(1, 0.18 * inch))

        story.append(HRFlowable(width="100%", thickness=0.8, color=light_border))
        story.append(Spacer(1, 0.08 * inch))

        footer_style = ParagraphStyle(
            "FooterStyle",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#5f6b7a"),
            alignment=1,
            fontName=_FONT_NAME,
        )
        disclaimer = Paragraph(
            "<i>Bu rapor klinik değerlendirme ile birlikte yorumlanmalıdır. Kesin tanı için uzman doktor görüşü gereklidir.</i>",
            footer_style,
        )
        story.append(disclaimer)

        signature_style = ParagraphStyle(
            "SignatureStyle",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#23303f"),
            alignment=0,
            spaceBefore=12,
            fontName=_FONT_NAME,
        )
        story.append(Paragraph(f"Dr. {doctor_name}", signature_style))

        doc.build(story)
        return True

    except Exception as exc:
        print(f"Error generating PDF: {exc}")
        return False


def generate_report_filename(scan_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"report_scan_{scan_id}_{timestamp}.pdf"
