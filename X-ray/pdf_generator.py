from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import os

def generate_pdf_report(scan_data, output_path):
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2d0052'),
            spaceAfter=30,
            alignment=1
        )
        
        title = Paragraph("Akciger Roentgen Tarama Raporu", title_style)
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        patient_info_style = ParagraphStyle(
            'PatientInfo',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12
        )
        
        story.append(Paragraph("<b>Hasta Bilgileri:</b>", patient_info_style))
        
        patient_data = [
            ['Ad Soyad:', scan_data.get('patient_name', 'Belirtilmemis')],
            ['Yas:', str(scan_data.get('age', 'Belirtilmemis'))],
            ['Cinsiyet:', scan_data.get('gender', 'Belirtilmemis')],
            ['TC Kimlik No:', scan_data.get('tc_kimlik', 'Belirtilmemis')],
            ['Telefon:', scan_data.get('phone', 'Belirtilmemis')],
            ['Tarama Tarihi:', scan_data.get('scan_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 0.4*inch))
        
        story.append(Paragraph("<b>Tarama Sonuclari:</b>", patient_info_style))
        
        prediction = scan_data.get('prediction', 'UNKNOWN')
        confidence = scan_data.get('confidence', 0)
        
        result_color = colors.green if prediction == 'NORMAL' else colors.red
        
        prediction_tr = 'NORMAL' if prediction == 'NORMAL' else 'ZATORRE (Pnomoni)'
        
        result_data = [
            ['Sonuc:', prediction_tr],
            ['Guven Orani:', f'{confidence}%'],
            ['Tarama No:', f"#{scan_data.get('scan_id', 'N/A')}"]
        ]
        
        result_table = Table(result_data, colWidths=[2*inch, 4*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (1, 0), (1, 0), result_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTSIZE', (1, 0), (1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        story.append(result_table)
        story.append(Spacer(1, 0.4*inch))
        
        story.append(Paragraph("<b>Goruntuler:</b>", patient_info_style))
        story.append(Spacer(1, 0.2*inch))
        
        if 'image_path' in scan_data and scan_data['image_path'] and os.path.exists(scan_data['image_path']):
            story.append(Paragraph("Orijinal Goruntu:", styles['Normal']))
            img = Image(scan_data['image_path'], width=3*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
        
        if 'heatmap_path' in scan_data and scan_data['heatmap_path'] and os.path.exists(scan_data['heatmap_path']):
            story.append(Paragraph("Isil Harita (Grad-CAM):", styles['Normal']))
            heatmap_img = Image(scan_data['heatmap_path'], width=3*inch, height=3*inch)
            story.append(heatmap_img)
            story.append(Spacer(1, 0.3*inch))
        
        if scan_data.get('notes'):
            story.append(Paragraph("<b>Notlar:</b>", patient_info_style))
            notes_para = Paragraph(scan_data['notes'], styles['Normal'])
            story.append(notes_para)
            story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("<b>Tibbi Tavsiyeler:</b>", patient_info_style))
        story.append(Spacer(1, 0.1*inch))
        
        if prediction == 'PNEUMONIA':
            recommendations = [
                "En kisa surede bir uzman hekime danismaniz onerilir",
                "Teshisi dogrulamak icin ek testlere ihtiyaciniz olabilir",
                "Oksuruk, ates veya nefes darligi gibi belirtileri gormezden gelmeyin",
                "Tedaviyle ilgili doktor talimatlarini dikkatlice takip edin"
            ]
        else:
            recommendations = [
                "Sonuclar akciger iltihabinin acik isaretlerini gostermemektedir",
                "Doktorunuzla duzenli takip onerilir",
                "Saglikli bir yasam tarzi surdurun ve sigaradan kacinin",
                "Herhangi bir belirti ortaya cikarsa hemen doktora basvurun"
            ]
        
        bullet_list = ListFlowable(
            [ListItem(Paragraph(rec, styles['Normal']), bulletColor=colors.blue) for rec in recommendations],
            bulletType='bullet',
            start='•'
        )
        story.append(bullet_list)
        
        story.append(Spacer(1, 0.5*inch))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=1
        )
        
        disclaimer = Paragraph(
            "<i>Uyari: Bu rapor bir yapay zeka sistemi tarafindan olusturulmustur ve profesyonel tibbi teshisin yerini almaz. "
            "Kesin teshis icin lutfen uzman bir hekime danisin.</i>",
            disclaimer_style
        )
        story.append(disclaimer)
        
        doc.build(story)
        
        return True
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

def generate_report_filename(scan_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"report_scan_{scan_id}_{timestamp}.pdf"
