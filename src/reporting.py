"""PDF rapor oluşturma modülü."""
from fpdf import FPDF
import pandas as pd
import numpy as np
from datetime import datetime
import base64

class EDAReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "CRISP-DM Veri Analizi Raporu", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}", 0, 0, "C")

def generate_pdf_report(df, target, model=None):
    pdf = EDAReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    pdf.write(5, f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
    pdf.write(5, f"Veri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun\n")
    pdf.write(5, f"Hedef değişken: {target}\n\n")
    pdf.write(5, "Temel İstatistikler:\n")
    desc = df.describe(include="all").to_string()
    pdf.set_font("Courier", size=8)
    pdf.multi_cell(0, 4, desc)
    # Model bilgileri varsa ekle
    if model:
        pdf.set_font("Arial", size=11)
        pdf.write(5, "\n\nModel Başarıyla Eğitildi.")
    return pdf.output(dest='S').encode('latin-1')
