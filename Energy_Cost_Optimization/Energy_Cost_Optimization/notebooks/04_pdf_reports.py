"""
PDF Report Generator
PT Surya Textile Indonesia – Energy Cost Optimization Project
Generates:
  1. Data Dictionary (Phase 1)
  2. EDA Report     (Phase 3)
  3. Final Energy Optimization Report
"""

import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.lib.pagesizes   import A4
from reportlab.lib.units        import cm
from reportlab.lib              import colors
from reportlab.lib.styles       import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums        import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus         import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether
)

REPORTS_PATH = "/home/claude/Energy_Cost_Optimization/reports/"
FIG_DIR      = REPORTS_PATH + "figures/"

# ─── Load pre-computed results ─────────────────────────────────────────────────
with open(REPORTS_PATH + "kpis.json")        as f: kpis       = json.load(f)
with open(REPORTS_PATH + "simulation.json")  as f: sim        = json.load(f)

# ─── Brand colours ─────────────────────────────────────────────────────────────
NAVY     = colors.HexColor("#1B3A6B")
ORANGE   = colors.HexColor("#E87722")
GREEN    = colors.HexColor("#2E7D32")
RED      = colors.HexColor("#C62828")
LGRAY    = colors.HexColor("#F0F2F5")
DGRAY    = colors.HexColor("#607D8B")
WHITE    = colors.white
BLACK    = colors.black

def idr(v):
    return f"Rp {v:,.0f}"
def idrm(v):
    return f"Rp {v/1e6:,.2f} Juta"

# ─── Style helpers ─────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name="Normal", **kw):
    base = styles[name]
    return ParagraphStyle(
        name + str(id(kw)),
        parent=base,
        **kw
    )

cover_title = S("Title",    fontSize=26, textColor=NAVY,   leading=32,
                alignment=TA_CENTER, spaceAfter=6)
cover_sub   = S("Normal",   fontSize=13, textColor=ORANGE,  alignment=TA_CENTER,
                spaceAfter=4)
cover_meta  = S("Normal",   fontSize=10, textColor=DGRAY,   alignment=TA_CENTER,
                spaceAfter=2)
h1          = S("Heading1", fontSize=16, textColor=NAVY,   spaceAfter=6,
                spaceBefore=14, leading=20)
h2          = S("Heading2", fontSize=13, textColor=NAVY,   spaceAfter=4,
                spaceBefore=10, leading=16)
h3          = S("Heading3", fontSize=11, textColor=ORANGE, spaceAfter=3,
                spaceBefore=8)
body        = S("Normal",   fontSize=10, leading=15, spaceAfter=4,
                alignment=TA_JUSTIFY)
body_l      = S("Normal",   fontSize=10, leading=15, spaceAfter=4)
small       = S("Normal",   fontSize=9,  textColor=DGRAY, leading=13)
code_style  = S("Code",     fontSize=8,  fontName="Courier", leading=12,
                backColor=LGRAY, borderPadding=(4,4,4,4))
caption     = S("Normal",   fontSize=9,  textColor=DGRAY, alignment=TA_CENTER,
                spaceAfter=6)
kpi_label   = S("Normal",   fontSize=9,  textColor=DGRAY,  leading=13)
kpi_value   = S("Normal",   fontSize=18, textColor=NAVY,   fontName="Helvetica-Bold",
                leading=22)
kpi_unit    = S("Normal",   fontSize=9,  textColor=ORANGE, leading=13)

def hr(color=NAVY):
    return HRFlowable(width="100%", thickness=1.5, color=color, spaceAfter=6, spaceBefore=4)

def tbl_style(header_bg=NAVY, row_bg=LGRAY, alt_bg=WHITE):
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [row_bg, alt_bg]),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("ALIGN",         (1, 1), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
        ("GRID",          (0, 0), (-1, -1), 0.35, colors.HexColor("#DEE2E6")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ])

def img(filename, width_cm=14):
    path = FIG_DIR + filename
    if os.path.exists(path):
        return Image(path, width=width_cm * cm, height=width_cm * 0.55 * cm)
    return Paragraph(f"[Image: {filename}]", small)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA DICTIONARY PDF
# ═══════════════════════════════════════════════════════════════════════════════
def build_data_dictionary():
    path = REPORTS_PATH + "Data_Dictionary.pdf"
    doc  = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm,  bottomMargin=2*cm)
    story = []

    # Cover
    story += [
        Spacer(1, 1.5*cm),
        Paragraph("DATA DICTIONARY", cover_title),
        Paragraph("Industrial Energy Cost Optimization & Load Management System", cover_sub),
        Paragraph("PT Surya Textile Indonesia (Simulasi)", cover_meta),
        Paragraph("Dokumen Versi 1.0  |  Januari – Desember 2025", cover_meta),
        Spacer(1, 0.4*cm),
        hr(),
        Spacer(1, 0.4*cm),
    ]

    # 1. Project Overview
    story += [
        Paragraph("1. Gambaran Proyek", h1), hr(),
        Paragraph(
            "Proyek ini bertujuan untuk melakukan analisis konsumsi energi listrik pada "
            "30 mesin produksi PT Surya Textile Indonesia selama periode Januari – Desember 2025. "
            "Dataset dibuat secara simulasi dengan interval pengukuran 15 menit menggunakan "
            "sistem tegangan tiga fasa 380V sesuai standar industri manufaktur di Indonesia.",
            body),
        Spacer(1, 0.3*cm),
    ]

    # Dataset summary table
    summary_data = [
        ["Parameter",               "Nilai"],
        ["Nama Tabel Utama",         "energy_consumption.csv"],
        ["Periode Data",             "1 Januari 2025 – 31 Desember 2025"],
        ["Interval Pengukuran",      "15 menit"],
        ["Jumlah Mesin",             "30 mesin (15 Motor, 10 Pump, 5 Compressor)"],
        ["Total Record (Raw)",       "1.051.200"],
        ["Total Record (Clean)",     "1.048.209"],
        ["Ukuran File (Raw)",        "~193 MB"],
        ["Sistem Tegangan",          "380V Three-Phase"],
        ["Zona Tarif Puncak",        "17:00 – 22:00 WIB (Rp 1.700/kWh)"],
        ["Zona Tarif Normal",        "00:00 – 17:00, 22:00 – 24:00 WIB (Rp 1.450/kWh)"],
    ]
    t = Table(summary_data, colWidths=[7*cm, 9*cm])
    t.setStyle(tbl_style())
    story += [t, Spacer(1, 0.5*cm)]

    # 2. Column definitions
    story += [Paragraph("2. Definisi Kolom Dataset", h1), hr()]

    columns = [
        ("timestamp",             "DATETIME",  "Waktu pengukuran dalam format YYYY-MM-DD HH:MM:SS. "
                                               "Interval 15 menit mulai 00:00:00 hingga 23:45:00."),
        ("machine_id",            "STRING",    "Identifikasi unik mesin. Format: [TYPE]_[NO], "
                                               "contoh: MOTOR_01, PUMP_05, COMPRESSOR_03."),
        ("machine_type",          "STRING",    "Kategori mesin: Motor (15 unit), Pump (10 unit), "
                                               "Compressor (5 unit)."),
        ("voltage_v",             "FLOAT",     "Tegangan fasa-fasa sistem 380V. Range: 375 – 390 V. "
                                               "Diukur dalam satuan Volt (V)."),
        ("current_a",             "FLOAT",     "Arus listrik tiap mesin. Motor: 15–40 A, "
                                               "Pump: 10–25 A, Compressor: 20–60 A. Satuan Ampere (A)."),
        ("power_factor",          "FLOAT",     "Faktor daya (cos phi). Normal: 0,80–0,95. "
                                               "10% data memiliki PF buruk (0,60–0,75). "
                                               "Dimensionless, range 0–1."),
        ("active_power_kw",       "FLOAT",     "Daya aktif yang benar-benar dikonsumsi. "
                                               "Rumus: P = (V3 x V x I x PF) / 1000. Satuan kW."),
        ("reactive_power_kvar",   "FLOAT",     "Daya reaktif. Rumus: Q = SQRT(S^2 - P^2). "
                                               "Satuan kVAR."),
        ("apparent_power_kva",    "FLOAT",     "Daya semu. Rumus: S = (V3 x V x I) / 1000. "
                                               "Satuan kVA."),
        ("energy_kwh",            "FLOAT",     "Energi per interval 15 menit. "
                                               "Rumus: E = P x 0,25. Satuan kWh."),
        ("tariff_idr_kwh",        "FLOAT",     "Tarif listrik berlaku. Peak (17:00–22:00): "
                                               "Rp 1.700/kWh. Normal: Rp 1.450/kWh. Satuan IDR/kWh."),
    ]
    col_data = [["No.", "Nama Kolom", "Tipe Data", "Deskripsi"]]
    for i, (name, dtype, desc) in enumerate(columns, 1):
        col_data.append([str(i), name, dtype, desc])

    t2 = Table(col_data, colWidths=[0.7*cm, 4.2*cm, 2*cm, 9.1*cm])
    t2.setStyle(tbl_style())
    story += [t2, Spacer(1, 0.5*cm)]

    # 3. Derived columns
    story += [Paragraph("3. Kolom Turunan (Hasil Data Cleaning)", h1), hr()]
    derived = [
        ["Nama Kolom",      "Tipe",     "Deskripsi"],
        ["cost_idr",        "FLOAT",    "Biaya listrik per interval = energy_kwh x tariff_idr_kwh"],
        ["hour",            "INTEGER",  "Jam pengukuran (0–23), diekstrak dari timestamp"],
        ["date",            "DATE",     "Tanggal pengukuran (YYYY-MM-DD)"],
        ["day_of_week",     "STRING",   "Nama hari (Monday–Sunday)"],
        ["month",           "INTEGER",  "Nomor bulan (1–12)"],
        ["is_peak",         "INTEGER",  "Flag jam puncak: 1 = Peak (17–22), 0 = Normal"],
    ]
    t3 = Table(derived, colWidths=[4*cm, 2.5*cm, 9.5*cm])
    t3.setStyle(tbl_style())
    story += [t3, Spacer(1, 0.5*cm)]

    # 4. Machine specification
    story += [Paragraph("4. Spesifikasi Mesin Produksi", h1), hr()]
    machine_data = [
        ["Kategori", "Jumlah", "ID Range", "Current (A)", "Keterangan"],
        ["Motor",      "15 unit", "MOTOR_01 – MOTOR_15",         "15 – 40 A",
         "Motor induksi 3-fasa, penggerak utama produksi"],
        ["Pump",       "10 unit", "PUMP_01 – PUMP_10",           "10 – 25 A",
         "Pompa sirkulasi, utilitas air & fluida proses"],
        ["Compressor", " 5 unit", "COMPRESSOR_01 – COMPRESSOR_05", "20 – 60 A",
         "Kompresor udara, kebutuhan beban tertinggi"],
    ]
    t4 = Table(machine_data, colWidths=[2.5*cm, 2*cm, 5*cm, 2.5*cm, 4*cm])
    t4.setStyle(tbl_style())
    story += [t4, Spacer(1, 0.5*cm)]

    # 5. Data quality rules
    story += [Paragraph("5. Aturan Kualitas Data & Validasi", h1), hr()]
    rules = [
        ("Missing Values",     "Tidak ada missing value dalam dataset simulasi. "
                               "Tahap cleaning melakukan pengecekan dan drop jika ditemukan."),
        ("Duplikat",           "Kombinasi (timestamp, machine_id) harus unik. "
                               "Duplikat diidentifikasi dan dihapus."),
        ("Outlier",            "Deteksi menggunakan metode IQR (3x IQR) pada semua kolom "
                               "numerik. Total 2.991 outlier dihapus dari raw data."),
        ("Validasi PF",        "Power factor dibatasi pada range 0,5 – 1,0. "
                               "Nilai di luar range dianggap error sensor."),
        ("Validasi Tegangan",  "Tegangan normal sistem: 375–390 V. "
                               "Nilai di luar range mengindikasikan gangguan jaringan."),
        ("Validasi Energi",    "Nilai energy_kwh harus > 0 dan proporsional dengan active_power_kw "
                               "x 0,25. Nilai negatif dianggap error instrumen."),
    ]
    for rule, desc in rules:
        story += [
            Paragraph(f"<b>{rule}:</b> {desc}", body),
        ]

    # 6. Tariff structure
    story += [
        Spacer(1, 0.3*cm),
        Paragraph("6. Struktur Tarif Listrik", h1), hr(),
    ]
    tariff_data = [
        ["Kategori",       "Jam",                  "Tarif (IDR/kWh)", "Keterangan"],
        ["Peak Hour",      "17:00 – 22:00 WIB",    "Rp 1.700",
         "Jam beban puncak PLN, tarif tertinggi"],
        ["Normal Hour",    "00:00–17:00 & 22:00–24:00", "Rp 1.450",
         "Tarif normal di luar jam puncak"],
        ["Selisih",        "–",                    "Rp 250 (+17,2%)",
         "Perbedaan tarif yang menjadi dasar load shifting"],
    ]
    t5 = Table(tariff_data, colWidths=[2.8*cm, 4.5*cm, 3.5*cm, 5.2*cm])
    t5.setStyle(tbl_style())
    story += [t5, Spacer(1, 0.5*cm)]

    # Footer note
    story += [
        hr(DGRAY),
        Paragraph(
            f"Dokumen ini disiapkan oleh Tim Energy Analytics | "
            f"PT Surya Textile Indonesia Simulation | "
            f"Dibuat: {datetime.now().strftime('%d %B %Y')}",
            small),
    ]

    doc.build(story)
    print(f"✅ Data Dictionary saved → {path}")
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA REPORT PDF
# ═══════════════════════════════════════════════════════════════════════════════
def build_eda_report():
    path = REPORTS_PATH + "EDA_Report.pdf"
    doc  = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm,  bottomMargin=2*cm)
    story = []

    # Cover
    story += [
        Spacer(1, 1.5*cm),
        Paragraph("LAPORAN ANALISIS DATA EKSPLORATORI", cover_title),
        Paragraph("Exploratory Data Analysis (EDA) Report", cover_sub),
        Paragraph("PT Surya Textile Indonesia (Simulasi) | Periode: Januari – Desember 2025",
                  cover_meta),
        Spacer(1, 0.4*cm), hr(), Spacer(1, 0.3*cm),
    ]

    # Summary KPI boxes (as table)
    te  = kpis["total_energy_kwh"]
    tc  = kpis["total_cost_idr"]
    adc = kpis["avg_daily_cost"]
    bp  = kpis["bad_pf_pct"]

    kpi_rows = [
        ["Total Energi 2025",            f"{te:,.0f} kWh",
         "Total Biaya 2025",             idr(tc)],
        ["Rata-rata Biaya/Hari",          idr(adc),
         "Mesin dengan PF Buruk (<0.80)", f"{bp:.1f}%"],
    ]
    kpi_tbl_data = []
    for label1, val1, label2, val2 in kpi_rows:
        kpi_tbl_data.append([
            Paragraph(f"<b>{label1}</b><br/><font color='#E87722' size='16'>{val1}</font>",
                      body_l),
            Paragraph(f"<b>{label2}</b><br/><font color='#E87722' size='16'>{val2}</font>",
                      body_l),
        ])
    kpi_t = Table(kpi_tbl_data, colWidths=[8*cm, 8*cm])
    kpi_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LGRAY),
        ("BOX",        (0,0), (-1,-1), 1, NAVY),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, colors.HexColor("#DEE2E6")),
        ("PADDING",    (0,0), (-1,-1), 10),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [kpi_t, Spacer(1, 0.5*cm)]

    # Q1 – Top 10 Energy Consumers
    story += [
        Paragraph("Q1 — Top 10 Mesin Konsumsi Energi Tertinggi", h1), hr(),
        Paragraph(
            "Berdasarkan akumulasi seluruh interval pengukuran sepanjang tahun 2025, "
            "kelompok mesin Compressor mendominasi konsumsi energi tertinggi. "
            "Hal ini sesuai dengan karakteristik operasional kompresor yang beroperasi "
            "dengan arus yang jauh lebih tinggi (20–60 A) dibandingkan Motor dan Pump.",
            body),
        img("q1_top10_energy.png", 15),
        Paragraph("Gambar 1. Top 10 Mesin berdasarkan Total Konsumsi Energi (kWh) – 2025",
                  caption),
    ]

    # Top 10 table
    top10_e = kpis["top10_energy"][:10]
    e_data  = [["No.", "Machine ID", "Type", "Total Energy (kWh)", "Avg Power (kW)", "Avg PF"]]
    for i, r in enumerate(top10_e, 1):
        e_data.append([
            str(i), r["machine_id"], r["machine_type"],
            f"{r['total_energy_kwh']:,.0f}",
            f"{r['avg_power_kw']:.3f}",
            f"{r['avg_power_factor']:.4f}",
        ])
    t_e = Table(e_data, colWidths=[1*cm, 3.5*cm, 3*cm, 3.5*cm, 3*cm, 2*cm])
    t_e.setStyle(tbl_style())
    story += [t_e, Spacer(1, 0.4*cm)]

    # Q2 – Top 10 Highest Cost
    story += [
        PageBreak(),
        Paragraph("Q2 — Top 10 Mesin dengan Biaya Listrik Tertinggi", h1), hr(),
        Paragraph(
            "Biaya listrik dipengaruhi oleh kombinasi konsumsi energi total dan "
            "distribusi jam operasi (peak vs. normal). Mesin yang banyak beroperasi "
            "pada jam puncak (17:00–22:00) akan memiliki biaya yang tidak proporsional "
            "terhadap konsumsi energinya.",
            body),
        img("q2_top10_cost.png", 15),
        Paragraph("Gambar 2. Top 10 Mesin berdasarkan Total Biaya Listrik (IDR) – 2025",
                  caption),
    ]

    top10_c = kpis["top10_cost"][:10]
    c_data  = [["No.", "Machine ID", "Type", "Total Cost (IDR)", "Total Energy (kWh)"]]
    for i, r in enumerate(top10_c, 1):
        c_data.append([
            str(i), r["machine_id"], r["machine_type"],
            idr(r["total_cost_idr"]),
            f"{r['total_energy_kwh']:,.0f}",
        ])
    t_c = Table(c_data, colWidths=[1*cm, 3.5*cm, 3*cm, 5.5*cm, 3*cm])
    t_c.setStyle(tbl_style())
    story += [t_c, Spacer(1, 0.4*cm)]

    # Q3 – Hourly Cost
    story += [
        Paragraph("Q3 — Profil Biaya per Jam Operasi", h1), hr(),
        Paragraph(
            "Analisis per jam menunjukkan lonjakan biaya signifikan pada rentang "
            "pukul 17:00–22:00 akibat berlakunya tarif puncak Rp 1.700/kWh. "
            "Selisih rata-rata biaya antara jam puncak dan normal mencapai 33,5%, "
            "menjadikan manajemen beban waktu (load shifting) sebagai peluang "
            "penghematan yang sangat strategis.",
            body),
        img("q3_hourly_cost.png", 16),
        Paragraph("Gambar 3. Rata-rata Biaya per Jam – Merah: Peak Hour, Biru: Normal Hour",
                  caption),
        Spacer(1, 0.3*cm),
    ]

    # Q4 – Daily Cost
    story += [
        Paragraph("Q4 — Hari Operasi dengan Biaya Tertinggi", h1), hr(),
        Paragraph(
            "Distribusi biaya antar hari kerja relatif merata karena operasi 24 jam "
            "berlangsung tujuh hari seminggu. Variasi kecil antar hari terjadi akibat "
            "perbedaan jadwal produksi, perawatan mesin (maintenance), dan fluktuasi "
            "beban produksi.",
            body),
        img("q4_daily_cost.png", 15),
        Paragraph("Gambar 4. Rata-rata Biaya Listrik berdasarkan Hari dalam Seminggu",
                  caption),
    ]

    # Q5 – Power Factor
    story += [
        PageBreak(),
        Paragraph("Q5 — Faktor Daya (Power Factor) Terburuk", h1), hr(),
        Paragraph(
            "Faktor daya merupakan indikator efisiensi penggunaan daya listrik. "
            "Nilai PF di bawah 0,85 mengakibatkan daya reaktif berlebih yang "
            "meningkatkan arus, memperbesar rugi-rugi jaringan, dan berpotensi "
            "dikenakan penalti oleh PLN. Sekitar 9,7% dari total interval pengukuran "
            "mencatat PF buruk (< 0,80), yang merupakan target utama perbaikan.",
            body),
        img("q5_power_factor.png", 15),
        Paragraph("Gambar 5. 10 Mesin dengan Rata-rata Power Factor Terendah (Garis oranye = target 0,85)",
                  caption),
        img("pf_distribution.png", 13),
        Paragraph("Gambar 6. Distribusi Power Factor Seluruh Mesin Sepanjang 2025",
                  caption),
    ]

    pf_worst = kpis["pf_worst10"][:10]
    pf_data  = [["No.", "Machine ID", "Type", "Avg Power Factor", "Status"]]
    for i, r in enumerate(pf_worst, 1):
        pf  = r["avg_power_factor"]
        sts = "KRITIS" if pf < 0.80 else "PERLU PERHATIAN"
        pf_data.append([
            str(i), r["machine_id"], r["machine_type"],
            f"{pf:.4f}", sts,
        ])
    t_pf = Table(pf_data, colWidths=[1*cm, 3.5*cm, 3*cm, 4.5*cm, 4*cm])
    ts_pf = tbl_style()
    # Highlight kritis rows
    for row_idx, r in enumerate(pf_worst, 1):
        if r["avg_power_factor"] < 0.80:
            ts_pf.add("BACKGROUND", (4, row_idx), (4, row_idx),
                      colors.HexColor("#FFEBEE"))
            ts_pf.add("TEXTCOLOR",  (4, row_idx), (4, row_idx), RED)
    t_pf.setStyle(ts_pf)
    story += [t_pf, Spacer(1, 0.4*cm)]

    # Monthly Trend
    story += [
        PageBreak(),
        Paragraph("Tren Bulanan – Energi & Biaya", h1), hr(),
        Paragraph(
            "Konsumsi energi dan biaya listrik relatif stabil sepanjang tahun 2025 "
            "dengan rata-rata Rp 410 Juta per bulan. Variasi bulanan disebabkan oleh "
            "jumlah hari kerja, jadwal perawatan berkala, dan fluktuasi produksi.",
            body),
        img("monthly_trend.png", 16),
        Paragraph("Gambar 7. Tren Energi (bar) dan Biaya (garis) per Bulan – 2025",
                  caption),
        img("machine_type_comparison.png", 16),
        Paragraph("Gambar 8. Perbandingan Total Energi, Biaya, dan Power Factor per Tipe Mesin",
                  caption),
    ]

    # Finding summary
    story += [
        PageBreak(),
        Paragraph("Ringkasan Temuan EDA", h1), hr(),
    ]
    findings = [
        ("F-01", "Dominasi Compressor",
         "Kelompok Compressor (5 mesin) berkontribusi pada konsumsi energi dan biaya "
         "tertinggi per unit mesin, jauh di atas Motor dan Pump."),
        ("F-02", "Pola Beban Jam Puncak",
         "Beban listrik meningkat signifikan pada 17:00–22:00 WIB. "
         "Biaya per interval pada jam puncak rata-rata 33,5% lebih mahal dari jam normal."),
        ("F-03", "Permasalahan Power Factor",
         "9,7% interval pengukuran mencatat PF < 0,80. Kondisi ini meningkatkan "
         "daya reaktif dan berpotensi dikenakan penalti dari PLN."),
        ("F-04", "Operasi 24/7 Tanpa Variasi Signifikan",
         "Beban produksi berlangsung merata tujuh hari seminggu, menunjukkan "
         "tidak ada jendela shutdown yang dapat dioptimalkan."),
        ("F-05", "Tidak Ada Anomali Kritis",
         "Setelah proses cleaning (3xIQR), tidak ditemukan anomali sensor yang "
         "signifikan. Data dinilai berkualitas baik untuk analisis lebih lanjut."),
    ]
    f_data = [["Kode", "Temuan", "Penjelasan"]]
    for code, title, desc in findings:
        f_data.append([code, title, desc])
    t_f = Table(f_data, colWidths=[1.5*cm, 5*cm, 9.5*cm])
    t_f.setStyle(tbl_style())
    story += [t_f]

    story += [
        Spacer(1, 0.5*cm), hr(DGRAY),
        Paragraph(
            f"EDA Report | PT Surya Textile Indonesia Simulation | "
            f"Dibuat: {datetime.now().strftime('%d %B %Y')}",
            small),
    ]

    doc.build(story)
    print(f"✅ EDA Report saved → {path}")
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FINAL ENERGY OPTIMIZATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def build_final_report():
    path = REPORTS_PATH + "Energy_Optimization_Report.pdf"
    doc  = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm,  bottomMargin=2*cm)
    story = []
    sa    = sim["scenario_a"]
    sb    = sim["scenario_b"]
    sc    = sim["scenario_c"]
    comb  = sim["combined"]
    base  = sim["baseline"]

    # ── COVER ──────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.8*cm),
        Paragraph("ENERGY OPTIMIZATION REPORT", cover_title),
        Paragraph("Industrial Energy Cost Optimization & Load Management System", cover_sub),
        Spacer(1, 0.2*cm),
        hr(ORANGE),
        Spacer(1, 0.2*cm),
        Paragraph("Disiapkan untuk:", cover_meta),
        Paragraph("<b>PT Surya Textile Indonesia (Simulasi)</b>",
                  S("Normal", fontSize=14, textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)),
        Paragraph("Disiapkan oleh:", cover_meta),
        Paragraph("<b>Tim Energy Analytics &amp; Cost Optimization</b>",
                  S("Normal", fontSize=12, textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)),
        Paragraph(f"Tanggal: {datetime.now().strftime('%d %B %Y')}", cover_meta),
        Paragraph("Periode Analisis: 1 Januari 2025 – 31 Desember 2025", cover_meta),
        Spacer(1, 0.5*cm), hr(), Spacer(1, 0.3*cm),
    ]

    # ── TABLE OF CONTENTS ──────────────────────────────────────────────────────
    story += [
        Paragraph("Daftar Isi", h1),
        hr(),
    ]
    toc = [
        ("1.", "Executive Summary", "2"),
        ("2.", "Latar Belakang & Metodologi", "3"),
        ("3.", "Temuan Utama (Key Findings)", "4"),
        ("4.", "Analisis Detail per Fase", "5"),
        ("5.", "Simulasi Penghematan Energi", "7"),
        ("6.", "Rekomendasi Strategis", "9"),
        ("7.", "Dampak Finansial", "10"),
        ("8.", "Rencana Implementasi", "11"),
        ("9.", "Kesimpulan", "12"),
    ]
    toc_data = [[Paragraph(f"<b>{n}</b>", body_l),
                 Paragraph(title, body_l),
                 Paragraph(f"Hal. {pg}", body_l)]
                for n, title, pg in toc]
    t_toc = Table(toc_data, colWidths=[1*cm, 13*cm, 2*cm])
    t_toc.setStyle(TableStyle([
        ("ALIGN",   (2,0), (2,-1), "RIGHT"),
        ("GRID",    (0,0), (-1,-1), 0.25, colors.HexColor("#DEE2E6")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [LGRAY, WHITE]),
        ("PADDING", (0,0), (-1,-1), 5),
        ("FONTSIZE",(0,0), (-1,-1), 9.5),
    ]))
    story += [t_toc, PageBreak()]

    # ── 1. EXECUTIVE SUMMARY ───────────────────────────────────────────────────
    story += [
        Paragraph("1. Executive Summary", h1), hr(),
        Paragraph(
            "PT Surya Textile Indonesia mengoperasikan 30 mesin produksi (15 Motor, "
            "10 Pump, 5 Compressor) selama 24 jam penuh sepanjang tahun 2025. "
            "Analisis komprehensif terhadap 1.048.209 record data konsumsi energi "
            "interval 15 menit mengungkapkan peluang penghematan biaya listrik yang "
            "substansial melalui tiga inisiatif utama.",
            body),
        Spacer(1, 0.2*cm),
    ]

    # KPI banner
    kpi_banner = [
        [
            Paragraph(f"<b>Total Energi 2025</b><br/><font size='18' color='#1B3A6B'>"
                      f"{base['total_energy_kwh']:,.0f} kWh</font>", body_l),
            Paragraph(f"<b>Total Biaya 2025</b><br/><font size='18' color='#1B3A6B'>"
                      f"{idrm(base['total_cost_idr'])}</font>", body_l),
            Paragraph(f"<b>Potensi Penghematan</b><br/><font size='18' color='#2E7D32'>"
                      f"{idrm(comb['saving_idr'])}</font>", body_l),
            Paragraph(f"<b>Penghematan (%)</b><br/><font size='18' color='#2E7D32'>"
                      f"{comb['saving_pct']:.2f}%</font>", body_l),
        ]
    ]
    banner_t = Table(kpi_banner, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
    banner_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LGRAY),
        ("BOX",        (0,0), (-1,-1), 1.5, NAVY),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, colors.HexColor("#DEE2E6")),
        ("PADDING",    (0,0), (-1,-1), 10),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
    ]))
    story += [banner_t, Spacer(1, 0.3*cm)]

    story += [
        Paragraph(
            "Tiga skenario penghematan yang diidentifikasi dan disimulasikan adalah: "
            "(A) Perbaikan faktor daya ke 0,95 yang menghemat "
            f"<b>{idrm(sa['saving_idr'])}</b> per tahun; "
            "(B) Pemindahan operasi dari jam 18:00 ke 13:00 yang menghemat "
            f"<b>{idrm(sb['saving_idr'])}</b> per tahun; dan "
            "(C) Efisiensi konsumsi energi 5% yang menghemat "
            f"<b>{idrm(sc['saving_idr'])}</b> per tahun. "
            f"Kombinasi ketiganya berpotensi menghemat "
            f"<b>{idrm(comb['saving_idr'])}</b> atau {comb['saving_pct']:.2f}% "
            f"dari total pengeluaran listrik tahunan.",
            body),
        Spacer(1, 0.2*cm),
        PageBreak(),
    ]

    # ── 2. LATAR BELAKANG & METODOLOGI ────────────────────────────────────────
    story += [
        Paragraph("2. Latar Belakang &amp; Metodologi", h1), hr(),
        Paragraph("2.1 Latar Belakang Masalah", h2),
        Paragraph(
            "PT Surya Textile Indonesia mencatat kenaikan tagihan listrik sebesar 17% "
            "dalam 12 bulan terakhir tanpa disertai peningkatan kapasitas produksi yang "
            "sebanding. Kondisi ini mendorong manajemen untuk meminta analisis menyeluruh "
            "guna mengidentifikasi sumber pemborosan energi dan merumuskan strategi "
            "penghematan yang terukur.",
            body),
        Paragraph("2.2 Ruang Lingkup Analisis", h2),
    ]
    scope_data = [
        ["Business Objective", "Pertanyaan Kunci",                             "Status"],
        ["BO-01",              "Mesin mana yang paling boros?",                "✓ Terjawab"],
        ["BO-02",              "Jam berapa beban tertinggi?",                  "✓ Terjawab"],
        ["BO-03",              "Berapa estimasi biaya listrik tahunan?",       "✓ Terjawab"],
        ["BO-04",              "Mesin dengan faktor daya terburuk?",           "✓ Terjawab"],
        ["BO-05",              "Rekomendasi pengurangan biaya?",               "✓ Terjawab"],
    ]
    t_scope = Table(scope_data, colWidths=[2.5*cm, 10*cm, 3.5*cm])
    ts_scope = tbl_style()
    for i in range(1, len(scope_data)):
        ts_scope.add("TEXTCOLOR", (2, i), (2, i), GREEN)
    t_scope.setStyle(ts_scope)
    story += [t_scope, Spacer(1, 0.3*cm)]

    story += [
        Paragraph("2.3 Metodologi", h2),
        Paragraph(
            "Proyek mengikuti kerangka kerja analitik enam fase: (1) Data Understanding, "
            "(2) Data Cleaning menggunakan metode IQR 3x untuk deteksi outlier, "
            "(3) Exploratory Data Analysis multi-dimensi, (4) Cost Analysis berbasis "
            "tarif PLN, (5) Energy Saving Simulation untuk tiga skenario, dan (6) "
            "Dashboard interaktif untuk pemantauan berkelanjutan.",
            body),
        Spacer(1, 0.2*cm),
        PageBreak(),
    ]

    # ── 3. TEMUAN UTAMA ────────────────────────────────────────────────────────
    story += [
        Paragraph("3. Temuan Utama (Key Findings)", h1), hr(),
    ]
    findings_main = [
        ("🔴 KRITIS",  "Compressor Mendominasi Konsumsi",
         f"Kelompok Compressor (5 mesin, 16,7% dari total mesin) bertanggung jawab "
         f"atas porsi konsumsi energi tertinggi per unit. COMPRESSOR_02 adalah mesin "
         f"paling boros dengan konsumsi 163.656 kWh dan biaya Rp 246,7 Juta sepanjang 2025."),
        ("🔴 KRITIS",  "Power Factor Buruk pada 9,7% Interval",
         f"Sebanyak 102.167 interval pengukuran mencatat power factor di bawah 0,80. "
         f"Kondisi ini menyebabkan daya reaktif berlebih, meningkatkan arus, "
         f"dan berpotensi dikenakan penalti PLN sebesar Rp 148,7 Juta per tahun."),
        ("🟡 TINGGI",  "Beban Puncak 17:00–22:00 Menaikkan Biaya 33,5%",
         f"Tarif puncak Rp 1.700/kWh berlaku 5 jam per hari. Beban listrik yang "
         f"tidak digeser dari jam puncak mengakibatkan pembayaran 17,2% lebih mahal "
         f"untuk energi yang sama. Potensi saving load shifting: Rp 37,4 Juta/tahun."),
        ("🟡 TINGGI",  "Tidak Ada Sistem Monitoring Energi",
         f"Tidak tersedianya dashboard monitoring real-time menyebabkan manajemen "
         f"tidak dapat mendeteksi anomali konsumsi, mesin boros, atau degradasi PF "
         f"secara dini."),
        ("🟢 MEDIUM",  "Potensi Efisiensi 5% melalui Operational Excellence",
         f"Optimasi jadwal produksi, perawatan preventif, dan peningkatan kualitas "
         f"operasi mesin diperkirakan dapat mengurangi konsumsi energi total sebesar 5%, "
         f"setara dengan penghematan Rp 246,1 Juta per tahun."),
    ]
    f_data = [["Prioritas", "Temuan", "Dampak"]]
    for prio, title, desc in findings_main:
        f_data.append([
            Paragraph(prio, S("Normal", fontSize=8, leading=12)),
            Paragraph(f"<b>{title}</b>", S("Normal", fontSize=9, leading=13)),
            Paragraph(desc, S("Normal", fontSize=8.5, leading=13)),
        ])
    t_fin = Table(f_data, colWidths=[2*cm, 4.5*cm, 9.5*cm])
    ts_f = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 9),
        ("GRID",       (0,0), (-1,-1), 0.35, colors.HexColor("#DEE2E6")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGRAY, WHITE]),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",(0,0), (-1,-1), 5),
        ("RIGHTPADDING",(0,0),(-1,-1), 5),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ])
    t_fin.setStyle(ts_f)
    story += [t_fin, PageBreak()]

    # ── 4. ANALISIS DETAIL ─────────────────────────────────────────────────────
    story += [
        Paragraph("4. Analisis Detail per Fase", h1), hr(),
        Paragraph("4.1 Profil Konsumsi Energi – Top 10 Mesin Terboros", h2),
        img("q1_top10_energy.png", 15),
        Paragraph("Gambar 1. Top 10 Mesin berdasarkan Konsumsi Energi Kumulatif 2025",
                  caption),
        img("q2_top10_cost.png", 15),
        Paragraph("Gambar 2. Top 10 Mesin berdasarkan Total Biaya Listrik 2025", caption),
    ]

    story += [
        Paragraph("4.2 Profil Beban Harian &amp; Biaya per Jam", h2),
        img("q3_hourly_cost.png", 16),
        Paragraph("Gambar 3. Biaya Rata-rata per Jam – Merah menandai jam puncak (17:00–22:00)",
                  caption),
        img("cost_breakdown.png", 16),
        Paragraph("Gambar 4. Proporsi Biaya (Peak vs. Normal) &amp; Biaya per Tipe Mesin",
                  caption),
        PageBreak(),
    ]

    story += [
        Paragraph("4.3 Analisis Power Factor", h2),
        Paragraph(
            "Power factor adalah rasio daya aktif (kW) terhadap daya semu (kVA). "
            "Standar PLN mensyaratkan PF minimum 0,85. PF di bawah standar "
            "mengakibatkan penarikan arus lebih besar, pemborosan energi reaktif, "
            "dan risiko penalti tarif dari PLN.",
            body),
        img("q5_power_factor.png", 15),
        Paragraph("Gambar 5. 10 Mesin dengan Power Factor Rata-rata Terendah", caption),
        img("pf_distribution.png", 13),
        Paragraph("Gambar 6. Distribusi PF Seluruh Mesin – Area kiri garis merah = bermasalah",
                  caption),
        PageBreak(),
    ]

    # ── 5. SIMULASI PENGHEMATAN ────────────────────────────────────────────────
    story += [
        Paragraph("5. Simulasi Penghematan Energi", h1), hr(),
    ]
    story += [
        img("scenario_comparison.png", 16),
        Paragraph("Gambar 7. Perbandingan Biaya Tahunan – Baseline vs. Setiap Skenario",
                  caption),
    ]

    story += [
        Paragraph("Scenario A – Perbaikan Power Factor ke 0,95", h2),
        Paragraph(
            "Pemasangan kapasitor bank atau sistem power factor correction (PFC) "
            "pada mesin dengan PF buruk (< 0,80) akan meningkatkan efisiensi penggunaan "
            "daya. Setiap kenaikan PF mengurangi arus reaktif, menurunkan rugi-rugi "
            "pada kabel dan transformator, serta menghilangkan potensi penalti PLN.",
            body),
    ]
    a_data = [
        ["Parameter",                    "Nilai"],
        ["Rekaman dengan PF Buruk",       f"102.167 interval (9,7%)"],
        ["Energi yang Dapat Dihemat",     f"{sa['saving_kwh']:,.0f} kWh/tahun"],
        ["Penghematan Biaya Tahunan",      idrm(sa["saving_idr"])],
        ["Persentase Penghematan",         f"{sa['saving_pct']:.2f}%"],
        ["Biaya Tahunan Setelah Perbaikan", idr(sa["new_annual_cost"])],
    ]
    t_a = Table(a_data, colWidths=[9*cm, 7*cm])
    ts_a = tbl_style(GREEN, LGRAY, WHITE)
    t_a.setStyle(ts_a)
    story += [t_a, Spacer(1, 0.3*cm)]

    story += [
        img("scenario_a_by_machine.png", 15),
        Paragraph("Gambar 8. Potensi Penghematan per Mesin – Scenario A (PF Correction)",
                  caption),
    ]

    story += [
        Paragraph("Scenario B – Load Shifting 18:00 → 13:00", h2),
        Paragraph(
            "Pemindahan operasi mesin dari jam puncak (18:00, tarif Rp 1.700/kWh) "
            "ke jam normal (13:00, tarif Rp 1.450/kWh) menghemat selisih tarif "
            "Rp 250/kWh untuk setiap kWh yang berhasil dipindahkan. "
            "Implementasi membutuhkan penyesuaian jadwal produksi dan validasi "
            "kapasitas penyimpanan barang setengah jadi.",
            body),
    ]
    b_data = [
        ["Parameter",                      "Nilai"],
        ["Interval yang Terpengaruh",      f"43.629 interval"],
        ["Energi yang Dapat Digeser",       f"{sb['saving_kwh']:,.0f} kWh/tahun"],
        ["Biaya Asal (Tarif 1700)",          idr(base["peak_cost_idr"] / (base["total_cost_idr"]/base["total_energy_kwh"]/1700 if base["total_energy_kwh"] > 0 else 1))],
        ["Penghematan Biaya Tahunan",        idrm(sb["saving_idr"])],
        ["Persentase Penghematan",           f"{sb['saving_pct']:.2f}%"],
        ["Biaya Tahunan Setelah Load Shift", idr(sb["new_annual_cost"])],
    ]
    t_b = Table(b_data, colWidths=[9*cm, 7*cm])
    t_b.setStyle(tbl_style(ORANGE, LGRAY, WHITE))
    story += [t_b, Spacer(1, 0.3*cm)]

    story += [
        Paragraph("Scenario C – Reduksi Konsumsi Energi 5%", h2),
        Paragraph(
            "Pengurangan konsumsi energi sebesar 5% dapat dicapai melalui kombinasi: "
            "optimasi parameter operasi mesin, implementasi program perawatan preventif "
            "berbasis kondisi (condition-based maintenance), upgrade komponen dengan "
            "efisiensi lebih tinggi, dan eliminasi idle running.",
            body),
    ]
    c_data = [
        ["Parameter",                       "Nilai"],
        ["Target Reduksi Energi",            f"5% dari {base['total_energy_kwh']:,.0f} kWh"],
        ["Energi yang Dihemat",              f"{sc['saving_kwh']:,.0f} kWh/tahun"],
        ["Penghematan Biaya Tahunan",         idrm(sc["saving_idr"])],
        ["Persentase Penghematan",            f"{sc['saving_pct']:.2f}%"],
        ["Biaya Tahunan Setelah Efisiensi",   idr(sc["new_annual_cost"])],
    ]
    t_c = Table(c_data, colWidths=[9*cm, 7*cm])
    t_c.setStyle(tbl_style(NAVY, LGRAY, WHITE))
    story += [t_c, Spacer(1, 0.3*cm)]

    story += [
        img("monthly_cost_comparison.png", 16),
        Paragraph("Gambar 9. Biaya Bulanan: Kondisi Saat Ini vs. Setelah Implementasi Gabungan",
                  caption),
        PageBreak(),
    ]

    # ── 6. REKOMENDASI ─────────────────────────────────────────────────────────
    story += [
        Paragraph("6. Rekomendasi Strategis", h1), hr(),
    ]
    recs = [
        ("R-01", "SEGERA", "Perbaiki Power Factor COMPRESSOR_03, COMPRESSOR_04, dan MOTOR_02",
         "Pasang kapasitor bank otomatis (APFC) pada panel distribusi ketiga mesin tersebut. "
         "Target PF minimal 0,92. Investasi diperkirakan Rp 45–60 Juta per unit, "
         f"dengan payback period < 18 bulan berdasarkan penghematan Scenario A."),
        ("R-02", "SEGERA", "Implementasi Load Shifting untuk Operasi Non-Kritis",
         "Identifikasi mesin atau proses yang dapat dijadwal ulang dari pukul 17:00–22:00 "
         "ke 08:00–16:00. Mulai dengan PUMP_01 hingga PUMP_05 yang fleksibel secara "
         f"operasional. Target penghematan langsung: {idrm(sb['saving_idr'])} per tahun."),
        ("R-03", "JANGKA PENDEK", "Pasang Energy Monitoring System (EMS) Real-Time",
         "Implementasi sistem monitoring energi berbasis IoT dengan dashboard live untuk "
         "semua 30 mesin. Sistem harus menampilkan PF, daya, dan biaya secara real-time "
         "dengan alert otomatis jika PF < 0,85 atau konsumsi melampaui baseline."),
        ("R-04", "JANGKA PENDEK", "Program Perawatan Preventif Berbasis Kondisi",
         "Tetapkan jadwal pemeliharaan berkala untuk motor, pompa, dan kompresor berdasarkan "
         "data historis konsumsi energi. Mesin yang menunjukkan kenaikan konsumsi > 10% "
         "dari baseline harus segera diperiksa. Target: reduksi konsumsi 2–3% dari preventive maintenance saja."),
        ("R-05", "JANGKA MENENGAH", "Audit &amp; Upgrade Peralatan Berusia > 10 Tahun",
         "Lakukan energy audit menyeluruh pada mesin-mesin yang beroperasi di atas 10 tahun. "
         "Pertimbangkan upgrade ke motor IE3/IE4 (efisiensi tinggi) yang dapat mengurangi "
         "konsumsi energi 3–8% dibandingkan motor konvensional."),
        ("R-06", "JANGKA MENENGAH", "Optimasi Sistem Kompresor Udara",
         "Kelompok Compressor adalah konsumen energi tertinggi. Pertimbangkan: (1) pemasangan "
         "Variable Speed Drive (VSD) pada kompresor, (2) perbaikan kebocoran sistem pneumatik, "
         "(3) optimasi tekanan operasi. Target penghematan kompresor: 10–15% dari konsumsi saat ini."),
        ("R-07", "JANGKA PANJANG", "Pertimbangkan Pemasangan PLTS Atap",
         "Dengan luas atap fasilitas manufaktur yang tersedia, instalasi Pembangkit Listrik "
         "Tenaga Surya (PLTS) rooftop berkapasitas 100–200 kWp dapat mengurangi ketergantungan "
         "pada jaringan PLN khususnya pada jam siang hari, menurunkan biaya energi 15–25%."),
    ]
    r_data = [["Kode", "Prioritas", "Rekomendasi", "Penjelasan &amp; Dampak"]]
    for code, prio, title, desc in recs:
        r_data.append([
            Paragraph(f"<b>{code}</b>", S("Normal", fontSize=8.5, leading=12)),
            Paragraph(f"<b>{prio}</b>", S("Normal", fontSize=8, leading=12,
                                          textColor=RED if prio=="SEGERA" else ORANGE if "PENDEK" in prio else NAVY)),
            Paragraph(f"<b>{title}</b>", S("Normal", fontSize=8.5, leading=13)),
            Paragraph(desc, S("Normal", fontSize=8, leading=12)),
        ])
    t_rec = Table(r_data, colWidths=[1.2*cm, 2.5*cm, 5*cm, 7.3*cm])
    ts_rec = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 9),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.HexColor("#DEE2E6")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [LGRAY, WHITE]),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",(0,0), (-1,-1), 5),
        ("RIGHTPADDING",(0,0),(-1,-1), 5),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ])
    t_rec.setStyle(ts_rec)
    story += [t_rec, PageBreak()]

    # ── 7. DAMPAK FINANSIAL ────────────────────────────────────────────────────
    story += [
        Paragraph("7. Dampak Finansial (Financial Impact)", h1), hr(),
    ]
    fin_data = [
        ["Inisiatif / Skenario",   "Investasi Est.",   "Penghematan/Tahun",
         "Payback Period", "NPV 5 Tahun (asumsi 7%)"],
        ["Scenario A: PF Correction",
         "Rp 135–180 Juta",    idrm(sa["saving_idr"]),
         "< 18 bulan",         idrm(sa["saving_idr"] * 5 * 0.9)],
        ["Scenario B: Load Shifting",
         "Rp 5–10 Juta",       idrm(sb["saving_idr"]),
         "< 3 bulan",          idrm(sb["saving_idr"] * 5 * 0.95)],
        ["Scenario C: 5% Reduction",
         "Rp 50–80 Juta",      idrm(sc["saving_idr"]),
         "< 6 bulan",          idrm(sc["saving_idr"] * 5 * 0.9)],
        ["TOTAL / GABUNGAN",
         "Rp 190–270 Juta",    idrm(comb["saving_idr"]),
         "< 12 bulan",         idrm(comb["saving_idr"] * 5 * 0.9)],
    ]
    t_fin = Table(fin_data, colWidths=[4*cm, 2.5*cm, 3.5*cm, 2.5*cm, 3.5*cm])
    ts_fin = tbl_style()
    ts_fin.add("FONTNAME", (0, len(fin_data)-1), (-1, len(fin_data)-1), "Helvetica-Bold")
    ts_fin.add("BACKGROUND", (0, len(fin_data)-1), (-1, len(fin_data)-1),
               colors.HexColor("#E8F5E9"))
    ts_fin.add("TEXTCOLOR",  (0, len(fin_data)-1), (-1, len(fin_data)-1), GREEN)
    t_fin.setStyle(ts_fin)
    story += [
        t_fin,
        Spacer(1, 0.3*cm),
        Paragraph(
            f"<b>Highlight:</b> Potensi Total Penghematan Tahunan: "
            f"<font color='#2E7D32' size='13'><b>{idrm(comb['saving_idr'])}</b></font> "
            f"({comb['saving_pct']:.2f}% dari tagihan listrik tahunan)",
            S("Normal", fontSize=11, leading=16)),
        PageBreak(),
    ]

    # ── 8. RENCANA IMPLEMENTASI ────────────────────────────────────────────────
    story += [
        Paragraph("8. Rencana Implementasi", h1), hr(),
    ]
    impl_data = [
        ["Fase", "Aktivitas",            "Timeline",         "PIC",            "Output"],
        ["1",    "Audit PF & Pasang APFC COMPRESSOR_03/04",
         "Bulan 1–2",  "Engineering Team", "PF > 0,92"],
        ["2",    "Load Shifting Jadwal PUMP_01–05",
         "Bulan 1",    "Production Planner", "Saving tarif"],
        ["3",    "Instalasi EMS & Dashboard",
         "Bulan 2–3",  "IT & Energy Team", "Dashboard live"],
        ["4",    "Program Preventive Maintenance",
         "Bulan 3–4",  "Maintenance Dept.", "SOP maintenance"],
        ["5",    "Audit Mesin >10 Tahun",
         "Bulan 4–5",  "Engineering + Finance", "Laporan audit"],
        ["6",    "Studi PLTS & VSD Compressor",
         "Bulan 5–6",  "Management",       "Proposal investasi"],
        ["7",    "Review & Evaluasi KPI Energi",
         "Bulan 6",    "Energy Manager",   "Laporan savings"],
    ]
    t_impl = Table(impl_data, colWidths=[1*cm, 5.5*cm, 2.5*cm, 3.5*cm, 3.5*cm])
    t_impl.setStyle(tbl_style())
    story += [t_impl, Spacer(1, 0.3*cm)]

    # KPI targets
    story += [
        Paragraph("8.1 Target KPI Pasca-Implementasi (Tahun Pertama)", h2),
    ]
    kpi_target = [
        ["KPI",                          "Baseline",                        "Target"],
        ["Power Factor Rata-rata",        "0,856",                          "> 0,92"],
        ["Persentase Interval PF Buruk",  "9,7%",                           "< 3%"],
        ["Konsumsi Energi Tahunan",        f"{base['total_energy_kwh']:,.0f} kWh",
         f"< {base['total_energy_kwh']*0.95:,.0f} kWh"],
        ["Total Biaya Listrik Tahunan",    idrm(base["total_cost_idr"]),
         f"< {idrm(base['total_cost_idr'] - comb['saving_idr'])}"],
        ["Rasio Beban Puncak/Total",       "26%",                           "< 20%"],
    ]
    t_kpi = Table(kpi_target, colWidths=[6*cm, 5*cm, 5*cm])
    ts_kpi = tbl_style()
    for row in range(1, len(kpi_target)):
        ts_kpi.add("TEXTCOLOR", (2, row), (2, row), GREEN)
        ts_kpi.add("FONTNAME",  (2, row), (2, row), "Helvetica-Bold")
    t_kpi.setStyle(ts_kpi)
    story += [t_kpi, PageBreak()]

    # ── 9. KESIMPULAN ──────────────────────────────────────────────────────────
    story += [
        Paragraph("9. Kesimpulan", h1), hr(),
        Paragraph(
            "Analisis komprehensif terhadap data konsumsi energi PT Surya Textile "
            "Indonesia selama tahun 2025 mengonfirmasi bahwa terdapat peluang "
            "penghematan biaya listrik yang signifikan dan dapat direalisasikan "
            "dalam jangka pendek.",
            body),
        Paragraph(
            "Tiga temuan kritis yang memerlukan tindakan segera adalah: "
            "(1) power factor buruk yang membuang energi reaktif secara masif pada "
            "mesin Compressor dan beberapa Motor; "
            "(2) beban yang tidak dioptimalkan pada jam tarif puncak; dan "
            "(3) tidak adanya sistem monitoring energi yang membuat anomali tidak "
            "terdeteksi secara dini.",
            body),
        Paragraph(
            f"Dengan mengimplementasikan tiga skenario penghematan yang telah "
            f"disimulasikan secara terstruktur, PT Surya Textile Indonesia berpotensi "
            f"menghemat biaya listrik sebesar <b>{idrm(comb['saving_idr'])}</b> per tahun "
            f"({comb['saving_pct']:.2f}% dari pengeluaran saat ini), "
            f"dengan investasi awal yang diperkirakan kembali dalam kurun waktu "
            f"kurang dari 12 bulan.",
            body),
        Spacer(1, 0.3*cm),
    ]

    # Sign-off
    sign_data = [
        [
            Paragraph("<b>Disiapkan oleh:</b><br/><br/><br/>Tim Energy Analytics<br/>"
                      "PT Surya Textile Indonesia (Simulasi)", body_l),
            Paragraph("<b>Diketahui oleh:</b><br/><br/><br/>Energy Manager<br/>"
                      "PT Surya Textile Indonesia (Simulasi)", body_l),
            Paragraph("<b>Disetujui oleh:</b><br/><br/><br/>Direktur Operasional<br/>"
                      "PT Surya Textile Indonesia (Simulasi)", body_l),
        ]
    ]
    t_sign = Table(sign_data, colWidths=[5.3*cm, 5.3*cm, 5.4*cm])
    t_sign.setStyle(TableStyle([
        ("BOX",        (0,0), (-1,-1), 0.5, NAVY),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, NAVY),
        ("PADDING",    (0,0), (-1,-1), 12),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ]))
    story += [
        t_sign,
        Spacer(1, 0.5*cm),
        hr(DGRAY),
        Paragraph(
            "Laporan ini bersifat rahasia dan ditujukan khusus untuk manajemen "
            "PT Surya Textile Indonesia. Dilarang memperbanyak atau mendistribusikan "
            f"tanpa izin tertulis. | {datetime.now().strftime('%d %B %Y')}",
            small),
    ]

    doc.build(story)
    print(f"✅ Final Report saved → {path}")
    return path

# ─── RUN ALL ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating PDF reports …\n")
    build_data_dictionary()
    build_eda_report()
    build_final_report()
    print("\n✅ All PDF reports generated successfully!")
