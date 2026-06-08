# ⚡ Industrial Energy Cost Optimization & Load Management System

**Client:** PT Surya Textile Indonesia (Simulasi)  
**Project Type:** Energy Analytics & Cost Optimization  
**Industry:** Manufacturing (Textile)  
**Periode Analisis:** Januari – Desember 2025

---

## 📋 Deskripsi Proyek

Proyek ini merupakan analisis komprehensif konsumsi energi listrik pada 30 mesin produksi PT Surya Textile Indonesia. Tujuan utama adalah mengidentifikasi sumber pemborosan energi dan merumuskan strategi penghematan biaya listrik yang terukur dan dapat diimplementasikan.

### Latar Belakang
- Tagihan listrik meningkat **17%** dalam 12 bulan terakhir
- Tidak ada sistem monitoring energi real-time
- Tidak diketahui mesin mana yang paling boros atau jam operasi paling mahal
- Manajemen menargetkan penghematan minimal **5%**

---

## 🎯 Business Objectives

| Kode  | Pertanyaan                                            | Status      |
|-------|-------------------------------------------------------|-------------|
| BO-01 | Mesin mana yang mengonsumsi energi paling besar?     | ✅ Terjawab |
| BO-02 | Jam berapa beban listrik tertinggi terjadi?           | ✅ Terjawab |
| BO-03 | Berapa estimasi biaya listrik tahunan?                | ✅ Terjawab |
| BO-04 | Mesin mana yang memiliki faktor daya terburuk?        | ✅ Terjawab |
| BO-05 | Apa rekomendasi untuk mengurangi biaya listrik?       | ✅ Terjawab |

---

## 📊 Key Results

| KPI                         | Nilai                          |
|-----------------------------|--------------------------------|
| Total Energi 2025           | 3.264.910 kWh                  |
| Total Biaya 2025            | Rp 4.922.073.901               |
| Rata-rata Biaya/Hari        | Rp 13.485.134                  |
| Mesin Paling Boros          | COMPRESSOR_02 (163.656 kWh)    |
| Interval PF Buruk           | 9,7% (target < 3%)             |
| **Potensi Penghematan**     | **Rp 432.213.922 / tahun**     |
| **Persentase Penghematan**  | **8,78%**                      |

---

## 🏗️ Struktur Folder

```
Energy_Cost_Optimization/
├── data/
│   ├── raw/
│   │   └── energy_consumption.csv      # Dataset raw (1.051.200 records)
│   └── processed/
│       └── clean_energy_data.csv       # Dataset setelah cleaning (1.048.209 records)
├── notebooks/
│   ├── 01_data_generation.py           # Phase 1: Generasi dataset simulasi
│   ├── 02_cleaning_eda.py              # Phase 2 & 3: Cleaning + EDA
│   ├── 03_cost_simulation.py           # Phase 4 & 5: Cost Analysis + Scenarios
│   └── 04_pdf_reports.py              # Generator laporan PDF
├── dashboard/
│   └── app.py                          # Streamlit dashboard (4 halaman)
├── reports/
│   ├── Data_Dictionary.pdf             # Phase 1 output
│   ├── EDA_Report.pdf                  # Phase 3 output
│   ├── Energy_Optimization_Report.pdf  # Laporan final konsultan energi
│   ├── kpis.json                       # KPI & ringkasan EDA
│   ├── simulation.json                 # Hasil simulasi penghematan
│   └── figures/                        # Chart PNG untuk laporan
│       ├── q1_top10_energy.png
│       ├── q2_top10_cost.png
│       ├── q3_hourly_cost.png
│       ├── q4_daily_cost.png
│       ├── q5_power_factor.png
│       ├── pf_distribution.png
│       ├── monthly_trend.png
│       ├── machine_type_comparison.png
│       ├── cost_breakdown.png
│       ├── scenario_comparison.png
│       ├── scenario_a_by_machine.png
│       └── monthly_cost_comparison.png
└── README.md
```

---

## 🔧 Spesifikasi Mesin

| Kategori    | Jumlah | ID Range                           | Arus (A) |
|-------------|--------|------------------------------------|----------|
| Motor       | 15     | MOTOR_01 – MOTOR_15                | 15–40 A  |
| Pump        | 10     | PUMP_01 – PUMP_10                  | 10–25 A  |
| Compressor  | 5      | COMPRESSOR_01 – COMPRESSOR_05      | 20–60 A  |

---

## 💡 Simulasi Skenario Penghematan

| Skenario | Deskripsi                          | Penghematan/Tahun    | % Saving |
|----------|------------------------------------|----------------------|----------|
| A        | PF Correction ke 0,95              | Rp 148.670.042       | 3,02%    |
| B        | Load Shift 18:00 → 13:00           | Rp 37.440.184        | 0,76%    |
| C        | Reduksi konsumsi energi 5%         | Rp 246.103.695       | 5,00%    |
| **A+B+C**| **Gabungan semua skenario**        | **Rp 432.213.922**   | **8,78%**|

---

## 📦 Deliverables

| Phase | Output                                 | Status |
|-------|----------------------------------------|--------|
| 1     | `Data_Dictionary.pdf`                  | ✅     |
| 2     | `clean_energy_data.csv`                | ✅     |
| 3     | `EDA_Report.pdf`                       | ✅     |
| 4     | Cost KPI Analysis                      | ✅     |
| 5     | Saving Simulation (3 scenarios)        | ✅     |
| 6     | Streamlit Dashboard (4 pages)          | ✅     |
| Final | `Energy_Optimization_Report.pdf`       | ✅     |

---

## 🚀 Cara Menjalankan

### 1. Instalasi Dependencies
```bash
pip install pandas numpy matplotlib seaborn reportlab streamlit plotly
```

### 2. Generate Dataset (Phase 1)
```bash
python notebooks/01_data_generation.py
```

### 3. Data Cleaning & EDA (Phase 2 & 3)
```bash
python notebooks/02_cleaning_eda.py
```

### 4. Cost Analysis & Simulation (Phase 4 & 5)
```bash
python notebooks/03_cost_simulation.py
```

### 5. Generate PDF Reports
```bash
python notebooks/04_pdf_reports.py
```

### 6. Jalankan Dashboard
```bash
cd Energy_Cost_Optimization
streamlit run dashboard/app.py
```
Dashboard akan terbuka di browser pada `http://localhost:8501`

---

## 📐 Metodologi

### Data Generation
- **Periode:** 1 Jan – 31 Des 2025
- **Interval:** 15 menit (96 titik/hari/mesin)
- **Total record:** 35.040 timestamp × 30 mesin = **1.051.200 records**
- **Sistem tegangan:** 380V tiga fasa

### Rumus Kalkulasi
```
S (kVA) = √3 × V × I / 1000        # Apparent Power
P (kW)  = S × PF                    # Active Power  
Q (kVAR)= √(S² - P²)               # Reactive Power
E (kWh) = P × 0,25                  # Energy per 15-min interval
```

### Data Cleaning
- Missing values: Drop rows (tidak ada dalam simulasi)
- Duplikat: Drop berdasarkan (timestamp, machine_id)
- Outlier: Metode IQR 3× pada semua kolom numerik → **2.991 records dihapus**
- Validasi PF: Range 0,5 – 1,0

### Tarif Listrik
- **Peak Hour** (17:00–22:00): Rp 1.700/kWh
- **Normal Hour**: Rp 1.450/kWh

---

## 🏆 Top 5 Rekomendasi

1. **[SEGERA]** Pasang kapasitor bank APFC pada COMPRESSOR_03, COMPRESSOR_04, MOTOR_02
2. **[SEGERA]** Implementasi load shifting PUMP_01–05 dari pukul 18:00 ke 13:00
3. **[JANGKA PENDEK]** Instalasi Energy Monitoring System (EMS) real-time semua 30 mesin
4. **[JANGKA PENDEK]** Program perawatan preventif berbasis kondisi (condition-based maintenance)
5. **[JANGKA MENENGAH]** Audit & upgrade mesin berusia > 10 tahun ke motor IE3/IE4

---

## 👥 Tim Proyek

| Peran                | Tanggung Jawab                                          |
|----------------------|---------------------------------------------------------|
| Energy Analyst       | Data generation, cleaning, EDA, cost analysis           |
| Data Engineer        | Pipeline data, penyimpanan, dan otomasi                 |
| Energy Consultant    | Interpretasi temuan, rekomendasi strategis              |
| Dashboard Developer  | Streamlit dashboard, visualisasi interaktif             |

---

## 📄 Laporan

Laporan lengkap tersedia di folder `reports/`:
- **Data_Dictionary.pdf** – Dokumentasi lengkap dataset dan kamus data
- **EDA_Report.pdf** – Laporan analisis eksploratif dengan 8 visualisasi
- **Energy_Optimization_Report.pdf** – Laporan konsultan energi (9 bab, 12 halaman)

---

*Proyek ini merupakan simulasi untuk keperluan analitik internal. Data bersifat simulasi dan tidak merepresentasikan konsumsi energi aktual PT Surya Textile Indonesia.*

**Versi:** 1.0 | **Tanggal:** 2025 | **Lisensi:** Internal Use Only
