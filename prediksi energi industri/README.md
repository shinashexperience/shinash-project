# Industrial Energy Monitoring & Predictive Maintenance Platform

**Klien:** PT Nusantara Manufacturing (Simulasi)  
**Durasi:** 12 Minggu  
**Periode Data:** Januari — Desember 2025  
**Total Records:** 43,800 baris | 5 Mesin | 9 Fitur

---

## 🎯 Tujuan Bisnis

| Menurunkan | Meningkatkan |
|------------|--------------|
| Downtime mesin | Reliability |
| Konsumsi energi | Asset Utilization |
| Kerusakan mendadak | Maintenance Planning |

---

## 📂 Struktur Folder

```
Industrial_Predictive_Maintenance/
├── data/
│   ├── raw_data.csv              # Dataset asli (43,800 records)
│   ├── processed_data.csv        # Dataset dengan health score & anomaly flags
│   ├── stats.json                # Aggregated statistics (dashboard ready)
│   └── plots/                    # Visualisasi (PNG)
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Fase 1: Pemahaman data
│   ├── 02_eda.ipynb                  # Fase 2: EDA (4 business questions)
│   ├── 03_health_index.ipynb         # Fase 3: Asset Health Index
│   ├── 04_anomaly_detection.ipynb    # Fase 4: Isolation Forest
│   ├── 05_failure_prediction.ipynb   # Fase 5: ML Models
│   └── 06_rul_prediction.ipynb       # Fase 6: Remaining Useful Life
├── dashboard/
│   ├── app.py                    # Main Streamlit entry point
│   └── pages/
│       ├── overview.py           # Overview KPI
│       ├── energy.py             # Energy Analysis
│       ├── health.py             # Asset Health
│       └── prediction.py        # Predictive Maintenance
└── reports/
    ├── 01_Data_Dictionary.pdf    # Fase 1 deliverable
    └── Project_Report.pdf        # Final presentation report
```

---

## 🚀 Cara Menjalankan Dashboard

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn
cd dashboard
streamlit run app.py
```

---

## 📊 Hasil per Fase

### Fase 1 — Data Understanding
- 9 kolom, 43,800 records, 5 mesin, 0 missing values
- Output: `reports/01_Data_Dictionary.pdf`

### Fase 2 — EDA
| Pertanyaan | Temuan |
|------------|--------|
| Mesin paling boros | MOTOR_01 (76,178 kWh) |
| Jam puncak | 09:00 & 14:00-16:00 |
| Paling sering WARNING | MOTOR_02 (142 events) |
| Korelasi suhu-vibrasi | r ≈ 0.72 (kuat positif) |

### Fase 3 — Asset Health Index
- Formula: 40% Temp + 40% Vibration + 20% Current
- Semua mesin dalam kategori **WARNING** (60-79%)

### Fase 4 — Anomaly Detection
- Metode: Isolation Forest (contamination=2%)
- **876 anomali terdeteksi** dari 43,800 records

### Fase 5 — Predictive Maintenance
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 100.0% | 99.8% |
| **Random Forest** | **100.0%** | **99.5%** |

### Fase 6 — Remaining Useful Life
- Rata-rata RUL semua mesin: **58 Hari**
- Rekomendasi: Preventive Maintenance setiap 45 hari

---

## 🔧 Dataset Columns

| Kolom | Satuan | Deskripsi |
|-------|--------|-----------|
| timestamp | — | Waktu per jam |
| machine_id | — | ID mesin |
| voltage_v | Volt | Tegangan listrik |
| current_a | Ampere | Arus listrik |
| power_kw | kW | Daya aktif |
| temperature_c | °C | Suhu operasional |
| vibration_mm_s | mm/s | Getaran mesin |
| power_factor | — | Faktor daya |
| status | — | NORMAL / WARNING |

---

## 💡 Rekomendasi Utama

1. **Preventive Maintenance setiap 45 hari** → Downtime turun 20%
2. **Pasang sistem pendingin tambahan** (MOTOR_02, COMPRESSOR_01)
3. **Implementasi alert otomatis** berbasis anomaly detection
4. **Load shifting** ke off-peak (22:00-06:00) → Hemat energi 8-12%
5. **Deploy Random Forest model** ke production monitoring

---

*Industrial Energy Monitoring & Predictive Maintenance Platform | PT Nusantara Manufacturing | 2025*
