# 🏭 Smart Industrial IoT Monitoring Platform

**PT Nusantara Steel Manufacturing** · Simulasi Proyek Portofolio

> Sistem monitoring aset industri berbasis IoT yang mensimulasikan pipeline data lengkap dari sensor ke dashboard — mulai dari pengumpulan data real-time via MQTT, penyimpanan ke PostgreSQL, hingga analitik lanjutan dengan Machine Learning.

---

## 📸 Dashboard Preview

| Executive Summary | Machine Health | Power Monitor |
|:-:|:-:|:-:|
| KPI + Energy Trend | Gauge + Vibration | Voltage + PF |

| Alarm Center | Maintenance | Analytics |
|:-:|:-:|:-:|
| Heatmap + History | MTBF + MTTR | Correlation + Forecast |

---

## 🎯 Project Objectives

| # | Tujuan | Status |
|---|--------|--------|
| 1 | Mengumpulkan data sensor otomatis (simulasi) | ✅ |
| 2 | Menyimpan 1.900.000+ record ke CSV / PostgreSQL | ✅ |
| 3 | Dashboard real-time multi-page Streamlit | ✅ |
| 4 | Sistem alarm otomatis berdasarkan threshold | ✅ |
| 5 | Riwayat sensor dengan grafik interaktif | ✅ |
| 6 | Kalkulasi konsumsi energi (kWh, Power Factor) | ✅ |
| 7 | Health score per mesin dengan algoritma custom | ✅ |
| 8 | Anomaly detection (Isolation Forest) | ✅ |
| 9 | Failure prediction (Random Forest) | ✅ |
| 10 | Energy forecast (Linear Regression) | ✅ |
| 11 | MQTT pipeline simulator (publish/subscribe) | ✅ |
| 12 | MTBF, MTTR, downtime analysis | ✅ |

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                    SENSOR LAYER                          │
│  Voltage · Current · Temperature · Vibration · Humidity  │
│  Flow Rate · Pressure · Frequency · Power Factor         │
└──────────────────────┬──────────────────────────────────┘
                       │
             ┌─────────▼─────────┐
             │  PLC / ESP32 Sim  │  ← mqtt_simulator.py
             └─────────┬─────────┘
                       │  MQTT Protocol
             ┌─────────▼─────────┐
             │   Mosquitto MQTT  │  ← localhost:1883
             │     Broker        │
             └─────────┬─────────┘
                       │
             ┌─────────▼─────────┐
             │  Python Subscriber │  ← mqtt_subscriber.py
             └─────────┬─────────┘
                       │
          ┌────────────▼────────────┐
          │   PostgreSQL Database   │
          │   (CSV fallback)        │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Analytics Engine     │
          │  Isolation Forest · RF  │
          │  Linear Regression      │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Streamlit Dashboard   │
          │  6 halaman interaktif   │
          └─────────────────────────┘
```

---

## 🏭 Aset yang Dimonitor

| Tipe          | Jumlah | ID Range          | Keterangan                        |
|---------------|--------|-------------------|-----------------------------------|
| Motor Induksi | 15     | MTR-001 – MTR-015 | 7.5 – 75 kW, usia 2–15 tahun     |
| Pompa         | 8      | PMP-001 – PMP-008 | Dengan sensor flow & pressure     |
| Compressor    | 6      | CMP-001 – CMP-006 | Arus tinggi 20–70 A               |
| Conveyor      | 4      | CNV-001 – CNV-004 | Vibrasi tinggi karena beban       |
| Boiler        | 2      | BLR-001 – BLR-002 | Suhu terpanas (>100°C)            |
| Cooling Tower | 1      | CLT-001           | Flow rate terbesar (>150 L/min)   |
| **Total**     | **36** |                   | **1.897.344 records sensor**      |

---

## 📡 Spesifikasi Sensor

| Sensor       | Satuan  | Normal          | Warning   | Critical  |
|--------------|---------|-----------------|-----------|-----------|
| Voltage      | V       | 370 – 390       | < 365     | < 358     |
| Current      | A       | Varies per type | —         | —         |
| Temperature  | °C      | 35 – 65         | > 70      | > 80      |
| Vibration    | mm/s    | 0.8 – 2.8       | > 3.5     | > 4.5     |
| Power Factor | —       | 0.85 – 0.98     | < 0.83    | < 0.80    |
| Pressure     | Bar     | 0 – 12          | > 11.5    | > 12.5    |
| Flow Rate    | L/min   | 15 – 220        | —         | —         |
| Humidity     | %       | 30 – 75         | —         | —         |
| Frequency    | Hz      | 49.7 – 50.3     | —         | —         |

---

## 🔢 Data Generation Rules (7 Aturan Fisika)

Data tidak dibuat secara acak murni, melainkan mengikuti hukum fisika dan engineering:

| Rule | Kondisi | Efek |
|------|---------|------|
| **Rule 1** | ↑ Current | → ↑ Temperature (termal dari I²R) |
| **Rule 2** | ↑ Usia mesin | → ↑ Vibrasi (keausan bearing) |
| **Rule 3** | ↑ Vibrasi | → ↑ Probabilitas alarm |
| **Rule 4** | ↑ Current | → ↑ Power → ↑ Energy |
| **Rule 5** | ↑ Humidity | → ↑ Risiko korosi → ↑ failure probability |
| **Rule 6** | Boiler | → Suhu jauh lebih tinggi dari tipe lain |
| **Rule 7** | Cooling Tower | → Flow rate terbesar dari semua aset |

---

## 📊 Dashboard Pages

### 🏠 Home — Executive Summary
- Total mesin, online/offline, active alarm, daily energy
- Status mesin per tipe (stacked bar)
- Distribusi alarm (pie chart)
- Tren energi harian + Moving Average 7 hari
- Mesin health score terburuk
- Alarm terbaru
- Health score bar chart semua mesin

### 🔧 Machine Health
- Gauge health score dengan warna dinamis
- KPI: avg temp, max temp, avg vibration, total alarm
- Tren suhu harian (avg + max)
- Tren vibrasi harian (avg + max)
- Box plot distribusi suhu per jam (pola shift)
- Riwayat alarm per mesin

### ⚡ Power Monitor
- KPI: voltage, current, PF, power, total kWh
- Grafik voltage & current (dual axis)
- Trend daya (kW) vs nominal
- Power factor trend dengan batas threshold
- Energi kumulatif + harian
- Perbandingan energi bulanan per tipe mesin
- Ranking mesin terboros

### 🚨 Alarm Center
- KPI: total alarm, critical, warning, mesin terdampak
- Tren alarm harian (stacked bar: Critical/Warning)
- Distribusi tipe alarm (pie chart)
- Top 15 mesin dengan alarm terbanyak
- **Alarm Heatmap**: jam × hari dalam seminggu
- Tabel alarm dengan search & filter
- Export CSV

### 🛠️ Maintenance
- KPI: total event, preventive, corrective, MTBF, MTTR, total biaya
- Jadwal maintenance mendatang (estimasi next PM)
- Breakdown preventive vs corrective
- Komponen yang paling sering diganti
- MTBF bar chart per mesin
- Biaya maintenance bulanan
- Durasi maintenance box plot
- Downtime ranking + availability %

### 📊 Analytics
- **Correlation Matrix**: heatmap korelasi antar semua parameter sensor
- **Rule Validation**: scatter plot usia vs vibrasi (konfirmasi Rule 2)
- Health score vs alarm count scatter
- **Anomaly Detection**: Isolation Forest (contamination=5%)
- **Energy Forecast**: Linear Regression + 3 bulan ke depan
- **Failure Risk Score**: Random Forest dengan feature importance
- Tabel prioritas maintenance otomatis

---

## 🚀 Cara Menjalankan

### Prasyarat
```bash
Python 3.10+
pip install -r requirements.txt
```

### 1. Generate Dataset
```bash
cd Smart_Industrial_IoT
python generate_data.py
# Output: 1.897.344 records sensor + 195.641 alarm + 259 maintenance
```

### 2. Jalankan Dashboard
```bash
cd dashboard
streamlit run Home.py
# Buka: http://localhost:8501
```

### 3. Jalankan MQTT Simulator (Opsional)
```bash
# Terminal 1: Start MQTT Broker
mosquitto -v

# Terminal 2: Start Subscriber (simpan ke DB)
python simulator/mqtt_subscriber.py

# Terminal 3: Start Sensor Simulator
python simulator/mqtt_simulator.py
```

### 4. Setup PostgreSQL (Opsional)
```bash
# Buat database
createdb iot_nusantara

# Buat tabel
psql -d iot_nusantara -f sql/create_tables.sql
```

---

## 📁 Struktur Proyek

```
Smart_Industrial_IoT/
│
├── generate_data.py              ← Generator dataset (jalankan dulu!)
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── sensor_log.csv        ← 1.897.344 records
│   │   ├── alarm_history.csv     ← 195.641 alarm events
│   │   └── maintenance_history.csv
│   ├── processed/
│   │   ├── sensor_hourly.csv     ← Agregasi per jam
│   │   ├── sensor_daily.csv      ← Agregasi per hari
│   │   ├── energy_monthly.csv    ← Energi per bulan
│   │   ├── machine_status.csv    ← Health score 36 mesin
│   │   └── machine_master.csv
│   └── master/
│       ├── machine_master.csv
│       └── operator_shift.csv
│
├── dashboard/
│   ├── Home.py                   ← Executive Summary
│   └── pages/
│       ├── 1_Machine_Health.py
│       ├── 2_Power_Monitor.py
│       ├── 3_Alarm_Center.py
│       ├── 4_Maintenance.py
│       └── 5_Analytics.py
│
├── simulator/
│   ├── mqtt_simulator.py         ← MQTT publisher (sensor)
│   └── mqtt_subscriber.py        ← MQTT subscriber → DB
│
└── sql/
    ├── create_tables.sql          ← Schema PostgreSQL
    └── sample_queries.sql         ← 16 analytical queries
```

---

## 🤖 Machine Learning Components

| Model | Task | Library |
|-------|------|---------|
| Isolation Forest | Anomaly Detection | scikit-learn |
| Random Forest Classifier | Failure Prediction | scikit-learn |
| Linear Regression | Energy Forecast | scikit-learn |
| StandardScaler | Feature Scaling | scikit-learn |

---

## 💡 Business Questions yang Dijawab

### Asset Health
- ✅ Mesin mana yang paling sering alarm? → Alarm Center
- ✅ Mesin mana yang health score terburuk? → Home + Machine Health
- ✅ Apakah usia mesin mempengaruhi vibrasi? → Analytics (r > 0.4)

### Energy
- ✅ Mesin mana yang paling boros? → Power Monitor (Ranking)
- ✅ Jam operasi paling mahal? → SQL Query Q6
- ✅ Power factor terburuk? → Power Monitor

### Maintenance
- ✅ Komponen apa yang paling sering diganti? → Maintenance
- ✅ Berapa MTBF? → Maintenance (per mesin)
- ✅ Berapa MTTR? → Maintenance (rata-rata)
- ✅ Mesin mana prioritas maintenance? → Analytics (Risk Score)

### IoT
- ✅ Berapa data dikirim per hari? → SQL Query Q13
- ✅ Sensor mana yang paling sering anomali? → Analytics
- ✅ Deteksi anomali otomatis? → Isolation Forest

---

## 🧠 Key Technical Skills Demonstrated

| Domain | Skill |
|--------|-------|
| **Electrical Engineering** | Pemahaman sensor industri, PF, MTBF, sistem 3-fasa |
| **IoT Architecture** | MQTT protocol, publish-subscribe pattern, edge computing |
| **Data Engineering** | 1.9M record generation, correlated data, time-series |
| **Database** | PostgreSQL schema design, indexing, analytical queries |
| **Data Analytics** | Correlation analysis, health scoring, anomaly detection |
| **Machine Learning** | Isolation Forest, Random Forest, Linear Regression |
| **Dashboard** | Streamlit multi-page, Plotly interactive charts |
| **Software Engineering** | Modular code, error handling, documentation |

---

## 👨‍💻 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue)
![MQTT](https://img.shields.io/badge/MQTT-Mosquitto-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-yellow)

---

## 📄 License

MIT License — Free for portfolio and educational use.

---

## 🙋‍♂️ About

**Proyek Portofolio #3** — Industrial Digital Transformation Series

Dirancang untuk memperlihatkan kemampuan **Industrial IoT Engineer** dengan
memahami bagaimana data dihasilkan dari sensor fisik, mengalir melalui pipeline
IoT, dan berakhir sebagai insight operasional yang actionable.

> *"Lulusan Teknik Elektro yang bisa bicara dari sensor sampai dashboard."*
