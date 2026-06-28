#!/usr/bin/env python3
"""
Smart Industrial IoT — MQTT Subscriber
PT Nusantara Steel Manufacturing

Subscribe ke semua topik sensor dan simpan ke PostgreSQL.
Berjalan bersamaan dengan mqtt_simulator.py.

Dependensi tambahan:
  pip install paho-mqtt psycopg2-binary
"""

import json
import signal
import sys
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
    MQTT_OK = True
except ImportError:
    MQTT_OK = False
    print("[WARN] paho-mqtt tidak tersedia.")

try:
    import psycopg2
    from psycopg2.extras import execute_values
    PG_OK = True
except ImportError:
    PG_OK = False
    print("[WARN] psycopg2 tidak tersedia. Data hanya disimpan ke file CSV.")

import csv, os

# ─────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC_SUB   = "nusantara/steel/sensor/#"   # Wildcard: subscribe semua mesin

PG_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "iot_nusantara",
    "user":     "postgres",
    "password": "postgres",
}

# Fallback CSV jika PostgreSQL tidak tersedia
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_LIVE   = os.path.join(BASE_DIR, "data", "raw", "live_sensor_log.csv")
CSV_ALARMS = os.path.join(BASE_DIR, "data", "raw", "live_alarm_log.csv")

# ─────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────
conn = None

def init_pg():
    global conn
    if not PG_OK:
        return False
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cur  = conn.cursor()
        # Buat tabel jika belum ada
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sensor_live (
                id            SERIAL PRIMARY KEY,
                timestamp     TIMESTAMPTZ NOT NULL,
                machine_id    VARCHAR(20)  NOT NULL,
                voltage       NUMERIC(6,1),
                current       NUMERIC(7,2),
                temperature   NUMERIC(6,1),
                vibration     NUMERIC(6,3),
                humidity      NUMERIC(5,1),
                pressure      NUMERIC(6,2),
                flow_rate     NUMERIC(7,1),
                frequency     NUMERIC(5,2),
                power_factor  NUMERIC(5,3),
                power_kw      NUMERIC(8,2)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alarm_live (
                id          SERIAL PRIMARY KEY,
                timestamp   TIMESTAMPTZ NOT NULL,
                machine_id  VARCHAR(20) NOT NULL,
                alarm_type  VARCHAR(60),
                severity    VARCHAR(20),
                value       NUMERIC(10,3)
            );
        """)
        conn.commit()
        cur.close()
        print("[PG] Connected & tables ready.")
        return True
    except Exception as e:
        print(f"[PG] Koneksi gagal: {e}")
        conn = None
        return False


def save_to_pg(data):
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sensor_live
              (timestamp, machine_id, voltage, current, temperature, vibration,
               humidity, pressure, flow_rate, frequency, power_factor, power_kw)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data["timestamp"], data["machine_id"],
            data["voltage"], data["current"], data["temperature"],
            data["vibration"], data["humidity"], data["pressure"],
            data["flow_rate"], data["frequency"],
            data["power_factor"], data["power_kw"],
        ))
        for alarm in data.get("alarms", []):
            cur.execute("""
                INSERT INTO alarm_live (timestamp, machine_id, alarm_type, severity, value)
                VALUES (%s,%s,%s,%s,%s)
            """, (data["timestamp"], data["machine_id"],
                  alarm["type"], alarm["severity"], alarm["value"]))
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"[PG] Insert error: {e}")
        conn.rollback()


# ─────────────────────────────────────────────────────────
# CSV FALLBACK
# ─────────────────────────────────────────────────────────
def init_csv():
    sensor_header = ["timestamp","machine_id","voltage","current","temperature",
                     "vibration","humidity","pressure","flow_rate","frequency",
                     "power_factor","power_kw"]
    alarm_header  = ["timestamp","machine_id","alarm_type","severity","value"]

    for path, header in [(CSV_LIVE, sensor_header), (CSV_ALARMS, alarm_header)]:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)
    print(f"[CSV] Fallback aktif → {CSV_LIVE}")

def save_to_csv(data):
    with open(CSV_LIVE, "a", newline="") as f:
        csv.writer(f).writerow([
            data["timestamp"], data["machine_id"],
            data["voltage"], data["current"], data["temperature"],
            data["vibration"], data["humidity"], data["pressure"],
            data["flow_rate"], data["frequency"],
            data["power_factor"], data["power_kw"],
        ])
    for alarm in data.get("alarms", []):
        with open(CSV_ALARMS, "a", newline="") as f:
            csv.writer(f).writerow([
                data["timestamp"], data["machine_id"],
                alarm["type"], alarm["severity"], alarm["value"],
            ])


# ─────────────────────────────────────────────────────────
# MQTT CALLBACKS
# ─────────────────────────────────────────────────────────
msg_count  = 0
alarm_count = 0

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected | Subscribe: {TOPIC_SUB}")
        client.subscribe(TOPIC_SUB, qos=1)
    else:
        print(f"[MQTT] Connect failed (rc={rc})")

def on_message(client, userdata, msg):
    global msg_count, alarm_count
    try:
        data = json.loads(msg.payload.decode("utf-8"))

        if PG_OK and conn:
            save_to_pg(data)
        else:
            save_to_csv(data)

        msg_count  += 1
        alarm_count += len(data.get("alarms", []))

        # Log setiap 10 pesan
        if msg_count % 10 == 0:
            print(f"  [MSG {msg_count:>5}] {data['machine_id']:<10} "
                  f"T={data['temperature']:>5.1f}°C  V={data['vibration']:>5.3f}mm/s  "
                  f"⚡{data['power_kw']:>7.2f}kW  | Total alarm: {alarm_count}")

        # Cetak alarm langsung
        for alarm in data.get("alarms", []):
            icon = "🔴" if alarm["severity"] == "Critical" else "🟡"
            print(f"  {icon} ALARM [{alarm['severity']:>8}] {data['machine_id']} — "
                  f"{alarm['type']}: {alarm['value']}")

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode: {e}")
    except Exception as e:
        print(f"[ERROR] on_message: {e}")

def on_disconnect(client, userdata, rc):
    print(f"[MQTT] Disconnected (rc={rc})")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Smart Industrial IoT — MQTT Subscriber             ║")
    print("║   PT Nusantara Steel Manufacturing                   ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Init storage
    pg_ok = init_pg()
    if not pg_ok:
        init_csv()
        print("[INFO] Data akan disimpan ke CSV (PostgreSQL tidak tersedia)")

    if not MQTT_OK:
        print("[ERROR] paho-mqtt diperlukan. Jalankan: pip install paho-mqtt")
        sys.exit(1)

    client = mqtt.Client(client_id="nusantara_subscriber")
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    # Graceful shutdown
    def shutdown(sig, frame):
        print(f"\n[INFO] Total pesan diterima: {msg_count} | Total alarm: {alarm_count}")
        client.loop_stop()
        client.disconnect()
        if conn: conn.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        print(f"[MQTT] Menunggu data dari broker {BROKER_HOST}:{BROKER_PORT}...\n")
        client.loop_forever()
    except ConnectionRefusedError:
        print(f"[ERROR] Broker MQTT tidak berjalan di {BROKER_HOST}:{BROKER_PORT}")
        print("[INFO] Jalankan Mosquitto dulu: mosquitto -v")

if __name__ == "__main__":
    main()
