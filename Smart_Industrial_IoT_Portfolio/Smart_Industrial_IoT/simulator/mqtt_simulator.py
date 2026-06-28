#!/usr/bin/env python3
"""
Smart Industrial IoT — MQTT Sensor Simulator
PT Nusantara Steel Manufacturing

Mensimulasikan aliran data sensor real-time melalui MQTT broker.
Arsitektur: Sensor → MQTT Broker → Subscriber → Database

Cara menjalankan:
1. Install Mosquitto: sudo apt-get install mosquitto mosquitto-clients
2. Jalankan broker: mosquitto -v
3. Jalankan simulator ini: python simulator/mqtt_simulator.py
4. Jalankan subscriber:    python simulator/mqtt_subscriber.py
"""

import json
import time
import random
import numpy as np
from datetime import datetime
import os

# MQTT tersedia jika diinstall. Script ini juga bisa dijalankan
# tanpa broker (mode DRY RUN) untuk keperluan demo.
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[WARN] paho-mqtt tidak tersedia. Jalankan dalam mode DRY RUN.\n")

# ─────────────────────────────────────────────────────────
# KONFIGURASI MQTT
# ─────────────────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC_BASE  = "nusantara/steel/sensor"   # → nusantara/steel/sensor/MTR-001
QOS         = 1
PUBLISH_INTERVAL = 5   # detik antar publish (simulasi 10 menit dikompresi)

# ─────────────────────────────────────────────────────────
# DEFINISI MESIN (subset 10 mesin untuk demo real-time)
# ─────────────────────────────────────────────────────────
DEMO_MACHINES = [
    {"id":"MTR-001","type":"Motor",       "i_min":10,"i_max":32,"t_base":50,"v_base":1.50,"pf_min":0.82},
    {"id":"MTR-004","type":"Motor",       "i_min":25,"i_max":45,"t_base":58,"v_base":2.00,"pf_min":0.82},
    {"id":"PMP-001","type":"Pump",        "i_min": 4,"i_max":16,"t_base":40,"v_base":1.10,"pf_min":0.83},
    {"id":"PMP-003","type":"Pump",        "i_min": 6,"i_max":24,"t_base":44,"v_base":1.30,"pf_min":0.82},
    {"id":"CMP-001","type":"Compressor",  "i_min":20,"i_max":40,"t_base":58,"v_base":1.80,"pf_min":0.82},
    {"id":"CMP-005","type":"Compressor",  "i_min":45,"i_max":70,"t_base":62,"v_base":2.10,"pf_min":0.82},
    {"id":"CNV-001","type":"Conveyor",    "i_min": 6,"i_max":20,"t_base":45,"v_base":2.00,"pf_min":0.82},
    {"id":"BLR-001","type":"Boiler",      "i_min":30,"i_max":75,"t_base":100,"v_base":0.70,"pf_min":0.86},
    {"id":"CLT-001","type":"Cooling Tower","i_min":15,"i_max":38,"t_base":30,"v_base":0.90,"pf_min":0.84},
    {"id":"MTR-012","type":"Motor",       "i_min":12,"i_max":38,"t_base":47,"v_base":1.10,"pf_min":0.85},
]

# ─────────────────────────────────────────────────────────
# SENSOR SIMULATION ENGINE
# ─────────────────────────────────────────────────────────
class SensorSimulator:
    def __init__(self, machine_config):
        self.m  = machine_config
        self.lf = 0.85      # load factor awal
        self._fault_active   = False
        self._fault_counter  = 0
        self._prev_temp      = machine_config["t_base"]
        self._prev_vib       = machine_config["v_base"]

    def _update_load_factor(self):
        hour = datetime.now().hour
        if 7 <= hour < 18:
            target_lf = 0.85 + random.gauss(0, 0.05)
        elif 18 <= hour < 22:
            target_lf = 0.70 + random.gauss(0, 0.05)
        else:
            target_lf = 0.50 + random.gauss(0, 0.05)
        # Smooth transition
        self.lf = 0.9 * self.lf + 0.1 * np.clip(target_lf, 0.3, 1.1)

    def _maybe_trigger_fault(self):
        """Fault acak — 1% kemungkinan setiap siklus."""
        if not self._fault_active and random.random() < 0.01:
            self._fault_active  = True
            self._fault_counter = random.randint(3, 12)   # berlangsung 3-12 siklus
        if self._fault_active:
            self._fault_counter -= 1
            if self._fault_counter <= 0:
                self._fault_active = False

    def generate(self):
        self._update_load_factor()
        self._maybe_trigger_fault()

        m = self.m
        fault = self._fault_active

        # ── Voltage ──
        voltage = 380 + random.gauss(0, 2.0)
        if fault: voltage -= random.uniform(10, 25)
        voltage = round(np.clip(voltage, 355, 405), 1)

        # ── Current (ATURAN 4) ──
        i_range = m["i_max"] - m["i_min"]
        current = m["i_min"] + i_range * self.lf + random.gauss(0, i_range * 0.03)
        if fault: current *= random.uniform(1.2, 1.4)
        current = round(np.clip(current, m["i_min"] * 0.6, m["i_max"] * 1.5), 2)

        # ── Temperature (ATURAN 1: current → temp, lag effect) ──
        curr_norm = (current - m["i_min"]) / (i_range + 1e-6)
        target_temp = m["t_base"] + 22 * curr_norm + random.gauss(0, 2.5)
        if fault: target_temp += random.uniform(15, 30)
        self._prev_temp = 0.85 * self._prev_temp + 0.15 * target_temp
        temperature = round(np.clip(self._prev_temp, 15, 180), 1)

        # ── Vibration (ATURAN 2+3: age + fault) ──
        target_vib = m["v_base"] + 0.4 * curr_norm + random.gauss(0, 0.12)
        if fault: target_vib *= random.uniform(2.0, 3.5)
        self._prev_vib = 0.80 * self._prev_vib + 0.20 * target_vib
        vibration = round(np.clip(self._prev_vib, 0.05, 12.0), 3)

        # ── Power Factor ──
        pf = m["pf_min"] + (0.96 - m["pf_min"]) * self.lf + random.gauss(0, 0.01)
        if fault: pf -= random.uniform(0.05, 0.12)
        power_factor = round(np.clip(pf, 0.65, 1.00), 3)

        # ── Frequency ──
        frequency = round(50.0 + random.gauss(0, 0.08), 2)

        # ── Humidity ──
        hour = datetime.now().hour
        base_hum = 52 + (8 if (5 <= datetime.now().month <= 10) else 0)
        humidity  = round(np.clip(base_hum + random.gauss(0, 4), 20, 98), 1)

        # ── Pressure (jika relevan) ──
        pressure = 0.0
        if m["type"] in ["Pump","Compressor","Boiler"]:
            pressure = round(np.clip(6.0 * self.lf + random.gauss(0, 0.3), 0, 13), 2)

        # ── Flow Rate ──
        flow_rate = 0.0
        if m["type"] in ["Pump","Boiler","Cooling Tower"]:
            if m["type"] == "Cooling Tower":
                flow_rate = round(np.clip(180 * self.lf + random.gauss(0, 8), 120, 225), 1)
            else:
                flow_rate = round(np.clip(80 * self.lf + random.gauss(0, 5), 15, 160), 1)

        # ── Power kW ──
        power_kw = round(np.sqrt(3) * voltage * current * power_factor / 1000, 2)

        # ── Alarm check ──
        alarms = []
        if temperature > 80:  alarms.append({"type":"High Temperature","severity":"Critical","value":temperature})
        elif temperature > 70: alarms.append({"type":"High Temperature","severity":"Warning","value":temperature})
        if vibration > 4.5:   alarms.append({"type":"High Vibration","severity":"Critical","value":vibration})
        elif vibration > 3.5: alarms.append({"type":"High Vibration","severity":"Warning","value":vibration})
        if voltage < 360:     alarms.append({"type":"Under Voltage","severity":"Critical","value":voltage})
        elif voltage < 368:   alarms.append({"type":"Under Voltage","severity":"Warning","value":voltage})
        if power_factor < 0.80: alarms.append({"type":"Low Power Factor","severity":"Critical","value":power_factor})
        elif power_factor < 0.83: alarms.append({"type":"Low Power Factor","severity":"Warning","value":power_factor})

        return {
            "timestamp":    datetime.now().isoformat(),
            "machine_id":   m["id"],
            "machine_type": m["type"],
            "voltage":      voltage,
            "current":      current,
            "temperature":  temperature,
            "vibration":    vibration,
            "humidity":     humidity,
            "pressure":     pressure,
            "flow_rate":    flow_rate,
            "frequency":    frequency,
            "power_factor": power_factor,
            "power_kw":     power_kw,
            "load_factor":  round(self.lf, 3),
            "fault_active": fault,
            "alarms":       alarms,
        }


# ─────────────────────────────────────────────────────────
# MAIN SIMULATOR
# ─────────────────────────────────────────────────────────
def run_dry():
    """Jalankan tanpa broker MQTT — hanya print ke konsol."""
    print("\n" + "="*60)
    print("  MQTT SENSOR SIMULATOR — DRY RUN MODE")
    print("  (Tanpa broker MQTT, data ditampilkan di konsol)")
    print("="*60)

    simulators = [SensorSimulator(m) for m in DEMO_MACHINES]
    cycle = 0

    try:
        while True:
            cycle += 1
            print(f"\n{'─'*60}")
            print(f"  SIKLUS #{cycle}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'─'*60}")
            print(f"  {'MESIN':<12} {'TIPE':<15} {'VOLT':>5} {'AMP':>6} {'TEMP':>6} {'VIB':>6} {'PF':>5} {'kW':>7} {'ALARM'}")
            print(f"  {'─'*10:<12} {'─'*13:<15} {'─'*5:>5} {'─'*6:>6} {'─'*6:>6} {'─'*6:>6} {'─'*5:>5} {'─'*7:>7}")

            for sim in simulators:
                data = sim.generate()
                alarm_str = f"🔴{len(data['alarms'])}" if data['alarms'] else "  ✅"
                fault_str = " ⚠FAULT" if data["fault_active"] else ""
                print(f"  {data['machine_id']:<12} {data['machine_type']:<15} "
                      f"{data['voltage']:>5.1f} {data['current']:>6.2f} "
                      f"{data['temperature']:>6.1f} {data['vibration']:>6.3f} "
                      f"{data['power_factor']:>5.3f} {data['power_kw']:>7.2f} "
                      f"{alarm_str}{fault_str}")

            print(f"\n  ⏱  Berikutnya dalam {PUBLISH_INTERVAL} detik... (Ctrl+C untuk berhenti)")
            time.sleep(PUBLISH_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n  Simulator dihentikan.")


def run_mqtt():
    """Jalankan dengan broker MQTT."""
    client = mqtt.Client(client_id="nusantara_simulator")
    client.on_connect = lambda c,u,f,rc: print(f"[MQTT] Connected (rc={rc})")
    client.on_disconnect = lambda c,u,rc: print(f"[MQTT] Disconnected (rc={rc})")

    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        client.loop_start()
    except Exception as e:
        print(f"[ERROR] Tidak bisa terhubung ke broker MQTT: {e}")
        print("Beralih ke DRY RUN mode...\n")
        run_dry()
        return

    simulators = [SensorSimulator(m) for m in DEMO_MACHINES]
    cycle = 0

    print(f"\n[MQTT] Broker: {BROKER_HOST}:{BROKER_PORT}")
    print(f"[MQTT] Topic : {TOPIC_BASE}/<machine_id>")
    print(f"[MQTT] Interval: {PUBLISH_INTERVAL}s")
    print("[MQTT] Simulator berjalan... (Ctrl+C untuk berhenti)\n")

    try:
        while True:
            cycle += 1
            total_alarms = 0

            for sim in simulators:
                data   = sim.generate()
                topic  = f"{TOPIC_BASE}/{data['machine_id']}"
                payload = json.dumps(data)
                result  = client.publish(topic, payload, qos=QOS)
                total_alarms += len(data["alarms"])

                if data["fault_active"]:
                    print(f"  ⚠ FAULT: {data['machine_id']} | Temp={data['temperature']}°C "
                          f"Vib={data['vibration']}mm/s")

            print(f"[Cycle {cycle:>4}] {datetime.now().strftime('%H:%M:%S')} "
                  f"| {len(simulators)} mesin | {total_alarms} alarm aktif")
            time.sleep(PUBLISH_INTERVAL)

    except KeyboardInterrupt:
        print("\n[MQTT] Simulator dihentikan.")
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Smart Industrial IoT — MQTT Sensor Simulator      ║")
    print("║   PT Nusantara Steel Manufacturing                   ║")
    print("╚══════════════════════════════════════════════════════╝")

    if MQTT_AVAILABLE:
        run_mqtt()
    else:
        run_dry()
