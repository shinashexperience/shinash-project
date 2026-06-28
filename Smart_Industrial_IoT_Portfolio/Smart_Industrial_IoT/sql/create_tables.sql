-- ============================================================
--  Smart Industrial IoT — Database Schema
--  PT Nusantara Steel Manufacturing
--  Database: PostgreSQL 15+
-- ============================================================

-- Buat database (jalankan sebagai superuser)
-- CREATE DATABASE iot_nusantara;

-- ────────────────────────────────────────────────────────
-- 1. MACHINE MASTER
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS machine_master (
    machine_id      VARCHAR(20)  PRIMARY KEY,
    machine_type    VARCHAR(30)  NOT NULL,
    location        VARCHAR(60)  NOT NULL,
    rated_power_kw  NUMERIC(8,2) NOT NULL,
    install_year    SMALLINT     NOT NULL
);

-- ────────────────────────────────────────────────────────
-- 2. SENSOR LOG (data historis)
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sensor_log (
    id              BIGSERIAL    PRIMARY KEY,
    timestamp       TIMESTAMPTZ  NOT NULL,
    machine_id      VARCHAR(20)  NOT NULL REFERENCES machine_master(machine_id),
    voltage         NUMERIC(6,1),
    current         NUMERIC(7,2),
    temperature     NUMERIC(6,1),
    vibration       NUMERIC(6,3),
    humidity        NUMERIC(5,1),
    pressure        NUMERIC(6,2),
    flow_rate       NUMERIC(7,1),
    frequency       NUMERIC(5,2),
    power_factor    NUMERIC(5,3),
    power_kw        NUMERIC(8,2),
    energy_kwh      NUMERIC(8,4)
);

-- Index untuk performa query
CREATE INDEX idx_sensor_machine   ON sensor_log(machine_id);
CREATE INDEX idx_sensor_timestamp ON sensor_log(timestamp DESC);
CREATE INDEX idx_sensor_machine_ts ON sensor_log(machine_id, timestamp DESC);

-- Partisi bulanan (opsional untuk skala besar)
-- Gunakan TimescaleDB untuk time-series yang lebih efisien

-- ────────────────────────────────────────────────────────
-- 3. ALARM HISTORY
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alarm_history (
    alarm_id    VARCHAR(20)  PRIMARY KEY,
    timestamp   TIMESTAMPTZ  NOT NULL,
    machine_id  VARCHAR(20)  NOT NULL REFERENCES machine_master(machine_id),
    alarm_type  VARCHAR(60)  NOT NULL,
    severity    VARCHAR(20)  NOT NULL CHECK (severity IN ('Warning','Critical')),
    value       VARCHAR(30),
    acknowledged BOOLEAN     DEFAULT FALSE,
    ack_by      VARCHAR(50),
    ack_at      TIMESTAMPTZ
);

CREATE INDEX idx_alarm_machine   ON alarm_history(machine_id);
CREATE INDEX idx_alarm_timestamp ON alarm_history(timestamp DESC);
CREATE INDEX idx_alarm_severity  ON alarm_history(severity, timestamp DESC);

-- ────────────────────────────────────────────────────────
-- 4. MAINTENANCE HISTORY
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS maintenance_history (
    maintenance_id  VARCHAR(20)  PRIMARY KEY,
    machine_id      VARCHAR(20)  NOT NULL REFERENCES machine_master(machine_id),
    date            DATE         NOT NULL,
    type            VARCHAR(80)  NOT NULL,
    technician      VARCHAR(60),
    duration_hours  NUMERIC(6,1),
    component       VARCHAR(80),
    cost_idr        BIGINT,
    notes           TEXT
);

CREATE INDEX idx_maint_machine ON maintenance_history(machine_id);
CREATE INDEX idx_maint_date    ON maintenance_history(date DESC);

-- ────────────────────────────────────────────────────────
-- 5. MACHINE STATUS (materialized — refresh periodik)
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS machine_status (
    machine_id       VARCHAR(20) PRIMARY KEY REFERENCES machine_master(machine_id),
    machine_type     VARCHAR(30),
    status           VARCHAR(20) CHECK (status IN ('Good','Warning','Critical','Offline')),
    health_score     NUMERIC(5,1),
    avg_temp_30d     NUMERIC(6,1),
    avg_vib_30d      NUMERIC(6,3),
    avg_pf_30d       NUMERIC(5,3),
    alarm_count_30d  INTEGER,
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────
-- 6. OPERATOR SHIFT
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS operator_shift (
    id         SERIAL      PRIMARY KEY,
    operator   VARCHAR(60) NOT NULL,
    shift      VARCHAR(30) NOT NULL,
    area       VARCHAR(60) NOT NULL
);

-- ────────────────────────────────────────────────────────
-- 7. SENSOR LIVE (real-time dari MQTT)
-- ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sensor_live (
    id           BIGSERIAL   PRIMARY KEY,
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    machine_id   VARCHAR(20) NOT NULL,
    voltage      NUMERIC(6,1),
    current      NUMERIC(7,2),
    temperature  NUMERIC(6,1),
    vibration    NUMERIC(6,3),
    humidity     NUMERIC(5,1),
    pressure     NUMERIC(6,2),
    flow_rate    NUMERIC(7,1),
    frequency    NUMERIC(5,2),
    power_factor NUMERIC(5,3),
    power_kw     NUMERIC(8,2)
);

CREATE INDEX idx_live_machine ON sensor_live(machine_id, timestamp DESC);
