-- ============================================================
--  Smart Industrial IoT — Sample Analytical Queries
--  PT Nusantara Steel Manufacturing
-- ============================================================


-- ────────────────────────────────────────────────────────
-- ASSET HEALTH QUERIES
-- ────────────────────────────────────────────────────────

-- Q1: Mesin dengan alarm terbanyak (Top 10)
SELECT
    machine_id,
    COUNT(*)                                      AS total_alarm,
    COUNT(*) FILTER (WHERE severity = 'Critical') AS critical,
    COUNT(*) FILTER (WHERE severity = 'Warning')  AS warning
FROM alarm_history
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY machine_id
ORDER BY total_alarm DESC
LIMIT 10;


-- Q2: Mesin dengan health score terburuk
SELECT
    ms.machine_id,
    mm.machine_type,
    mm.location,
    ms.health_score,
    ms.status,
    ms.avg_temp_30d,
    ms.avg_vib_30d,
    ms.alarm_count_30d,
    2024 - mm.install_year AS age_years
FROM machine_status ms
JOIN machine_master  mm ON ms.machine_id = mm.machine_id
ORDER BY ms.health_score ASC
LIMIT 10;


-- Q3: Apakah usia mesin mempengaruhi vibrasi? (Pearson r)
SELECT
    CORR(2024 - mm.install_year, ms.avg_vib_30d) AS corr_age_vibration
FROM machine_status ms
JOIN machine_master  mm ON ms.machine_id = mm.machine_id;


-- Q4: Trend suhu rata-rata per minggu per tipe mesin
SELECT
    DATE_TRUNC('week', timestamp)  AS week,
    mm.machine_type,
    AVG(temperature)               AS avg_temp,
    MAX(temperature)               AS max_temp,
    STDDEV(temperature)            AS stddev_temp
FROM sensor_log sl
JOIN machine_master mm ON sl.machine_id = mm.machine_id
WHERE timestamp >= '2024-01-01'
GROUP BY 1, 2
ORDER BY 1, 2;


-- ────────────────────────────────────────────────────────
-- ENERGY QUERIES
-- ────────────────────────────────────────────────────────

-- Q5: Mesin paling boros energi (total kWh 2024)
SELECT
    sl.machine_id,
    mm.machine_type,
    mm.rated_power_kw,
    SUM(sl.energy_kwh)         AS total_kwh,
    AVG(sl.power_factor)       AS avg_pf,
    AVG(sl.power_kw)           AS avg_kw
FROM sensor_log sl
JOIN machine_master mm ON sl.machine_id = mm.machine_id
WHERE sl.timestamp BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY sl.machine_id, mm.machine_type, mm.rated_power_kw
ORDER BY total_kwh DESC
LIMIT 15;


-- Q6: Jam operasi paling mahal (konsumsi energi per jam)
SELECT
    EXTRACT(HOUR FROM timestamp) AS hour_of_day,
    AVG(power_kw)                AS avg_power_kw,
    SUM(energy_kwh)              AS total_kwh,
    COUNT(DISTINCT machine_id)   AS machines_active
FROM sensor_log
WHERE timestamp BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY 1
ORDER BY total_kwh DESC;


-- Q7: Power Factor terburuk per mesin
SELECT
    machine_id,
    AVG(power_factor)   AS avg_pf,
    MIN(power_factor)   AS min_pf,
    COUNT(*) FILTER (WHERE power_factor < 0.82) AS low_pf_events
FROM sensor_log
GROUP BY machine_id
ORDER BY avg_pf ASC
LIMIT 10;


-- Q8: Konsumsi energi bulanan per tipe mesin
SELECT
    TO_CHAR(timestamp, 'YYYY-MM')  AS month,
    mm.machine_type,
    SUM(sl.energy_kwh)             AS total_kwh,
    AVG(sl.power_factor)           AS avg_pf
FROM sensor_log sl
JOIN machine_master mm ON sl.machine_id = mm.machine_id
GROUP BY 1, 2
ORDER BY 1, 2;


-- ────────────────────────────────────────────────────────
-- MAINTENANCE QUERIES
-- ────────────────────────────────────────────────────────

-- Q9: Komponen yang paling sering diganti
SELECT
    component,
    COUNT(*)            AS frequency,
    SUM(cost_idr)       AS total_cost,
    AVG(duration_hours) AS avg_duration_h
FROM maintenance_history
WHERE type <> 'Preventive Maintenance'
GROUP BY component
ORDER BY frequency DESC
LIMIT 10;


-- Q10: MTBF per mesin (Mean Time Between Failures)
WITH failures AS (
    SELECT
        machine_id,
        COUNT(*) AS n_failures
    FROM maintenance_history
    WHERE type <> 'Preventive Maintenance'
    GROUP BY machine_id
)
SELECT
    f.machine_id,
    mm.machine_type,
    f.n_failures,
    ROUND(8760.0 / f.n_failures, 1) AS mtbf_hours,   -- 8760h = 1 tahun
    2024 - mm.install_year           AS age_years
FROM failures f
JOIN machine_master mm ON f.machine_id = mm.machine_id
ORDER BY mtbf_hours ASC
LIMIT 15;


-- Q11: MTTR per mesin (Mean Time To Repair)
SELECT
    machine_id,
    COUNT(*)             AS n_repairs,
    AVG(duration_hours)  AS mttr_hours,
    MAX(duration_hours)  AS max_repair_hours,
    SUM(cost_idr)        AS total_cost_idr
FROM maintenance_history
WHERE type <> 'Preventive Maintenance'
GROUP BY machine_id
ORDER BY mttr_hours DESC;


-- Q12: Biaya maintenance per bulan
SELECT
    TO_CHAR(date, 'YYYY-MM')   AS month,
    SUM(cost_idr)              AS total_cost,
    SUM(cost_idr) FILTER (WHERE type = 'Preventive Maintenance') AS preventive_cost,
    SUM(cost_idr) FILTER (WHERE type <> 'Preventive Maintenance') AS corrective_cost,
    COUNT(*)                   AS total_events
FROM maintenance_history
GROUP BY 1
ORDER BY 1;


-- ────────────────────────────────────────────────────────
-- IoT / SENSOR ANALYTICS
-- ────────────────────────────────────────────────────────

-- Q13: Berapa banyak data dikirim per hari?
SELECT
    DATE(timestamp)  AS date,
    COUNT(*)         AS records,
    COUNT(DISTINCT machine_id) AS active_machines
FROM sensor_log
GROUP BY 1
ORDER BY 1 DESC
LIMIT 30;


-- Q14: Deteksi anomali sederhana (Z-score > 3)
WITH stats AS (
    SELECT
        machine_id,
        AVG(temperature)    AS mean_temp,
        STDDEV(temperature) AS std_temp
    FROM sensor_log
    GROUP BY machine_id
)
SELECT
    sl.timestamp,
    sl.machine_id,
    sl.temperature,
    s.mean_temp,
    s.std_temp,
    ABS(sl.temperature - s.mean_temp) / NULLIF(s.std_temp, 0) AS z_score
FROM sensor_log sl
JOIN stats s ON sl.machine_id = s.machine_id
WHERE ABS(sl.temperature - s.mean_temp) / NULLIF(s.std_temp, 0) > 3
ORDER BY z_score DESC
LIMIT 20;


-- Q15: Korelasi Current → Temperature (per mesin)
SELECT
    machine_id,
    CORR(current, temperature) AS corr_current_temp,
    CORR(vibration, temperature) AS corr_vib_temp,
    CORR(current, power_kw) AS corr_current_power
FROM sensor_log
WHERE timestamp >= '2024-01-01'
GROUP BY machine_id
ORDER BY corr_current_temp DESC;


-- Q16: Alarm heatmap — jam × hari
SELECT
    EXTRACT(DOW  FROM timestamp) AS day_of_week,  -- 0=Sun, 6=Sat
    EXTRACT(HOUR FROM timestamp) AS hour,
    COUNT(*)                     AS alarm_count,
    COUNT(*) FILTER (WHERE severity='Critical') AS critical_count
FROM alarm_history
GROUP BY 1, 2
ORDER BY 1, 2;


-- ────────────────────────────────────────────────────────
-- MAINTENANCE PRIORITY VIEW
-- ────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_maintenance_priority AS
WITH alarm_30d AS (
    SELECT machine_id, COUNT(*) AS alarm_count
    FROM alarm_history
    WHERE timestamp >= NOW() - INTERVAL '30 days'
    GROUP BY machine_id
),
last_maint AS (
    SELECT machine_id, MAX(date) AS last_pm_date
    FROM maintenance_history
    WHERE type = 'Preventive Maintenance'
    GROUP BY machine_id
)
SELECT
    ms.machine_id,
    mm.machine_type,
    mm.location,
    ms.health_score,
    ms.status,
    2024 - mm.install_year                        AS age_years,
    COALESCE(a.alarm_count, 0)                    AS alarm_30d,
    lm.last_pm_date,
    NOW()::DATE - lm.last_pm_date                 AS days_since_pm,
    CASE
        WHEN ms.health_score < 50  THEN 'P1 — CRITICAL'
        WHEN ms.health_score < 65  THEN 'P2 — HIGH'
        WHEN days_since_pm > 100   THEN 'P3 — PM DUE'
        ELSE 'P4 — MONITOR'
    END AS priority
FROM machine_status ms
JOIN machine_master mm  ON ms.machine_id = mm.machine_id
LEFT JOIN alarm_30d a   ON ms.machine_id = a.machine_id
LEFT JOIN last_maint lm ON ms.machine_id = lm.machine_id
ORDER BY ms.health_score ASC;
