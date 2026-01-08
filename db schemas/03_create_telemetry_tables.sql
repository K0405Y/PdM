-- Telemetry Schema - Time Series Data
CREATE TABLE IF NOT EXISTS telemetry.gas_turbine_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    turbine_id INT NOT NULL REFERENCES master_data.gas_turbines(turbine_id) ON DELETE CASCADE,
    sample_time TIMESTAMP NOT NULL,
    operating_hours FLOAT,
    speed_rpm FLOAT,
    egt_celsius FLOAT,
    oil_temp_celsius FLOAT,
    fuel_flow_kg_s FLOAT,
    compressor_discharge_temp_celsius FLOAT,
    compressor_discharge_pressure_kpa FLOAT,
    vibration_rms_mm_s FLOAT,
    vibration_peak_mm_s FLOAT,
    efficiency_fraction FLOAT,
    health_hgp FLOAT,
    health_blade FLOAT,
    health_bearing FLOAT,
    health_fuel FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telemetry.centrifugal_compressor_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    compressor_id INT NOT NULL REFERENCES master_data.centrifugal_compressors(compressor_id) ON DELETE CASCADE,
    sample_time TIMESTAMP NOT NULL,
    operating_hours FLOAT,
    speed_rpm FLOAT,
    flow_m3h FLOAT,
    head_kj_kg FLOAT,
    discharge_pressure_kpa FLOAT,
    discharge_temp_celsius FLOAT,
    surge_margin_percent FLOAT,
    vibration_amplitude_mm FLOAT,
    average_gap_mm FLOAT,
    sync_amplitude_mm FLOAT,
    bearing_temp_de_celsius FLOAT,
    bearing_temp_nde_celsius FLOAT,
    thrust_bearing_temp_celsius FLOAT,
    seal_health_primary FLOAT,
    seal_health_secondary FLOAT,
    seal_leakage_rate FLOAT,
    health_impeller FLOAT,
    health_bearing FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telemetry.centrifugal_pump_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    pump_id INT NOT NULL REFERENCES master_data.centrifugal_pumps(pump_id) ON DELETE CASCADE,
    sample_time TIMESTAMP NOT NULL,
    operating_hours FLOAT,
    speed_rpm FLOAT,
    flow_m3h FLOAT,
    head_m FLOAT,
    efficiency_fraction FLOAT,
    power_kw FLOAT,
    suction_pressure_kpa FLOAT,
    discharge_pressure_kpa FLOAT,
    fluid_temp_celsius FLOAT,
    npsh_available_m FLOAT,
    npsh_required_m FLOAT,
    cavitation_margin_m FLOAT,
    cavitation_severity INT,
    vibration_mm_s FLOAT,
    bearing_temp_de_celsius FLOAT,
    bearing_temp_nde_celsius FLOAT,
    motor_current_amps FLOAT,
    seal_health FLOAT,
    seal_leakage_rate FLOAT,
    health_impeller FLOAT,
    health_seal FLOAT,
    health_bearing_de FLOAT,
    health_bearing_nde FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_gt_telemetry_turbine_time ON telemetry.gas_turbine_telemetry(turbine_id, sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_gt_telemetry_time ON telemetry.gas_turbine_telemetry(sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_cc_telemetry_compressor_time ON telemetry.centrifugal_compressor_telemetry(compressor_id, sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_cc_telemetry_time ON telemetry.centrifugal_compressor_telemetry(sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_cp_telemetry_pump_time ON telemetry.centrifugal_pump_telemetry(pump_id, sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_cp_telemetry_time ON telemetry.centrifugal_pump_telemetry(sample_time DESC);

-- Create partitions by month for telemetry tables (optional, for very large datasets)
-- This is left commented as PostgreSQL partitioning requires additional setup
