-- Telemetry Schema - Time Series Data
-- Updated to include ALL simulator output modes (environmental, maintenance, faults, upsets, enhanced vibration)
CREATE TABLE IF NOT EXISTS telemetry.gas_turbine_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    turbine_id INT NOT NULL REFERENCES master_data.gas_turbines(turbine_id) ON DELETE CASCADE,
    sample_time TIMESTAMP NOT NULL,
    operating_hours FLOAT,
    -- Core measurements
    speed_rpm FLOAT,
    speed_target_rpm FLOAT,
    egt_celsius FLOAT,
    oil_temp_celsius FLOAT,
    fuel_flow_kg_s FLOAT,
    compressor_discharge_temp_celsius FLOAT,
    compressor_discharge_pressure_kpa FLOAT,
    efficiency_fraction FLOAT,
    -- Environmental conditions (enable_environmental=True)
    ambient_temp_celsius FLOAT,
    ambient_pressure_kpa FLOAT,
    -- Vibration metrics
    vibration_rms_mm_s FLOAT,
    vibration_peak_mm_s FLOAT,
    vibration_crest_factor FLOAT,
    vibration_kurtosis FLOAT,
    -- Health indicators
    health_hgp FLOAT,
    health_blade FLOAT,
    health_bearing FLOAT,
    health_fuel FLOAT,
    -- Fault tracking (enable_faults=True)
    num_active_faults INT,
    total_faults_initiated INT,
    -- Process upset tracking (enable_upsets=True)
    upset_active BOOLEAN,
    upset_type VARCHAR(50),
    upset_severity FLOAT,
    -- Derived features (DERIVED_FEATURES output mode)
    vibration_trend_7d FLOAT,
    temp_variation_24h FLOAT,
    speed_stability FLOAT,
    efficiency_degradation_rate FLOAT,
    load_factor FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telemetry.compressor_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    compressor_id INT NOT NULL REFERENCES master_data.compressors(compressor_id) ON DELETE CASCADE,
    sample_time TIMESTAMP NOT NULL,
    operating_hours FLOAT,
    -- Core measurements
    speed_rpm FLOAT,
    speed_target_rpm FLOAT,
    flow_m3h FLOAT,
    head_kj_kg FLOAT,
    efficiency_fraction FLOAT,
    power_kw FLOAT,
    -- Pressure and temperature
    suction_pressure_kpa FLOAT,
    suction_temp_celsius FLOAT,
    discharge_pressure_kpa FLOAT,
    discharge_temp_celsius FLOAT,
    -- Surge protection
    surge_margin_percent FLOAT,
    surge_alarm BOOLEAN,
    -- Vibration and shaft dynamics
    vibration_amplitude_mm FLOAT,
    sync_amplitude_mm FLOAT,
    shaft_x_displacement_mm FLOAT,
    shaft_y_displacement_mm FLOAT,
    -- Bearing temperatures
    bearing_temp_de_celsius FLOAT,
    bearing_temp_nde_celsius FLOAT,
    thrust_bearing_temp_celsius FLOAT,
    -- Seal condition
    seal_health_primary FLOAT,
    seal_health_secondary FLOAT,
    primary_seal_leakage_kg_s FLOAT,
    secondary_seal_leakage_kg_s FLOAT,
    -- Health indicators
    health_impeller FLOAT,
    health_bearing FLOAT,
    -- Surge event tracking
    surge_active BOOLEAN,
    surge_cycle_count INT,
    -- Fault tracking (enable_faults=True)
    num_active_faults INT,
    total_faults_initiated INT,
    -- Process upset tracking (enable_upsets=True)
    upset_active BOOLEAN,
    upset_type VARCHAR(50),
    upset_severity FLOAT,
    -- Derived features (DERIVED_FEATURES output mode)
    vibration_trend_7d FLOAT,
    temp_variation_24h FLOAT,
    speed_stability FLOAT,
    efficiency_degradation_rate FLOAT,
    pressure_ratio FLOAT,
    load_factor FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telemetry.pump_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    pump_id INT NOT NULL REFERENCES master_data.pumps(pump_id) ON DELETE CASCADE,
    sample_time TIMESTAMP NOT NULL,
    operating_hours FLOAT,
    -- Core measurements
    speed_rpm FLOAT,
    speed_target_rpm FLOAT,
    flow_m3h FLOAT,
    head_m FLOAT,
    efficiency_fraction FLOAT,
    power_kw FLOAT,
    -- Pressure and temperature
    suction_pressure_kpa FLOAT,
    discharge_pressure_kpa FLOAT,
    fluid_temp_celsius FLOAT,
    -- NPSH and cavitation
    npsh_available_m FLOAT,
    npsh_required_m FLOAT,
    cavitation_margin_m FLOAT,
    cavitation_severity INT,
    -- Vibration metrics
    vibration_rms_mm_s FLOAT,
    vibration_peak_mm_s FLOAT,
    -- Bearings
    bearing_temp_de_celsius FLOAT,
    bearing_temp_nde_celsius FLOAT,
    -- Motor
    motor_current_amps FLOAT,
    motor_current_ratio FLOAT,
    -- Seal condition
    seal_health FLOAT,
    seal_leakage_rate FLOAT,
    -- Performance
    bep_deviation_percent FLOAT,
    -- Health indicators
    health_impeller FLOAT,
    health_seal FLOAT,
    health_bearing_de FLOAT,
    health_bearing_nde FLOAT,
    -- Derived features (DERIVED_FEATURES output mode)
    temp_variation_24h FLOAT,
    vibration_trend_7d FLOAT,
    speed_stability FLOAT,
    efficiency_degradation_rate FLOAT,
    pressure_ratio FLOAT,
    load_factor FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_gt_telemetry_turbine_time ON telemetry.gas_turbine_telemetry(turbine_id, sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_gt_telemetry_time ON telemetry.gas_turbine_telemetry(sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_cc_telemetry_compressor_time ON telemetry.compressor_telemetry(compressor_id, sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_cc_telemetry_time ON telemetry.compressor_telemetry(sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_cp_telemetry_pump_time ON telemetry.pump_telemetry(pump_id, sample_time DESC);
CREATE INDEX IF NOT EXISTS idx_cp_telemetry_time ON telemetry.pump_telemetry(sample_time DESC);
