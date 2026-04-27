-- Failure Events Schema - Log all equipment failures
CREATE TABLE IF NOT EXISTS failure_events.failure_modes (
    failure_mode_id SERIAL PRIMARY KEY,
    equipment_type VARCHAR(50) NOT NULL,
    mode_code VARCHAR(50) NOT NULL,
    description TEXT,
    UNIQUE(equipment_type, mode_code)
);

CREATE TABLE IF NOT EXISTS failure_events.turbine_failures (
    failure_id BIGSERIAL PRIMARY KEY,
    turbine_id INT NOT NULL REFERENCES master_data.turbines(turbine_id) ON DELETE CASCADE,
    failure_time TIMESTAMP NOT NULL,
    operating_hours_at_failure FLOAT,
    failure_mode_code VARCHAR(50),
    failure_description TEXT,
    speed_rpm_at_failure FLOAT,
    egt_celsius_at_failure FLOAT,
    vibration_mm_s_at_failure FLOAT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS failure_events.compressor_failures (
    failure_id BIGSERIAL PRIMARY KEY,
    compressor_id INT NOT NULL REFERENCES master_data.compressors(compressor_id) ON DELETE CASCADE,
    failure_time TIMESTAMP NOT NULL,
    operating_hours_at_failure FLOAT,
    failure_mode_code VARCHAR(50),
    failure_description TEXT,
    speed_rpm_at_failure FLOAT,
    surge_margin_at_failure FLOAT,
    surge_cycles_at_failure INT,
    vibration_amplitude_at_failure FLOAT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS failure_events.pump_failures (
    failure_id BIGSERIAL PRIMARY KEY,
    pump_id INT NOT NULL REFERENCES master_data.pumps(pump_id) ON DELETE CASCADE,
    failure_time TIMESTAMP NOT NULL,
    operating_hours_at_failure FLOAT,
    failure_mode_code VARCHAR(50),
    failure_description TEXT,
    speed_rpm_at_failure FLOAT,
    vibration_mm_s_at_failure FLOAT,
    cavitation_margin_at_failure FLOAT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_gt_failures_turbine ON failure_events.turbine_failures(turbine_id);
CREATE INDEX IF NOT EXISTS idx_gt_failures_time ON failure_events.turbine_failures(failure_time DESC);
CREATE INDEX IF NOT EXISTS idx_cc_failures_compressor ON failure_events.compressor_failures(compressor_id);
CREATE INDEX IF NOT EXISTS idx_cc_failures_time ON failure_events.compressor_failures(failure_time DESC);
CREATE INDEX IF NOT EXISTS idx_cp_failures_pump ON failure_events.pump_failures(pump_id);
CREATE INDEX IF NOT EXISTS idx_cp_failures_time ON failure_events.pump_failures(failure_time DESC);

-- Populate failure modes reference table
INSERT INTO failure_events.failure_modes (equipment_type, mode_code, description) VALUES
    ('turbine', 'F_HGP', 'Hot Gas Path Degradation - Combustion liner cracking'),
    ('turbine', 'F_BLADE_COMPRESSOR', 'Compressor Blade Fouling/Erosion - Discharge temp loss'),
    ('turbine', 'F_BLADE_TURBINE', 'Turbine Blade Tip Clearance/Rub - Integer harmonic vibration'),
    ('turbine', 'F_BEARING', 'Bearing Failure - Lubrication/mechanical degradation'),
    ('turbine', 'F_FUEL', 'Fuel System Fouling - Nozzle blockage'),
    ('turbine', 'F_COMPRESSOR_FOULING', 'Compressor Fouling - Airborne deposit buildup'),
    ('compressor', 'F_IMPELLER', 'Impeller Degradation - Erosion or fouling'),
    ('compressor', 'F_BEARING', 'Bearing Failure - Journal bearing damage'),
    ('compressor', 'F_SEAL_PRIMARY', 'Primary Dry Gas Seal Failure'),
    ('compressor', 'F_SEAL_SECONDARY', 'Secondary Dry Gas Seal Failure'),
    ('compressor', 'F_HIGH_VIBRATION', 'High Vibration Trip - Shaft orbit amplitude exceeded safety limits'),
    ('compressor', 'F_SURGE', 'Compressor Surge - Anti-surge protection failure, flow reversal damage'),
    ('compressor', 'F_BEARING_THRUST', 'Thrust Bearing Failure - Axial load capacity degradation'),
    ('compressor', 'F_ROTOR_CRACK', 'Rotor Crack - Fatigue crack, stiffness asymmetry'),
    ('pump', 'F_IMPELLER', 'Impeller Degradation - Erosion, corrosion, or damage'),
    ('pump', 'F_WEAR_RING', 'Wear Ring Degradation - Clearance increase shifts BEP'),
    ('pump', 'F_SEAL', 'Mechanical Seal Failure - Wear, thermal damage, or contamination'),
    ('pump', 'F_BEARING_DRIVE_END', 'Drive End Bearing Failure - Fatigue, lubrication, or contamination'),
    ('pump', 'F_BEARING_NON_DRIVE_END', 'Non-Drive End Bearing Failure'),
    ('pump', 'F_CAVITATION', 'Severe Cavitation - NPSH margin critical (sustained 30s)')
ON CONFLICT (equipment_type, mode_code) DO NOTHING;
