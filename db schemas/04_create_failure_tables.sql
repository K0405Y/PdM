-- Failure Events Schema - Log all equipment failures
CREATE TABLE IF NOT EXISTS failure_events.failure_modes (
    failure_mode_id SERIAL PRIMARY KEY,
    equipment_type VARCHAR(50) NOT NULL,
    mode_code VARCHAR(50) NOT NULL,
    description TEXT,
    UNIQUE(equipment_type, mode_code)
);

CREATE TABLE IF NOT EXISTS failure_events.gas_turbine_failures (
    failure_id BIGSERIAL PRIMARY KEY,
    turbine_id INT NOT NULL REFERENCES master_data.gas_turbines(turbine_id) ON DELETE CASCADE,
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

CREATE TABLE IF NOT EXISTS failure_events.centrifugal_compressor_failures (
    failure_id BIGSERIAL PRIMARY KEY,
    compressor_id INT NOT NULL REFERENCES master_data.centrifugal_compressors(compressor_id) ON DELETE CASCADE,
    failure_time TIMESTAMP NOT NULL,
    operating_hours_at_failure FLOAT,
    failure_mode_code VARCHAR(50),
    failure_description TEXT,
    speed_rpm_at_failure FLOAT,
    surge_margin_at_failure FLOAT,
    vibration_amplitude_at_failure FLOAT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS failure_events.centrifugal_pump_failures (
    failure_id BIGSERIAL PRIMARY KEY,
    pump_id INT NOT NULL REFERENCES master_data.centrifugal_pumps(pump_id) ON DELETE CASCADE,
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
CREATE INDEX IF NOT EXISTS idx_gt_failures_turbine ON failure_events.gas_turbine_failures(turbine_id);
CREATE INDEX IF NOT EXISTS idx_gt_failures_time ON failure_events.gas_turbine_failures(failure_time DESC);
CREATE INDEX IF NOT EXISTS idx_cc_failures_compressor ON failure_events.centrifugal_compressor_failures(compressor_id);
CREATE INDEX IF NOT EXISTS idx_cc_failures_time ON failure_events.centrifugal_compressor_failures(failure_time DESC);
CREATE INDEX IF NOT EXISTS idx_cp_failures_pump ON failure_events.centrifugal_pump_failures(pump_id);
CREATE INDEX IF NOT EXISTS idx_cp_failures_time ON failure_events.centrifugal_pump_failures(failure_time DESC);

-- Populate failure modes reference table
INSERT INTO failure_events.failure_modes (equipment_type, mode_code, description) VALUES
    ('gas_turbine', 'F_HGP', 'Hot Gas Path Degradation - Combustion liner cracking'),
    ('gas_turbine', 'F_BLADE', 'Blade Erosion - Leading edge degradation'),
    ('gas_turbine', 'F_BEARING', 'Bearing Failure - Lubrication/mechanical degradation'),
    ('gas_turbine', 'F_FUEL', 'Fuel System Fouling - Nozzle blockage'),
    ('centrifugal_compressor', 'F_IMPELLER', 'Impeller Degradation - Erosion or fouling'),
    ('centrifugal_compressor', 'F_BEARING', 'Bearing Failure - Journal or thrust bearing damage'),
    ('centrifugal_compressor', 'F_SEAL_PRIMARY', 'Primary Dry Gas Seal Failure'),
    ('centrifugal_compressor', 'F_SEAL_SECONDARY', 'Secondary Dry Gas Seal Failure'),
    ('centrifugal_compressor', 'F_SURGE', 'Surge Event - Violent flow reversal'),
    ('centrifugal_compressor', 'F_HIGH_VIBRATION', 'High Vibration Trip - Shaft orbit amplitude exceeded safety limits'),
    ('centrifugal_compressor', 'F_BEARING_TEMP', 'High Bearing Temperature Trip - Temperature exceeded limit'),
    ('centrifugal_pump', 'F_IMPELLER', 'Impeller Degradation - Erosion, corrosion, or damage'),
    ('centrifugal_pump', 'F_SEAL', 'Mechanical Seal Failure - Wear, thermal damage, or contamination'),
    ('centrifugal_pump', 'F_BEARING_DRIVE_END', 'Drive End Bearing Failure - Fatigue, lubrication, or contamination'),
    ('centrifugal_pump', 'F_BEARING_NON_DRIVE_END', 'Non-Drive End Bearing Failure'),
    ('centrifugal_pump', 'F_BEARING_OVERTEMP', 'Bearing Overtemperature - Excessive friction or cooling failure'),
    ('centrifugal_pump', 'F_HIGH_VIBRATION', 'High Vibration Trip - Mechanical instability'),
    ('centrifugal_pump', 'F_CAVITATION', 'Severe Cavitation - NPSH margin critical'),
    ('centrifugal_pump', 'F_MOTOR_OVERLOAD', 'Motor Overload - Excessive current draw')
ON CONFLICT (equipment_type, mode_code) DO NOTHING;
