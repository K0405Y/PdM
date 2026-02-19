-- Maintenance Events Schema - Log all corrective maintenance actions
CREATE TABLE IF NOT EXISTS maintenance_events.gas_turbine_maintenance (
    maintenance_id BIGSERIAL PRIMARY KEY,
    turbine_id INT NOT NULL REFERENCES master_data.gas_turbines(turbine_id) ON DELETE CASCADE,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    failure_code VARCHAR(50),
    downtime_hours FLOAT,
    repaired_components JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS maintenance_events.compressor_maintenance (
    maintenance_id BIGSERIAL PRIMARY KEY,
    compressor_id INT NOT NULL REFERENCES master_data.compressors(compressor_id) ON DELETE CASCADE,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    failure_code VARCHAR(50),
    downtime_hours FLOAT,
    repaired_components JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS maintenance_events.pump_maintenance (
    maintenance_id BIGSERIAL PRIMARY KEY,
    pump_id INT NOT NULL REFERENCES master_data.pumps(pump_id) ON DELETE CASCADE,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    failure_code VARCHAR(50),
    downtime_hours FLOAT,
    repaired_components JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_gt_maint_turbine ON maintenance_events.gas_turbine_maintenance(turbine_id);
CREATE INDEX IF NOT EXISTS idx_gt_maint_time ON maintenance_events.gas_turbine_maintenance(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_cc_maint_compressor ON maintenance_events.compressor_maintenance(compressor_id);
CREATE INDEX IF NOT EXISTS idx_cc_maint_time ON maintenance_events.compressor_maintenance(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_cp_maint_pump ON maintenance_events.pump_maintenance(pump_id);
CREATE INDEX IF NOT EXISTS idx_cp_maint_time ON maintenance_events.pump_maintenance(start_time DESC);
