-- Master Data Schema - Equipment Registry and Specifications
CREATE TABLE IF NOT EXISTS master_data.equipment_types (
    type_id SERIAL PRIMARY KEY,
    type_name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS master_data.gas_turbines (
    turbine_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    serial_number VARCHAR(100),
    location VARCHAR(200),
    installed_date DATE,
    design_speed_rpm INT,
    ambient_temp_celsius FLOAT DEFAULT 25.0,
    ambient_pressure_kpa FLOAT DEFAULT 101.3,
    initial_health_hgp FLOAT DEFAULT 0.92,
    initial_health_blade FLOAT DEFAULT 0.95,
    initial_health_bearing FLOAT DEFAULT 0.90,
    initial_health_fuel FLOAT DEFAULT 0.93,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master_data.compressors (
    compressor_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    serial_number VARCHAR(100),
    location VARCHAR(200),
    installed_date DATE,
    design_flow_m3h FLOAT,
    design_head_kj_kg FLOAT,
    suction_pressure_kpa FLOAT DEFAULT 2000.0,
    suction_temp_celsius FLOAT DEFAULT 35.0,
    initial_health_impeller FLOAT DEFAULT 0.92,
    initial_health_bearing FLOAT DEFAULT 0.88,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master_data.pumps (
    pump_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    serial_number VARCHAR(100),
    service_type VARCHAR(100),
    location VARCHAR(200),
    installed_date DATE,
    design_flow_m3h FLOAT,
    design_head_m FLOAT,
    design_speed_rpm INT DEFAULT 3000,
    fluid_density_kg_m3 FLOAT DEFAULT 850.0,
    npsh_available_m FLOAT DEFAULT 8.0,
    initial_health_impeller FLOAT DEFAULT 0.94,
    initial_health_seal FLOAT DEFAULT 0.93,
    initial_health_bearing_de FLOAT DEFAULT 0.90,
    initial_health_bearing_nde FLOAT DEFAULT 0.92,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes on master tables
CREATE INDEX IF NOT EXISTS idx_gas_turbines_status ON master_data.gas_turbines(status);
CREATE INDEX IF NOT EXISTS idx_compressors_status ON master_data.compressors(status);
CREATE INDEX IF NOT EXISTS idx_pumps_status ON master_data.pumps(status);
CREATE INDEX IF NOT EXISTS idx_pumps_service_type ON master_data.pumps(service_type);
