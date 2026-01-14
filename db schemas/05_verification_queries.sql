-- Verification and Summary Queries for PdM Data Integrity


-- MASTER DATA VERIFICATION

-- Summary of equipment count by type
SELECT 
    'Gas Turbines' as equipment_type,
    COUNT(*) as total_count,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_count
FROM master_data.gas_turbines
UNION ALL
SELECT 
    'Centrifugal Compressors' as equipment_type,
    COUNT(*) as total_count,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_count
FROM master_data.centrifugal_compressors
UNION ALL
SELECT 
    'Centrifugal Pumps' as equipment_type,
    COUNT(*) as total_count,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_count
FROM master_data.centrifugal_pumps
ORDER BY equipment_type;

-- Gas turbines with health statistics
SELECT 
    name,
    serial_number,
    location,
    installed_date,
    ROUND(initial_health_hgp::numeric, 3) as hgp_health,
    ROUND(initial_health_blade::numeric, 3) as blade_health,
    ROUND(initial_health_bearing::numeric, 3) as bearing_health,
    ROUND(initial_health_fuel::numeric, 3) as fuel_health,
    status,
    created_at
FROM master_data.gas_turbines
ORDER BY turbine_id
LIMIT 10;

-- Centrifugal compressors with design specifications
SELECT 
    name,
    serial_number,
    location,
    design_flow_m3h,
    design_head_kj_kg,
    ROUND(initial_health_impeller::numeric, 3) as impeller_health,
    ROUND(initial_health_bearing::numeric, 3) as bearing_health,
    status
FROM master_data.centrifugal_compressors
ORDER BY compressor_id
LIMIT 10;

-- Centrifugal pumps by service type
SELECT 
    service_type,
    COUNT(*) as pump_count,
    ROUND(AVG(design_flow_m3h)::numeric, 1) as avg_design_flow,
    ROUND(AVG(design_head_m)::numeric, 1) as avg_design_head,
    MIN(initial_health_impeller) as min_impeller_health,
    MAX(initial_health_impeller) as max_impeller_health
FROM master_data.centrifugal_pumps
GROUP BY service_type
ORDER BY pump_count DESC;

-- TELEMETRY DATA VERIFICATION

-- Telemetry record counts by equipment type
SELECT 
    'Gas Turbine' as equipment_type,
    COUNT(*) as total_records,
    COUNT(DISTINCT turbine_id) as unique_equipment,
    MIN(sample_time) as earliest_sample,
    MAX(sample_time) as latest_sample
FROM telemetry.gas_turbine_telemetry
UNION ALL
SELECT 
    'Centrifugal Compressor' as equipment_type,
    COUNT(*) as total_records,
    COUNT(DISTINCT compressor_id) as unique_equipment,
    MIN(sample_time) as earliest_sample,
    MAX(sample_time) as latest_sample
FROM telemetry.centrifugal_compressor_telemetry
UNION ALL
SELECT 
    'Centrifugal Pump' as equipment_type,
    COUNT(*) as total_records,
    COUNT(DISTINCT pump_id) as unique_equipment,
    MIN(sample_time) as earliest_sample,
    MAX(sample_time) as latest_sample
FROM telemetry.centrifugal_pump_telemetry
ORDER BY equipment_type;

-- Gas turbine telemetry coverage by machine
SELECT 
    gt.turbine_id,
    gt.name,
    COUNT(gtt.telemetry_id) as record_count,
    MIN(gtt.sample_time) as first_sample,
    MAX(gtt.sample_time) as last_sample,
    ROUND((EXTRACT(EPOCH FROM (MAX(gtt.sample_time) - MIN(gtt.sample_time))) / 3600)::numeric, 1) as hours_of_data,
    ROUND(AVG(gtt.operating_hours)::numeric, 1) as avg_operating_hours,
    ROUND(AVG(gtt.speed_rpm)::numeric, 1) as avg_speed_rpm,
    ROUND(AVG(gtt.egt_celsius)::numeric, 1) as avg_egt
FROM master_data.gas_turbines gt
LEFT JOIN telemetry.gas_turbine_telemetry gtt ON gt.turbine_id = gtt.turbine_id
GROUP BY gt.turbine_id, gt.name
ORDER BY gt.turbine_id;

-- Compressor telemetry coverage by machine
SELECT 
    cc.compressor_id,
    cc.name,
    COUNT(cct.telemetry_id) as record_count,
    MIN(cct.sample_time) as first_sample,
    MAX(cct.sample_time) as last_sample,
    ROUND(AVG(cct.operating_hours)::numeric, 1) as avg_operating_hours,
    ROUND(AVG(cct.speed_rpm)::numeric, 1) as avg_speed_rpm,
    ROUND(AVG(cct.flow_m3h)::numeric, 1) as avg_flow
FROM master_data.centrifugal_compressors cc
LEFT JOIN telemetry.centrifugal_compressor_telemetry cct ON cc.compressor_id = cct.compressor_id
GROUP BY cc.compressor_id, cc.name
ORDER BY cc.compressor_id;

-- Pump telemetry sampling rate verification (should be ~10-minute intervals)
WITH pump_intervals AS (
    SELECT 
        pump_id,
        LAG(sample_time) OVER (PARTITION BY pump_id ORDER BY sample_time) as prev_time,
        sample_time,
        EXTRACT(EPOCH FROM (sample_time - LAG(sample_time) OVER (PARTITION BY pump_id ORDER BY sample_time)))/60 as interval_minutes
    FROM telemetry.centrifugal_pump_telemetry
)
SELECT 
    pump_id,
    COUNT(*) as samples_checked,
    ROUND(AVG(interval_minutes)::numeric, 2) as avg_interval_min,
    MIN(interval_minutes)::integer as min_interval_min,
    MAX(interval_minutes)::integer as max_interval_min,
    ROUND((AVG(interval_minutes) - 10)::numeric, 2) as deviation_from_10min
FROM pump_intervals
WHERE interval_minutes IS NOT NULL
GROUP BY pump_id
LIMIT 10;

-- HEALTH DEGRADATION VERIFICATION

-- Gas turbine health degradation over time (sample from first and last 100 records)
WITH ranked_telemetry AS (
    SELECT 
        turbine_id,
        sample_time,
        health_hgp,
        health_blade,
        health_bearing,
        health_fuel,
        ROW_NUMBER() OVER (PARTITION BY turbine_id ORDER BY sample_time) as rn,
        COUNT(*) OVER (PARTITION BY turbine_id) as total_records
    FROM telemetry.gas_turbine_telemetry
)
SELECT 
    turbine_id,
    'Initial' as phase,
    ROUND(AVG(health_hgp)::numeric, 4) as avg_hgp_health,
    ROUND(AVG(health_blade)::numeric, 4) as avg_blade_health,
    ROUND(AVG(health_bearing)::numeric, 4) as avg_bearing_health,
    ROUND(AVG(health_fuel)::numeric, 4) as avg_fuel_health
FROM ranked_telemetry
WHERE rn <= 100
GROUP BY turbine_id
UNION ALL
SELECT 
    turbine_id,
    'Final' as phase,
    ROUND(AVG(health_hgp)::numeric, 4) as avg_hgp_health,
    ROUND(AVG(health_blade)::numeric, 4) as avg_blade_health,
    ROUND(AVG(health_bearing)::numeric, 4) as avg_bearing_health,
    ROUND(AVG(health_fuel)::numeric, 4) as avg_fuel_health
FROM ranked_telemetry
WHERE rn > (total_records - 100)
GROUP BY turbine_id
ORDER BY turbine_id, phase;

-- Pump health degradation over time
WITH ranked_telemetry AS (
    SELECT 
        pump_id,
        sample_time,
        health_impeller,
        health_seal,
        health_bearing_de,
        health_bearing_nde,
        ROW_NUMBER() OVER (PARTITION BY pump_id ORDER BY sample_time) as rn,
        COUNT(*) OVER (PARTITION BY pump_id) as total_records
    FROM telemetry.centrifugal_pump_telemetry
)
SELECT 
    pump_id,
    'Initial' as phase,
    ROUND(AVG(health_impeller)::numeric, 4) as avg_impeller,
    ROUND(AVG(health_seal)::numeric, 4) as avg_seal,
    ROUND(AVG(health_bearing_de)::numeric, 4) as avg_bearing_de,
    ROUND(AVG(health_bearing_nde)::numeric, 4) as avg_bearing_nde
FROM ranked_telemetry
WHERE rn <= 50
GROUP BY pump_id
LIMIT 10;

-- FAILURE EVENT VERIFICATION

-- Summary of failures by equipment type
SELECT 
    'Gas Turbine' as equipment_type,
    COUNT(*) as total_failures,
    COUNT(DISTINCT turbine_id) as equipment_with_failures,
    MIN(failure_time) as earliest_failure,
    MAX(failure_time) as latest_failure
FROM failure_events.gas_turbine_failures
UNION ALL
SELECT 
    'Centrifugal Compressor' as equipment_type,
    COUNT(*) as total_failures,
    COUNT(DISTINCT compressor_id) as equipment_with_failures,
    MIN(failure_time) as earliest_failure,
    MAX(failure_time) as latest_failure
FROM failure_events.centrifugal_compressor_failures
UNION ALL
SELECT 
    'Centrifugal Pump' as equipment_type,
    COUNT(*) as total_failures,
    COUNT(DISTINCT pump_id) as equipment_with_failures,
    MIN(failure_time) as earliest_failure,
    MAX(failure_time) as latest_failure
FROM failure_events.centrifugal_pump_failures
ORDER BY equipment_type;

-- Top failure modes by frequency
SELECT 
    failure_mode_code,
    COUNT(*) as frequency,
    'Gas Turbine' as equipment_type
FROM failure_events.gas_turbine_failures
GROUP BY failure_mode_code
UNION ALL
SELECT 
    failure_mode_code,
    COUNT(*) as frequency,
    'Compressor' as equipment_type
FROM failure_events.centrifugal_compressor_failures
GROUP BY failure_mode_code
UNION ALL
SELECT 
    failure_mode_code,
    COUNT(*) as frequency,
    'Pump' as equipment_type
FROM failure_events.centrifugal_pump_failures
GROUP BY failure_mode_code
ORDER BY equipment_type, frequency DESC;

-- Failure statistics by equipment
SELECT 
    gt.name,
    COUNT(gtf.failure_id) as failure_count,
    AVG(gtf.operating_hours_at_failure)::numeric(10,1) as avg_hours_to_failure,
    STRING_AGG(DISTINCT gtf.failure_mode_code, ', ') as failure_modes
FROM master_data.gas_turbines gt
LEFT JOIN failure_events.gas_turbine_failures gtf ON gt.turbine_id = gtf.turbine_id
GROUP BY gt.turbine_id, gt.name
ORDER BY failure_count DESC, gt.name;

-- DATA QUALITY CHECKS

-- Check for null values in critical telemetry fields
SELECT 
    'gas_turbine_telemetry' as table_name,
    SUM(CASE WHEN turbine_id IS NULL THEN 1 ELSE 0 END) as null_turbine_id,
    SUM(CASE WHEN sample_time IS NULL THEN 1 ELSE 0 END) as null_sample_time,
    SUM(CASE WHEN operating_hours IS NULL THEN 1 ELSE 0 END) as null_operating_hours,
    SUM(CASE WHEN speed_rpm IS NULL THEN 1 ELSE 0 END) as null_speed_rpm
FROM telemetry.gas_turbine_telemetry
UNION ALL
SELECT 
    'centrifugal_compressor_telemetry' as table_name,
    SUM(CASE WHEN compressor_id IS NULL THEN 1 ELSE 0 END) as null_compressor_id,
    SUM(CASE WHEN sample_time IS NULL THEN 1 ELSE 0 END) as null_sample_time,
    SUM(CASE WHEN operating_hours IS NULL THEN 1 ELSE 0 END) as null_operating_hours,
    SUM(CASE WHEN speed_rpm IS NULL THEN 1 ELSE 0 END) as null_speed_rpm
FROM telemetry.centrifugal_compressor_telemetry
UNION ALL
SELECT 
    'centrifugal_pump_telemetry' as table_name,
    SUM(CASE WHEN pump_id IS NULL THEN 1 ELSE 0 END) as null_pump_id,
    SUM(CASE WHEN sample_time IS NULL THEN 1 ELSE 0 END) as null_sample_time,
    SUM(CASE WHEN operating_hours IS NULL THEN 1 ELSE 0 END) as null_operating_hours,
    SUM(CASE WHEN speed_rpm IS NULL THEN 1 ELSE 0 END) as null_speed_rpm
FROM telemetry.centrifugal_pump_telemetry
ORDER BY table_name;

-- Check for orphaned telemetry records (foreign key integrity)
SELECT 
    'Gas Turbine' as equipment_type,
    COUNT(*) as orphaned_records
FROM telemetry.gas_turbine_telemetry
WHERE turbine_id NOT IN (SELECT turbine_id FROM master_data.gas_turbines)
UNION ALL
SELECT 
    'Centrifugal Compressor' as equipment_type,
    COUNT(*) as orphaned_records
FROM telemetry.centrifugal_compressor_telemetry
WHERE compressor_id NOT IN (SELECT compressor_id FROM master_data.centrifugal_compressors)
UNION ALL
SELECT 
    'Centrifugal Pump' as equipment_type,
    COUNT(*) as orphaned_records
FROM telemetry.centrifugal_pump_telemetry
WHERE pump_id NOT IN (SELECT pump_id FROM master_data.centrifugal_pumps)
ORDER BY equipment_type;

-- DATASET STATISTICS SUMMARY

-- Comprehensive data statistics
WITH totals AS (
    SELECT 
        (SELECT COUNT(*) FROM master_data.gas_turbines WHERE status = 'active') as active_turbines,
        (SELECT COUNT(*) FROM master_data.centrifugal_compressors WHERE status = 'active') as active_compressors,
        (SELECT COUNT(*) FROM master_data.centrifugal_pumps WHERE status = 'active') as active_pumps,
        (SELECT COUNT(*) FROM telemetry.gas_turbine_telemetry) as gt_telemetry,
        (SELECT COUNT(*) FROM telemetry.centrifugal_compressor_telemetry) as cc_telemetry,
        (SELECT COUNT(*) FROM telemetry.centrifugal_pump_telemetry) as cp_telemetry,
        (SELECT COUNT(*) FROM failure_events.gas_turbine_failures) as gt_failures,
        (SELECT COUNT(*) FROM failure_events.centrifugal_compressor_failures) as cc_failures,
        (SELECT COUNT(*) FROM failure_events.centrifugal_pump_failures) as cp_failures
)
SELECT 
    'Equipment Count' as metric,
    (active_turbines + active_compressors + active_pumps)::text as value,
    FORMAT('%d turbines, %d compressors, %d pumps', active_turbines, active_compressors, active_pumps) as details
FROM totals
UNION ALL
SELECT 
    'Total Telemetry Records' as metric,
    (gt_telemetry + cc_telemetry + cp_telemetry)::text as value,
    FORMAT('%d turbine, %d compressor, %d pump', gt_telemetry, cc_telemetry, cp_telemetry) as details
FROM totals
UNION ALL
SELECT 
    'Total Failures Logged' as metric,
    (gt_failures + cc_failures + cp_failures)::text as value,
    FORMAT('%d turbine, %d compressor, %d pump', gt_failures, cc_failures, cp_failures) as details
FROM totals
UNION ALL
SELECT 
    'Avg Records per Equipment' as metric,
    ROUND(((gt_telemetry + cc_telemetry + cp_telemetry)::numeric / 
            (active_turbines + active_compressors + active_pumps)), 0)::text as value,
    '6 months @ 10-min intervals = ~26,280 per equipment' as details
FROM totals;
