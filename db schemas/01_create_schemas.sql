-- Active: 1767633787465@@127.0.0.1@5432@pdm
CREATE SCHEMA IF NOT EXISTS master_data;
CREATE SCHEMA IF NOT EXISTS telemetry;
CREATE SCHEMA IF NOT EXISTS failure_events;
CREATE SCHEMA IF NOT EXISTS maintenance_events;
CREATE SCHEMA IF NOT EXISTS test_data;
CREATE SCHEMA IF NOT EXISTS ml_jobs;
-- Set search path
ALTER ROLE postgres SET search_path TO master_data, telemetry, failure_events, maintenance_events, test_data, ml_jobs, public;