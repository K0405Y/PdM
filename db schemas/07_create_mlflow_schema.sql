-- Active: 1767633787465@@127.0.0.1@5432@pdm_mlflow

-- MLflow PostgreSQL Backend Store Schema

CREATE SCHEMA IF NOT EXISTS mlflow;

-- Update search path to include mlflow schema
ALTER ROLE postgres SET search_path TO mlflow, public;
