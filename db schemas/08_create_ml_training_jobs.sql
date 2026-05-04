-- Active: 1767633787465@@127.0.0.1@5432@pdm

-- ML Training Jobs — API-layer queue + audit/correlation log.
--
-- Scope boundary:
--   This table tracks API-observable job state, idempotency, HTTP-caller
--   attribution, and cache-key linkage. MLflow remains the source of truth
--   for params/metrics/artifacts; this table only stores a denormalized
--   "headline" metrics snapshot to keep the listing endpoint fast.
--
-- Audit policy: rows are never deleted. Use the `archived` flag to hide
-- finished jobs from default listings while preserving history.

CREATE SCHEMA IF NOT EXISTS ml_jobs;

-- Lifecycle states the API exposes to callers. Distinct from MLflow's run
-- states because:
--   - `queued` exists before any MLflow run is started.
--   - `failed` may fire pre-MLflow (DB unavailable, validation, etc.) where
--     no MLflow run will ever be created.
--   - `cancelled` is API-driven; MLflow has no equivalent.
DO $body$
BEGIN
    CREATE TYPE ml_jobs.job_status AS ENUM (
        'queued',
        'running',
        'succeeded',
        'failed',
        'cancelled'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$body$;

DO $body$
BEGIN
    CREATE TYPE ml_jobs.job_task AS ENUM (
        'classification',
        'regression',
        'features_precompute'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$body$;


CREATE TABLE IF NOT EXISTS ml_jobs.training_jobs (
    -- Surrogate key. UUID rather than serial so the API can mint one
    -- before the row is inserted (useful for idempotent re-submits and
    -- for echoing a Location header on 202 Accepted before commit).
    job_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    --  What the caller asked for 
    task                ml_jobs.job_task NOT NULL,
    equipment_type      VARCHAR(32) NOT NULL,           -- turbine|compressor|pump
    config              JSONB NOT NULL,                 -- full request body (PipelineConfig + flags)
    --  Lifecycle state 
    status              ml_jobs.job_status NOT NULL DEFAULT 'queued',
    error               TEXT,                           -- exception message + abbreviated trace on failure
    progress_message    TEXT,                           -- last human-readable phase (e.g. "engineering features")
    --  Timing (API-observable, independent of MLflow) 
    submitted_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at          TIMESTAMPTZ,                    -- worker pickup
    finished_at         TIMESTAMPTZ,                    -- terminal transition
    --  HTTP-layer attribution (no auth yet, but capture what we have) 
    submitted_by        VARCHAR(128),                   -- X-User-Id header; nullable until auth lands
    request_id          UUID,                           -- X-Request-Id (echoed in responses for tracing)
    idempotency_key     VARCHAR(128),                   -- Idempotency-Key header
    client_ip           INET,                           -- best-effort; behind proxy use X-Forwarded-For
    user_agent          TEXT,
    --  Cross-system pointers 
    -- Cache-key links to the on-disk engineered-features cache produced
    -- by train.py. Multiple jobs share a cache_key when configs match.
    cache_key           CHAR(16),
    -- mlflow_run_id is null until the worker calls mlflow.start_run().
    -- Stored as TEXT (MLflow uses 32-char hex) to avoid coupling to a
    -- specific MLflow internal type.
    mlflow_run_id       TEXT,
    mlflow_experiment_id TEXT,
    --  Denormalized snapshot of headline metrics (NOT source of truth) 
    -- Populated on terminal success; query MLflow for full params/metrics/artifacts.
    -- Shape:
    --   classification: {"accuracy": 0.94, "macro_f1": 0.81, "roc_auc_macro": 0.96, ...}
    --   regression:     {"health_hgp": {"r2": 0.82, "mae": 0.04}, "health_bearing": {...}, ...}
    metrics             JSONB,
    --  Audit / housekeeping 
    archived            BOOLEAN NOT NULL DEFAULT FALSE, -- soft-hide; rows are never deleted
    schema_version      SMALLINT NOT NULL DEFAULT 1,    -- bump when row shape evolves
    --  Constraints 
    CONSTRAINT training_jobs_terminal_has_finished CHECK (
        (status IN ('succeeded','failed','cancelled')) = (finished_at IS NOT NULL)
    ),
    CONSTRAINT training_jobs_running_has_started CHECK (
        status = 'queued' OR started_at IS NOT NULL
    ),
    CONSTRAINT training_jobs_equipment_known CHECK (
        equipment_type IN ('turbine','compressor','pump')
    )
);

-- Idempotency: a given (idempotency_key, submitted_by) pair must resolve to
-- exactly one job. Partial unique index allows nulls (callers who don't send
-- the header).
CREATE UNIQUE INDEX IF NOT EXISTS training_jobs_idem_uniq
    ON ml_jobs.training_jobs (idempotency_key, submitted_by)
    WHERE idempotency_key IS NOT NULL;

-- Hot path for `GET /ml/jobs?status=running` and worker pickup polling.
CREATE INDEX IF NOT EXISTS training_jobs_status_submitted_idx
    ON ml_jobs.training_jobs (status, submitted_at DESC)
    WHERE archived = FALSE;

-- Fast lookup from MLflow side ("what API job spawned this run?")
CREATE INDEX IF NOT EXISTS training_jobs_mlflow_run_idx
    ON ml_jobs.training_jobs (mlflow_run_id)
    WHERE mlflow_run_id IS NOT NULL;

-- Listing by caller.
CREATE INDEX IF NOT EXISTS training_jobs_submitted_by_idx
    ON ml_jobs.training_jobs (submitted_by, submitted_at DESC)
    WHERE submitted_by IS NOT NULL;

-- Cache-key fan-out (which jobs reused this cache?).
CREATE INDEX IF NOT EXISTS training_jobs_cache_key_idx
    ON ml_jobs.training_jobs (cache_key)
    WHERE cache_key IS NOT NULL;

-- Convenience view: active (non-archived) jobs only.
CREATE OR REPLACE VIEW ml_jobs.training_jobs_active AS
    SELECT * FROM ml_jobs.training_jobs WHERE archived = FALSE;

COMMENT ON TABLE ml_jobs.training_jobs IS
    'API-layer queue + audit log for ML training jobs. MLflow is the source of truth for params/metrics/artifacts; this table links via mlflow_run_id and stores only a denormalized headline metrics snapshot.';
COMMENT ON COLUMN ml_jobs.training_jobs.metrics IS
    'Denormalized headline metrics snapshot. NOT source of truth — query MLflow with mlflow_run_id for full data.';
COMMENT ON COLUMN ml_jobs.training_jobs.config IS
    'Full request body (PipelineConfig + CLI-equivalent flags). Lets the worker replay or audit the exact request.';
COMMENT ON COLUMN ml_jobs.training_jobs.idempotency_key IS
    'Per-caller idempotency key (Idempotency-Key header). Unique together with submitted_by.';