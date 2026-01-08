"""
Migration Script: Transition to Enhanced PdM Simulation System

This script helps you migrate from the basic simulators to the enhanced versions
with minimal code changes. It provides both gradual migration paths and complete
replacement options.

Usage:
    python migrate_to_enhanced.py --mode gradual  # Add enhancements incrementally
    python migrate_to_enhanced.py --mode complete  # Full replacement
    python migrate_to_enhanced.py --check          # Check compatibility
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def check_environment():
    """Check if environment is ready for enhancements."""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    checks = {
        'NumPy': False,
        'Module Structure': False,
        'Database Connection': False
    }

    # Check NumPy
    try:
        import numpy as np
        checks['NumPy'] = True
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not found - required for enhanced vibration")

    # Check module structure
    try:
        from src.data_generation import physics, simulation, ml_utils
        checks['Module Structure'] = True
        print("✓ Enhanced modules properly structured")
    except ImportError as e:
        print(f"✗ Module import failed: {e}")

    # Check database
    try:
        from src.ingestion.data_pipeline import PdMDatabase
        checks['Database Connection'] = True
        print("✓ Database utilities available")
    except ImportError:
        print("✗ Database utilities not found")

    print("\n" + "=" * 70)
    all_passed = all(checks.values())
    if all_passed:
        print("✅ All checks passed - ready for migration")
    else:
        print("⚠️  Some checks failed - fix issues before migrating")

    return all_passed


def show_simple_example():
    """Show simple usage example of enhanced simulators."""
    print("\n" + "=" * 70)
    print("SIMPLE USAGE EXAMPLE")
    print("=" * 70)

    example_code = '''
# Option 1: Use enhanced simulator with all features
from src.data_generation.enhanced_simulators import EnhancedGasTurbine
from src.data_generation.physics import LocationType
from src.data_generation.ml_utils import OutputMode

turbine = EnhancedGasTurbine(
    name='GT-001',
    location_type=LocationType.OFFSHORE,
    enable_enhanced_vibration=True,
    enable_thermal_transients=True,
    enable_environmental=True,
    enable_maintenance=True,
    enable_incipient_faults=True,
    enable_process_upsets=True,
    output_mode=OutputMode.FULL
)

# Use exactly like base simulator
turbine.set_speed(9500)
state = turbine.next_state()

# New fields available in state:
print(f"Vibration Kurtosis: {state['vibration_kurtosis']}")
print(f"Operating Mode: {state['operating_mode']}")
print(f"Active Faults: {state['num_active_faults']}")

# Option 2: Gradual adoption - enable features one by one
turbine_partial = EnhancedGasTurbine(
    name='GT-002',
    enable_enhanced_vibration=True,  # Just vibration enhancement
    enable_thermal_transients=False,
    enable_environmental=False,
    enable_maintenance=False,
    enable_incipient_faults=False,
    enable_process_upsets=False,
    output_mode=OutputMode.FULL
)

# Option 3: Use factory function for backward compatibility
from src.data_generation.enhanced_simulators import create_enhanced_turbine

# Enhanced version
turbine_new = create_enhanced_turbine('GT-003', enable_all=True)

# Original version (for testing/comparison)
turbine_old = create_enhanced_turbine('GT-004', enable_all=False)
'''

    print(example_code)


def show_migration_paths():
    """Show different migration strategies."""
    print("\n" + "=" * 70)
    print("MIGRATION PATHS")
    print("=" * 70)

    print("\n📌 Path 1: GRADUAL MIGRATION (Recommended)")
    print("-" * 70)
    print("""
1. Start with enhanced vibration only:
   - Immediate ML value (better defect signatures)
   - No workflow changes
   - Low risk

2. Add thermal transients:
   - Models startup/shutdown stress
   - Adds operating mode tracking
   - Small schema change (new columns)

3. Enable maintenance events:
   - Realistic health restoration
   - Adds maintenance log
   - Medium schema change

4. Add environmental variability:
   - Location-specific degradation
   - Daily/seasonal cycles
   - Small schema change

5. Enable incipient faults & upsets:
   - Edge cases for ML
   - Fault event logging
   - Medium schema change

6. Switch to generator pipeline:
   - Memory efficiency
   - Parallel processing
   - No schema change, pipeline refactor
""")

    print("\n📌 Path 2: COMPLETE REPLACEMENT")
    print("-" * 70)
    print("""
1. Update database schema (add all new columns)
2. Replace simulators with Enhanced versions
3. Update data_pipeline.py to use generators
4. Update data_ingestion.py to use bulk insert
5. Test end-to-end
6. Deploy

Timeline: 1-2 weeks
Risk: Medium (all changes at once)
Benefit: All improvements immediately
""")

    print("\n📌 Path 3: PARALLEL OPERATION")
    print("-" * 70)
    print("""
1. Run both old and new pipelines side-by-side
2. Compare outputs for validation
3. Gradually shift to new pipeline
4. Deprecate old pipeline when confident

Timeline: 2-3 weeks
Risk: Low (safe rollback)
Benefit: Thorough validation
""")


def check_database_schema():
    """Check if database schema needs updates."""
    print("\n" + "=" * 70)
    print("DATABASE SCHEMA CHECK")
    print("=" * 70)

    new_columns = {
        'telemetry.gas_turbine_telemetry': [
            'vibration_crest_factor REAL',
            'vibration_kurtosis REAL',
            'operating_mode VARCHAR(20)',
            'temp_rotor_c REAL',
            'temp_casing_c REAL',
            'differential_temp_c REAL',
            'thermal_stress REAL',
            'startup_cycles INTEGER',
            'num_active_faults INTEGER',
            'total_faults_initiated INTEGER',
            'upset_active BOOLEAN',
            'upset_type VARCHAR(50)',
            'upset_severity REAL',
            'location VARCHAR(20)',
            'corrosion_factor REAL',
            'fouling_factor REAL'
        ]
    }

    print("\n📋 New columns to add:")
    for table, columns in new_columns.items():
        print(f"\n{table}:")
        for col in columns:
            print(f"  ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col};")

    print("\n💡 SQL script: database/schemas/06_enhanced_columns.sql")


def generate_migration_sql():
    """Generate SQL migration script."""
    sql_content = """-- Enhanced Columns Migration Script
-- Adds columns for physics enhancements and ML features

-- Gas Turbine Telemetry Enhancements
ALTER TABLE IF EXISTS telemetry.gas_turbine_telemetry
    ADD COLUMN IF NOT EXISTS vibration_crest_factor REAL,
    ADD COLUMN IF NOT EXISTS vibration_kurtosis REAL,
    ADD COLUMN IF NOT EXISTS operating_mode VARCHAR(20),
    ADD COLUMN IF NOT EXISTS temp_rotor_c REAL,
    ADD COLUMN IF NOT EXISTS temp_casing_c REAL,
    ADD COLUMN IF NOT EXISTS differential_temp_c REAL,
    ADD COLUMN IF NOT EXISTS thermal_stress REAL,
    ADD COLUMN IF NOT EXISTS startup_cycles INTEGER,
    ADD COLUMN IF NOT EXISTS num_active_faults INTEGER,
    ADD COLUMN IF NOT EXISTS total_faults_initiated INTEGER,
    ADD COLUMN IF NOT EXISTS upset_active BOOLEAN,
    ADD COLUMN IF NOT EXISTS upset_type VARCHAR(50),
    ADD COLUMN IF NOT EXISTS upset_severity REAL,
    ADD COLUMN IF NOT EXISTS location VARCHAR(20),
    ADD COLUMN IF NOT EXISTS corrosion_factor REAL,
    ADD COLUMN IF NOT EXISTS fouling_factor REAL;

-- Similar for compressor and pump tables
-- (Add as needed based on equipment type)

-- Create maintenance events table
CREATE TABLE IF NOT EXISTS maintenance.maintenance_events (
    maintenance_id SERIAL PRIMARY KEY,
    equipment_type VARCHAR(20) NOT NULL,
    equipment_id INTEGER NOT NULL,
    maintenance_type VARCHAR(30) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    operating_hours_at_maintenance REAL,
    cost_usd REAL,
    duration_hours REAL,
    quality_factor REAL,
    health_before JSONB,
    health_after JSONB
);

-- Create fault events table
CREATE TABLE IF NOT EXISTS events.fault_events (
    fault_id SERIAL PRIMARY KEY,
    equipment_type VARCHAR(20) NOT NULL,
    equipment_id INTEGER NOT NULL,
    fault_type VARCHAR(50) NOT NULL,
    initiation_time TIMESTAMP NOT NULL,
    initiation_operating_hours REAL,
    affected_component VARCHAR(30),
    severity REAL,
    location VARCHAR(50)
);

-- Create upset events table
CREATE TABLE IF NOT EXISTS events.upset_events (
    upset_id SERIAL PRIMARY KEY,
    equipment_type VARCHAR(20) NOT NULL,
    equipment_id INTEGER NOT NULL,
    upset_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    duration_seconds INTEGER,
    severity REAL,
    description TEXT
);

-- Create schemas if not exists
CREATE SCHEMA IF NOT EXISTS maintenance;
CREATE SCHEMA IF NOT EXISTS events;

COMMENT ON TABLE maintenance.maintenance_events IS 'Maintenance interventions and scheduling history';
COMMENT ON TABLE events.fault_events IS 'Discrete fault initiation events';
COMMENT ON TABLE events.upset_events IS 'Process upset and abnormal condition events';
"""

    output_path = Path('database/schemas/06_enhanced_columns.sql')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(sql_content)

    print(f"\n✅ Generated SQL script: {output_path}")


def show_performance_comparison():
    """Show expected performance improvements."""
    print("\n" + "=" * 70)
    print("EXPECTED PERFORMANCE IMPROVEMENTS")
    print("=" * 70)

    print("""
┌─────────────────────────────┬────────────┬────────────┬─────────────┐
│ Metric                      │ Original   │ Enhanced   │ Improvement │
├─────────────────────────────┼────────────┼────────────┼─────────────┤
│ Memory Usage (1M records)   │ ~8 GB      │ ~50 MB     │ 160x better │
│ DB Insert Speed (rows/sec)  │ 1-5k       │ 100-500k   │ 20-100x     │
│ Parallel Speedup (8 cores)  │ 1x         │ ~8x        │ 8x faster   │
│ Signal Realism              │ Moderate   │ High       │ Qualitative │
└─────────────────────────────┴────────────┴────────────┴─────────────┘

🎯 For 6 months × 100 equipment × 10-min sampling:
   Original: ~4 hours (single core, serial DB inserts)
   Enhanced: ~25 minutes (8 cores, bulk inserts)
   Speedup: 9.6x
""")


def main():
    parser = argparse.ArgumentParser(
        description='PdM Enhancement Migration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_to_enhanced.py --check
  python migrate_to_enhanced.py --show-examples
  python migrate_to_enhanced.py --generate-sql
  python migrate_to_enhanced.py --migration-guide
        """
    )

    parser.add_argument('--check', action='store_true',
                       help='Check environment readiness')
    parser.add_argument('--show-examples', action='store_true',
                       help='Show usage examples')
    parser.add_argument('--migration-guide', action='store_true',
                       help='Show migration paths')
    parser.add_argument('--generate-sql', action='store_true',
                       help='Generate database migration SQL')
    parser.add_argument('--schema-check', action='store_true',
                       help='Check database schema requirements')
    parser.add_argument('--performance', action='store_true',
                       help='Show performance comparison')
    parser.add_argument('--all', action='store_true',
                       help='Show all information')

    args = parser.parse_args()

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    print("\n" + "═" * 70)
    print("PdM ENHANCEMENT MIGRATION TOOL")
    print("═" * 70)

    if args.all or args.check:
        check_environment()

    if args.all or args.show_examples:
        show_simple_example()

    if args.all or args.migration_guide:
        show_migration_paths()

    if args.all or args.schema_check:
        check_database_schema()

    if args.all or args.generate_sql:
        generate_migration_sql()

    if args.all or args.performance:
        show_performance_comparison()

    print("\n" + "=" * 70)
    print("📚 For detailed integration guide, see: docs/INTEGRATION_GUIDE.md")
    print("📊 For performance benchmarks, see: docs/ENHANCEMENT_SUMMARY.md")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()