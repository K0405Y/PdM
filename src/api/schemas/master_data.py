"""
Pydantic schemas for master data endpoints.
Equipment create/update/response models, seeding, and failure mode metadata.
"""
from typing import List, Optional
from datetime import date
from enum import Enum
from pydantic import BaseModel, Field

# Enums
class EquipmentType(str, Enum):
    turbine = "turbine"
    compressor = "compressor"
    pump = "pump"

class EquipmentStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    maintenance = "maintenance"

class PumpServiceType(str, Enum):
    crude_booster = "Crude Booster"
    seawater_injection = "Seawater Injection"
    process_water = "Process Water"
    methanol_pump = "Methanol Pump"
    fire_water = "Fire Water"

# Gas Turbine
class GasTurbineCreate(BaseModel):
    """Create a new gas turbine."""
    name: str
    serial_number: Optional[str] = None
    location: Optional[str] = None
    installed_date: Optional[date] = None
    design_speed_rpm: Optional[int] = None
    ambient_temp_celsius: float = 25.0
    ambient_pressure_kpa: float = 101.3
    initial_health_hgp: float = Field(0.92, ge=0.0, le=1.0)
    initial_health_blade_compressor: float = Field(0.95, ge=0.0, le=1.0)
    initial_health_blade_turbine: float = Field(0.95, ge=0.0, le=1.0)
    initial_health_bearing: float = Field(0.90, ge=0.0, le=1.0)
    initial_health_fuel: float = Field(0.93, ge=0.0, le=1.0)
    initial_health_compressor_fouling: float = Field(0.98, ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "GE LM2500 Unit A",
                "serial_number": "GT-2024-001",
                "location": "Platform Alpha",
                "installed_date": "2023-06-15",
                "design_speed_rpm": 3600,
                "ambient_temp_celsius": 30.0,
                "ambient_pressure_kpa": 101.3,
                "initial_health_hgp": 0.92,
                "initial_health_blade_compressor": 0.95,
                "initial_health_blade_turbine": 0.95,
                "initial_health_bearing": 0.90,
                "initial_health_fuel": 0.93,
                "initial_health_compressor_fouling": 0.98,
            }]
        }
    }


class GasTurbineUpdate(BaseModel):
    """Partial update for a gas turbine. Supply only the fields to change."""
    name: Optional[str] = None
    location: Optional[str] = None
    status: Optional[EquipmentStatus] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "location": "Platform Bravo",
                "status": "maintenance",
            }]
        }
    }


class GasTurbineResponse(BaseModel):
    turbine_id: int
    name: Optional[str] = None
    serial_number: Optional[str] = None
    location: Optional[str] = None
    installed_date: Optional[date] = None
    design_speed_rpm: Optional[int] = None
    ambient_temp_celsius: Optional[float] = None
    ambient_pressure_kpa: Optional[float] = None
    initial_health_hgp: Optional[float] = None
    initial_health_blade_compressor: Optional[float] = None
    initial_health_blade_turbine: Optional[float] = None
    initial_health_bearing: Optional[float] = None
    initial_health_fuel: Optional[float] = None
    initial_health_compressor_fouling: Optional[float] = None
    status: Optional[EquipmentStatus] = None


# Compressor
class CompressorCreate(BaseModel):
    """Create a new compressor."""
    name: str
    serial_number: Optional[str] = None
    location: Optional[str] = None
    installed_date: Optional[date] = None
    design_flow_m3h: Optional[float] = None
    design_head_kj_kg: Optional[float] = None
    suction_pressure_kpa: float = 2000.0
    suction_temp_celsius: float = 35.0
    initial_health_impeller: float = Field(0.92, ge=0.0, le=1.0)
    initial_health_bearing: float = Field(0.88, ge=0.0, le=1.0)
    initial_health_seal_primary: float = Field(0.95, ge=0.0, le=1.0)
    initial_health_seal_secondary: float = Field(0.98, ge=0.0, le=1.0)
    initial_health_bearing_thrust: float = Field(0.90, ge=0.0, le=1.0)
    initial_health_rotor_crack: float = Field(0.98, ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "Export Gas Compressor B",
                "serial_number": "CMP-2024-010",
                "location": "Module 3",
                "design_flow_m3h": 5000.0,
                "design_head_kj_kg": 120.0,
                "suction_pressure_kpa": 2500.0,
                "suction_temp_celsius": 38.0,
                "initial_health_impeller": 0.92,
                "initial_health_bearing": 0.88,
                "initial_health_seal_primary": 0.95,
                "initial_health_seal_secondary": 0.98,
                "initial_health_bearing_thrust": 0.90,
                "initial_health_rotor_crack": 0.98,
            }]
        }
    }

class CompressorUpdate(BaseModel):
    """Partial update for a compressor. Supply only the fields to change."""
    name: Optional[str] = None
    location: Optional[str] = None
    status: Optional[EquipmentStatus] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "Export Gas Compressor B (Overhauled)",
                "status": "active",
            }]
        }
    }

class CompressorResponse(BaseModel):
    compressor_id: int
    name: Optional[str] = None
    serial_number: Optional[str] = None
    location: Optional[str] = None
    installed_date: Optional[date] = None
    design_flow_m3h: Optional[float] = None
    design_head_kj_kg: Optional[float] = None
    suction_pressure_kpa: Optional[float] = None
    suction_temp_celsius: Optional[float] = None
    initial_health_impeller: Optional[float] = None
    initial_health_bearing: Optional[float] = None
    initial_health_seal_primary: Optional[float] = None
    initial_health_seal_secondary: Optional[float] = None
    initial_health_bearing_thrust: Optional[float] = None
    initial_health_rotor_crack: Optional[float] = None
    status: Optional[EquipmentStatus] = None



# Pump
class PumpCreate(BaseModel):
    """Create a new pump."""
    name: str
    serial_number: Optional[str] = None
    service_type: Optional[PumpServiceType] = None
    location: Optional[str] = None
    installed_date: Optional[date] = None
    design_flow_m3h: Optional[float] = None
    design_head_m: Optional[float] = None
    design_speed_rpm: int = 3000
    fluid_density_kg_m3: float = 850.0
    npsh_available_m: float = 8.0
    initial_health_impeller: float = Field(0.94, ge=0.0, le=1.0)
    initial_health_wear_ring: float = Field(0.95, ge=0.0, le=1.0)
    initial_health_seal: float = Field(0.93, ge=0.0, le=1.0)
    initial_health_bearing_de: float = Field(0.90, ge=0.0, le=1.0)
    initial_health_bearing_nde: float = Field(0.92, ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "Crude Booster Pump P-101A",
                "serial_number": "PMP-2024-050",
                "service_type": "Crude Booster",
                "location": "Wellhead Platform",
                "design_flow_m3h": 350.0,
                "design_head_m": 180.0,
                "design_speed_rpm": 2950,
                "fluid_density_kg_m3": 870.0,
                "npsh_available_m": 6.5,
                "initial_health_impeller": 0.94,
                "initial_health_wear_ring": 0.95,
                "initial_health_seal": 0.93,
                "initial_health_bearing_de": 0.90,
                "initial_health_bearing_nde": 0.92,
            }]
        }
    }


class PumpUpdate(BaseModel):
    """Partial update for a pump. Supply only the fields to change."""
    name: Optional[str] = None
    service_type: Optional[PumpServiceType] = None
    location: Optional[str] = None
    status: Optional[EquipmentStatus] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "service_type": "Seawater Injection",
                "location": "Module 5",
            }]
        }
    }

class PumpResponse(BaseModel):
    pump_id: int
    name: Optional[str] = None
    serial_number: Optional[str] = None
    service_type: Optional[PumpServiceType] = None
    location: Optional[str] = None
    installed_date: Optional[date] = None
    design_flow_m3h: Optional[float] = None
    design_head_m: Optional[float] = None
    design_speed_rpm: Optional[int] = None
    fluid_density_kg_m3: Optional[float] = None
    npsh_available_m: Optional[float] = None
    initial_health_impeller: Optional[float] = None
    initial_health_wear_ring: Optional[float] = None
    initial_health_seal: Optional[float] = None
    initial_health_bearing_de: Optional[float] = None
    initial_health_bearing_nde: Optional[float] = None
    status: Optional[EquipmentStatus] = None

# Seeding
class SeedRequest(BaseModel):
    """Batch-seed equipment master data with random realistic defaults."""
    turbine_count: int = Field(10, ge=0, le=1000)
    compressor_count: int = Field(10, ge=0, le=1000)
    pump_count: int = Field(50, ge=0, le=5000)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "turbine_count": 5,
                "compressor_count": 3,
                "pump_count": 20,
            }]
        }
    }

class SeedResponse(BaseModel):
    turbine_ids: List[int]
    compressor_ids: List[int]
    pump_ids: List[int]
    message: str

# Failure modes (enriched with ML metadata)
class FailureModeDetail(BaseModel):
    mode_code: str
    equipment_type: str
    description: str
    failure_threshold: float
    severity: str  # safety_critical, availability, performance
    primary_indicators: List[str]
    lagging_indicators: List[str]
    typical_lead_time_hours: str
