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
    name: str
    serial_number: Optional[str] = None
    location: Optional[str] = None
    installed_date: Optional[date] = None
    design_speed_rpm: Optional[int] = None
    ambient_temp_celsius: float = 25.0
    ambient_pressure_kpa: float = 101.3
    initial_health_hgp: float = Field(0.92, ge=0.0, le=1.0)
    initial_health_blade: float = Field(0.95, ge=0.0, le=1.0)
    initial_health_bearing: float = Field(0.90, ge=0.0, le=1.0)
    initial_health_fuel: float = Field(0.93, ge=0.0, le=1.0)


class GasTurbineUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    status: Optional[EquipmentStatus] = None


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
    initial_health_blade: Optional[float] = None
    initial_health_bearing: Optional[float] = None
    initial_health_fuel: Optional[float] = None
    status: Optional[str] = None


# Compressor
class CompressorCreate(BaseModel):
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

class CompressorUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    status: Optional[EquipmentStatus] = None

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
    status: Optional[str] = None



# Pump
class PumpCreate(BaseModel):
    name: str
    serial_number: Optional[str] = None
    service_type: Optional[str] = None
    location: Optional[str] = None
    installed_date: Optional[date] = None
    design_flow_m3h: Optional[float] = None
    design_head_m: Optional[float] = None
    design_speed_rpm: int = 3000
    fluid_density_kg_m3: float = 850.0
    npsh_available_m: float = 8.0
    initial_health_impeller: float = Field(0.94, ge=0.0, le=1.0)
    initial_health_seal: float = Field(0.93, ge=0.0, le=1.0)
    initial_health_bearing_de: float = Field(0.90, ge=0.0, le=1.0)
    initial_health_bearing_nde: float = Field(0.92, ge=0.0, le=1.0)


class PumpUpdate(BaseModel):
    name: Optional[str] = None
    service_type: Optional[str] = None
    location: Optional[str] = None
    status: Optional[EquipmentStatus] = None

class PumpResponse(BaseModel):
    pump_id: int
    name: Optional[str] = None
    serial_number: Optional[str] = None
    service_type: Optional[str] = None
    location: Optional[str] = None
    installed_date: Optional[date] = None
    design_flow_m3h: Optional[float] = None
    design_head_m: Optional[float] = None
    design_speed_rpm: Optional[int] = None
    fluid_density_kg_m3: Optional[float] = None
    npsh_available_m: Optional[float] = None
    initial_health_impeller: Optional[float] = None
    initial_health_seal: Optional[float] = None
    initial_health_bearing_de: Optional[float] = None
    initial_health_bearing_nde: Optional[float] = None
    status: Optional[str] = None

# Seeding
class SeedRequest(BaseModel):
    turbine_count: int = Field(10, ge=0, le=1000)
    compressor_count: int = Field(10, ge=0, le=1000)
    pump_count: int = Field(50, ge=0, le=5000)

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