from typing import List, Optional

from pydantic import BaseModel


class Drug(BaseModel):
    drug_name: str
    dose: str


class Treatment(BaseModel):
    treatment_name: str
    start_date: str
    end_date: Optional[str] = None
    drugs: List[Drug] = []


class PersonalInfo(BaseModel):
    age: Optional[int] = None
    name: Optional[str] = None


class Entity(BaseModel):
    """Base entity with identity fields shared by all entity types."""
    entity_name: str
    entity_guid: str


class Patient(Entity):
    """Patient entity with clinical data: personal information and treatments."""
    personal_info: Optional[PersonalInfo] = None
    treatments: List[Treatment] = []
