from pydantic import BaseModel
from typing import List, Optional, Any

class Drug(BaseModel):
    drug_name: str
    dose: str

class Treatment(BaseModel):
    treatment_name: str
    start_date: str
    end_date: Any = None
    drugs: List[Drug] = []

class PersonalInfo(BaseModel):
    age: Optional[int] = None
    name: Optional[str] = None

class Entity(BaseModel):
    """
    Representa una entidad estandarizada. 
    A futuro se pueden crear otras entidades heredando de una base.
    """
    entity_name: str
    entity_guid: str
    personal_info: Optional[PersonalInfo] = None
    treatments: List[Treatment] = []
