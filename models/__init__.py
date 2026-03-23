# Expose common API and Entity models for easy importing
from .api.chat import (
    ChatMessage, 
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatChoice, 
    ChatCompletionResponseUsage
)

from .entities.patient import (
    Entity,
    Treatment,
    Drug,
    PersonalInfo
)
