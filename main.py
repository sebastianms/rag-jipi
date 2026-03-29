from fastapi import FastAPI, HTTPException
import time
import uuid
import os
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()

from models.api.chat import (
    ChatCompletionRequest, ChatCompletionResponse, 
    ChatChoice, ChatMessage, ChatCompletionResponseUsage
)
from models.entities.patient import Entity

from database import vector_store
from litellm import completion
from langchain_core.documents import Document

app = FastAPI(title="Agnostic RAG Engine")

def dict_to_text(data: dict, indent_level: int = 0) -> str:
    """Recursively converts a dictionary to a clear, readable text format."""
    lines = []
    indent = "  " * indent_level
    for key, value in data.items():
        if value is None or value == "":
            continue
        
        clean_key = str(key).replace("_", " ").title()
        
        if isinstance(value, dict):
            lines.append(f"{indent}{clean_key}:")
            lines.append(dict_to_text(value, indent_level + 1))
        elif isinstance(value, list):
            if not value:
                continue
            lines.append(f"{indent}{clean_key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{indent}  -")
                    nested_text = dict_to_text(item, indent_level + 2)
                    lines.append(nested_text)
                else:
                    lines.append(f"{indent}  - {item}")
        else:
            lines.append(f"{indent}{clean_key}: {value}")
    
    return "\\n".join(lines)

def convert_entity_to_text(entity: Any) -> str:
    """
    Converts any Pydantic model or dictionary to a text representation 
    suitable for RAG context ingestion.
    """
    if hasattr(entity, "model_dump"):
        data = entity.model_dump()
    elif hasattr(entity, "dict"):
        data = entity.dict()
    elif isinstance(entity, dict):
        data = entity
    else:
        return str(entity)
            
    return dict_to_text(data)


@app.post("/v1/entities")
async def ingest_entity(entity: Entity):
    """
    Ingests a standard entity into the vector store.
    """
    try:
        text_content = convert_entity_to_text(entity)
        doc = Document(
            page_content=text_content,
            metadata=entity.model_dump()
        )
        vector_store.add_documents([doc])
        return {"status": "success", "message": f"Entity {entity.entity_guid} ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint that does RAG and routes to any LLM using LiteLLM.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    user_message = req.messages[-1].content
    
    # Retrieve context
    docs = vector_store.similarity_search(user_message, k=3)
    
    context = ""
    if docs:
        context = "Relevant context from the database:\\n"
        for doc in docs:
            context += f"---\\n{doc.page_content}\\n"
        context += "---\\n\\n"
    
    # Prepend context to last message
    augmented_messages = req.messages.copy()
    augmented_messages[-1] = ChatMessage(
        role="user", 
        content=f"{context}User Query: {user_message}"
    )

    try:
        litellm_messages = [{"role": m.role, "content": m.content} for m in augmented_messages]
        
        # Use litellm to call the target model (OpenAI, Anthropic, Ollama, etc.)
        response = completion(
            model=req.model,
            messages=litellm_messages,
            temperature=req.temperature,
            stream=req.stream
        )
        
        if req.stream:
            raise HTTPException(status_code=400, detail="Streaming is not implemented.")
            
        # Parse usage explicitly, litellm returns its own structured dict
        usage = getattr(response, "usage", None)
        resp_usage = ChatCompletionResponseUsage()
        if usage:
            resp_usage.prompt_tokens = getattr(usage, "prompt_tokens", 0)
            resp_usage.completion_tokens = getattr(usage, "completion_tokens", 0)
            resp_usage.total_tokens = getattr(usage, "total_tokens", 0)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response.choices[0].message.content
                    ),
                    finish_reason=response.choices[0].finish_reason or "stop"
                )
            ],
            usage=resp_usage
        )

    try:
        text_content = convert_entity_to_text(entity)
        doc = Document(
            page_content=text_content,
            metadata=entity.model_dump()
        )
        vector_store.add_documents([doc])
        return {"status": "success", "message": f"Entity {entity.entity_guid} ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
