from fastapi import FastAPI, HTTPException
import time
import uuid
import os
from typing import Optional
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

def convert_entity_to_text(entity: Entity) -> str:
    parts = []
    parts.append(f"Entity Type: {entity.entity_name}")
    parts.append(f"Entity GUID: {entity.entity_guid}")
    if entity.personal_info:
        parts.append(f"Patient Name: {entity.personal_info.name}, Age: {entity.personal_info.age}")
    
    for t in entity.treatments:
        treatment_str = f"Treatment: {t.treatment_name} from {t.start_date} to {t.end_date}."
        drug_strs = [f"{d.drug_name} ({d.dose})" for d in t.drugs]
        if drug_strs:
            treatment_str += f" Drugs: {', '.join(drug_strs)}."
        parts.append(treatment_str)
    
    return "\\n".join(parts)


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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
