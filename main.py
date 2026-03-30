import logging
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from litellm import completion

import database
from models.api.chat import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseUsage,
    ChatMessage,
)
from models.entities.patient import Patient
from utils.text import convert_entity_to_text

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.vector_store = database.init_vector_store()
    yield


app = FastAPI(title="Agnostic RAG Engine", lifespan=lifespan)


@app.post("/v1/entities")
async def ingest_entity(entity: Patient):
    """Ingests a patient entity into the vector store."""
    if database.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store is not initialized.")
    try:
        text_content = convert_entity_to_text(entity)
        doc = Document(
            page_content=text_content,
            metadata=entity.model_dump(),
        )
        database.vector_store.add_documents([doc])
        return {"status": "success", "message": f"Entity {entity.entity_guid} ingested successfully."}
    except Exception:
        logger.exception("Failed to ingest entity '%s'", entity.entity_guid)
        raise HTTPException(status_code=500, detail="Failed to ingest entity.")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible endpoint that does RAG and routes to any LLM using LiteLLM."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented.")

    if database.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store is not initialized.")

    user_message = req.messages[-1].content

    docs = database.vector_store.similarity_search(user_message, k=3)

    context = ""
    if docs:
        context = "Relevant context from the database:\n"
        for doc in docs:
            context += f"---\n{doc.page_content}\n"
        context += "---\n\n"

    augmented_messages = req.messages.copy()
    augmented_messages[-1] = ChatMessage(
        role="user",
        content=f"{context}User Query: {user_message}",
    )

    try:
        litellm_messages = [{"role": m.role, "content": m.content} for m in augmented_messages]

        response = completion(
            model=req.model,
            messages=litellm_messages,
            temperature=req.temperature,
            stream=False,
        )

        # Parse usage explicitly — litellm returns its own structured object, not a plain dict
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
                        content=response.choices[0].message.content,
                    ),
                    finish_reason=response.choices[0].finish_reason or "stop",
                )
            ],
            usage=resp_usage,
        )

    except Exception:
        logger.exception("LLM call failed for model '%s'", req.model)
        raise HTTPException(status_code=500, detail="LLM request failed.")
