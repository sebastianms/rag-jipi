# RAG Engine (Agnostic LLM)

Este proyecto es un motor RAG (Retrieval-Augmented Generation) construido con **FastAPI**, **Qdrant** y **LangChain**, diseñado para ser completamente agnóstico al modelo de lenguaje (LLM) subyacente gracias a la perfecta integración con **LiteLLM**. 

La aplicación permite ingestar entidades de pacientes médicos altamente estructuradas y exponer endpoints totalmente compatibles con el formato de la API de OpenAI (e.g., `/v1/chat/completions`), posibilitando que cualquier chatbot o interfaz moderna se conecte a él sin notar la diferencia.

---

## 🚀 Características Principales

- **Endpoints Compatibles con OpenAI:** Los clientes pueden consumir la API usando las mismas librerías y esquemas que usarían para conectarse a OpenAI.
- **Ruteo Agnóstico de Modelos (LiteLLM):** Envía tus consultas a OpenAI (`gpt-4o`), Anthropic (`claude-3-opus`), Ollama (`ollama/llama3`), o Google Gemini (`gemini/gemini-pro`) con solo cambiar el string `model` en la request.
- **Base de Datos Vectorial Local:** Qdrant está configurado para ejecutarse vía almacenamiento local persistente en `./qdrant_data`.
- **Embeddings Completamente Locales:** Se utiliza `all-MiniLM-L6-v2` de LangChain (HuggingFace) para que la vectorización no incurra en gastos de red externa ni costos de API.
- **Formato Estándar de Entidades:** El endpoint `/v1/entities` formatea modelos complejos JSON a modelos de RAG textuales y asegura que sus metadatos se recuperen sin pérdidas.

---

## 🛠 Entorno y Prerrequisitos

Este proyecto está desarrollado para sistemas basados en Unix (Linux/Ubuntu/Debian).

Para poder construir los entornos virtuales, asegúrate de tener instalados los paquetes de Python nivel de sistema:
```bash
sudo apt update
sudo apt install -y python3.13-venv python3-pip
```

---

## 📦 Instalación

1. **Clonar/Abrir el directorio del proyecto**
   ```bash
   cd /home/seba/Apps/my_rag
   ```

2. **Crear y activar un entorno virtual**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar las dependencias**
   ```bash
   pip install fastapi uvicorn qdrant-client langchain langchain-qdrant langchain-community sentence-transformers litellm google-cloud-aiplatform google-auth pydantic python-dotenv httpx
   ```

---

## ⚙️ Configuración (.env)

En la raíz del proyecto, debes tener un archivo `.env`. Este archivo es detectado automáticamente por LiteLLM para configurar la autenticación hacia el LLM del proveedor final.

Ejemplo de `.env`:
```env
# Ejemplo para usar OpenAI
OPENAI_API_KEY="sk-tu-clave-openai"

# Ejemplo para usar Anthropic
ANTHROPIC_API_KEY="sk-ant-tu-clave-anthropic"

# Para cambiar el modelo de Embeddings (Opcional)
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Ruta de Qdrant (Opcional)
QDRANT_PATH="./qdrant_data"
```

---

## 🏃 Módulo de Ejecución

Para iniciar el servidor de manera local en el puerto `8000`:
```bash
source venv/bin/activate
uvicorn main:app --reload
```
Una vez iniciado, tendrás acceso a la documentación interactiva en:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 Documentación de API Endpoints

### 1. Endpoint de Ingestión de Entidades
**`POST /v1/entities`**

Este endpoint permite enviar la entidad estructurada (por ejemplo, pacientes médicos con sus tratamientos) para ser embebida en la base de datos vectorial Qdrant.

*Ejemplo de Consulta (cURL):*
```bash
curl -X POST "http://127.0.0.1:8000/v1/entities" \
     -H "Content-Type: application/json" \
     -d '{
           "entity_name": "Patient",
           "entity_guid": "1102753c-0992-42d5-b28d-d7d2ddb74a0e",
           "personal_info": { "age": 24, "name": "John" },
           "treatments": [
             {
               "treatment_name": "Extraction",
               "start_date": "2026-01-01",
               "end_date": "2026-02-01",
               "drugs": [{"drug_name": "Foo", "dose": "30 gr at morning"}]
             }
           ]
         }'
```

*Respuesta Esperada:*
```json
{
  "status": "success",
  "message": "Entity 1102753c-0992-42d5-b28d-d7d2ddb74a0e ingested successfully."
}
```

### 2. Endpoint Chat - Formato OpenAI
**`POST /v1/chat/completions`**

Actúa como proxy para el LLM; realiza un RAG contra la colección embebida, pre-apende el contexto a tu mensaje y hace uso de `LiteLLM` para derivar la consulta al modelo que proveas.

*Ejemplo 1: OpenAI (`gpt-4o`)*
```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "gpt-4o",
           "messages": [
             {"role": "user", "content": "¿Qué tratamiento tuvo el paciente John?"}
           ],
           "temperature": 0.7
         }'
```
*(Requiere: `OPENAI_API_KEY` en el `.env`)*

*Ejemplo 2: Anthropic (`claude-3-opus-20240229`)*
```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "claude-3-opus-20240229",
           "messages": [
             {"role": "user", "content": "¿Qué tratamiento tuvo el paciente John?"}
           ],
           "temperature": 0.7
         }'
```
*(Requiere: `ANTHROPIC_API_KEY` en el `.env`)*

*Ejemplo 3: Servidor Local (Ollama con `llama3`)*
```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "ollama/llama3",
           "messages": [
             {"role": "user", "content": "¿Qué tratamiento tuvo el paciente John?"}
           ],
           "temperature": 0.7
         }'
```
*(Requiere: Ollama ejecutándose de fondo, no precisa API Key de pago. Opcional `OLLAMA_API_BASE` en el `.env`)*

*Ejemplo 4: Google Gemini (`gemini/gemini-pro`)*
```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "gemini/gemini-pro",
           "messages": [
             {"role": "user", "content": "¿Qué tratamiento tuvo el paciente John?"}
           ],
           "temperature": 0.7
         }'
```
*(Requiere: `GEMINI_API_KEY` en el `.env`)*

---

## 🔌 Integración en tu Aplicación

### Caso 1: Tu app ya usa el formato OpenAI

Si tu aplicación está construida sobre el SDK de OpenAI o consume directamente `/v1/chat/completions`, la integración es inmediata: solo apunta la `base_url` / `baseURL` a rag-jipi en lugar de a OpenAI. No requiere ningún adaptador.

**Python** (con `openai` SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no-se-usa",  # rag-jipi no valida API key; las claves van en .env del servidor
)

response = client.chat.completions.create(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "¿Qué tratamiento tuvo el paciente John?"}],
)
print(response.choices[0].message.content)
```

**JavaScript / Node.js** (con `openai` SDK)
```js
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "no-se-usa",
});

const response = await client.chat.completions.create({
  model: "gemini/gemini-2.0-flash",
  messages: [{ role: "user", content: "¿Qué tratamiento tuvo el paciente John?" }],
});
console.log(response.choices[0].message.content);
```

**Fetch directo** (cualquier frontend o backend sin SDK)
```js
const response = await fetch("http://localhost:8000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "gemini/gemini-2.0-flash",
    messages: [{ role: "user", content: "¿Qué tratamiento tuvo el paciente John?" }],
  }),
});
const data = await response.json();
console.log(data.choices[0].message.content);
```

---

### Caso 2: Tu app usa un formato de respuesta propio

Si tu aplicación espera un formato distinto al de OpenAI (por ejemplo, `{ textResponse, sources, error }` como AnythingLLM), necesitás un adaptador en el backend que consuma rag-jipi y transforme la respuesta.

El patrón recomendado es agregar un endpoint proxy en tu servidor que:
1. Reciba el mensaje en el formato que tu app ya entiende
2. Llame a rag-jipi con el formato OpenAI
3. Mapee `choices[0].message.content` al campo que tu app espera

**Ejemplo con Express.js**
```js
// Adaptador: convierte respuesta OpenAI → formato propio de tu app
app.post("/api/chat", async (req, res) => {
  const { message, model = "gemini/gemini-2.0-flash" } = req.body;

  const ragRes = await fetch("http://localhost:8000/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: message }],
    }),
  });

  const data = await ragRes.json();
  const textResponse = data.choices?.[0]?.message?.content ?? null;

  // Devolvés en el formato que tu app espera
  res.json({ textResponse, sources: [], error: null });
});
```

**Ejemplo con FastAPI (Python)**
```python
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.post("/api/chat")
async def chat(body: dict):
    message = body.get("message", "")
    model = body.get("model", "gemini/gemini-2.0-flash")

    async with httpx.AsyncClient() as client:
        rag_res = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": message}]},
        )

    data = rag_res.json()
    text_response = data["choices"][0]["message"]["content"]

    # Devolvés en el formato que tu app espera
    return {"textResponse": text_response, "sources": [], "error": None}
```

---

## 📁 Estructura del Proyecto

- `main.py`: Archivo raíz de la aplicación FastAPI y controladores de rutas.
- `database.py`: Creación del cliente Qdrant y configuración de Langchain `QdrantVectorStore`.
- `models.py`: Schemas Pydantic del dominio de Pacientes y schemas de Chat Completion.
- `verify.py`: Script con el framework TestClient de FastAPI para verificar la ingesta local fácilmente.
- `qdrant_data/`: Directorio autogenerado persistente para la vector store local de Qdrant.
- `.env`: Gestor de credenciales superpuestas (API Keys).

---

## ✨ Frameworks Utilizados
- **FastAPI**: Backend web de alto rendimiento.
- **Qdrant**: Motor RAG y bases de datos vectorial.
- **LangChain**: Orquestación y recuperación de Embeddings.
- **LiteLLM**: Interfaz unificada multiproveedor (OpenAI, Anthropic, Gemini, Ollama, etc).
- **Sentence-Transformers**: Modelos locales NLP.
