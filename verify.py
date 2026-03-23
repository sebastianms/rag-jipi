from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_ingest():
    entity_payload = {
        "entity_name": "Patient",
        "entity_guid": "1102753c-0992-42d5-b28d-d7d2ddb74a0e",
        "personal_info": { "age": 24, "name": "John"},
        "treatments": [{
            "treatment_name": "Extraction",
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
            "drugs": [{"drug_name": "Foo", "dose": "30 gr at morning"}]
        }]
    }

    print("Ingesting entity...")
    try:
        response = client.post("/v1/entities", json=entity_payload)
        print("Ingest response:", response.json())
        assert response.status_code == 200
        print("Entity ingested properly!")
    except Exception as e:
        print("Error during ingestion:", e)

if __name__ == "__main__":
    test_ingest()
    print("\\nVerification script ready. To fully test chat_completions, ensure you have an API key (e.g. OPENAI_API_KEY) in the .env file and run the server using 'uvicorn main:app --reload'.")
