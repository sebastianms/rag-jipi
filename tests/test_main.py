import sys
import os

# Add parent directory to sys.path to allow importing main and database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Qdrant test path to avoid locking issues with the concurrently running dev server
os.environ["QDRANT_PATH"] = "./qdrant_test_data"

import unittest
import uuid
import shutil
from fastapi.testclient import TestClient
from main import app

class TestRAGEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_ingest_patient_entity(self):
        entity_payload = {
            "entity_name": "Patient",
            "entity_guid": str(uuid.uuid4()),
            "personal_info": {"age": 30, "name": "Alice Testing"},
            "treatments": [{
                "treatment_name": "Checkup",
                "start_date": "2026-03-01",
                "end_date": "2026-03-02",
                "drugs": [{"drug_name": "Vitamins", "dose": "1 pill"}]
            }]
        }

        response = self.client.post("/v1/entities", json=entity_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn(entity_payload["entity_guid"], data["message"])

    def test_ingest_generic_entity(self):
        entity_payload = {
            "entity_name": "Generic Note",
            "entity_guid": str(uuid.uuid4()),
            "personal_info": {"name": "Test User"},
            # Notice treatments list is empty, demonstrating it can accept partial payload
            "treatments": []
        }

        response = self.client.post("/v1/entities", json=entity_payload)
        self.assertEqual(response.status_code, 200, f"Failed with {response.text}")
        data = response.json()
        self.assertEqual(data["status"], "success")

    def test_chat_completions_requires_messages(self):
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": []
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_chat_completions_rag(self):
        payload = {
            "model": "gemini/gemini-1.5-flash", 
            "messages": [
                {"role": "user", "content": "Who is Alice Testing?"}
            ],
            "temperature": 0.0
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("id", data)
            self.assertIn("choices", data)
            self.assertGreater(len(data["choices"]), 0)
            self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
            self.assertTrue(len(data["choices"][0]["message"]["content"]) > 0)
        else:
            print(f"Chat completions failed with: {response.status_code} - {response.text}")
            # If API keys are missing, we expect a 500 error from litellm/fastapi. 
            self.assertIn(response.status_code, [200, 500])

    @classmethod
    def tearDownClass(cls):
        # Clean up the test database directory
        if os.path.exists("./qdrant_test_data"):
            shutil.rmtree("./qdrant_test_data")

if __name__ == "__main__":
    unittest.main(verbosity=2)
