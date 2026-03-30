import os

# Must be set before importing main, which triggers database initialization via lifespan
os.environ["QDRANT_PATH"] = "./qdrant_test_data"

import shutil
import unittest
import uuid

import pytest
from fastapi.testclient import TestClient

from main import app

_TEST_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qdrant_test_data")

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

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
    def test_chat_completions_rag(self):
        payload = {
            "model": "gemini/gemini-1.5-flash",
            "messages": [
                {"role": "user", "content": "Who is Alice Testing?"}
            ],
            "temperature": 0.0
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"]), 0)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertTrue(len(data["choices"][0]["message"]["content"]) > 0)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(_TEST_DB_PATH):
            shutil.rmtree(_TEST_DB_PATH)

if __name__ == "__main__":
    unittest.main(verbosity=2)
