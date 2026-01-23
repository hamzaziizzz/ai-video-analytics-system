from fastapi.testclient import TestClient

from ai_video_analytics.api.main import get_app
from ai_video_analytics.core import processing as processing_module


class DummyProcessing:
    async def extract(self, *args, **kwargs):
        return {
            "took": {"total_ms": 1.0, "read_imgs_ms": 0.1, "detect_all_ms": 0.2},
            "data": [
                {
                    "status": "ok",
                    "took_ms": 1.0,
                    "people": [
                        {
                            "bbox": [0, 0, 10, 10],
                            "prob": 0.9,
                            "class_id": 0,
                            "class_name": "person",
                            "num_det": 1,
                        }
                    ],
                }
            ],
        }


async def _override_processing():
    return DummyProcessing()


def test_detect_schema_matches_contract():
    app = get_app()
    app.dependency_overrides[processing_module.get_processing] = _override_processing
    client = TestClient(app)

    resp = client.post("/detect", json={"images": {"urls": ["test_images/person.jpg"]}})
    assert resp.status_code == 200
    payload = resp.json()

    assert "took" in payload
    assert "data" in payload
    assert payload["data"]
    first = payload["data"][0]
    assert "status" in first
    assert "people" in first
    assert first["people"]
    person = first["people"][0]
    for key in ("bbox", "prob", "class_id", "class_name"):
        assert key in person
