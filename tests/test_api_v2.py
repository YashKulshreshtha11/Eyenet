from fastapi.testclient import TestClient

from backend.app_server import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "classes" in payload


def test_predict_endpoint_with_mock(monkeypatch):
    from backend.routes import api_v2

    monkeypatch.setattr(
        api_v2.predict_svc,
        "predict",
        lambda *_args, **_kwargs: {
            "model_name": "EyeNet Ensemble",
            "weights_loaded": True,
            "predicted_class": "Normal",
            "class_index": 3,
            "confidence": 0.91,
            "probabilities": {
                "Diabetic Retinopathy": 0.03,
                "Glaucoma": 0.02,
                "Cataract": 0.04,
                "Normal": 0.91,
            },
            "top_predictions": [
                {"class_name": "Normal", "probability": 0.91},
                {"class_name": "Cataract", "probability": 0.04},
            ],
            "quality_metrics": {"brightness": 55.2, "contrast": 41.3, "sharpness": 38.7},
            "quality_verdict": "Excellent",
            "quality_advisory": "Image quality is strong enough for reliable analysis.",
            "preprocess_preview_base64": "abc",
            "gradcam_base64": "abc",
            "overlay_base64": "abc",
            "inference_time_ms": 123.4,
        },
    )
    monkeypatch.setattr(api_v2, "save_prediction", lambda **_kwargs: None)
    monkeypatch.setattr(api_v2, "log_event", lambda *_args, **_kwargs: None)

    client = TestClient(app)
    response = client.post(
        "/api/v1/predict",
        files={"file": ("sample.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "Normal"
    assert payload["confidence"] == 0.91
