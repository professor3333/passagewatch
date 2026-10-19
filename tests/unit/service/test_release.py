from __future__ import annotations

import json

from passagewatch.service.release import not_passagewatch, service_release

MODEL_INFO = {
    "pipeline_version": "passagewatch-0.3.2",
    "pipeline_config_sha256": "a946937d",
    "preprocessing_version": "letterbox-temporal3-v1",
    "counting_policy": "cfc-compatible-v1",
}


def test_a_passagewatch_model_info_names_its_release() -> None:
    assert service_release(200, json.dumps(MODEL_INFO).encode()) == "passagewatch-0.3.2"
    smoke = {**MODEL_INFO, "pipeline_version": "smoke-0"}
    assert service_release(200, json.dumps(smoke).encode()) == "smoke-0"


def test_another_program_on_the_port_is_not_mistaken_for_passagewatch() -> None:
    assert service_release(404, b'{"detail":"Not Found"}') is None
    assert service_release(200, b"<html>another app</html>") is None
    assert service_release(200, b'["a list"]') is None
    other = {"status": "ready", "checks": {"model_release": "beltwatch-0.1.0"}}
    assert service_release(200, json.dumps(other).encode()) is None
    partial = {k: v for k, v in MODEL_INFO.items() if k != "counting_policy"}
    assert service_release(200, json.dumps(partial).encode()) is None


def test_the_message_says_how_to_move_to_another_port() -> None:
    message = not_passagewatch("http://127.0.0.1:8000")
    assert "http://127.0.0.1:8000" in message and "PASSAGEWATCH_HTTP_PORT=8010" in message
