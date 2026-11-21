"""
The desktop this project must run on has no internet. These tests make that
a checked property instead of a claim in the README:

* every network entry point in the process (DNS lookups and socket connects)
  is replaced by a function that fails the test, then each baseline file is
  scanned through both HTTP endpoints. If any analyzer ever reaches for the
  network, this fails;
* "scan by URL", the one feature that needs a network, fails with a normal
  structured error instead of hanging or crashing;
* the OpenCV face detector (a toolkit some builds lack) is present, so the
  portrait signal is not silently switched off.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.analyzers import gradients

BASELINE = Path(__file__).resolve().parents[2] / "baseline"
FILES = (
    "jpg/real-photo.jpeg",
    "png/ai-generated.png",
    "png/fireplace-generated.png",
    "pdf/baseline_sample.pdf",
    "pdf/anomaly_metadata_sample.pdf",
)

pytestmark = pytest.mark.skipif(not BASELINE.exists(), reason="baseline files not present")


class NetworkTouched(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    """Any DNS lookup or outgoing connection raises NetworkTouched."""

    def blocked(*args, **kwargs):
        raise NetworkTouched(f"network access attempted: {args[:2]}")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@pytest.fixture
def offline_like_a_real_machine(monkeypatch):
    """What a machine with no connectivity does: DNS fails, connects are refused."""

    def no_dns(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "Temporary failure in name resolution")

    def refused(*args, **kwargs):
        raise ConnectionRefusedError("network unreachable")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    monkeypatch.setattr(socket, "create_connection", refused)
    monkeypatch.setattr(socket.socket, "connect", refused)


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.mark.parametrize("rel", FILES)
def test_scan_api_uses_no_network(client, no_network, rel):
    path = BASELINE / rel
    resp = client.post("/api/v1/scans", files={"file": (path.name, path.read_bytes())})
    assert resp.status_code == 200, resp.text
    assert resp.json()["summary"]["status"] == "completed"


@pytest.mark.parametrize("rel", FILES)
def test_check_endpoint_uses_no_network(client, no_network, rel):
    path = BASELINE / rel
    resp = client.post(
        "/1.0/check.json", files={"media": (path.name, path.read_bytes())}, data={"models": "genai"}
    )
    assert resp.status_code == 200, resp.text


def test_scan_by_url_fails_cleanly_without_a_network(client, offline_like_a_real_machine):
    resp = client.post("/api/v1/scans", data={"url": "https://example.com/photo.jpg"})
    assert resp.status_code in (502, 504), resp.text
    summary = resp.json()["summary"]
    assert summary["status"] == "failed"
    assert summary["error"]["code"].startswith("url_")


def test_face_detector_toolkit_is_available():
    """OpenCV 5.x has no CascadeClassifier; without it portrait analysis is skipped and scores shift."""
    assert gradients._load_cascades() is not None, (
        "face detector could not be loaded: install the pinned OpenCV 4.x (pip install -r requirements.txt)"
    )
