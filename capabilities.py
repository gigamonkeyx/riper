"""Capability Registry for RIPER-Ω System
Detects availability of optional subsystems and returns a structured report.

Usage:
    from capabilities import get_capabilities, print_capabilities
    caps = get_capabilities()
    print(caps)

CLI:
    python -m capabilities
"""
from __future__ import annotations
import importlib
import json
import os
import shutil
import socket
from typing import Dict, Any

CAPABILITY_VERSION = "1.0.0"

_DEF_OLLAMA_HOST = "localhost"
_DEF_OLLAMA_PORT = 11434


def _safe_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _check_ollama() -> Dict[str, Any]:
    host, port = _DEF_OLLAMA_HOST, _DEF_OLLAMA_PORT
    status = {"available": False, "host": host, "port": port}
    try:
        with socket.create_connection((host, port), timeout=1.5):
            status["available"] = True
    except Exception:
        pass
    return status


def _check_cuda() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch
        info["available"] = torch.cuda.is_available()
        if info["available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
    except Exception as e:
        info["error"] = str(e)
    return info


def _check_openrouter_key() -> Dict[str, Any]:
    key = os.getenv("OPENROUTER_API_KEY")
    return {"configured": bool(key), "masked": f"{key[:8]}..." if key else None}


def _check_drive_d() -> Dict[str, Any]:
    # Windows specific: ensure D: exists & writable
    drive = "D:/"
    ok = os.path.exists(drive)
    writable = False
    if ok:
        test_path = os.path.join(drive, ".riper_capability_test")
        try:
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
            writable = True
        except Exception:
            writable = False
    return {"exists": ok, "writable": writable}


def get_capabilities() -> Dict[str, Any]:
    caps: Dict[str, Any] = {
        "capability_version": CAPABILITY_VERSION,
        "python": {
            "version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        },
        "modules": {
            "torch": _safe_import("torch"),
            "evotorch": _safe_import("evotorch"),
            "deap": _safe_import("deap"),
            "ollama_module": _safe_import("ollama"),
            "bark": _safe_import("bark"),
            "camel": _safe_import("camel"),
        },
        "cuda": _check_cuda(),
        "ollama_service": _check_ollama(),
        "openrouter": _check_openrouter_key(),
        "drive_d": _check_drive_d(),
    }

    # Derive summary / warnings
    warnings = []
    if not caps["cuda"].get("available"):
        warnings.append("CUDA not available - running in CPU mode")
    if not caps["ollama_service"]["available"]:
        warnings.append("Ollama service not reachable on localhost:11434")
    if not caps["openrouter"]["configured"]:
        warnings.append("OPENROUTER_API_KEY not configured")
    if not caps["drive_d"]["exists"]:
        warnings.append("Drive D: missing - path compliance features limited")

    caps["summary"] = {
        "ready_for_gpu": caps["cuda"].get("available", False),
        "local_llm_ready": caps["ollama_service"]["available"],
        "hybrid_mode_ready": caps["ollama_service"]["available"] and caps["openrouter"]["configured"],
        "warnings": warnings,
    }
    return caps


def print_capabilities() -> None:
    print(json.dumps(get_capabilities(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    print_capabilities()
