#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


RUN_DIR = Path("ai_worker_v1/training/ball_v0/rfdetr_s_ball_v0_t1_runpod_wide_20260716T210444Z")
INFRA_DIR = RUN_DIR / "infrastructure"
AVAILABILITY_PATH = INFRA_DIR / "runpod_gpu_availability.json"

GPU_ORDER = [
    "NVIDIA L4",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA L40",
    "NVIDIA L40S",
]


def safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def candidate_clouds(gpu_info: dict) -> list[tuple[str, dict]]:
    choices: list[tuple[str, dict]] = []
    for cloud, key in [("COMMUNITY", "community"), ("SECURE", "secure")]:
        lowest_price = gpu_info.get(key) or {}
        stock = lowest_price.get("stockStatus")
        if stock and stock != "None":
            choices.append((cloud, lowest_price))
    return choices


def make_payload(gpu: str, cloud: str) -> dict:
    return {
        "name": "oneframe-rfdetr-ball-v0-training",
        "cloudType": cloud,
        "computeType": "GPU",
        "gpuTypeIds": [gpu],
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "imageName": "runpod/pytorch:1.0.7-rc.138-cu1281-torch280-ubuntu2204",
        "containerDiskInGb": 30,
        "volumeInGb": 50,
        "volumeMountPath": "/workspace",
        "ports": ["8888/http", "22/tcp"],
        "interruptible": False,
        "locked": False,
        "minRAMPerGPU": 16,
        "minVCPUPerGPU": 4,
        "allowedCudaVersions": ["12.8", "12.7", "12.6", "12.5", "12.4"],
        "supportPublicIp": True,
        "env": {},
    }


def create_pod(api_key: str, payload_path: Path) -> dict:
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            "https://rest.runpod.io/v1/pods",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            f"@{payload_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(proc.stdout)
    except Exception:
        return {"parse_error": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


def main() -> int:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY missing", file=sys.stderr)
        return 2
    availability = json.loads(AVAILABILITY_PATH.read_text(encoding="utf-8"))
    stock_by_gpu = {
        item["gpu_id"]: (item["response"].get("data", {}).get("gpuTypes") or [{}])[0]
        for item in availability
    }
    attempts = []
    created = None
    INFRA_DIR.mkdir(parents=True, exist_ok=True)
    for gpu in GPU_ORDER:
        gpu_info = stock_by_gpu.get(gpu, {})
        for cloud, stock in candidate_clouds(gpu_info):
            payload = make_payload(gpu, cloud)
            request_name = f"pod_creation_request_{safe_name(gpu)}_{cloud.lower()}.json"
            response_name = f"pod_create_response_{safe_name(gpu)}_{cloud.lower()}.json"
            request_path = INFRA_DIR / request_name
            response_path = INFRA_DIR / response_name
            request_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            response = create_pod(api_key, request_path)
            response_path.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            attempts.append(
                {
                    "gpu": gpu,
                    "cloudType": cloud,
                    "stock": stock,
                    "request_file": request_name,
                    "response_file": response_name,
                    "response": response,
                }
            )
            print(json.dumps({"gpu": gpu, "cloudType": cloud, "id": response.get("id"), "error": response.get("error"), "status": response.get("status")}, sort_keys=True))
            if response.get("id"):
                created = response
                break
            time.sleep(1)
        if created:
            break
    (INFRA_DIR / "pod_creation_attempts.json").write_text(json.dumps(attempts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if created:
        (INFRA_DIR / "selected_pod.json").write_text(json.dumps(created, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"created_pod_id": created.get("id")}, sort_keys=True))
        return 0
    print(json.dumps({"created_pod_id": None, "status": "no_pod_created"}, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
