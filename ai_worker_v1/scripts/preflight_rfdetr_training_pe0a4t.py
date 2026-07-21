#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EXPECTED_DATASET_HASH = "0d26f09f6c48733efd65d5401193504235f6530acb05213900a17211a8a8a4ff"
DATASET_ID = "OneFrame_Ball_v0"

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
DATASET_DIR = AI_WORKER_ROOT / "datasets" / DATASET_ID
TRAINING_ROOT = AI_WORKER_ROOT / "training" / "ball_v0"
sys.path.insert(0, str(AI_WORKER_ROOT / "src"))

SPLITS = ("train", "valid", "test")
CANONICAL_HASH_EXCLUDES = {"dataset_hash.txt", "artifact_manifest.json"}

from dataset_hashing import compute_training_payload_hash, explain_hash_scope  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_dataset_hash(dataset_dir: Path) -> Dict[str, Any]:
    payload = compute_training_payload_hash(dataset_dir)
    digest = payload["training_payload_hash"]
    stored_legacy = (dataset_dir / "dataset_hash.txt").read_text(encoding="utf-8").strip()
    expected_v1_path = dataset_dir / "training_payload_hash_v1.txt"
    expected_v1 = expected_v1_path.read_text(encoding="utf-8").strip() if expected_v1_path.exists() else ""
    return {
        "status": "passed" if expected_v1 and digest == expected_v1 else "failed",
        "hash_spec_version": payload["hash_spec_version"],
        "legacy_dataset_hash": EXPECTED_DATASET_HASH,
        "stored_legacy_dataset_hash": stored_legacy,
        "expected_training_payload_hash_v1": expected_v1,
        "recalculated_hash": digest,
        "file_count_hashed": payload["entry_count"],
        "hash_scope": explain_hash_scope()["training_payload_hash"],
        "note": "Training gates use HASH_SPEC_V1 training_payload_hash, not legacy bundle/dataset_hash.txt.",
    }


def validate_coco(dataset_dir: Path) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    split_stats: Dict[str, Dict[str, int]] = {}
    seen_files: Dict[str, str] = {}
    sequence_splits: Dict[str, str] = {}
    category_report: Dict[str, Any] = {}

    for split in SPLITS:
        annotation_path = dataset_dir / "annotations" / f"instances_{split}.json"
        if not annotation_path.exists():
            errors.append({"split": split, "error": "missing_coco_file", "path": str(annotation_path)})
            continue
        coco = read_json(annotation_path)
        categories = coco.get("categories", [])
        category_report[split] = categories
        if categories != [{"id": 1, "name": "ball"}]:
            errors.append({"split": split, "error": "invalid_categories", "categories": categories})

        image_ids = set()
        image_files = set()
        image_sequences = {}
        for image in coco.get("images", []):
            image_id = image.get("id")
            image_ids.add(image_id)
            rel = image.get("file_name", "")
            image_files.add(rel)
            image_path = dataset_dir / "images" / rel
            if not image_path.exists():
                errors.append({"split": split, "error": "missing_image", "file_name": rel})
            if image.get("width") != 512 or image.get("height") != 512:
                errors.append({"split": split, "error": "invalid_image_size", "file_name": rel, "image": image})
            if rel in seen_files and seen_files[rel] != split:
                errors.append({"split": split, "error": "image_shared_between_splits", "file_name": rel, "other_split": seen_files[rel]})
            seen_files[rel] = split
            seq = image.get("sequence_id", "")
            image_sequences[image_id] = seq
            if seq in sequence_splits and sequence_splits[seq] != split:
                errors.append({"split": split, "error": "sequence_shared_between_splits", "sequence_id": seq, "other_split": sequence_splits[seq]})
            sequence_splits[seq] = split

        annotated_images = set()
        ann_ids = set()
        for ann in coco.get("annotations", []):
            ann_id = ann.get("id")
            if ann_id in ann_ids:
                errors.append({"split": split, "error": "duplicate_annotation_id", "annotation_id": ann_id})
            ann_ids.add(ann_id)
            image_id = ann.get("image_id")
            annotated_images.add(image_id)
            if image_id not in image_ids:
                errors.append({"split": split, "error": "annotation_missing_image", "annotation_id": ann_id, "image_id": image_id})
            if ann.get("category_id") != 1:
                errors.append({"split": split, "error": "invalid_category_id", "annotation_id": ann_id, "category_id": ann.get("category_id")})
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append({"split": split, "error": "invalid_bbox_shape", "annotation_id": ann_id, "bbox": bbox})
                continue
            x, y, w, h = [float(v) for v in bbox]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 512 or y + h > 512:
                errors.append({"split": split, "error": "bbox_out_of_bounds", "annotation_id": ann_id, "bbox": bbox})
            if ann.get("pseudo_label") is True:
                errors.append({"split": split, "error": "pseudo_label_included", "annotation_id": ann_id})
            if ann.get("ground_truth") is not True:
                errors.append({"split": split, "error": "annotation_not_ground_truth", "annotation_id": ann_id})

        split_stats[split] = {
            "images": len(image_ids),
            "annotations": len(ann_ids),
            "positives": len(annotated_images),
            "negatives": len(image_ids) - len(annotated_images),
            "sequences": len(set(image_sequences.values())),
        }

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "split_stats": split_stats,
        "categories": category_report,
    }


def validate_splits(dataset_dir: Path) -> Dict[str, Any]:
    source_manifest = read_json(dataset_dir / "manifests" / "source_frames_manifest.json")["items"]
    split_counts: Dict[str, Dict[str, int]] = {}
    sequence_splits: Dict[str, set] = {}
    image_splits: Dict[str, set] = {}
    errors: List[Dict[str, Any]] = []

    for item in source_manifest:
        split = item["split"]
        status = item["status"]
        if split in SPLITS:
            split_counts.setdefault(split, {"ball": 0, "no_ball": 0, "uncertain": 0, "total": 0})
            split_counts[split]["total"] += 1
            if status == "reviewed_ball":
                split_counts[split]["ball"] += 1
            elif status == "reviewed_no_ball":
                split_counts[split]["no_ball"] += 1
            elif status == "reviewed_uncertain":
                split_counts[split]["uncertain"] += 1
            sequence_splits.setdefault(item["sequence_id"], set()).add(split)
            image_splits.setdefault(item["image_path"], set()).add(split)

    for seq, splits in sorted(sequence_splits.items()):
        if len(splits) > 1:
            errors.append({"error": "sequence_shared_between_splits", "sequence_id": seq, "splits": sorted(splits)})
    for image_path, splits in sorted(image_splits.items()):
        if len(splits) > 1:
            errors.append({"error": "image_shared_between_splits", "image_path": image_path, "splits": sorted(splits)})

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "split_counts": split_counts,
        "sequence_count": len(sequence_splits),
    }


def package_version(name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", "unknown"), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "version": None, "error": repr(exc)}


def inspect_rfdetr_api() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    api: Dict[str, Any] = {
        "package": package_version("rfdetr"),
        "RFDETRSmall": {"available": False},
        "train_signature": None,
        "error": None,
    }
    config = {
        "architecture": "RFDETRSmall",
        "classes": ["ball"],
        "num_classes": 1,
        "input_resolution": 512,
        "max_epochs": {"conceptual": 60, "argument": None, "supported": None},
        "mixed_precision": {"conceptual": True, "argument": None, "supported": None},
        "gradient_accumulation": {"conceptual": "if supported", "argument": None, "supported": None},
        "early_stopping": {"conceptual": "if supported", "argument": None, "supported": None},
        "selection_split": "valid",
        "test_usage": "prohibited_until_threshold_and_epoch_frozen",
    }
    try:
        from rfdetr import RFDETRSmall  # type: ignore

        api["RFDETRSmall"] = {"available": True, "repr": repr(RFDETRSmall)}
        sig = inspect.signature(RFDETRSmall.train)
        api["train_signature"] = str(sig)
        params = set(sig.parameters)
        config["max_epochs"] = {"conceptual": 60, "argument": "epochs" if "epochs" in params else None, "supported": "epochs" in params}
        for candidate in ["amp", "mixed_precision", "use_amp"]:
            if candidate in params:
                config["mixed_precision"] = {"conceptual": True, "argument": candidate, "supported": True}
                break
        else:
            config["mixed_precision"]["supported"] = False
        for candidate in ["grad_accum_steps", "gradient_accumulation_steps", "accumulate"]:
            if candidate in params:
                config["gradient_accumulation"] = {"conceptual": "if supported", "argument": candidate, "supported": True}
                break
        else:
            config["gradient_accumulation"]["supported"] = False
        for candidate in ["early_stopping", "patience"]:
            if candidate in params:
                config["early_stopping"] = {"conceptual": "if supported", "argument": candidate, "supported": True}
                break
        else:
            config["early_stopping"]["supported"] = False
    except Exception as exc:  # noqa: BLE001
        api["error"] = repr(exc)
    return api, config


def environment_manifest() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "created_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "os": os.name,
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "rfdetr": package_version("rfdetr"),
        "cuda": {"available": False, "version": None, "device_count": 0, "devices": []},
        "cudnn": None,
        "nvidia_smi": {"available": False, "output": None, "error": None},
    }
    try:
        import torch

        env["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "vram_bytes": torch.cuda.get_device_properties(idx).total_memory,
                }
                for idx in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
        env["cudnn"] = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
    except Exception as exc:  # noqa: BLE001
        env["cuda"]["error"] = repr(exc)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run([nvidia_smi], check=False, capture_output=True, text=True, timeout=10)
            env["nvidia_smi"] = {
                "available": proc.returncode == 0,
                "output": proc.stdout[-4000:],
                "error": proc.stderr[-4000:] if proc.stderr else None,
            }
        except Exception as exc:  # noqa: BLE001
            env["nvidia_smi"] = {"available": False, "output": None, "error": repr(exc)}
    else:
        env["nvidia_smi"] = {"available": False, "output": None, "error": "nvidia-smi not found"}
    return env


def make_report(run_dir: Path, state: Dict[str, Any]) -> None:
    lines = [
        "# PE-0A4-T Training Preflight",
        "",
        f"- status: `{state['status']}`",
        f"- dataset_hash: `{state['dataset_hash']['recalculated_hash']}`",
        f"- hash_verified: `{state['dataset_hash']['status'] == 'passed'}`",
        f"- coco_validation: `{state['coco']['status']}`",
        f"- split_validation: `{state['splits']['status']}`",
        f"- rfdetr_api_available: `{state['resolved_training_api']['RFDETRSmall'].get('available')}`",
        f"- cuda_available: `{state['environment']['cuda']['available']}`",
        f"- nvidia_smi_available: `{state['environment']['nvidia_smi']['available']}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    if state["blocking_reasons"]:
        lines.extend(f"- `{reason}`" for reason in state["blocking_reasons"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not start full training unless status is `ready_for_gpu_training` and a GPU environment is explicitly confirmed.",
            "",
        ]
    )
    write_text(run_dir / "training_preflight" / "TRAINING_PREFLIGHT_REPORT.md", "\n".join(lines))


def main() -> None:
    run_id = f"rfdetr_s_ball_v0_{compact_now()}"
    run_dir = TRAINING_ROOT / run_id
    preflight_dir = run_dir / "training_preflight"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "plots").mkdir()
    (run_dir / "dataset_snapshot").mkdir()

    dataset_hash = canonical_dataset_hash(DATASET_DIR)
    coco = validate_coco(DATASET_DIR)
    splits = validate_splits(DATASET_DIR)
    api, config = inspect_rfdetr_api()
    env = environment_manifest()

    blocking = []
    if dataset_hash["status"] != "passed":
        blocking.append("blocked_dataset_hash_mismatch")
    if coco["status"] != "passed":
        blocking.append("blocked_coco_validation")
    if splits["status"] != "passed":
        blocking.append("blocked_split_validation")
    if not api["RFDETRSmall"].get("available"):
        blocking.append("blocked_rfdetr_api_unavailable")
    if not env["cuda"]["available"] or not env["nvidia_smi"]["available"]:
        blocking.append("blocked_gpu_infrastructure_unavailable")

    status = "ready_for_gpu_training" if not blocking else "blocked"
    snapshot = {
        "dataset_id": DATASET_ID,
        "dataset_path": str(DATASET_DIR),
        "dataset_hash": dataset_hash["recalculated_hash"],
        "snapshot_type": "manifest_only_pre_gpu",
        "immutable_source": True,
        "note": "Full dataset copy is deferred until GPU training starts; source dataset is hash-verified.",
    }
    state = {
        "phase": "PE-0A4-T RF-DETR-S BALL SPECIALIST V0 TRAINING",
        "created_at": utc_now(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status,
        "blocking_reasons": blocking,
        "dataset_hash": dataset_hash,
        "coco": coco,
        "splits": splits,
        "resolved_training_api": api,
        "resolved_training_config": config,
        "environment": env,
        "dataset_snapshot": snapshot,
    }

    write_json(preflight_dir / "dataset_hash_validation.json", dataset_hash)
    write_json(preflight_dir / "coco_validation.json", coco)
    write_json(preflight_dir / "split_validation.json", splits)
    write_json(preflight_dir / "resolved_training_api.json", api)
    write_json(preflight_dir / "resolved_training_config.json", config)
    write_json(run_dir / "environment_manifest.json", env)
    write_json(run_dir / "dataset_snapshot" / "dataset_snapshot_manifest.json", snapshot)
    write_json(run_dir / "training_summary.json", state)
    if blocking:
        write_json(run_dir / "failure_report.json", {"created_at": utc_now(), "status": status, "blocking_reasons": blocking})
    make_report(run_dir, state)
    write_json(run_dir / "artifact_manifest.json", {"created_at": utc_now(), "run_id": run_id, "artifacts": sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file())})
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
