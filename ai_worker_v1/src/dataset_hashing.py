from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


HASH_SPEC_VERSION = "hash_spec_v1"
TRAINING_SPLITS = ("train", "valid", "test")
TRAINING_MANIFESTS = (
    "train_manifest.json",
    "valid_manifest.json",
    "test_manifest.json",
    "split_manifest.json",
    "crop_manifest.json",
    "source_frames_manifest.json",
)
IGNORED_NAMES = {".DS_Store"}
IGNORED_PREFIXES = ("._",)
IGNORED_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
TEMP_SUFFIXES = (".tmp", ".temp", ".swp")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_ignored(path: Path) -> bool:
    if path.name in IGNORED_NAMES:
        return True
    if any(path.name.startswith(prefix) for prefix in IGNORED_PREFIXES):
        return True
    if path.suffix in TEMP_SUFFIXES:
        return True
    if any(part in IGNORED_DIR_NAMES for part in path.parts):
        return True
    return False


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json_digest(path: Path) -> str:
    return sha256_bytes(canonical_json_bytes(_read_json(path)))


def _category_definition(dataset_dir: Path) -> Dict[str, Any]:
    categories_by_split = {}
    for split in TRAINING_SPLITS:
        annotation_path = dataset_dir / "annotations" / f"instances_{split}.json"
        if annotation_path.exists():
            categories_by_split[split] = _read_json(annotation_path).get("categories", [])
    canonical_categories = categories_by_split.get("train") or next(iter(categories_by_split.values()), [])
    return {
        "num_classes": len(canonical_categories),
        "categories": [
            {
                "id": item.get("id"),
                "name": item.get("name"),
            }
            for item in canonical_categories
        ],
        "categories_by_split": categories_by_split,
    }


def enumerate_training_payload(dataset_dir: Path) -> List[Dict[str, Any]]:
    dataset_dir = dataset_dir.resolve()
    entries: List[Dict[str, Any]] = []

    for split in TRAINING_SPLITS:
        image_root = dataset_dir / "images" / split
        if image_root.exists():
            for path in sorted(image_root.rglob("*")):
                if path.is_file() and not _is_ignored(path):
                    entries.append(
                        {
                            "path": _relative(path, dataset_dir),
                            "artifact_type": "training_image",
                            "semantic": True,
                            "hash_type": "sha256_bytes",
                            "sha256": sha256_file(path),
                            "size_bytes": path.stat().st_size,
                        }
                    )

    for split in TRAINING_SPLITS:
        path = dataset_dir / "annotations" / f"instances_{split}.json"
        if path.exists() and not _is_ignored(path):
            entries.append(
                {
                    "path": _relative(path, dataset_dir),
                    "artifact_type": "training_annotation_coco",
                    "semantic": True,
                    "hash_type": "sha256_canonical_json",
                    "sha256": _canonical_json_digest(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    for manifest_name in TRAINING_MANIFESTS:
        path = dataset_dir / "manifests" / manifest_name
        if path.exists() and not _is_ignored(path):
            entries.append(
                {
                    "path": _relative(path, dataset_dir),
                    "artifact_type": "training_manifest",
                    "semantic": True,
                    "hash_type": "sha256_canonical_json",
                    "sha256": _canonical_json_digest(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    category_payload = _category_definition(dataset_dir)
    entries.append(
        {
            "path": "__category_definition__.json",
            "artifact_type": "category_definition",
            "semantic": True,
            "hash_type": "sha256_canonical_json",
            "sha256": sha256_bytes(canonical_json_bytes(category_payload)),
            "size_bytes": len(canonical_json_bytes(category_payload)),
            "payload": category_payload,
        }
    )
    return sorted(entries, key=lambda item: item["path"])


def _hash_entries(entries: Iterable[Dict[str, Any]], *, include_artifact_type: bool = True) -> str:
    rows = []
    for entry in sorted(entries, key=lambda item: item["path"]):
        row = {
            "path": entry["path"],
            "sha256": entry["sha256"],
        }
        if include_artifact_type:
            row["artifact_type"] = entry.get("artifact_type")
            row["hash_type"] = entry.get("hash_type")
        rows.append(row)
    return sha256_bytes(canonical_json_bytes(rows))


def compute_training_payload_hash(dataset_dir: Path) -> Dict[str, Any]:
    entries = enumerate_training_payload(dataset_dir)
    return {
        "hash_spec_version": HASH_SPEC_VERSION,
        "training_payload_hash": _hash_entries(entries),
        "entries": entries,
        "entry_count": len(entries),
    }


def provenance_inputs(dataset_dir: Path, training_payload_hash: Optional[str] = None) -> List[Dict[str, Any]]:
    dataset_dir = dataset_dir.resolve()
    ai_worker_root = Path(__file__).resolve().parents[1]
    repo_root = ai_worker_root.parent
    if training_payload_hash is None:
        training_payload_hash = compute_training_payload_hash(dataset_dir)["training_payload_hash"]
    relpaths = [
        "audit/original_review_frozen_manifest.json",
        "audit/supplemental_review_frozen_manifest.json",
        "audit/correction_review_frozen_manifest.json",
        "audit/positive_completion_frozen_manifest.json",
        "audit/decision_precedence_report.json",
        "audit/annotation_audit.json",
        "audit/combined_review_manifest.json",
        "audit/leakage_report.json",
        "DATASET_QA_REPORT.json",
    ]
    entries: List[Dict[str, Any]] = [
        {
            "path": "__training_payload_hash__.txt",
            "artifact_type": "training_payload_hash",
            "hash_type": "literal_sha256",
            "sha256": sha256_bytes(training_payload_hash.encode("ascii")),
        },
        {
            "path": "__hash_spec_version__.txt",
            "artifact_type": "hash_spec_version",
            "hash_type": "literal_sha256",
            "sha256": sha256_bytes(HASH_SPEC_VERSION.encode("utf-8")),
        }
    ]
    exporter_path = repo_root / "ai_worker_v1" / "scripts" / "finalize_oneframe_ball_v0_pe0a4.py"
    if exporter_path.exists():
        entries.append(
            {
                "path": "__exporter_code__/finalize_oneframe_ball_v0_pe0a4.py",
                "artifact_type": "exporter_code",
                "hash_type": "sha256_bytes",
                "sha256": sha256_file(exporter_path),
                "size_bytes": exporter_path.stat().st_size,
            }
        )
    source_manifest_path = dataset_dir / "manifests" / "source_frames_manifest.json"
    source_payload: Dict[str, Any] = {}
    if source_manifest_path.exists():
        source_items = _read_json(source_manifest_path).get("items", [])
        source_payload = {
            "source_review_freeze_ids": sorted({row.get("source_freeze_id") for row in source_items if row.get("source_freeze_id")}),
            "source_review_hashes": sorted({row.get("source_review_hash") for row in source_items if row.get("source_review_hash")}),
            "source_reviews": sorted({row.get("source_review") for row in source_items if row.get("source_review")}),
        }
    export_config_payload = {
        "dataset_id": dataset_dir.name,
        "exporter": "finalize_oneframe_ball_v0_pe0a4.py",
        "exporter_contract": "PE-0A4 DATASET FINALIZATION",
        "exporter_version": "pe0a4_finalize_v1",
        "crop_size": 512,
        "category_id": 1,
        "category_name": "ball",
        "splits": list(TRAINING_SPLITS),
        "training_manifests": list(TRAINING_MANIFESTS),
        "calibration_profile_references": source_payload,
    }
    entries.append(
        {
            "path": "__export_config__.json",
            "artifact_type": "export_config",
            "hash_type": "sha256_canonical_json",
            "sha256": sha256_bytes(canonical_json_bytes(export_config_payload)),
            "payload": export_config_payload,
        }
    )
    for relpath in relpaths:
        path = dataset_dir / relpath
        if not path.exists() or _is_ignored(path):
            continue
        hash_type = "sha256_canonical_json" if path.suffix == ".json" else "sha256_bytes"
        digest = _canonical_json_digest(path) if path.suffix == ".json" else sha256_file(path)
        entries.append(
            {
                "path": relpath,
                "artifact_type": "provenance_artifact",
                "hash_type": hash_type,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return sorted(entries, key=lambda item: item["path"])


def compute_provenance_hash(dataset_dir: Path, training_payload_hash: Optional[str] = None) -> Dict[str, Any]:
    entries = provenance_inputs(dataset_dir, training_payload_hash)
    return {
        "hash_spec_version": HASH_SPEC_VERSION,
        "provenance_hash": _hash_entries(entries),
        "entries": entries,
        "entry_count": len(entries),
    }


def enumerate_bundle(dataset_dir: Path, *, exclude_paths: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    dataset_dir = dataset_dir.resolve()
    excluded = set(exclude_paths or [])
    entries = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or _is_ignored(path):
            continue
        relpath = _relative(path, dataset_dir)
        if relpath in excluded:
            continue
        entries.append(
            {
                "path": relpath,
                "artifact_type": "bundle_artifact",
                "hash_type": "sha256_bytes",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entries


def compute_bundle_hash(dataset_dir: Path, *, exclude_paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    entries = enumerate_bundle(dataset_dir, exclude_paths=exclude_paths)
    return {
        "hash_spec_version": HASH_SPEC_VERSION,
        "bundle_hash": _hash_entries(entries),
        "entries": entries,
        "entry_count": len(entries),
    }


def explain_hash_scope() -> Dict[str, Any]:
    return {
        "hash_spec_version": HASH_SPEC_VERSION,
        "training_payload_hash": {
            "includes": [
                "images/train/**",
                "images/valid/**",
                "images/test/**",
                "annotations/instances_train.json",
                "annotations/instances_valid.json",
                "annotations/instances_test.json",
                *[f"manifests/{name}" for name in TRAINING_MANIFESTS],
                "category definition derived from COCO categories",
            ],
            "excludes": [
                "dataset_hash.txt",
                "artifact_manifest.json",
                "reports",
                "contact_sheets",
                "caches",
                ".DS_Store",
                "temporary files",
                "filesystem timestamps",
                "permissions",
                "absolute paths",
            ],
        },
        "provenance_hash": {
            "includes": [
                "human review freeze manifests",
                "decision precedence report",
                "annotation audit",
                "QA/leakage provenance",
                "training_payload_hash",
            ]
        },
        "bundle_hash": {
            "includes": ["all immutable dataset artifacts"],
            "excludes": ["itself", "caches", "temporary files", ".DS_Store"],
            "training_gate": False,
        },
    }


def compare_dataset_payloads(left_dir: Path, right_dir: Path) -> Dict[str, Any]:
    left = {entry["path"]: entry for entry in enumerate_training_payload(left_dir)}
    right = {entry["path"]: entry for entry in enumerate_training_payload(right_dir)}
    left_paths = set(left)
    right_paths = set(right)
    common = left_paths & right_paths
    modified = [
        {
            "path": path,
            "left_sha256": left[path]["sha256"],
            "right_sha256": right[path]["sha256"],
            "artifact_type": left[path].get("artifact_type"),
        }
        for path in sorted(common)
        if left[path]["sha256"] != right[path]["sha256"]
    ]
    return {
        "left_training_payload_hash": compute_training_payload_hash(left_dir)["training_payload_hash"],
        "right_training_payload_hash": compute_training_payload_hash(right_dir)["training_payload_hash"],
        "equivalent": not (left_paths - right_paths or right_paths - left_paths or modified),
        "added": sorted(right_paths - left_paths),
        "removed": sorted(left_paths - right_paths),
        "modified": modified,
    }
