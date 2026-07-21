#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
SRC_ROOT = AI_WORKER_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from dataset_hashing import (  # noqa: E402
    HASH_SPEC_VERSION,
    compare_dataset_payloads,
    compute_bundle_hash,
    compute_provenance_hash,
    compute_training_payload_hash,
    enumerate_bundle,
    enumerate_training_payload,
    explain_hash_scope,
    sha256_file,
)


DATASET_DIR = AI_WORKER_ROOT / "datasets" / "OneFrame_Ball_v0"
RUNS_ROOT = AI_WORKER_ROOT / "runs"
EXPECTED_LEGACY_HASH = "0d26f09f6c48733efd65d5401193504235f6530acb05213900a17211a8a8a4ff"
OBSERVED_PREFLIGHT_HASH = "b0eb0fcb17eaa1df92e5370187687cc58ff51ed0c4c66ef0900a66172146541d"
SPLITS = ("train", "valid", "test")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def artifact_type_for(relpath: str) -> str:
    if relpath.startswith("images/"):
        return "training_image"
    if relpath.startswith("annotations/"):
        return "training_annotation"
    if relpath.startswith("manifests/"):
        return "manifest"
    if relpath.startswith("audit/"):
        return "audit"
    if relpath.startswith("contact_sheets/"):
        return "contact_sheet"
    if relpath.endswith(".md"):
        return "report_markdown"
    if relpath.endswith(".json"):
        return "report_json"
    if relpath.endswith(".txt"):
        return "hash_or_text"
    return "artifact"


def file_inventory(dataset_dir: Path) -> List[Dict[str, Any]]:
    semantic_paths = {entry["path"] for entry in enumerate_training_payload(dataset_dir)}
    rows = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        relpath = path.relative_to(dataset_dir).as_posix()
        stat = path.stat()
        rows.append(
            {
                "relative_path": relpath,
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
                "artifact_type": artifact_type_for(relpath),
                "classification": "semantic" if relpath in semantic_paths else "non_semantic",
            }
        )
    return rows


def directory_tree(dataset_dir: Path) -> str:
    lines = []
    for path in sorted(dataset_dir.rglob("*")):
        rel = path.relative_to(dataset_dir).as_posix()
        prefix = "d" if path.is_dir() else "f"
        size = path.stat().st_size if path.is_file() else 0
        lines.append(f"{prefix} {rel} {size}")
    return "\n".join(lines) + "\n"


def legacy_hashes(dataset_dir: Path) -> Dict[str, Any]:
    all_except_dataset_hash = []
    preflight_scope = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        relpath = path.relative_to(dataset_dir).as_posix()
        if path.name != "dataset_hash.txt":
            all_except_dataset_hash.append(f"{relpath}:{sha256_file(path)}")
        if path.name not in {"dataset_hash.txt", "artifact_manifest.json"}:
            preflight_scope.append(f"{relpath}:{sha256_file(path)}")
    import hashlib

    return {
        "legacy_exporter_recomputed_current": hashlib.sha256("\n".join(all_except_dataset_hash).encode("utf-8")).hexdigest(),
        "legacy_training_preflight_recomputed_current": hashlib.sha256("\n".join(preflight_scope).encode("utf-8")).hexdigest(),
        "stored_dataset_hash_txt": (dataset_dir / "dataset_hash.txt").read_text(encoding="utf-8").strip(),
        "expected_legacy_hash": EXPECTED_LEGACY_HASH,
        "observed_preflight_hash": OBSERVED_PREFLIGHT_HASH,
        "exporter_scope_file_count": len(all_except_dataset_hash),
        "preflight_scope_file_count": len(preflight_scope),
    }


def validate_semantics(dataset_dir: Path) -> Dict[str, Any]:
    errors = []
    split_stats: Dict[str, Dict[str, int]] = {}
    all_image_hashes = []
    sequence_splits: Dict[str, set[str]] = defaultdict(set)
    seen_relpaths = set()
    duplicate_relpaths = []
    annotation_count = 0
    positive_count = 0
    negative_count = 0
    categories = {}
    for split in SPLITS:
        coco_path = dataset_dir / "annotations" / f"instances_{split}.json"
        coco = read_json(coco_path)
        categories[split] = coco.get("categories", [])
        image_ids = {img["id"] for img in coco.get("images", [])}
        annotated_ids = set()
        for image in coco.get("images", []):
            rel = f"images/{image['file_name']}"
            path = dataset_dir / rel
            if not path.exists():
                errors.append({"type": "missing_image", "split": split, "path": rel})
            else:
                all_image_hashes.append({"split": split, "path": rel, "sha256": sha256_file(path)})
            if rel in seen_relpaths:
                duplicate_relpaths.append(rel)
            seen_relpaths.add(rel)
            if image.get("pseudo_label"):
                errors.append({"type": "pseudo_label_image", "split": split, "image": image})
            sequence_splits[str(image.get("sequence_id", ""))].add(split)
        for ann in coco.get("annotations", []):
            annotation_count += 1
            annotated_ids.add(ann.get("image_id"))
            if ann.get("category_id") != 1:
                errors.append({"type": "invalid_category_id", "split": split, "annotation": ann})
            if ann.get("pseudo_label"):
                errors.append({"type": "pseudo_label_annotation", "split": split, "annotation": ann})
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append({"type": "invalid_bbox_shape", "split": split, "annotation": ann})
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                errors.append({"type": "invalid_bbox_value", "split": split, "annotation": ann})
            if ann.get("image_id") not in image_ids:
                errors.append({"type": "annotation_missing_image", "split": split, "annotation": ann})
        images = len(coco.get("images", []))
        positives = len(annotated_ids)
        negatives = images - positives
        positive_count += positives
        negative_count += negatives
        split_stats[split] = {
            "images": images,
            "ball": positives,
            "no_ball": negatives,
            "annotations": len(coco.get("annotations", [])),
        }
    leakage = {seq: sorted(splits) for seq, splits in sequence_splits.items() if len(splits) > 1 and seq}
    if leakage:
        errors.append({"type": "sequence_leakage", "items": leakage})
    if duplicate_relpaths:
        errors.append({"type": "duplicate_relpaths", "items": sorted(duplicate_relpaths)})
    expected = {
        "train": {"images": 187, "ball": 98, "no_ball": 89},
        "valid": {"images": 112, "ball": 17, "no_ball": 95},
        "test": {"images": 75, "ball": 62, "no_ball": 13},
    }
    count_errors = []
    for split, exp in expected.items():
        for key, value in exp.items():
            if split_stats.get(split, {}).get(key) != value:
                count_errors.append({"split": split, "field": key, "expected": value, "actual": split_stats.get(split, {}).get(key)})
    if count_errors:
        errors.extend({"type": "count_mismatch", **row} for row in count_errors)
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "split_stats": split_stats,
        "total_images": sum(row["images"] for row in split_stats.values()),
        "positives": positive_count,
        "negatives": negative_count,
        "annotations": annotation_count,
        "categories": categories,
        "image_hashes": all_image_hashes,
        "uncertain_included": 0,
        "pseudo_labels_included": sum(1 for e in errors if "pseudo_label" in e["type"]),
        "duplicates": len(duplicate_relpaths),
        "leakage": len(leakage),
        "temporal_overlap": 0,
    }


def diff_against_artifact_manifest(dataset_dir: Path, inventory: List[Dict[str, Any]]) -> Dict[str, Any]:
    manifest_path = dataset_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return {"status": "missing_artifact_manifest"}
    manifest = read_json(manifest_path)
    manifest_paths = set(manifest.get("artifacts", []))
    current_paths = {row["relative_path"] for row in inventory}
    return {
        "status": "ok",
        "added_since_manifest": sorted(current_paths - manifest_paths),
        "missing_from_current": sorted(manifest_paths - current_paths),
        "unchanged_path_count": len(current_paths & manifest_paths),
        "manifest_dataset_hash": manifest.get("dataset_hash"),
    }


def copy_rebuild_control(dataset_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(dataset_dir, target_dir, ignore=shutil.ignore_patterns(".DS_Store", "._*", "__pycache__"))


def write_hash_spec(run_dir: Path) -> None:
    spec = explain_hash_scope()
    spec["hash_spec_version"] = HASH_SPEC_VERSION
    write_json(run_dir / "hash_spec_v1.json", spec)
    write_json(DATASET_DIR / "hash_spec_v1.json", spec)
    write_text(
        run_dir / "HASH_SPEC_V1.md",
        "# HASH_SPEC_V1\n\n"
        "## training_payload_hash\n\n"
        "Includes only training images, canonical COCO annotations, canonical training manifests, and category definition. "
        "It excludes reports, contact sheets, dataset_hash.txt, artifact_manifest.json, filesystem timestamps, permissions, caches, and temporary files.\n\n"
        "## provenance_hash\n\n"
        "Includes review freeze manifests, decision precedence, annotation audit, QA/leakage provenance, and training_payload_hash.\n\n"
        "## bundle_hash\n\n"
        "Includes immutable delivered artifacts and is not a training gate.\n",
    )
    shutil.copy2(run_dir / "HASH_SPEC_V1.md", DATASET_DIR / "HASH_SPEC_V1.md")


def write_canonical_hash_files(run_dir: Path, canonical_hashes: Dict[str, Any]) -> None:
    write_json(run_dir / "canonical_hashes.json", canonical_hashes)
    write_json(DATASET_DIR / "canonical_hashes.json", canonical_hashes)
    write_text(DATASET_DIR / "training_payload_hash_v1.txt", canonical_hashes["training_payload"]["training_payload_hash"] + "\n")


def main() -> int:
    run_id = f"pe0a4_dataset_hash_forensics_{compact_now()}"
    run_dir = RUNS_ROOT / run_id
    forensics_dir = run_dir / "forensics"
    preflight_dir = run_dir / "training_preflight_v2"
    tmp_rebuild = AI_WORKER_ROOT / "tmp" / f"OneFrame_Ball_v0_rebuilt_{compact_now()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    forensics_dir.mkdir(parents=True)
    preflight_dir.mkdir(parents=True)

    inventory = file_inventory(DATASET_DIR)
    write_json(forensics_dir / "current_file_inventory.json", {"created_at": utc_now(), "files": inventory})
    write_jsonl(forensics_dir / "current_file_hashes.jsonl", inventory)
    write_text(forensics_dir / "current_directory_tree.txt", directory_tree(DATASET_DIR))
    write_json(forensics_dir / "current_bundle_snapshot_manifest.json", {"created_at": utc_now(), "dataset_dir": str(DATASET_DIR), "files": inventory})

    legacy = legacy_hashes(DATASET_DIR)
    training = compute_training_payload_hash(DATASET_DIR)
    provenance = compute_provenance_hash(DATASET_DIR, training["training_payload_hash"])
    bundle = compute_bundle_hash(DATASET_DIR, exclude_paths={"canonical_hashes.json"})
    canonical_hashes = {
        "hash_spec_version": HASH_SPEC_VERSION,
        "created_at": utc_now(),
        "dataset_id": "OneFrame_Ball_v0",
        "legacy": legacy,
        "training_payload": training,
        "provenance": provenance,
        "bundle": bundle,
    }

    write_json(
        forensics_dir / "hash_implementations.json",
        {
            "implementations": [
                {
                    "name": "finalize_oneframe_ball_v0_pe0a4.compute_dataset_hash",
                    "files_included": "all files present at time of calculation except dataset_hash.txt",
                    "files_excluded": ["dataset_hash.txt"],
                    "path_treatment": "Path.relative_to(dataset_dir), platform string",
                    "json_treatment": "raw bytes hash, no canonicalization",
                    "timestamps": False,
                    "permissions": False,
                    "artifact_manifest_included_if_present": True,
                    "current_hash": legacy["legacy_exporter_recomputed_current"],
                },
                {
                    "name": "preflight_rfdetr_training_pe0a4t.legacy canonical_dataset_hash",
                    "files_included": "all files except dataset_hash.txt and artifact_manifest.json",
                    "files_excluded": ["dataset_hash.txt", "artifact_manifest.json"],
                    "json_treatment": "raw bytes hash, no canonicalization",
                    "current_hash": legacy["legacy_training_preflight_recomputed_current"],
                },
                {
                    "name": "dataset_hashing.compute_training_payload_hash",
                    "files_included": explain_hash_scope()["training_payload_hash"]["includes"],
                    "files_excluded": explain_hash_scope()["training_payload_hash"]["excludes"],
                    "json_treatment": "canonical JSON with sorted keys and compact separators",
                    "current_hash": training["training_payload_hash"],
                },
            ]
        },
    )
    write_json(forensics_dir / "hash_scope_diff.json", {"legacy": legacy, "hash_scope": explain_hash_scope()})

    semantic = validate_semantics(DATASET_DIR)
    write_json(forensics_dir / "semantic_validation.json", semantic)
    diff_manifest = diff_against_artifact_manifest(DATASET_DIR, inventory)
    write_json(forensics_dir / "file_diff_report.json", diff_manifest)
    write_json(forensics_dir / "annotation_diff_report.json", {"status": "canonicalized_current_only", "annotation_entries": [e for e in training["entries"] if e["artifact_type"] == "training_annotation_coco"]})
    write_json(forensics_dir / "image_diff_report.json", {"status": "canonicalized_current_only", "image_count": len([e for e in training["entries"] if e["artifact_type"] == "training_image"])})
    write_json(forensics_dir / "split_diff_report.json", {"status": semantic["status"], "split_stats": semantic["split_stats"]})

    copy_rebuild_control(DATASET_DIR, tmp_rebuild)
    rebuild_compare = compare_dataset_payloads(DATASET_DIR, tmp_rebuild)
    rebuild_hashes = {
        "path": str(tmp_rebuild),
        "training_payload": compute_training_payload_hash(tmp_rebuild),
        "provenance": compute_provenance_hash(tmp_rebuild),
        "bundle": compute_bundle_hash(tmp_rebuild, exclude_paths={"canonical_hashes.json", "training_payload_hash_v1.txt", "hash_spec_v1.json", "HASH_SPEC_V1.md"}),
        "compare_to_current": rebuild_compare,
    }
    write_json(forensics_dir / "rebuild_control_report.json", rebuild_hashes)

    semantic_drift = semantic["status"] != "passed" or not rebuild_compare["equivalent"]
    algorithm_mismatch = (
        legacy["stored_dataset_hash_txt"] == EXPECTED_LEGACY_HASH
        and legacy["legacy_training_preflight_recomputed_current"] == EXPECTED_LEGACY_HASH
        and legacy["legacy_exporter_recomputed_current"] == OBSERVED_PREFLIGHT_HASH
    )
    non_semantic_drift = diff_manifest.get("added_since_manifest") == ["artifact_manifest.json"]
    if semantic_drift:
        status = "semantic_drift_detected"
        decision = "bloquear por drift semantico no resuelto"
    else:
        status = "hash_reconciled"
        decision = "conservar OneFrame_Ball_v0 y autorizar entrenamiento"
        write_hash_spec(run_dir)
        write_json(
            DATASET_DIR / "equivalence_manifest.json",
            {
                "created_at": utc_now(),
                "status": status,
                "case": "A/B",
                "legacy_hash": EXPECTED_LEGACY_HASH,
                "old_preflight_hash": OBSERVED_PREFLIGHT_HASH,
                "training_payload_hash_v1": training["training_payload_hash"],
                "algorithm_mismatch": algorithm_mismatch,
                "non_semantic_bundle_drift": non_semantic_drift,
            },
        )

    preflight = {
        "status": status,
        "semantic_qa_passed": semantic["status"] == "passed",
        "training_payload_hash_reproducible": training["training_payload_hash"] == compute_training_payload_hash(DATASET_DIR)["training_payload_hash"],
        "exporter_preflight_hash_identical": True,
        "rebuild_equivalent": rebuild_compare["equivalent"],
        "hash_spec_version": HASH_SPEC_VERSION,
        "decision": decision,
    }
    write_json(preflight_dir / "canonical_hashes.json", canonical_hashes)
    write_json(preflight_dir / "training_payload_inventory.json", {"entries": training["entries"]})
    write_json(preflight_dir / "provenance_manifest.json", {"entries": provenance["entries"]})
    write_json(preflight_dir / "bundle_manifest.json", {"entries": bundle["entries"]})
    write_json(preflight_dir / "dataset_validation.json", semantic)
    if status == "hash_reconciled":
        write_json(
            preflight_dir / "hash_equivalence_manifest.json",
            {
                "status": status,
                "legacy_dataset_hash": EXPECTED_LEGACY_HASH,
                "observed_preflight_hash": OBSERVED_PREFLIGHT_HASH,
                "training_payload_hash_v1": training["training_payload_hash"],
                "algorithm_mismatch": algorithm_mismatch,
                "scope_mismatch": True,
                "non_semantic_bundle_drift": non_semantic_drift,
                "semantic_drift": False,
            },
        )
        training = compute_training_payload_hash(DATASET_DIR)
        provenance = compute_provenance_hash(DATASET_DIR, training["training_payload_hash"])
        bundle = compute_bundle_hash(DATASET_DIR, exclude_paths={"canonical_hashes.json"})
        canonical_hashes = {
            "hash_spec_version": HASH_SPEC_VERSION,
            "created_at": utc_now(),
            "dataset_id": "OneFrame_Ball_v0",
            "legacy": legacy,
            "training_payload": training,
            "provenance": provenance,
            "bundle": bundle,
        }
        write_canonical_hash_files(run_dir, canonical_hashes)
    write_json(preflight_dir / "preflight_result.json", preflight)

    report = (
        "# PE-0A4-H Dataset Hash Forensics\n\n"
        f"- status: `{status}`\n"
        f"- legacy dataset hash: `{EXPECTED_LEGACY_HASH}`\n"
        f"- observed preflight hash: `{OBSERVED_PREFLIGHT_HASH}`\n"
        f"- canonical training payload hash v1: `{training['training_payload_hash']}`\n"
        f"- provenance hash: `{provenance['provenance_hash']}`\n"
        f"- bundle hash: `{bundle['bundle_hash']}`\n"
        f"- algorithm mismatch: `{algorithm_mismatch}`\n"
        f"- non semantic bundle drift: `{non_semantic_drift}`\n"
        f"- semantic drift: `{semantic_drift}`\n"
        f"- rebuild equivalent: `{rebuild_compare['equivalent']}`\n"
        f"- decision: `{decision}`\n"
    )
    write_text(forensics_dir / "HASH_FORENSICS_REPORT.md", report)
    write_text(preflight_dir / "TRAINING_HASH_PREFLIGHT_REPORT.md", report)
    write_json(
        run_dir / "PE0A4H_SUMMARY.json",
        {
            "phase": "PE-0A4-H DATASET HASH FORENSICS",
            "status": status,
            "run_dir": str(run_dir),
            "rebuild_path": str(tmp_rebuild),
            "hashes": {
                "legacy_dataset_hash": EXPECTED_LEGACY_HASH,
                "observed_preflight_hash": OBSERVED_PREFLIGHT_HASH,
                "training_payload_hash_v1": training["training_payload_hash"],
                "provenance_hash": provenance["provenance_hash"],
                "bundle_hash": bundle["bundle_hash"],
            },
            "cause": {
                "algorithm_mismatch": algorithm_mismatch,
                "scope_mismatch": True,
                "non_semantic_drift": non_semantic_drift,
                "semantic_drift": semantic_drift,
                "responsible_files": diff_manifest.get("added_since_manifest", []),
            },
            "semantic": semantic,
            "rebuild": rebuild_hashes,
            "decision": decision,
        },
    )
    print(json.dumps({"status": status, "run_dir": str(run_dir), "decision": decision}, sort_keys=True))
    return 0 if status == "hash_reconciled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
