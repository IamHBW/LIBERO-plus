#!/usr/bin/env python3
"""Build the full replay-based LIBERO-Plus RGB-D replica dataset."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "data/libero_plus_rgbd_replica_v3"
DEFAULT_SOURCE_ROOT = ROOT / "libero/datasets"
DEFAULT_QWEN = ROOT.parents[1] / "ckpt/Qwen2.5-7B-Instruct-a09a3545"
IMAGE_SIZE = 256
MASTER_SEED = 0
SETTINGS = ("objects", "background", "light", "camera", "language", "noise")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUBDIMENSIONS = {
    "objects": ("O1",),
    "background": ("B1", "B2"),
    "light": ("L1", "L2", "L3", "L4"),
    "camera": ("C1", "C2", "C3"),
    "language": ("R1", "R2", "R3"),
    "noise": ("N1", "N2", "N3", "N4", "N5"),
}
TEMPORAL_MODES = {
    "objects": "episode_static",
    "background": "episode_static",
    "light": "episode_static",
    "camera": "episode_static",
    "language": "episode_static",
    "N1": "per_frame_random",
    "N2": "per_frame_deterministic",
    "N3": "per_frame_deterministic",
    "N4": "per_frame_random",
    "N5": "per_frame_random",
}
NOISE_ORDER = ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur")
NOISE_CODES = dict(zip(NOISE_ORDER, SUBDIMENSIONS["noise"]))
LIGHT_FIELDS = {"L1": "diffuse", "L2": "dir", "L3": "specular", "L4": "castshadow"}
BASE_SCENES = {
    "libero_tabletop_manipulation": "libero_tabletop_base_style.xml",
    "libero_floor_manipulation": "libero_floor_base_style.xml",
    "libero_kitchen_tabletop_manipulation": "libero_kitchen_tabletop_base_style.xml",
    "libero_living_room_tabletop_manipulation": "libero_living_room_tabletop_base_style.xml",
    "libero_study_tabletop_manipulation": "libero_study_base_style.xml",
    "libero_coffee_table_manipulation": "libero_coffee_table_base_style.xml",
}
SOURCE = {
    "repo": "yifengzhu-hf/LIBERO-datasets",
    "revision": "f13aa24a3da8c43c7225569f28c562979fa0e35a",
}
TASK_METADATA = {
    "repo": "Sylvest/libero_plus_lerobot",
    "revision": "22c57433fef692b5b9ecc0795344daac7fa867a5",
    "path": "meta/tasks.jsonl",
}
QWEN = {
    "repo": "Qwen/Qwen2.5-7B-Instruct",
    "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
}
PROTOCOL_VERSION = "libero-plus-rgbd-replica-v3"
NOOP_EPSILON = 1e-6
SMOKE_SOURCE_ORDINALS = (0, 50, 51, 52, 100, 101, 102, 150, 151, 152, 1959, 1983)


class AuditError(RuntimeError):
    pass


class AttemptFailure(RuntimeError):
    pass


class WorkerFailure(AttemptFailure):
    def __init__(self, message: str, worker_traceback: str = ""):
        super().__init__(message)
        self.worker_traceback = worker_traceback


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts) -> int:
    payload = canonical_json((MASTER_SEED, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def atomic_jsonl(path: Path, rows) -> None:
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def decoded_dataset(dataset) -> str:
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def action_key(actions) -> tuple[int, str]:
    import numpy as np

    actions = np.ascontiguousarray(actions, dtype="<f4")
    return len(actions), sha256_bytes(actions.tobytes())


def keep_action_mask(actions):
    import numpy as np

    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise AuditError(f"actions must have shape [T, >=6], got {actions.shape}")
    return np.max(np.abs(actions[:, :6]), axis=1) > NOOP_EPSILON


def bddl_block(text: str, name: str) -> str:
    match = re.search(rf"\(:{re.escape(name)}\b", text, re.IGNORECASE)
    if not match:
        raise AuditError(f"missing BDDL block: {name}")
    depth = 0
    for index in range(match.start(), len(text)):
        depth += text[index] == "("
        depth -= text[index] == ")"
        if depth == 0:
            return text[match.start() : index + 1]
    raise AuditError(f"unterminated BDDL block: {name}")


def bddl_objects(path: Path) -> list[tuple[str, str]]:
    block = bddl_block(path.read_text(), "objects")
    result = []
    for names, category in re.findall(r"^\s+(.+?)\s+-\s+(\S+)\s*$", block, re.MULTILINE):
        result.extend((name, category) for name in names.split())
    return result


def bddl_language(path: Path) -> str:
    match = re.search(r"\(:language\s+(.+?)\s*\)", path.read_text(), re.IGNORECASE)
    if not match:
        raise AuditError(f"missing :language in {path}")
    return " ".join(match.group(1).split())


def bddl_problem(path: Path) -> str:
    match = re.search(r"\(define\s+\(problem\s+([^)]+)\)", path.read_text(), re.IGNORECASE)
    if not match:
        raise AuditError(f"missing problem name in {path}")
    return match.group(1)


def canonical_bddl(suite: str, task: str) -> Path:
    normalized = task.lower().replace(" ", "_")
    folder = ROOT / "libero/libero/bddl_files" / suite
    direct = folder / f"{normalized}.bddl"
    if direct.is_file():
        return direct
    matches = [
        path
        for path in folder.glob(f"*_{normalized}.bddl")
        if not re.search(r"_(?:table|tb|language|light|add|level|moved|view)_", path.stem)
    ]
    if len(matches) != 1:
        raise AuditError(f"{suite}: cannot map task to one canonical BDDL: {task}")
    return matches[0]


def task_suite(task: str) -> tuple[str, Path]:
    matches = []
    for suite in SUITES:
        try:
            matches.append((suite, canonical_bddl(suite, task)))
        except AuditError:
            pass
    if len(matches) != 1:
        raise AuditError(f"task maps to {len(matches)} suites: {task}")
    return matches[0]


@lru_cache(maxsize=1)
def benchmark_map() -> dict:
    path = ROOT / "libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(path.read_text())
    return ast.literal_eval(tree.body[0].value)


@lru_cache(maxsize=None)
def task_test_names(suite: str, stem: str) -> list[str]:
    return sorted(name for name in benchmark_map()[suite] if name.startswith(stem + "_"))


def is_multistep_task(language: str) -> bool:
    lower = language.lower()
    return " and " in lower or "both " in lower or " then " in lower


def protocol() -> dict:
    value = {
        "version": PROTOCOL_VERSION,
        "source": SOURCE,
        "task_metadata": TASK_METADATA,
        "qwen": QWEN,
        "image_size": IMAGE_SIZE,
        "settings": list(SETTINGS),
        "subdimensions": SUBDIMENSIONS,
        "no_op_rule": f"max(abs(action[:6])) > {NOOP_EPSILON}",
        "frame_seed_rule": "stable_seed(job_seed, source_timestep)",
        "replay": "execute every source action; save only non-no-op post-step frames; retain final successes",
        "excluded": ["O2 Target Object Pose", "Robot Initial State"],
    }
    value["protocol_hash"] = sha256_bytes(canonical_json(value).encode())
    return value


def _download(repo: str, revision: str, relative: str, root: Path, repo_type="dataset") -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=relative,
            revision=revision,
            repo_type=repo_type,
            local_dir=root,
        )
    )


def source_demo_rows(path: Path) -> list[dict]:
    import h5py

    rows = []
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise AuditError(f"{path}: missing data group")
        names = sorted(handle["data"], key=lambda name: int(re.search(r"\d+", name).group()))
        for name in names:
            group = handle[f"data/{name}"]
            actions = group["actions"][()]
            keep = keep_action_mask(actions)
            model = group.attrs.get("model_file")
            if isinstance(model, bytes):
                model = model.decode()
            if not model or group["states"].shape[0] < 1 or actions.shape[1] != 7:
                raise AuditError(f"{path}:{name}: invalid source trajectory")
            rows.append(
                {
                    "source_demo": name,
                    "actions": int(len(actions)),
                    "saved_frames": int(keep.sum()),
                    "action_sha256": action_key(actions)[1],
                    "keep_mask_sha256": sha256_bytes(keep.tobytes()),
                    "source_model_xml_sha256": sha256_bytes(str(model).encode()),
                }
            )
    if not rows:
        raise AuditError(f"{path}: contains no demos")
    return rows


def _task_metadata(output: Path, no_download: bool) -> list[dict]:
    path = output / "metadata" / TASK_METADATA["path"]
    if no_download:
        if not path.is_file():
            raise AuditError(f"missing pinned task metadata: {path}")
    else:
        path = _download(
            TASK_METADATA["repo"], TASK_METADATA["revision"], TASK_METADATA["path"], output / "metadata"
        )
    rows = read_jsonl(path)
    if len(rows) != 40 or sorted(row["task_index"] for row in rows) != list(range(40)):
        raise AuditError("pinned task metadata must contain task indices 0..39")
    return sorted(rows, key=lambda row: row["task_index"])


def audit(args) -> None:
    output = args.output.resolve()
    source_root = args.source_root.resolve()
    tasks = _task_metadata(output, args.no_download)
    audited = []
    source_ordinal = 0
    for task in tasks:
        suite, bddl = task_suite(task["task"])
        relative = f"{suite}/{bddl.stem}_demo.hdf5"
        source_path = source_root / relative
        if args.no_download:
            if not source_path.is_file():
                raise AuditError(f"missing source HDF5: {source_path}")
        else:
            source_path = _download(SOURCE["repo"], SOURCE["revision"], relative, source_root)
        demos = source_demo_rows(source_path)
        for demo in demos:
            demo["source_ordinal"] = source_ordinal
            source_ordinal += 1
        audited.append(
            {
                "task_index": task["task_index"],
                "task": task["task"],
                "suite": suite,
                "task_stem": bddl.stem,
                "canonical_language": bddl_language(bddl),
                "multi_step": is_multistep_task(bddl_language(bddl)),
                "canonical_bddl": str(bddl.resolve()),
                "canonical_bddl_sha256": sha256_file(bddl),
                "source_file": str(source_path.resolve()),
                "source_bytes": source_path.stat().st_size,
                "demos": demos,
            }
        )
    if Counter(row["suite"] for row in audited) != Counter({suite: 10 for suite in SUITES}):
        raise AuditError("source tasks must contain ten tasks from each LIBERO suite")
    result = {
        "complete": True,
        "source": SOURCE,
        "source_root": str(source_root),
        "task_count": len(audited),
        "source_demo_count": source_ordinal,
        "tasks": audited,
    }
    atomic_json(output / "protocol.json", protocol())
    atomic_json(output / "audit.json", result)
    print(canonical_json({"audit": "complete", "tasks": len(audited), "source_demos": source_ordinal}))


LANGUAGE_PROMPTS = {
    "R1": (
        "Rewrite the robot instruction as a longer conversational request. Add one harmless, "
        "task-irrelevant contextual detail, but preserve the exact task goal and do not add an action."
    ),
    "R2": (
        "Rewrite the robot instruction using commonsense descriptions of the mentioned objects. "
        "You may replace the original entity words, but preserve the exact action, relations, and goal."
    ),
    "R3": (
        "Rewrite this multi-step robot instruction so the ordered reasoning chain is explicit. "
        "Preserve every action, its order, all spatial relations, and the final goal."
    ),
}
LANGUAGE_VARIANTS = {
    "R1": (
        'Begin exactly with "While there is no rush,".',
        'Begin exactly with "As a small favor,".',
        'Begin exactly with "For today’s household task,".',
        'Begin exactly with "When you have a moment,".',
        'Begin exactly with "To help keep things organized,".',
    ),
    "R2": (
        'Begin exactly with "Using visible characteristics," and avoid original entity nouns where possible.',
        'Begin exactly with "By their usual household functions," and avoid original entity nouns where possible.',
        'Begin exactly with "By category and distinguishing features,".',
        'Begin exactly with "Using natural indirect descriptions,".',
        'Begin exactly with "In concise commonsense terms,".',
    ),
    "R3": (
        'Begin exactly with "First," and connect later steps with "then".',
        'Begin exactly with "Begin by" and connect later steps with "after that".',
        'Begin exactly with "Step one is to" and explicitly name the next step.',
        'Begin exactly with "To complete the task," and make the ordered chain explicit.',
        'Begin exactly with "In order," and preserve the action sequence.',
    ),
}


def normalized_language(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def enforce_variant_prefix(instruction: str, variant: str) -> str:
    required = re.search(r'Begin exactly with "([^"]+)"', variant)
    if required and not normalized_language(instruction).startswith(normalized_language(required.group(1))):
        return f"{required.group(1)} {instruction}"
    return instruction


def _model_identity(model_path: Path) -> dict:
    required = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise AuditError(f"incomplete pinned Qwen snapshot at {model_path}: {missing}")
    value = {
        "repo": QWEN["repo"],
        "revision": QWEN["revision"],
        "files": {
            name: {"bytes": (model_path / name).stat().st_size, "sha256": sha256_file(model_path / name)}
            for name in required
        },
    }
    value["model_hash"] = sha256_bytes(canonical_json(value).encode())
    return value


def language_test_texts(task: dict) -> set[str]:
    result = set()
    canonical = Path(task["canonical_bddl"])
    for name in task_test_names(task["suite"], task["task_stem"]):
        path = canonical.with_name(name + ".bddl")
        if path.is_file() and "_language_" in name:
            result.add(normalized_language(bddl_language(path)))
    return result


def create_language_candidates(audit_data: dict, output: Path, model_path: Path, gpu: int) -> dict:
    path = output / "language_candidates.json"
    if path.is_file():
        return json.loads(path.read_text())

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    identity = _model_identity(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": gpu},
        attn_implementation="eager",
    ).eval()
    result = {"model": identity, "tasks": {}}
    try:
        for task in audit_data["tasks"]:
            excluded = language_test_texts(task)
            task_rows = {}
            for subtype in SUBDIMENSIONS["language"]:
                if subtype == "R3" and not task["multi_step"]:
                    continue
                candidates = []
                for variant_index, variant in enumerate(LANGUAGE_VARIANTS[subtype], 1):
                    prompt = (
                        f"{LANGUAGE_PROMPTS[subtype]}\n"
                        "Return only the rewritten instruction, without quotes or commentary.\n"
                        f"Variant style: {variant}\n"
                        f"Original instruction: {task['canonical_language']}"
                    )
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                            max_new_tokens=160,
                        )
                    instruction = tokenizer.decode(
                        generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    instruction = " ".join(instruction.strip(" \"'\n\t").split())
                    if (
                        not instruction
                        or normalized_language(instruction) == normalized_language(task["canonical_language"])
                        or normalized_language(instruction) in excluded
                    ):
                        continue
                    instruction = enforce_variant_prefix(instruction, variant)
                    if normalized_language(instruction) in {
                        normalized_language(row["instruction"]) for row in candidates
                    }:
                        continue
                    candidate_index = len(candidates) + 1
                    candidates.append(
                        {
                            "candidate_id": f"{subtype.lower()}-{candidate_index}",
                            "variant_index": variant_index,
                            "instruction": instruction,
                            "prompt": prompt,
                            "prompt_hash": sha256_bytes(prompt.encode()),
                            "model_hash": identity["model_hash"],
                        }
                    )
                    if len(candidates) == 3:
                        break
                if len(candidates) != 3:
                    raise AuditError(f"Qwen produced fewer than three isolated rewrites: {task['task_stem']}/{subtype}")
                task_rows[subtype] = candidates
            result["tasks"][task["task_stem"]] = task_rows
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(path, result)
    return result


def rotate_candidates(items: list[dict], seed: int, count=3) -> list[dict]:
    if not items:
        raise AuditError("candidate pool is empty")
    items = sorted(items, key=canonical_json)
    start = seed % len(items)
    return [dict(items[(start + index) % len(items)]) for index in range(count)]


def parse_camera_name(name: str) -> dict | None:
    match = re.search(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_(\d+)", name)
    if not match:
        return None
    azimuth, elevation, scale, yaw, pitch, initstate = map(int, match.groups())
    return {
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "distance_scale": scale / 100,
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": 0,
        "initstate": initstate,
    }


def _camera_rotation(config: dict):
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler(
        "zyx",
        [
            config.get("azimuth_deg", 0) + config.get("yaw_deg", 0),
            config.get("elevation_deg", 0) + config.get("pitch_deg", 0),
            config.get("roll_deg", 0),
        ],
        degrees=True,
    )


def camera_tuple_distance(left: dict, right: dict) -> float:
    rotation = (_camera_rotation(left).inv() * _camera_rotation(right)).magnitude()
    rotation_deg = math.degrees(rotation)
    scale_points = abs(left.get("distance_scale", 1) - right.get("distance_scale", 1)) * 100
    return max(rotation_deg, scale_points)


def camera_candidates(subtype: str, tests: list[str], seed: int) -> list[dict]:
    test_configs = [
        parsed
        for name in tests
        if (parsed := parse_camera_name(name)) is not None
    ]
    if subtype == "C1":
        pool = [
            {"distance_scale": scale, "azimuth_deg": 0, "elevation_deg": 0, "yaw_deg": 0, "pitch_deg": 0, "roll_deg": 0}
            for scale in (1.06, 1.17, 1.29, 1.41, 1.58, 1.73, 1.89)
        ]
    elif subtype == "C2":
        pool = [
            {"distance_scale": 1.0, "azimuth_deg": azimuth, "elevation_deg": elevation, "yaw_deg": 0, "pitch_deg": 0, "roll_deg": 0}
            for azimuth, elevation in ((-67, -21), (-53, 26), (-37, 18), (-23, -31), (19, 37), (31, -24), (47, 29), (63, -17))
        ]
    else:
        pool = [
            {"distance_scale": 1.0, "azimuth_deg": 0, "elevation_deg": 0, "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}
            for yaw, pitch, roll in ((2, 3, 6), (-3, 5, -7), (5, -4, 8), (-6, -2, 9), (7, 6, -10), (-8, 4, 7))
        ]
    for index, item in enumerate(pool, 1):
        item["candidate_id"] = f"{subtype.lower()}-{index}"
        item["subtype"] = subtype
        item["min_test_distance"] = min(
            (camera_tuple_distance(item, test) for test in test_configs), default=180.0
        )
        item["benchmark_tuple_equal"] = any(
            camera_tuple_distance(item, test) < 1e-9 for test in test_configs
        )
    pool = [item for item in pool if item["min_test_distance"] >= 5 - 1e-9]
    return rotate_candidates(pool, seed)


def noise_parameters(algorithm: str, severity: float, seed: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    if algorithm == "motion_blur":
        return {
            "radius": round(4.25 + 31.5 * severity, 4),
            "sigma": round(1.35 + 18.1 * severity, 4),
            "angle_mode": "per_frame_uniform_-45_45",
        }
    if algorithm == "gaussian_blur":
        return {"sigma": round(0.55 + 8.9 * severity, 4)}
    if algorithm == "zoom_blur":
        return {
            "minimum": 1.0,
            "maximum": round(1.055 + 0.47 * severity, 4),
            "step": round(0.007 + 0.019 * severity, 4),
        }
    if algorithm == "fog":
        return {"alpha": round(0.35 + 4.4 * severity, 4), "beta": round(3.15 - 1.7 * severity, 4)}
    return {
        "sigma": round(0.35 + 2.0 * severity, 4),
        "delta": int(1 + 4 * severity),
        "iterations": int(rng.choice((2, 3))),
    }


def noise_candidates(subtype: str, source_ordinal: int, seed: int) -> list[dict]:
    import numpy as np

    algorithm = NOISE_ORDER[SUBDIMENSIONS["noise"].index(subtype)]
    center = 0.12 + 0.76 * ((source_ordinal // len(NOISE_ORDER)) % 11) / 10
    severities = np.clip(center + np.asarray((-0.041, 0.017, 0.063)), 0.05, 0.95)
    result = []
    for index, severity in enumerate(severities, 1):
        parameters = noise_parameters(algorithm, float(severity), stable_seed(seed, index))
        result.append(
            {
                "candidate_id": f"{algorithm}-{index}",
                "algorithm": algorithm,
                "severity": round(float(severity), 6),
                "parameters": parameters,
                "benchmark_tuple_equal": False,
                "parameter_source": "training-only continuous range; excludes Appendix A Table 5 tuples",
            }
        )
    return result


def assign_language_subtypes(sources: list[dict]) -> dict[tuple[int, str], str]:
    counts = Counter()
    result = {}
    for row in sorted(sources, key=lambda item: item["source_ordinal"]):
        allowed = ["R1", "R2"] + (["R3"] if row["multi_step"] else [])
        minimum = min(counts[name] for name in allowed)
        tied = [name for name in allowed if counts[name] == minimum]
        subtype = tied[row["source_ordinal"] % len(tied)]
        counts[subtype] += 1
        result[(row["task_index"], row["source_demo"])] = subtype
    return result


def flattened_sources(audit_data: dict) -> list[dict]:
    rows = []
    for task in audit_data["tasks"]:
        for demo in task["demos"]:
            rows.append({**{key: value for key, value in task.items() if key != "demos"}, **demo})
    return sorted(rows, key=lambda row: row["source_ordinal"])


@lru_cache(maxsize=None)
def _variant_donors(suite: str, kind: str, base_problem: str) -> tuple[Path, ...]:
    folder = ROOT / "libero/libero/bddl_files" / suite
    tests = set(benchmark_map()[suite])
    donors = []
    marker = f"_{kind}_"
    for variant in folder.glob(f"*{marker}*.bddl"):
        if variant.stem in tests:
            continue
        canonical = variant.with_name(variant.stem.rsplit(marker, 1)[0] + ".bddl")
        if canonical.is_file() and bddl_problem(canonical).lower() == base_problem.lower():
            donors.append(variant.resolve())
    return tuple(sorted(donors))


@lru_cache(maxsize=None)
def _benchmark_bddl_hashes(suite: str) -> frozenset[str]:
    folder = ROOT / "libero/libero/bddl_files" / suite
    paths = (folder / f"{name}.bddl" for name in benchmark_map()[suite])
    return frozenset(sha256_file(path) for path in paths if path.is_file())


def _xml_light_signature(path: Path) -> dict[str, dict]:
    fields = ("diffuse", "dir", "specular", "castshadow")
    return {
        element.get("name", f"light-{index}"): {field: element.get(field) for field in fields}
        for index, element in enumerate(ET.parse(path).getroot().iter("light"))
    }


@lru_cache(maxsize=None)
def _light_bddl_changed_fields(canonical_name: str, variant_name: str) -> frozenset[str]:
    canonical_problem = bddl_problem(Path(canonical_name)).lower()
    variant_problem = bddl_problem(Path(variant_name))
    if canonical_problem not in BASE_SCENES or not variant_problem.lower().startswith(canonical_problem + "_"):
        return frozenset()
    scenes = ROOT / "libero/libero/assets/scenes"
    source = _xml_light_signature(scenes / BASE_SCENES[canonical_problem])
    target_path = scenes / "lights" / f"{variant_problem[len(canonical_problem) + 1:]}.xml"
    if not target_path.is_file():
        return frozenset()
    target = _xml_light_signature(target_path)
    return frozenset(
        field
        for name in set(source) | set(target)
        for field in LIGHT_FIELDS.values()
        if source.get(name, {}).get(field) != target.get(name, {}).get(field)
    )


@lru_cache(maxsize=None)
def _generated_scene_candidates_cached(
    suite: str, canonical_name: str, setting: str, subtype: str
) -> tuple[dict, ...]:
    canonical = Path(canonical_name)
    kind = "tb" if subtype == "B1" else "table" if subtype == "B2" else "light"
    donors = _variant_donors(suite, kind, bddl_problem(canonical))
    rows = []
    for donor in donors:
        if setting == "light" and LIGHT_FIELDS[subtype] not in _light_bddl_changed_fields(
            canonical_name, str(donor)
        ):
            continue
        text = re.sub(
            r"(\(define\s+\(problem\s+)[^)]+(\))",
            lambda match: match.group(1) + bddl_problem(donor) + match.group(2),
            canonical.read_text(),
            count=1,
            flags=re.IGNORECASE,
        )
        digest = sha256_bytes(text.encode())
        if digest in _benchmark_bddl_hashes(suite):
            continue
        row = {
            "candidate_id": f"generated-{donor.stem}",
            "bddl_text": text,
            "bddl_sha256": digest,
            "problem": bddl_problem(donor),
            "donor_bddl": str(donor),
            "donor_bddl_sha256": sha256_file(donor),
            "benchmark_bddl_excluded": True,
        }
        if setting == "background":
            row["variant"] = "scene_and_surface" if subtype == "B1" else "surface_only"
        else:
            row["required_xml_field"] = LIGHT_FIELDS[subtype]
        rows.append(row)
        if len(rows) == 3:
            break
    return tuple(rows)


def _generated_scene_candidates(task: dict, setting: str, subtype: str) -> list[dict]:
    return copy.deepcopy(
        _generated_scene_candidates_cached(
            task["suite"], task["canonical_bddl"], setting, subtype
        )
    )


def _added_objects(base_spec, variant_spec) -> list[tuple[str, object]]:
    extras = []
    by_region = {region.region_name: region for region in variant_spec.region_infos}
    workspace_prefix = variant_spec.workspace_name + "_"
    for category, count in sorted(variant_spec.object_num_info.items()):
        base_count = base_spec.object_num_info.get(category, 0)
        for index in range(base_count + 1, count + 1):
            name = f"{category}_{index}"
            init = next(
                (
                    state
                    for state in variant_spec.init_states
                    if len(state) >= 3 and state[0].lower() == "on" and state[1] == name
                ),
                None,
            )
            if init is None or not init[2].startswith(workspace_prefix):
                continue
            region = by_region.get(init[2][len(workspace_prefix) :])
            if region is not None:
                extras.append((category, region))
    return extras


def _distractor_centroid(region, extra_index: int, layout_index: int) -> tuple[float, float]:
    x, y = region.region_centroid_xy
    points = ((x, y), (-x, -y), (y, -x), (-y, x), (0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25))
    unique = []
    for point in points:
        point = tuple(max(-0.4, min(0.4, float(value))) for value in point)
        if point not in unique:
            unique.append(point)
    return unique[(layout_index * 2 + extra_index) % len(unique)]


@lru_cache(maxsize=None)
def _generated_object_candidates_cached(
    suite: str, canonical_name: str, extra_count: int
) -> tuple[dict, ...]:
    from libero.randomizer.bddl_operators import RegionInfo, bddl_spec2str, load_bddl_file

    canonical = Path(canonical_name)
    target = load_bddl_file(str(canonical))
    donors = _variant_donors(suite, "add", bddl_problem(canonical))
    rows = []
    for donor in donors:
        donor_canonical = donor.with_name(donor.stem.rsplit("_add_", 1)[0] + ".bddl")
        extras = _added_objects(load_bddl_file(str(donor_canonical)), load_bddl_file(str(donor)))
        if not extras:
            continue
        for layout_index in range(3):
            spec = copy.deepcopy(target)
            original_objects_of_interest = list(spec.task_info.objects_of_interest)
            valid_objects_of_interest = {
                f"{category}_{index}"
                for counts in (spec.fixture_num_info, spec.object_num_info)
                for category, count in counts.items()
                for index in range(1, count + 1)
            }
            spec.task_info.objects_of_interest = [
                name for name in original_objects_of_interest if name in valid_objects_of_interest
            ]
            for extra_index in range(extra_count):
                category, region = extras[extra_index % len(extras)]
                object_index = spec.object_num_info.get(category, 0) + 1
                spec.object_num_info[category] = object_index
                region_name = f"replica_distractor_{len(rows) + 1}_{extra_index + 1}"
                spec.region_infos.append(
                    RegionInfo(
                        region_centroid_xy=_distractor_centroid(region, extra_index, layout_index),
                        region_name=region_name,
                        target_name=spec.workspace_name,
                        region_half_len=region.region_half_len,
                        yaw_rotation=region.yaw_rotation,
                    )
                )
                spec.init_states.append(
                    ("On", f"{category}_{object_index}", f"{spec.workspace_name}_{region_name}")
                )
            text = bddl_spec2str(spec)
            text = text.replace(
                bddl_block(text, "obj_of_interest"),
                bddl_block(canonical.read_text(), "obj_of_interest"),
                1,
            )
            digest = sha256_bytes(text.encode())
            if digest in _benchmark_bddl_hashes(suite):
                continue
            rows.append(
                {
                    "candidate_id": f"generated-{donor.stem}-{extra_count}-layout{layout_index + 1}",
                    "bddl_text": text,
                    "bddl_sha256": digest,
                    "problem": bddl_problem(canonical),
                    "donor_bddl": str(donor),
                    "donor_bddl_sha256": sha256_file(donor),
                    "benchmark_bddl_excluded": True,
                    "extra_objects": extra_count,
                }
            )
            if len(rows) == 3:
                return tuple(rows)
    return tuple(rows)


def _generated_object_candidates(task: dict, extra_count: int) -> list[dict]:
    return copy.deepcopy(
        _generated_object_candidates_cached(task["suite"], task["canonical_bddl"], extra_count)
    )


def _bddl_candidates(task: dict, setting: str, subtype: str, source_ordinal: int) -> list[dict]:
    canonical = Path(task["canonical_bddl"])
    tests = set(task_test_names(task["suite"], task["task_stem"]))
    if setting == "objects":
        extra = source_ordinal % 3 + 1
        base_count = len(bddl_objects(canonical))
        paths = [
            path
            for path in canonical.parent.glob(f"{task['task_stem']}_add_*.bddl")
            if path.stem not in tests and len(bddl_objects(path)) - base_count == extra
        ]
    elif setting == "background":
        suffix = "tb" if subtype == "B1" else "table"
        paths = [
            path
            for path in canonical.parent.glob(f"{task['task_stem']}_{suffix}_*.bddl")
            if path.stem not in tests
        ]
    else:
        paths = [
            path
            for path in canonical.parent.glob(f"{task['task_stem']}_light_*.bddl")
            if path.stem not in tests
        ]
    rows = []
    for path in paths:
        if setting == "light" and LIGHT_FIELDS[subtype] not in _light_bddl_changed_fields(
            str(canonical), str(path)
        ):
            continue
        row = {
            "candidate_id": path.stem,
            "bddl": str(path.resolve()),
            "bddl_sha256": sha256_file(path),
            "problem": bddl_problem(path),
            "benchmark_bddl_excluded": True,
        }
        if setting == "objects":
            row["extra_objects"] = source_ordinal % 3 + 1
        if setting == "background":
            row["variant"] = "scene_and_surface" if subtype == "B1" else "surface_only"
        if setting == "light":
            row["required_xml_field"] = LIGHT_FIELDS[subtype]
        rows.append(row)
    if len(rows) < 3:
        if setting == "objects":
            rows.extend(_generated_object_candidates(task, source_ordinal % 3 + 1))
        else:
            rows.extend(_generated_scene_candidates(task, setting, subtype))
    return rotate_candidates(rows, stable_seed(task["task_index"], source_ordinal, setting, subtype))


def build_manifest_rows(audit_data: dict, languages: dict) -> list[dict]:
    sources = flattened_sources(audit_data)
    language_subtypes = assign_language_subtypes(sources)
    rows = []
    for source in sources:
        tests = task_test_names(source["suite"], source["task_stem"])
        source_key = (source["task_index"], source["source_demo"])
        for setting in SETTINGS:
            if setting == "objects":
                subtype = "O1"
            elif setting in ("background", "light", "camera", "noise"):
                values = SUBDIMENSIONS[setting]
                subtype = values[source["source_ordinal"] % len(values)]
            else:
                subtype = language_subtypes[source_key]

            seed = stable_seed(PROTOCOL_VERSION, source["source_ordinal"], setting)
            if setting in {"objects", "background", "light"}:
                candidates = _bddl_candidates(source, setting, subtype, source["source_ordinal"])
            elif setting == "camera":
                candidates = camera_candidates(subtype, tests, seed)
            elif setting == "language":
                candidates = [dict(item) for item in languages["tasks"][source["task_stem"]][subtype]]
                for candidate in candidates:
                    candidate.update(
                        {
                            "subtype": subtype,
                            "benchmark_language_excluded": normalized_language(candidate["instruction"])
                            not in language_test_texts(source),
                        }
                    )
            else:
                candidates = noise_candidates(subtype, source["source_ordinal"], seed)
            for index, candidate in enumerate(candidates, 1):
                candidate["candidate_index"] = index

            safe_demo = re.sub(r"[^a-zA-Z0-9_-]+", "-", source["source_demo"])
            job_id = f"t{source['task_index']:02d}-{safe_demo}-{setting}"
            temporal_mode = TEMPORAL_MODES[subtype] if subtype in TEMPORAL_MODES else TEMPORAL_MODES[setting]
            affected = {
                "objects": ["front_rgb", "wrist_rgb", "front_depth", "wrist_depth", "scene"],
                "background": ["front_rgb", "wrist_rgb", "front_depth", "wrist_depth", "scene"],
                "light": ["front_rgb", "wrist_rgb", "scene"],
                "camera": ["front_rgb", "front_depth", "front_extrinsics"],
                "language": ["language"],
                "noise": ["front_rgb"],
            }[setting]
            rows.append(
                {
                    "job_id": job_id,
                    "task_index": source["task_index"],
                    "suite": source["suite"],
                    "task": source["task"],
                    "task_stem": source["task_stem"],
                    "multi_step": source["multi_step"],
                    "canonical_language": source["canonical_language"],
                    "canonical_bddl": source["canonical_bddl"],
                    "canonical_bddl_sha256": source["canonical_bddl_sha256"],
                    "source_file": source["source_file"],
                    "source_revision": SOURCE["revision"],
                    "source_demo": source["source_demo"],
                    "source_ordinal": source["source_ordinal"],
                    "source_actions": source["actions"],
                    "saved_frames": source["saved_frames"],
                    "action_sha256": source["action_sha256"],
                    "keep_mask_sha256": source["keep_mask_sha256"],
                    "source_model_xml_sha256": source["source_model_xml_sha256"],
                    "setting": setting,
                    "subdimensions": [subtype],
                    "temporal_mode": temporal_mode,
                    "affected_modalities": affected,
                    "seed": seed,
                    "frame_seed_rule": "stable_seed(job_seed, source_timestep)",
                    "candidates": candidates,
                    "output_path": f"episodes/task-{source['task_index']:02d}/{job_id}.hdf5",
                }
            )
    return rows


def validate_manifest_rows(rows: list[dict], expected_sources: int | None = None) -> None:
    if expected_sources is not None and len(rows) != expected_sources * len(SETTINGS):
        raise AuditError(f"manifest has {len(rows)} jobs, expected {expected_sources * len(SETTINGS)}")
    if len({row["job_id"] for row in rows}) != len(rows):
        raise AuditError("manifest job_id values are not unique")
    grouped = defaultdict(set)
    for row in rows:
        grouped[(row["task_index"], row["source_demo"])].add(row["setting"])
        if len(row["candidates"]) != 3 or [item["candidate_index"] for item in row["candidates"]] != [1, 2, 3]:
            raise AuditError(f"{row['job_id']}: must contain three ordered candidates")
        if row["subdimensions"] == ["R3"] and not row["multi_step"]:
            raise AuditError(f"{row['job_id']}: R3 assigned to a single-step task")
        for candidate in row["candidates"]:
            if not candidate.get("candidate_id"):
                raise AuditError(f"{row['job_id']}: candidate_id is missing")
            if row["setting"] in {"objects", "background", "light"}:
                tests = set(benchmark_map()[row["suite"]])
                bddl_path = Path(candidate["bddl"]) if "bddl" in candidate else None
                donor_path = Path(candidate["donor_bddl"]) if "donor_bddl" in candidate else None
                generated_hash = sha256_bytes(candidate.get("bddl_text", "").encode())
                if (
                    not candidate["benchmark_bddl_excluded"]
                    or (bddl_path is not None and bddl_path.stem in tests)
                    or (donor_path is not None and donor_path.stem in tests)
                    or ("bddl_text" in candidate and generated_hash in _benchmark_bddl_hashes(row["suite"]))
                ):
                    raise AuditError(f"{row['job_id']}: benchmark BDDL leak")
            if row["setting"] == "camera" and (
                candidate["benchmark_tuple_equal"] or candidate["min_test_distance"] < 5 - 1e-9
            ):
                raise AuditError(f"{row['job_id']}: camera test tuple leak")
            if row["setting"] == "language" and (
                not candidate["benchmark_language_excluded"]
                or not candidate.get("prompt_hash")
                or not candidate.get("model_hash")
            ):
                raise AuditError(f"{row['job_id']}: benchmark language leak")
            if row["setting"] == "noise" and (
                candidate["benchmark_tuple_equal"] or "training-only" not in candidate.get("parameter_source", "")
            ):
                raise AuditError(f"{row['job_id']}: benchmark noise tuple leak")
    if any(settings != set(SETTINGS) for settings in grouped.values()):
        raise AuditError("each source demo must have exactly six setting jobs")
    covered = {code for row in rows for code in row["subdimensions"]}
    expected = {code for values in SUBDIMENSIONS.values() for code in values}
    if covered != expected:
        raise AuditError(f"manifest subdimension coverage mismatch: {sorted(expected - covered)} missing")


def manifest(args) -> None:
    output = args.output.resolve()
    audit_path = output / "audit.json"
    if not audit_path.is_file():
        raise AuditError("run audit before manifest")
    audit_data = json.loads(audit_path.read_text())
    if not audit_data.get("complete"):
        raise AuditError("audit is incomplete")
    configure_libero(output)
    languages = create_language_candidates(audit_data, output, args.qwen_path.expanduser().resolve(), args.gpu)
    rows = build_manifest_rows(audit_data, languages)
    validate_manifest_rows(rows, audit_data["source_demo_count"])
    path = output / "manifest.jsonl"
    content = "".join(canonical_json(row) + "\n" for row in rows)
    if path.is_file() and path.read_text() != content:
        raise AuditError("existing manifest differs from deterministic regeneration")
    atomic_text(path, content)
    manifest_hash = sha256_bytes(content.encode())
    atomic_json(
        output / "manifest_summary.json",
        {
            "jobs": len(rows),
            "source_demos": audit_data["source_demo_count"],
            "manifest_hash": manifest_hash,
            "per_setting": Counter(row["setting"] for row in rows),
            "per_subdimension": Counter(code for row in rows for code in row["subdimensions"]),
        },
    )
    print(canonical_json({"manifest": "complete", "jobs": len(rows), "sha256": manifest_hash}))


def configure_libero(output: Path) -> None:
    config_dir = output / "runtime/libero_config"
    benchmark = (ROOT / "libero/libero").resolve()
    atomic_json(
        config_dir / "config.yaml",
        {
            "benchmark_root": str(benchmark),
            "bddl_files": str(benchmark / "bddl_files"),
            "init_states": str(benchmark / "init_files"),
            "datasets": str(DEFAULT_SOURCE_ROOT.resolve()),
            "assets": str(benchmark / "assets"),
        },
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_metadata(gpu: int) -> dict:
    packages = {}
    try:
        import importlib.metadata

        packages = {
            name: importlib.metadata.version(name)
            for name in ("h5py", "mujoco", "numpy", "opencv-python", "robosuite", "scipy")
        }
    except importlib.metadata.PackageNotFoundError:
        pass
    return {
        "code_commit": command_output(["git", "rev-parse", "HEAD"]),
        "code_dirty": bool(command_output(["git", "status", "--porcelain", "--untracked-files=no"])),
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpu_index": gpu,
        "gpu": command_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"]
        ),
        "egl": {"MUJOCO_GL": os.environ.get("MUJOCO_GL"), "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM")},
    }


def resolve_source_model_xml(xml: str) -> tuple[str, list[dict]]:
    import robosuite

    root = ET.fromstring(xml)
    libero_assets = (ROOT / "libero/libero/assets").resolve()
    robosuite_root = Path(robosuite.__file__).resolve().parent
    inventory = []
    missing = []
    for element in root.iter():
        original = element.attrib.get("file")
        if not original:
            continue
        path = Path(original).expanduser()
        normalized = original.replace("\\", "/")
        if "/chiliocosm/assets/" in normalized:
            path = (libero_assets / normalized.split("/chiliocosm/assets/", 1)[1]).resolve()
        elif "robosuite" in path.parts:
            index = max(i for i, part in enumerate(path.parts) if part == "robosuite")
            path = robosuite_root.joinpath(*path.parts[index + 1 :]).resolve()
        elif not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.is_file():
            missing.append(original)
            continue
        element.set("file", str(path))
        inventory.append({"kind": element.tag, "id": element.get("name"), "file": str(path)})
    if missing:
        raise AuditError(f"source XML has unresolved assets: {missing[:3]}")
    return ET.tostring(root, encoding="unicode"), sorted(inventory, key=canonical_json)


def load_source_episode(row: dict) -> dict:
    import h5py
    import numpy as np

    with h5py.File(row["source_file"], "r") as handle:
        group = handle[f"data/{row['source_demo']}"]
        xml = group.attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode()
        processed, assets = resolve_source_model_xml(str(xml))
        obs = group.get("obs")
        return {
            "actions": np.asarray(group["actions"], dtype=np.float64),
            "states": np.asarray(group["states"], dtype=np.float64),
            "joint_states": np.asarray(obs["joint_states"], dtype=np.float64) if obs is not None and "joint_states" in obs else None,
            "ee_pos": np.asarray(obs["ee_pos"], dtype=np.float64) if obs is not None and "ee_pos" in obs else None,
            "ee_ori": np.asarray(obs["ee_ori"], dtype=np.float64) if obs is not None and "ee_ori" in obs else None,
            "source_xml": str(xml),
            "source_xml_processed": processed,
            "source_assets": assets,
        }


def execution_bddl(row: dict, candidate: dict, output: Path) -> Path:
    if row["setting"] in {"objects", "background", "light"}:
        if "bddl" in candidate:
            return Path(candidate["bddl"])
        path = output / "runtime/generated_bddl" / f"{row['job_id']}.{candidate['candidate_index']}.bddl"
        atomic_text(path, candidate["bddl_text"])
        return path
    if row["setting"] != "language":
        return Path(row["canonical_bddl"])
    text = Path(row["canonical_bddl"]).read_text()
    text, count = re.subn(
        r"(\(:language\s+)(.+?)(\s*\))",
        lambda match: match.group(1) + candidate["instruction"] + match.group(3),
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    if count != 1:
        raise AttemptFailure("could not replace BDDL language")
    path = output / "runtime/generated_bddl" / f"{row['job_id']}.bddl"
    atomic_text(path, text)
    return path


def make_env(bddl: Path, gpu: int, horizon: int):
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        robots=["Panda"],
        initialization_noise=None,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
        camera_depths=True,
        render_gpu_device_id=gpu,
        control_freq=20,
        horizon=horizon,
        ignore_done=True,
        hard_reset=True,
    )


def seed_environment(env, seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)


def decoded_names(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def named_state_snapshot_from_xml(source: dict) -> dict:
    import numpy as np
    from robosuite.utils.binding_utils import MjSim

    sim = MjSim.from_xml_string(source["source_xml_processed"])
    try:
        sim.set_state_from_flattened(source["states"][0])
        sim.forward()
        joints = {
            name: {
                "qpos": np.asarray(sim.data.get_joint_qpos(name)).tolist(),
                "qvel": np.asarray(sim.data.get_joint_qvel(name)).tolist(),
            }
            for name in decoded_names(sim.model.joint_names)
        }
        mocap = {}
        for name in decoded_names(sim.model.body_names):
            body_id = sim.model.body_name2id(name)
            if sim.model.body_mocapid[body_id] >= 0:
                mocap[name] = {
                    "pos": np.asarray(sim.data.get_mocap_pos(name)).tolist(),
                    "quat": np.asarray(sim.data.get_mocap_quat(name)).tolist(),
                }
        return {"time": float(sim.data.time), "joints": joints, "mocap": mocap}
    finally:
        sim.free()


def apply_named_state(snapshot: dict, env) -> dict:
    import numpy as np

    source_joints = set(snapshot["joints"])
    target_joints = set(decoded_names(env.sim.model.joint_names))
    common_joints = sorted(source_joints & target_joints)
    for name in common_joints:
        env.sim.data.set_joint_qpos(name, np.asarray(snapshot["joints"][name]["qpos"]))
        env.sim.data.set_joint_qvel(name, np.asarray(snapshot["joints"][name]["qvel"]))
    target_bodies = set(decoded_names(env.sim.model.body_names))
    common_mocap = []
    for name in sorted(set(snapshot["mocap"]) & target_bodies):
        body_id = env.sim.model.body_name2id(name)
        if env.sim.model.body_mocapid[body_id] >= 0:
            env.sim.data.set_mocap_pos(name, np.asarray(snapshot["mocap"][name]["pos"]))
            env.sim.data.set_mocap_quat(name, np.asarray(snapshot["mocap"][name]["quat"]))
            common_mocap.append(name)
    env.sim.data.time = snapshot["time"]
    env.sim.forward()
    return {
        "method": "joint_and_mocap_body_names",
        "common_joints": common_joints,
        "common_mocap_bodies": common_mocap,
        "source_only_joints": sorted(source_joints - target_joints),
        "target_only_joints": sorted(target_joints - source_joints),
    }


def set_source_initial_state(env, row: dict, source: dict) -> dict:
    initial = source["states"][0]
    if row["setting"] in {"camera", "language", "noise"}:
        env.reset_from_xml_string(source["source_xml_processed"])
        env.sim.reset()
        if env.sim.get_state().flatten().shape != initial.shape:
            raise AttemptFailure("source XML state layout differs from source initial state")
        env.set_state(initial)
        env.sim.forward()
        return {"method": "source_xml_flat_state", "state_size": int(initial.size)}
    if row["setting"] == "objects":
        return apply_named_state(named_state_snapshot_from_xml(source), env)
    if env.sim.get_state().flatten().shape != initial.shape:
        raise AttemptFailure("environment variant changed MuJoCo state layout")
    env.set_state(initial)
    env.sim.forward()
    return {"method": "flat_state_exact", "state_size": int(initial.size)}


def apply_camera_perturbation(env, candidate: dict) -> dict:
    import numpy as np
    from scipy.spatial.transform import Rotation

    camera_id = env.sim.model.camera_name2id("agentview")
    before_pos = np.asarray(env.sim.model.cam_pos[camera_id], dtype=float).copy()
    before_quat = np.asarray(env.sim.model.cam_quat[camera_id], dtype=float).copy()
    position = before_pos.copy()
    current = Rotation.from_quat([before_quat[1], before_quat[2], before_quat[3], before_quat[0]])
    subtype = candidate["subtype"]
    pivot = np.asarray([0.0, 0.0, 0.8])
    if subtype == "C1":
        position = pivot + (position - pivot) * candidate["distance_scale"]
        rotation = current
    elif subtype == "C2":
        turn = Rotation.from_euler(
            "zy", [candidate["azimuth_deg"], -candidate["elevation_deg"]], degrees=True
        )
        position = turn.apply(position - pivot) + pivot
        rotation = turn * current
    else:
        turn = Rotation.from_euler(
            "zyx", [candidate["yaw_deg"], candidate["pitch_deg"], candidate["roll_deg"]], degrees=True
        )
        rotation = turn * current
    xyzw = rotation.as_quat()
    quaternion = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    env.sim.model.cam_pos[camera_id] = position
    env.sim.model.cam_quat[camera_id] = quaternion
    env.sim.forward()
    return {
        "before_position": before_pos.tolist(),
        "after_position": position.tolist(),
        "before_quaternion_wxyz": before_quat.tolist(),
        "after_quaternion_wxyz": quaternion.tolist(),
        "position_changed": not np.allclose(position, before_pos),
        "orientation_changed": not np.allclose(quaternion, before_quat) and not np.allclose(quaternion, -before_quat),
    }


def as_rgb(image):
    import numpy as np

    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image * (255 if image.max(initial=0) <= 1 else 1), 0, 255).astype(np.uint8)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise AttemptFailure(f"unexpected RGB shape: {image.shape}")
    return image


def _zoom_blur(image, maximum: float, step: float):
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    layers = [image.astype(np.float32)]
    for factor in np.arange(1 + step, maximum + step / 2, step):
        resized = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
        y, x = (resized.shape[0] - height) // 2, (resized.shape[1] - width) // 2
        layers.append(resized[y : y + height, x : x + width])
    return np.mean(layers, axis=0).astype(np.uint8)


def apply_noise(image, config: dict, seed: int):
    import cv2
    import numpy as np

    image = as_rgb(image)
    rng = np.random.default_rng(seed)
    algorithm = config["algorithm"]
    params = config["parameters"]
    if algorithm == "motion_blur":
        size = max(3, int(round(params["radius"]))) | 1
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2] = cv2.getGaussianKernel(size, max(params["sigma"], 0.1)).ravel()
        angle = float(rng.uniform(-45, 45))
        matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1)
        kernel = cv2.warpAffine(kernel, matrix, (size, size))
        kernel /= max(float(kernel.sum()), 1e-9)
        return cv2.filter2D(image, -1, kernel).astype(np.uint8)
    if algorithm == "gaussian_blur":
        return cv2.GaussianBlur(image, (0, 0), params["sigma"]).astype(np.uint8)
    if algorithm == "zoom_blur":
        return _zoom_blur(image, params["maximum"], params["step"])
    if algorithm == "fog":
        field = rng.normal(size=image.shape[:2]).astype(np.float32)
        field = cv2.GaussianBlur(field, (0, 0), max(1.0, params["beta"] * 4))
        field = (field - field.min()) / max(float(field.max() - field.min()), 1e-6)
        alpha = min(0.8, params["alpha"] / 6)
        fog = np.repeat((field * 255)[..., None], 3, axis=2)
        return np.clip(image * (1 - alpha) + fog * alpha, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(image, (0, 0), params["sigma"])
    delta = params["delta"]
    grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    result = blurred
    for _ in range(params["iterations"]):
        dx = cv2.GaussianBlur(
            rng.uniform(-delta, delta, image.shape[:2]).astype(np.float32), (0, 0), 1
        )
        dy = cv2.GaussianBlur(
            rng.uniform(-delta, delta, image.shape[:2]).astype(np.float32), (0, 0), 1
        )
        result = cv2.remap(
            result,
            (grid_x + dx).astype(np.float32),
            (grid_y + dy).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
    return result.astype(np.uint8)


def metric_depth(sim, raw_depth):
    import numpy as np
    from robosuite.utils import camera_utils

    raw = np.squeeze(np.asarray(raw_depth, dtype=np.float32))
    valid = np.isfinite(raw) & (raw >= 0) & (raw <= 1)
    depth = camera_utils.get_real_depth_map(sim, np.where(valid, raw, 1).astype(np.float32)).astype(np.float32)
    depth[~valid] = 0
    return depth, valid


def asset_inventory(model_xml: str) -> list[dict]:
    result = []
    for element in ET.fromstring(model_xml).iter():
        if file_name := element.get("file"):
            result.append({"kind": element.tag, "id": element.get("name"), "file": file_name})
    return sorted(result, key=canonical_json)


def light_signature(model_xml: str) -> dict[str, dict]:
    fields = ("diffuse", "dir", "specular", "castshadow", "pos", "directional")
    return {
        element.get("name", f"light-{index}"): {field: element.get(field) for field in fields}
        for index, element in enumerate(ET.fromstring(model_xml).iter("light"))
    }


def changed_light_fields(source_xml: str, model_xml: str) -> list[str]:
    source = light_signature(source_xml)
    target = light_signature(model_xml)
    changed = set()
    for name in set(source) | set(target):
        for field in ("diffuse", "dir", "specular", "castshadow", "pos", "directional"):
            if source.get(name, {}).get(field) != target.get(name, {}).get(field):
                changed.add(field)
    return sorted(changed)


def perturbation_evidence(
    row: dict,
    candidate: dict,
    source: dict,
    model_xml: str,
    camera_actual: dict | None,
    bddl: Path,
) -> dict:
    setting = row["setting"]
    subtype = row["subdimensions"][0]
    if setting == "objects":
        canonical_names = {name for name, _ in bddl_objects(Path(row["canonical_bddl"]))}
        candidate_names = {name for name, _ in bddl_objects(bddl)}
        added = sorted(candidate_names - canonical_names)
        if len(added) != candidate["extra_objects"]:
            raise AttemptFailure("object candidate did not add the declared distractor count")
        actual = {"extra_object_count": len(added), "extra_object_names": added}
    elif setting == "background":
        source_files = {item["file"] for item in source["source_assets"]}
        target_files = {item["file"] for item in asset_inventory(model_xml)}
        changed = sorted(target_files - source_files)
        if not changed:
            raise AttemptFailure("background candidate did not change any model asset")
        actual = {"variant": candidate["variant"], "changed_assets": changed}
    elif setting == "light":
        changed = changed_light_fields(source["source_xml_processed"], model_xml)
        required = LIGHT_FIELDS[subtype]
        if required not in changed:
            raise AttemptFailure(f"light candidate did not change required XML field {required}")
        actual = {
            "changed_fields": changed,
            "source_lights": light_signature(source["source_xml_processed"]),
            "model_lights": light_signature(model_xml),
        }
    elif setting == "camera":
        if subtype == "C1" and (not camera_actual["position_changed"] or camera_actual["orientation_changed"]):
            raise AttemptFailure("C1 must change distance without changing orientation")
        if subtype == "C2" and not (camera_actual["position_changed"] and camera_actual["orientation_changed"]):
            raise AttemptFailure("C2 must change spherical position and orientation")
        if subtype == "C3" and (camera_actual["position_changed"] or not camera_actual["orientation_changed"]):
            raise AttemptFailure("C3 must change orientation at a fixed position")
        actual = camera_actual
    elif setting == "language":
        actual = {
            "instruction": candidate["instruction"],
            "prompt_hash": candidate["prompt_hash"],
            "model_hash": candidate["model_hash"],
        }
    else:
        actual = {
            "algorithm": candidate["algorithm"],
            "severity": candidate["severity"],
            "parameters": candidate["parameters"],
        }
    return {
        "setting": setting,
        "subdimensions": row["subdimensions"],
        "temporal_mode": row["temporal_mode"],
        "affected_modalities": row["affected_modalities"],
        "candidate_index": candidate["candidate_index"],
        "candidate_id": candidate["candidate_id"],
        "candidate": candidate,
        "actual_parameters": actual,
        "episode_seed": row["seed"],
        "frame_seed_rule": row["frame_seed_rule"],
        "protocol_hash": protocol()["protocol_hash"],
        "manifest_row_hash": sha256_bytes(canonical_json(row).encode()),
        "candidate_config_hash": sha256_bytes(canonical_json(candidate).encode()),
        "source_model_xml_hash": sha256_bytes(source["source_xml"].encode()),
        "model_xml_hash": sha256_bytes(model_xml.encode()),
        "bddl_hash": sha256_file(bddl),
    }


def frame_dataset(group, name: str, data) -> None:
    import numpy as np

    data = np.asarray(data)
    if not data.ndim or not len(data):
        group.create_dataset(name, data=data)
    else:
        group.create_dataset(name, data=data, chunks=(1, *data.shape[1:]), compression="lzf")


def text_dataset(group, name: str, value: str) -> None:
    import h5py

    group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))


def write_episode(path: Path, payload: dict) -> None:
    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w") as handle:
        observations = handle.create_group("observations")
        for name in (
            "front_rgb",
            "wrist_rgb",
            "front_depth_m",
            "wrist_depth_m",
            "depth_valid",
            "state",
            "joint_state",
        ):
            frame_dataset(observations, name, payload[name])
        if "front_rgb_clean" in payload:
            frame_dataset(observations, "front_rgb_clean", payload["front_rgb_clean"])
        frame_dataset(handle.create_group("sim"), "state", payload["sim_state"])
        for name in ("actions", "reward", "success", "terminal"):
            frame_dataset(handle, name, payload[name])

        source = handle.create_group("source")
        for name in ("actions_full", "keep_mask", "timestep", "initial_state"):
            frame_dataset(source, name, payload[name])

        camera = handle.create_group("camera")
        for short in ("front", "wrist"):
            group = camera.create_group(short)
            for name in ("K", "T_world_cam", "T_cam_world"):
                frame_dataset(group, name, payload["camera"][short][name])
            group.attrs["near_m"] = payload["near_m"]
            group.attrs["far_m"] = payload["far_m"]

        language = handle.create_group("language")
        text_dataset(language, "canonical", payload["row"]["canonical_language"])
        text_dataset(language, "instruction", payload["instruction"])

        perturbation = handle.create_group("perturbation")
        text_dataset(perturbation, "setting", payload["perturbation"]["setting"])
        text_dataset(perturbation, "subdimensions_json", canonical_json(payload["perturbation"]["subdimensions"]))
        text_dataset(perturbation, "temporal_mode", payload["perturbation"]["temporal_mode"])
        text_dataset(perturbation, "parameters_json", canonical_json(payload["perturbation"]))
        frame_dataset(perturbation, "frame_seeds", payload["frame_seeds"])
        perturbation.attrs["episode_seed"] = payload["row"]["seed"]
        perturbation.attrs["candidate_index"] = payload["candidate"]["candidate_index"]

        metadata = handle.create_group("metadata")
        for name, value in (
            ("bddl", payload["bddl"]),
            ("source_xml", payload["source_xml"]),
            ("model_xml", payload["model_xml"]),
            ("manifest_json", canonical_json(payload["row"])),
            ("candidate_json", canonical_json(payload["candidate"])),
            ("runtime_json", canonical_json(payload["runtime"])),
            ("state_mapping_json", canonical_json(payload["state_mapping"])),
            ("replay_error_json", canonical_json(payload["replay_error"])),
        ):
            text_dataset(metadata, name, value)

        handle.attrs["format"] = PROTOCOL_VERSION
        handle.attrs["job_id"] = payload["row"]["job_id"]
        handle.attrs["source_demo"] = payload["row"]["source_demo"]
        handle.attrs["executed_full_actions"] = len(payload["actions_full"])
        handle.attrs["saved_actions"] = len(payload["actions"])
        handle.attrs["final_success"] = True
        handle.attrs["depth_unit"] = "meter"
        handle.attrs["camera_convention"] = "OpenCV x-right y-down z-forward"
        handle.flush()
    os.replace(partial, path)


def run_attempt(row: dict, candidate: dict, output: Path, runtime: dict, gpu: int) -> dict:
    import mujoco
    import numpy as np
    from robosuite.utils import camera_utils

    source = load_source_episode(row)
    keep = keep_action_mask(source["actions"])
    if (
        len(source["actions"]) != row["source_actions"]
        or action_key(source["actions"])[1] != row["action_sha256"]
        or sha256_bytes(keep.tobytes()) != row["keep_mask_sha256"]
        or sha256_bytes(source["source_xml"].encode()) != row["source_model_xml_sha256"]
    ):
        raise AttemptFailure("source action, keep mask, or model XML fingerprint changed")
    if sha256_file(Path(row["canonical_bddl"])) != row["canonical_bddl_sha256"]:
        raise AttemptFailure("canonical BDDL fingerprint changed")
    bddl = execution_bddl(row, candidate, output)
    if row["setting"] in {"objects", "background", "light"} and sha256_file(bddl) != candidate["bddl_sha256"]:
        raise AttemptFailure("candidate BDDL fingerprint changed")
    env = make_env(bddl, gpu, len(source["actions"]) + 5)
    try:
        seed_environment(env, stable_seed(row["seed"], candidate["candidate_index"], "environment"))
        env.env.reset()
        state_mapping = set_source_initial_state(env, row, source)
        camera_actual = apply_camera_perturbation(env, candidate) if row["setting"] == "camera" else None

        render_context = env.sim._render_context_offscreen
        render_context.gl_ctx.make_current()
        for texture_id in range(env.sim.model.ntex):
            mujoco.mjr_uploadTexture(env.sim.model._model, render_context.con, texture_id)

        model_xml = env.sim.model.get_xml()
        evidence = perturbation_evidence(row, candidate, source, model_xml, camera_actual, bddl)
        near_m = float(env.sim.model.vis.map.znear * env.sim.model.stat.extent)
        far_m = float(env.sim.model.vis.map.zfar * env.sim.model.stat.extent)
        saved = defaultdict(list)
        cameras = {"front": defaultdict(list), "wrist": defaultdict(list)}
        replay_error = {
            "max_sim_state_error": 0.0,
            "max_joint_error": 0.0,
            "max_eef_position_error": 0.0,
            "max_eef_orientation_error": 0.0,
        }
        for timestep, action in enumerate(source["actions"]):
            obs, reward, _, _ = env.step(action)
            if row["setting"] != "objects" and timestep + 1 < len(source["states"]):
                current = env.sim.get_state().flatten()
                if current.shape == source["states"][timestep + 1].shape:
                    replay_error["max_sim_state_error"] = max(
                        replay_error["max_sim_state_error"],
                        float(np.max(np.abs(current - source["states"][timestep + 1]), initial=0)),
                    )
            if source["joint_states"] is not None and timestep < len(source["joint_states"]):
                replay_error["max_joint_error"] = max(
                    replay_error["max_joint_error"],
                    float(np.max(np.abs(obs["robot0_joint_pos"] - source["joint_states"][timestep]), initial=0)),
                )
            if source["ee_pos"] is not None and timestep < len(source["ee_pos"]):
                replay_error["max_eef_position_error"] = max(
                    replay_error["max_eef_position_error"],
                    float(np.max(np.abs(obs["robot0_eef_pos"] - source["ee_pos"][timestep]), initial=0)),
                )
            if source["ee_ori"] is not None and timestep < len(source["ee_ori"]):
                from robosuite.utils import transform_utils

                axis_angle = transform_utils.quat2axisangle(obs["robot0_eef_quat"])
                replay_error["max_eef_orientation_error"] = max(
                    replay_error["max_eef_orientation_error"],
                    float(np.max(np.abs(axis_angle - source["ee_ori"][timestep]), initial=0)),
                )
            if not keep[timestep]:
                continue

            frame_seed = stable_seed(row["seed"], timestep)
            front_clean = as_rgb(obs["agentview_image"])
            front = apply_noise(front_clean, candidate, frame_seed) if row["setting"] == "noise" else front_clean
            wrist = as_rgb(obs["robot0_eye_in_hand_image"])
            front_depth, front_valid = metric_depth(env.sim, obs["agentview_depth"])
            wrist_depth, wrist_valid = metric_depth(env.sim, obs["robot0_eye_in_hand_depth"])
            saved["front_rgb"].append(front)
            saved["wrist_rgb"].append(wrist)
            if row["setting"] == "noise":
                saved["front_rgb_clean"].append(front_clean)
            saved["front_depth_m"].append(front_depth)
            saved["wrist_depth_m"].append(wrist_depth)
            saved["depth_valid"].append(
                np.stack(
                    (
                        front_valid & (front_depth >= near_m) & (front_depth <= far_m),
                        wrist_valid & (wrist_depth >= near_m) & (wrist_depth <= far_m),
                    )
                )
            )
            saved["state"].append(env.env.get_robot_state_vector(obs))
            saved["joint_state"].append(
                np.concatenate((obs["robot0_joint_pos"], obs.get("robot0_gripper_qpos", [])))
            )
            saved["sim_state"].append(env.sim.get_state().flatten())
            saved["reward"].append(reward)
            saved["success"].append(env.check_success())
            saved["frame_seeds"].append(frame_seed)
            for short, camera_name in (("front", "agentview"), ("wrist", "robot0_eye_in_hand")):
                intrinsic = camera_utils.get_camera_intrinsic_matrix(env.sim, camera_name, IMAGE_SIZE, IMAGE_SIZE)
                world_cam = camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name)
                cameras[short]["K"].append(intrinsic)
                cameras[short]["T_world_cam"].append(world_cam)
                cameras[short]["T_cam_world"].append(np.linalg.inv(world_cam))

        final_success = bool(env.check_success())
        if not final_success:
            raise AttemptFailure("full source action replay did not finish successfully")
        if len(saved["front_rgb"]) != int(keep.sum()) or not len(saved["front_rgb"]):
            raise AttemptFailure("saved frame count differs from non-no-op mask")
        timesteps = np.flatnonzero(keep)
        actions = source["actions"][keep]
        path = output / row["output_path"]
        payload = {
            **{name: np.asarray(value) for name, value in saved.items() if name != "frame_seeds"},
            "actions": actions,
            "terminal": np.arange(len(actions)) == len(actions) - 1,
            "actions_full": source["actions"],
            "keep_mask": keep,
            "timestep": timesteps,
            "initial_state": source["states"][0],
            "camera": {
                short: {name: np.asarray(value) for name, value in values.items()}
                for short, values in cameras.items()
            },
            "near_m": near_m,
            "far_m": far_m,
            "instruction": candidate.get("instruction", row["canonical_language"]),
            "bddl": bddl.read_text(),
            "source_xml": source["source_xml"],
            "model_xml": model_xml,
            "row": row,
            "candidate": candidate,
            "runtime": runtime,
            "state_mapping": state_mapping,
            "replay_error": {**replay_error, "final_success": final_success},
            "perturbation": evidence,
            "frame_seeds": np.asarray(saved["frame_seeds"], dtype=np.int64),
        }
        write_episode(path, payload)
        return {
            "job_id": row["job_id"],
            "task_index": row["task_index"],
            "suite": row["suite"],
            "setting": row["setting"],
            "subdimensions": row["subdimensions"],
            "source_demo": row["source_demo"],
            "candidate_index": candidate["candidate_index"],
            "candidate_id": candidate["candidate_id"],
            "frames": len(actions),
            "path": str(path.resolve()),
            "relative_path": row["output_path"],
            "final_success": True,
            "replay_error": replay_error,
            "parameters": candidate,
        }
    finally:
        env.close()


def attempt_worker(row: dict, candidate: dict, output: str, runtime: dict, gpu: int, result_path: str) -> None:
    try:
        output_path = Path(output)
        configure_libero(output_path)
        result = run_attempt(row, candidate, output_path, runtime, gpu)
        atomic_json(Path(result_path), {"ok": True, "result": result})
    except BaseException as error:
        atomic_json(
            Path(result_path),
            {
                "ok": False,
                "exception": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=30),
            },
        )
    finally:
        # MuJoCo/EGL may abort during interpreter teardown after producing a valid result.
        os._exit(0)


def run_attempt_isolated(row: dict, candidate: dict, output: Path, runtime: dict, gpu: int) -> dict:
    import multiprocessing

    result_path = output / "runtime/results" / f"{row['job_id']}.{candidate['candidate_index']}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    process = multiprocessing.get_context("spawn").Process(
        target=attempt_worker,
        args=(row, candidate, str(output), runtime, gpu, str(result_path)),
    )
    process.start()
    process.join()
    if not result_path.is_file():
        raise WorkerFailure(f"attempt worker exited {process.exitcode} without a result")
    payload = json.loads(result_path.read_text())
    result_path.unlink()
    if process.exitcode != 0 or not payload.get("ok"):
        raise WorkerFailure(
            payload.get("exception", f"attempt worker exited {process.exitcode}"),
            payload.get("traceback", ""),
        )
    return payload["result"]


def episode_structurally_valid(path: Path, row: dict) -> bool:
    import h5py
    import numpy as np

    if not path.is_file() or path.with_suffix(path.suffix + ".partial").exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            required = (
                "observations/front_rgb",
                "observations/wrist_rgb",
                "observations/front_depth_m",
                "observations/wrist_depth_m",
                "source/actions_full",
                "source/keep_mask",
                "source/timestep",
                "actions",
                "camera/front/K",
                "camera/front/T_world_cam",
                "perturbation/frame_seeds",
                "metadata/manifest_json",
            )
            if any(name not in handle for name in required):
                return False
            manifest_row = json.loads(decoded_dataset(handle["metadata/manifest_json"]))
            length = len(handle["actions"])
            return bool(
                handle.attrs.get("format") == PROTOCOL_VERSION
                and handle.attrs.get("job_id") == row["job_id"]
                and handle.attrs.get("final_success")
                and manifest_row["job_id"] == row["job_id"]
                and length == row["saved_frames"]
                and handle["observations/front_rgb"].shape == (length, IMAGE_SIZE, IMAGE_SIZE, 3)
                and len(handle["source/actions_full"]) == row["source_actions"]
                and len(handle["source/keep_mask"]) == row["source_actions"]
                and len(handle["source/timestep"]) == length
                and len(handle["perturbation/frame_seeds"]) == length
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def select_smoke_jobs(rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if row["source_ordinal"] in SMOKE_SOURCE_ORDINALS]
    if len(selected) != 72:
        raise AuditError(f"smoke manifest must contain 72 jobs, found {len(selected)}")
    source_suites = Counter(
        (row["suite"], row["source_ordinal"])
        for row in selected
        if row["setting"] == SETTINGS[0]
    )
    if Counter(suite for suite, _ in source_suites) != Counter({suite: 3 for suite in SUITES}):
        raise AuditError("smoke manifest must contain three source demos from each suite")
    expected = {code for values in SUBDIMENSIONS.values() for code in values}
    covered = {code for row in selected for code in row["subdimensions"]}
    if covered != expected:
        raise AuditError(f"smoke subdimension coverage mismatch: {sorted(expected - covered)} missing")
    return selected


def _attempt_state(path: Path) -> tuple[set[tuple[str, int]], set[str]]:
    if not path.is_file():
        return set(), set()
    rows = read_jsonl(path)
    failed = {
        (row["job_id"], row["candidate_index"])
        for row in rows
        if row.get("status") == "failed" and "candidate_index" in row
    }
    rejected = {row["job_id"] for row in rows if row.get("status") == "rejected"}
    return failed, rejected


def _core_h5_hash(path: Path) -> str:
    import h5py
    import numpy as np

    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        for name in (
            "observations/front_rgb",
            "observations/wrist_rgb",
            "observations/front_depth_m",
            "observations/wrist_depth_m",
            "sim/state",
            "actions",
            "source/actions_full",
            "source/keep_mask",
            "perturbation/frame_seeds",
        ):
            dataset = handle[name]
            digest.update(name.encode())
            digest.update(str(dataset.shape).encode())
            if dataset.ndim and len(dataset):
                for index in range(len(dataset)):
                    digest.update(np.asarray(dataset[index]).tobytes())
            else:
                digest.update(np.asarray(dataset[()]).tobytes())
    return digest.hexdigest()


def create_smoke_evidence(output: Path, rows: list[dict]) -> None:
    import h5py
    import imageio.v2 as imageio

    hashes = []
    examples = {}
    for row in rows:
        path = output / row["output_path"]
        if not episode_structurally_valid(path, row):
            continue
        hashes.append({"job_id": row["job_id"], "path": row["output_path"], "content_hash": _core_h5_hash(path)})
        examples.setdefault(row["setting"], path)
    atomic_jsonl(output / "smoke/content_hashes.jsonl", hashes)
    preview_root = output / "smoke/previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    for setting, path in examples.items():
        final = preview_root / f"{setting}.mp4"
        partial = final.with_suffix(".partial.mp4")
        with h5py.File(path, "r") as handle, imageio.get_writer(partial, fps=20, codec="libx264", quality=7) as writer:
            for frame in handle["observations/front_rgb"]:
                writer.append_data(frame)
        os.replace(partial, final)


def generate(args) -> None:
    output = args.output.resolve()
    manifest_path = output / "manifest.jsonl"
    if not manifest_path.is_file():
        raise AuditError("run manifest before generate")
    rows = read_jsonl(manifest_path)
    validate_manifest_rows(rows)
    if args.smoke_only and args.task_index is not None:
        raise AuditError("--smoke-only and --task-index are mutually exclusive")
    if not args.smoke_only and args.task_index is None:
        raise AuditError("generate requires --task-index 0..39 unless --smoke-only is used")
    if args.task_index is not None and not 0 <= args.task_index < 40:
        raise AuditError("--task-index must be in 0..39")
    selected = select_smoke_jobs(rows) if args.smoke_only else [row for row in rows if row["task_index"] == args.task_index]
    if not selected:
        raise AuditError("selected generation partition is empty")
    if args.smoke_only:
        atomic_jsonl(output / "smoke/jobs.jsonl", selected)

    configure_libero(output)
    task_indices = sorted({row["task_index"] for row in selected})
    runtimes = {}
    for task_index in task_indices:
        path = output / f"runtime/task-{task_index:02d}.json"
        if path.is_file():
            runtimes[task_index] = json.loads(path.read_text())
        else:
            runtimes[task_index] = runtime_metadata(args.gpu)
            atomic_json(path, runtimes[task_index])

    statuses = Counter()
    for task_index in task_indices:
        task_rows = [row for row in selected if row["task_index"] == task_index]
        attempt_path = output / f"attempts/task-{task_index:02d}.jsonl"
        failed, rejected = _attempt_state(attempt_path)
        for row in task_rows:
            final = output / row["output_path"]
            if episode_structurally_valid(final, row):
                statuses["success"] += 1
                continue
            if row["job_id"] in rejected:
                statuses["rejected"] += 1
                continue
            success = None
            for candidate in row["candidates"]:
                key = (row["job_id"], candidate["candidate_index"])
                if key in failed:
                    continue
                try:
                    success = run_attempt_isolated(row, candidate, output, runtimes[task_index], args.gpu)
                    append_jsonl(
                        attempt_path,
                        {
                            "job_id": row["job_id"],
                            "candidate_index": candidate["candidate_index"],
                            "candidate_id": candidate["candidate_id"],
                            "status": "success",
                            "final_success": True,
                            "episode": success,
                        },
                    )
                    statuses["success"] += 1
                    print(canonical_json({"generated": row["job_id"], "candidate": candidate["candidate_index"]}))
                    break
                except Exception as error:
                    if episode_structurally_valid(final, row):
                        success = {
                            "job_id": row["job_id"],
                            "path": str(final.resolve()),
                            "recovered_after_worker_error": f"{type(error).__name__}: {error}",
                        }
                        append_jsonl(
                            attempt_path,
                            {
                                "job_id": row["job_id"],
                                "candidate_index": candidate["candidate_index"],
                                "candidate_id": candidate["candidate_id"],
                                "status": "success",
                                "final_success": True,
                                "episode": success,
                            },
                        )
                        statuses["success"] += 1
                        print(canonical_json({"recovered": row["job_id"], "candidate": candidate["candidate_index"]}))
                        break
                    append_jsonl(
                        attempt_path,
                        {
                            "job_id": row["job_id"],
                            "candidate_index": candidate["candidate_index"],
                            "candidate_id": candidate["candidate_id"],
                            "status": "failed",
                            "final_success": False,
                            "exception": f"{type(error).__name__}: {error}",
                            "traceback": getattr(error, "worker_traceback", "") or traceback.format_exc(limit=20),
                        },
                    )
                    failed.add(key)
                    print(canonical_json({"failed": row["job_id"], "candidate": candidate["candidate_index"], "error": str(error)}))
            if success is None:
                append_jsonl(
                    attempt_path,
                    {"job_id": row["job_id"], "status": "rejected", "final_success": False, "reason": "all three candidates failed"},
                )
                statuses["rejected"] += 1
        task_success = sum(episode_structurally_valid(output / row["output_path"], row) for row in task_rows)
        task_rejected = len(task_rows) - task_success
        atomic_json(
            output / f"status/task-{task_index:02d}.json",
            {
                "terminal": True,
                "task_index": task_index,
                "jobs": len(task_rows),
                "success": task_success,
                "rejected": task_rejected,
                "attempt_ledger": str(attempt_path.resolve()),
            },
        )
    if args.smoke_only:
        create_smoke_evidence(output, selected)
    print(canonical_json({"generation": "terminal", "jobs": len(selected), **statuses}))


def validate_episode(path: Path, row: dict) -> tuple[dict, list[str]]:
    import h5py
    import numpy as np

    errors = []
    with h5py.File(path, "r") as handle:
        candidate = json.loads(decoded_dataset(handle["metadata/candidate_json"]))
        perturbation = json.loads(decoded_dataset(handle["perturbation/parameters_json"]))
        replay_error = json.loads(decoded_dataset(handle["metadata/replay_error_json"]))
        full = np.asarray(handle["source/actions_full"])
        mask = np.asarray(handle["source/keep_mask"], dtype=bool)
        timesteps = np.asarray(handle["source/timestep"])
        actions = np.asarray(handle["actions"])
        length = len(actions)
        if action_key(full)[1] != row["action_sha256"] or len(full) != row["source_actions"]:
            errors.append(f"{row['job_id']}: source action fingerprint changed")
        expected_mask = keep_action_mask(full)
        if not np.array_equal(mask, expected_mask):
            errors.append(f"{row['job_id']}: no-op mask does not use action[:6] threshold")
        if not np.array_equal(timesteps, np.flatnonzero(mask)):
            errors.append(f"{row['job_id']}: saved timesteps differ from keep mask")
        if actions.shape != full[mask].shape or not np.array_equal(actions, full[mask]):
            errors.append(f"{row['job_id']}: saved actions differ from source actions")
        if int(handle.attrs["executed_full_actions"]) != len(full):
            errors.append(f"{row['job_id']}: not all source actions were executed")
        if not bool(handle.attrs["final_success"]):
            errors.append(f"{row['job_id']}: final success is false")
        terminal = np.asarray(handle["terminal"], dtype=bool)
        if length < 1 or terminal.sum() != 1 or not terminal[-1]:
            errors.append(f"{row['job_id']}: terminal must mark only the last saved frame")

        for name in ("front_rgb", "wrist_rgb"):
            if handle[f"observations/{name}"].shape != (length, IMAGE_SIZE, IMAGE_SIZE, 3):
                errors.append(f"{row['job_id']}: invalid {name} shape")
        depth_valid = np.asarray(handle["observations/depth_valid"], dtype=bool)
        for index, short in enumerate(("front", "wrist")):
            depth = np.asarray(handle[f"observations/{short}_depth_m"])
            near = float(handle[f"camera/{short}"].attrs["near_m"])
            far = float(handle[f"camera/{short}"].attrs["far_m"])
            valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
            if depth.shape != (length, IMAGE_SIZE, IMAGE_SIZE) or not np.array_equal(valid, depth_valid[:, index]):
                errors.append(f"{row['job_id']}: invalid {short} depth or validity mask")
            world_cam = np.asarray(handle[f"camera/{short}/T_world_cam"])
            cam_world = np.asarray(handle[f"camera/{short}/T_cam_world"])
            if world_cam.shape != (length, 4, 4) or not np.allclose(world_cam @ cam_world, np.eye(4), atol=1e-6):
                errors.append(f"{row['job_id']}: invalid {short} camera extrinsics")

        expected_seeds = np.asarray([stable_seed(row["seed"], int(timestep)) for timestep in timesteps])
        if not np.array_equal(expected_seeds, handle["perturbation/frame_seeds"]):
            errors.append(f"{row['job_id']}: frame seeds are not deterministically derived")
        if perturbation["setting"] != row["setting"] or perturbation["subdimensions"] != row["subdimensions"]:
            errors.append(f"{row['job_id']}: perturbation metadata differs from manifest")
        if perturbation["source_model_xml_hash"] != sha256_bytes(decoded_dataset(handle["metadata/source_xml"]).encode()):
            errors.append(f"{row['job_id']}: source XML hash mismatch")
        if perturbation["model_xml_hash"] != sha256_bytes(decoded_dataset(handle["metadata/model_xml"]).encode()):
            errors.append(f"{row['job_id']}: model XML hash mismatch")
        if (
            perturbation.get("protocol_hash") != protocol()["protocol_hash"]
            or perturbation.get("manifest_row_hash") != sha256_bytes(canonical_json(row).encode())
            or perturbation.get("candidate_config_hash") != sha256_bytes(canonical_json(candidate).encode())
            or perturbation.get("bddl_hash") != sha256_bytes(decoded_dataset(handle["metadata/bddl"]).encode())
        ):
            errors.append(f"{row['job_id']}: protocol, manifest, candidate, or BDDL hash mismatch")
        actual = perturbation["actual_parameters"]
        subtype = row["subdimensions"][0]
        if row["setting"] == "objects" and actual.get("extra_object_count") not in (1, 2, 3):
            errors.append(f"{row['job_id']}: missing O1 distractor evidence")
        elif row["setting"] == "background" and not actual.get("changed_assets"):
            errors.append(f"{row['job_id']}: missing background asset evidence")
        elif row["setting"] == "light" and LIGHT_FIELDS[subtype] not in actual.get("changed_fields", []):
            errors.append(f"{row['job_id']}: missing {subtype} XML evidence")
        elif row["setting"] == "camera":
            front_pose = np.asarray(handle["camera/front/T_world_cam"])
            if not np.allclose(front_pose, front_pose[0], atol=1e-9):
                errors.append(f"{row['job_id']}: camera extrinsics changed across frames")
            if subtype == "C3" and not 2 <= abs(candidate.get("roll_deg", 0)) <= 10:
                errors.append(f"{row['job_id']}: C3 lacks a 2-10 degree roll")
            if candidate.get("benchmark_tuple_equal") or candidate.get("min_test_distance", 0) < 5 - 1e-9:
                errors.append(f"{row['job_id']}: camera test tuple leaked")
        elif row["setting"] == "language":
            instruction = normalized_language(decoded_dataset(handle["language/instruction"]))
            if instruction in language_test_texts(row) or not candidate.get("prompt_hash") or not candidate.get("model_hash"):
                errors.append(f"{row['job_id']}: language isolation or provenance failed")
            if subtype == "R3" and not row["multi_step"]:
                errors.append(f"{row['job_id']}: R3 assigned to single-step task")
        elif row["setting"] == "noise":
            if candidate.get("benchmark_tuple_equal") is not False or "observations/front_rgb_clean" not in handle:
                errors.append(f"{row['job_id']}: noise isolation or clean evidence failed")
            elif np.array_equal(
                handle["observations/front_rgb"][()], handle["observations/front_rgb_clean"][()]
            ):
                errors.append(f"{row['job_id']}: noise did not alter front RGB")
        if not replay_error.get("final_success"):
            errors.append(f"{row['job_id']}: replay metadata does not record final success")
        index = {
            "job_id": row["job_id"],
            "task_index": row["task_index"],
            "suite": row["suite"],
            "setting": row["setting"],
            "subdimensions": row["subdimensions"],
            "source_demo": row["source_demo"],
            "candidate_index": candidate["candidate_index"],
            "candidate_id": candidate["candidate_id"],
            "frames": length,
            "path": row["output_path"],
            "parameters": candidate,
            "replay_error": replay_error,
        }
    return index, errors


def _validation_scope(output: Path, rows: list[dict]) -> tuple[str, list[dict]]:
    expected_per_task = Counter(row["task_index"] for row in rows)
    full_started = False
    for path in (output / "status").glob("task-*.json"):
        status = json.loads(path.read_text())
        task_index = status.get("task_index")
        if task_index in expected_per_task and status.get("jobs") == expected_per_task[task_index]:
            full_started = True
            break
    smoke_path = output / "smoke/jobs.jsonl"
    if smoke_path.is_file() and not full_started:
        return "smoke", read_jsonl(smoke_path)
    return "full", rows


def _terminal_state(output: Path, rows: list[dict]) -> tuple[dict[str, str], Counter]:
    state = {}
    attempts = Counter()
    by_job = {row["job_id"]: row for row in rows}
    for task_index in sorted({row["task_index"] for row in rows}):
        path = output / f"attempts/task-{task_index:02d}.jsonl"
        if not path.is_file():
            continue
        for attempt in read_jsonl(path):
            job_id = attempt.get("job_id")
            if job_id not in by_job:
                continue
            if "candidate_index" in attempt and attempt.get("status") in {"success", "failed"}:
                attempts[by_job[job_id]["subdimensions"][0]] += 1
            if attempt.get("status") == "rejected":
                state[job_id] = "rejected"
    for row in rows:
        if episode_structurally_valid(output / row["output_path"], row):
            state[row["job_id"]] = "success"
    return state, attempts


def summarize_parameters(code: str, episodes: list[dict]):
    candidates = [row["parameters"] for row in episodes if code in row["subdimensions"]]
    if not candidates:
        return None
    if code == "O1":
        values = sorted({item["extra_objects"] for item in candidates})
        return {"extra_objects": [min(values), max(values)]}
    if code in {"B1", "B2"}:
        return {"variants": sorted({item["variant"] for item in candidates})}
    if code in LIGHT_FIELDS:
        return {"verified_xml_field": LIGHT_FIELDS[code], "candidate_count": len({item["candidate_id"] for item in candidates})}
    if code == "C1":
        values = [item["distance_scale"] for item in candidates]
        return {"distance_scale": [min(values), max(values)]}
    if code == "C2":
        return {
            "azimuth_deg": [min(item["azimuth_deg"] for item in candidates), max(item["azimuth_deg"] for item in candidates)],
            "elevation_deg": [min(item["elevation_deg"] for item in candidates), max(item["elevation_deg"] for item in candidates)],
        }
    if code == "C3":
        return {
            name: [min(item[name] for item in candidates), max(item[name] for item in candidates)]
            for name in ("yaw_deg", "pitch_deg", "roll_deg")
        }
    if code in {"R1", "R2", "R3"}:
        return {"model_hashes": sorted({item["model_hash"] for item in candidates})}
    return {
        "algorithm": candidates[0]["algorithm"],
        "severity": [min(item["severity"] for item in candidates), max(item["severity"] for item in candidates)],
    }


def build_reports(rows: list[dict], episodes: list[dict], state: dict[str, str], attempts: Counter) -> tuple[dict, dict]:
    episode_by_job = {row["job_id"]: row for row in episodes}
    coverage_rows = []
    for setting in SETTINGS:
        for code in SUBDIMENSIONS[setting]:
            expected = [row for row in rows if code in row["subdimensions"]]
            successes = [episode_by_job[row["job_id"]] for row in expected if row["job_id"] in episode_by_job]
            rejected = sum(state.get(row["job_id"]) == "rejected" for row in expected)
            coverage_rows.append(
                {
                    "setting": setting,
                    "subdimension": code,
                    "jobs": len(expected),
                    "attempts": attempts[code],
                    "success": len(successes),
                    "rejected": rejected,
                    "retention_rate": len(successes) / len(expected) if expected else 0.0,
                    "parameters": summarize_parameters(code, successes),
                    "example": successes[0]["path"] if successes else None,
                }
            )
    coverage = {"subdimensions": coverage_rows}
    per_task = []
    for task_index in sorted({row["task_index"] for row in rows}):
        expected = [row for row in rows if row["task_index"] == task_index]
        success = sum(row["job_id"] in episode_by_job for row in expected)
        rejected = sum(state.get(row["job_id"]) == "rejected" for row in expected)
        per_task.append(
            {
                "task_index": task_index,
                "jobs": len(expected),
                "success": success,
                "rejected": rejected,
                "pending": len(expected) - success - rejected,
                "retention_rate": success / len(expected),
            }
        )
    total = len(rows)
    success = len(episodes)
    rejected = sum(value == "rejected" for value in state.values())
    retention = {
        "jobs": total,
        "candidate_attempts": sum(attempts.values()),
        "success": success,
        "rejected": rejected,
        "pending": total - success - rejected,
        "retention_rate": success / total if total else 0.0,
        "tasks": per_task,
    }
    return coverage, retention


IMPLEMENTATION = {
    "O1": "non-test _add_ BDDL plus named-state replay",
    "B1": "non-test _tb_ BDDL (scene and surface XML assets)",
    "B2": "non-test _table_ BDDL (working-surface XML asset)",
    "L1": "non-test light BDDL; final model XML diffuse verified",
    "L2": "non-test light BDDL; final model XML direction verified",
    "L3": "non-test light BDDL; final model XML specular verified",
    "L4": "non-test light BDDL; final model XML castshadow verified",
    "C1": "direct agentview distance scaling",
    "C2": "direct agentview spherical azimuth/elevation transform",
    "C3": "direct agentview quaternion yaw/pitch/roll transform",
    "R1": "deterministic Qwen distraction rewrite",
    "R2": "deterministic Qwen commonsense rewrite",
    "R3": "deterministic Qwen multi-step reasoning-chain rewrite",
    "N1": "per-frame motion blur",
    "N2": "fixed-parameter Gaussian blur per frame",
    "N3": "fixed-parameter zoom blur per frame",
    "N4": "per-frame random fog field",
    "N5": "per-frame random glass-distortion field",
}


def write_perturbations_doc(output: Path, coverage: dict, retention: dict, manifest_hash: str) -> None:
    protocol_data = json.loads((output / "protocol.json").read_text())
    report = {row["subdimension"]: row for row in coverage["subdimensions"]}
    row_by_code = {
        code: row
        for row in read_jsonl(output / "manifest.jsonl")
        for code in row["subdimensions"]
    }
    lines = [
        "# LIBERO-Plus RGB-D Replica v3 Perturbations",
        "",
        "This dataset covers the mechanisms used by the generalized LIBERO-Plus training set. It does not claim pixel-level identity or identical official sampling probabilities.",
        "",
        f"- Data version: `{PROTOCOL_VERSION}`",
        f"- Source revision: `{SOURCE['revision']}`",
        f"- Protocol hash: `{protocol_data['protocol_hash']}`",
        f"- Manifest hash: `{manifest_hash}`",
        f"- Generated: `{coverage['generated_at']}`",
        "",
        "| Category / item | Implementation | Temporal behavior | Actual parameter range | Affected modalities | Randomness | Test isolation | Attempts | Success | Rejected | Retention | Example |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for setting in SETTINGS:
        for code in SUBDIMENSIONS[setting]:
            item = report[code]
            manifest_row = row_by_code[code]
            randomness = (
                "episode seed + stable_seed(job_seed, timestep)"
                if manifest_row["temporal_mode"].startswith("per_frame")
                else "episode seed"
            )
            isolation = {
                "objects": "benchmark BDDL excluded",
                "background": "benchmark BDDL excluded",
                "light": "benchmark BDDL excluded",
                "camera": "test tuple excluded; >=5 degree/scale-point distance",
                "language": "test instruction text excluded",
                "noise": "Appendix A Table 5 exact tuples excluded",
            }[setting]
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"{setting} / {code}",
                        IMPLEMENTATION[code],
                        manifest_row["temporal_mode"],
                        f"`{canonical_json(item['parameters'])}`",
                        ", ".join(manifest_row["affected_modalities"]),
                        randomness,
                        isolation,
                        str(item["attempts"]),
                        str(item["success"]),
                        str(item["rejected"]),
                        f"{item['retention_rate']:.6f}",
                        item["example"] or "—",
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Scope and exclusions",
            "",
            "O2 Target Object Pose is excluded because the official generalized training set retained only confounding-object trajectories; pose replay was not reliable enough. Robot Initial State is likewise not one of the six generalized-training variants.",
            "",
            "Camera perturbations are episode-static: C1 changes distance, C2 changes spherical position, and C3 changes a fixed camera quaternion including roll. Apparent frame-to-frame visual variation belongs to N1/N4/N5, whose random direction or field is deterministically resampled per source timestep.",
            "",
            "## Retention",
            "",
            f"- Jobs: `{retention['jobs']}`",
            f"- Candidate attempts: `{retention['candidate_attempts']}`",
            f"- Successful episodes: `{retention['success']}`",
            f"- Rejected episodes: `{retention['rejected']}`",
            f"- Pending episodes: `{retention['pending']}`",
            f"- Retention rate: `{retention['retention_rate']:.6f}`",
        ]
    )
    atomic_text(output / "PERTURBATIONS.md", "\n".join(lines) + "\n")


def deterministic_smoke_rerun(output: Path, rows: list[dict], episodes: list[dict], gpu: int, skip: bool) -> tuple[dict, list[str]]:
    import tempfile

    result = {"requested": not skip, "expected": len(SETTINGS), "matched": 0}
    if skip:
        return result, []
    errors = []
    episode_by_setting = {}
    for episode in episodes:
        episode_by_setting.setdefault(episode["setting"], episode)
    row_by_job = {row["job_id"]: row for row in rows}
    with tempfile.TemporaryDirectory(dir=output / "runtime") as temporary:
        temporary = Path(temporary)
        for setting in SETTINGS:
            original = episode_by_setting.get(setting)
            if original is None:
                errors.append(f"smoke determinism lacks a successful {setting} episode")
                continue
            row = row_by_job[original["job_id"]]
            candidate = original["parameters"]
            runtime_path = output / f"runtime/task-{row['task_index']:02d}.json"
            runtime = json.loads(runtime_path.read_text())
            try:
                rerun = run_attempt_isolated(row, candidate, temporary, runtime, gpu)
                if _core_h5_hash(output / row["output_path"]) == _core_h5_hash(Path(rerun["path"])):
                    result["matched"] += 1
                else:
                    errors.append(f"smoke determinism content mismatch: {setting}")
            except Exception as error:
                errors.append(f"smoke determinism rerun failed for {setting}: {error}")
    return result, errors


def validate(args) -> None:
    output = args.output.resolve()
    manifest_path = output / "manifest.jsonl"
    if not manifest_path.is_file():
        raise AuditError("manifest.jsonl is missing")
    all_rows = read_jsonl(manifest_path)
    validate_manifest_rows(all_rows)
    scope, rows = _validation_scope(output, all_rows)
    state, attempts = _terminal_state(output, rows)
    errors = []
    episodes = []

    expected_per_task = Counter(row["task_index"] for row in rows)
    for task_index, count in expected_per_task.items():
        status_path = output / f"status/task-{task_index:02d}.json"
        if not status_path.is_file():
            errors.append(f"task-{task_index:02d}: terminal status is missing")
            continue
        status = json.loads(status_path.read_text())
        if not status.get("terminal") or status.get("jobs") != count:
            errors.append(f"task-{task_index:02d}: status is not terminal for {count} expected jobs")

    for row in rows:
        path = output / row["output_path"]
        if episode_structurally_valid(path, row):
            try:
                episode, episode_errors = validate_episode(path, row)
                episodes.append(episode)
                errors.extend(episode_errors)
            except Exception as error:
                errors.append(f"{row['job_id']}: validation exception: {type(error).__name__}: {error}")
        elif state.get(row["job_id"]) != "rejected":
            errors.append(f"{row['job_id']}: neither a valid success nor a tracked rejection")

    coverage, retention = build_reports(rows, episodes, state, attempts)
    generated_at = datetime.now(timezone.utc).isoformat()
    coverage.update(
        {
            "scope": scope,
            "version": PROTOCOL_VERSION,
            "source_revision": SOURCE["revision"],
            "protocol_hash": json.loads((output / "protocol.json").read_text())["protocol_hash"],
            "manifest_hash": sha256_file(manifest_path),
            "generated_at": generated_at,
        }
    )
    retention.update({"scope": scope, "generated_at": generated_at})
    for item in coverage["subdimensions"]:
        if item["success"] == 0:
            errors.append(f"{item['subdimension']}: no successful mechanism evidence")
    if retention["pending"]:
        errors.append(f"{retention['pending']} jobs have no terminal outcome")
    if scope == "full" and len(expected_per_task) != 40:
        errors.append(f"full validation covers {len(expected_per_task)} tasks instead of 40")
    if scope == "smoke" and retention["success"] != 72:
        errors.append(f"smoke retained {retention['success']}/72 jobs")

    deterministic = {"requested": False, "expected": 0, "matched": 0}
    if scope == "smoke" and not errors:
        deterministic, deterministic_errors = deterministic_smoke_rerun(
            output, rows, episodes, args.gpu, args.skip_determinism
        )
        errors.extend(deterministic_errors)
        hashes = output / "smoke/content_hashes.jsonl"
        if not hashes.is_file() or len(read_jsonl(hashes)) != len(episodes):
            errors.append("smoke content hashes are incomplete")
        previews = list((output / "smoke/previews").glob("*.mp4"))
        if len(previews) != len(SETTINGS):
            errors.append(f"smoke preview count {len(previews)} != {len(SETTINGS)}")

    report_root = output / ("smoke" if scope == "smoke" else "")
    atomic_json(report_root / "coverage.json", coverage)
    atomic_json(report_root / "retention.json", retention)
    atomic_jsonl(report_root / "index.jsonl", episodes)
    atomic_json(
        report_root / "validation.json",
        {"go": not errors, "scope": scope, "errors": errors, "determinism": deterministic},
    )
    if scope == "full":
        write_perturbations_doc(output, coverage, retention, coverage["manifest_hash"])
    if errors:
        raise AuditError(f"{scope} validation failed with {len(errors)} error(s); see {report_root / 'validation.json'}")
    print(canonical_json({"validation": "GO", "scope": scope, "success": len(episodes), "rejected": retention["rejected"]}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    audit_parser = commands.add_parser("audit", help="check/download all 40 source HDF5 files and enumerate demos")
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    audit_parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    audit_parser.add_argument("--no-download", "--no-download-sources", action="store_true")
    audit_parser.set_defaults(handler=audit)

    manifest_parser = commands.add_parser("manifest", help="freeze one six-setting job per source demo")
    manifest_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    manifest_parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN)
    manifest_parser.add_argument("--gpu", type=int, default=0)
    manifest_parser.set_defaults(handler=manifest)

    generate_parser = commands.add_parser("generate", help="replay one task partition or the fixed 72-job smoke set")
    generate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    generate_parser.add_argument("--task-index", type=int)
    generate_parser.add_argument("--gpu", type=int, default=0)
    generate_parser.add_argument("--smoke-only", action="store_true")
    generate_parser.set_defaults(handler=generate)

    validate_parser = commands.add_parser("validate", help="validate terminal ledgers and write coverage/retention/docs")
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser.add_argument("--gpu", type=int, default=0)
    validate_parser.add_argument("--skip-determinism", action="store_true")
    validate_parser.set_defaults(handler=validate)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
        return 0
    except (AuditError, AttemptFailure) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
