#!/usr/bin/env python3
"""Build and verify the fixed 72-episode LIBERO-Plus RGB-D sample."""

from __future__ import annotations

import argparse
import ast
import binascii
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "data/libero_plus_rgbd_sample_v2"
DEFAULT_QWEN = Path.home() / "workspace/ckpt/Qwen2.5-7B-Instruct-a09a3545"
MASTER_SEED = 0
IMAGE_SIZE = 256
SETTINGS = ("objects", "background", "light", "camera", "language", "noise")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
REVISIONS = {
    "lerobot": {
        "repo": "Sylvest/libero_plus_lerobot",
        "sha": "22c57433fef692b5b9ecc0795344daac7fa867a5",
    },
    "rlds": {
        "repo": "Sylvest/libero_plus_rlds",
        "sha": "fb0c7029b076030d5d57227229e4f7460def1f7c",
    },
    "camera": {
        "repo": "Sylvest/libero_plus_camparam_rlds",
        "sha": "dc60c70eb7bd63cb694a89d7c5ea53f2032d8807",
    },
    "segmentation": {
        "repo": "Sylvest/libero_plus_seg",
        "sha": "254ad63ac8a130049362a79b7c26ef9ff93766ad",
    },
    "source": {
        "repo": "yifengzhu-hf/LIBERO-datasets",
        "sha": "f13aa24a3da8c43c7225569f28c562979fa0e35a",
    },
    "qwen": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "sha": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
}
ARCHIVES = {
    "rlds": {
        "files": (
            ("libero_plus_mixdata.z01", 32212254720, "61928db465d2ae68994796174ddaaccd951c3728550e8454f3331b32ab555741"),
            ("libero_plus_mixdata.z02", 32212254720, "57abe4f928e80b5c40f5f3ffaabafe175ec2b301adf9e85c1b7779a9a6bdb2e0"),
            ("libero_plus_mixdata.zip", 11119747860, "56bf87a8decce0a0a331b09f3fe5011af3fb061a1867a761c755283342b0836e"),
        ),
        "shards": 1024,
    },
    "camera": {
        "files": (("libero_plus_camparam_rlds.zip", 16607835331, "a99466a1bb7eab4d0c55094d64d53ef6794ee835ba0db003fcee3e3fa6568e73"),),
        "shards": 256,
    },
    "segmentation": {
        "files": (
            ("libero_mix_seg.z01", 26843545600, "6549267cce1a8b50ce48c4d3e56c2b5049dc7d71e37ce95e16d14e9026e57b41"),
            ("libero_mix_seg.z02", 26843545600, "b7c10190af60713363893b18c1f47455a85fe82a65d1ed183a510e4fb0414c09"),
            ("libero_mix_seg.zip", 24535692812, "26a320b071cda59836c4a974983f3b8ca72fff31af014ed98f6e744d122f557e"),
        ),
        "shards": 1024,
    },
}
NOISE_ORDER = ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur")
QWEN_STYLES = (
    'Begin exactly with "Please," and use a concise polite request.',
    'Begin exactly with "Now," and use a direct imperative.',
    'Begin exactly with "First," and copy the original instruction verbatim after that prefix.',
    'Begin exactly with "Could you please" and use a household-assistant request.',
    'Begin exactly with "For this task," and use a concise instruction.',
    'Begin exactly with "Your goal is to" and copy the original instruction verbatim after that prefix.',
    'Begin exactly with "Would you" and use a conversational request.',
    'Begin exactly with "The desired end state is" and then state the action.',
)
ACTION_PHRASES = ("pick up", "place", "put", "open", "close", "turn on", "turn off")
RELATIONS = (
    "next to",
    "between",
    "to the left of",
    "to the right of",
    "in front of",
    "at the back",
    "on top of",
    "inside",
    "left",
    "right",
    "front",
    "back",
    "top",
    "bottom",
)
_SHA256_CACHE = {}
CALIBRATION_CACHE_VERSION = 1
PROBE_CACHE_VERSION = 2


class AuditError(RuntimeError):
    pass


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    path = path.resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key in _SHA256_CACHE:
        return _SHA256_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    _SHA256_CACHE[key] = result
    return result


def stable_seed(*parts) -> int:
    payload = canonical_json((MASTER_SEED, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def atomic_jsonl(path: Path, rows) -> None:
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_range(path: Path, start: int, length: int) -> bytes:
    if length < 0 or start < 0:
        raise AuditError("negative byte range")
    if not length:
        return b""
    with path.open("rb") as stream:
        stream.seek(start)
        payload = stream.read(length)
    if len(payload) != length:
        raise AuditError(f"{path}: read {len(payload)} bytes, expected {length}")
    return payload


def _zip64_values(extra: bytes, compressed: int, size: int, offset: int, disk: int):
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, length = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4 : cursor + 4 + length]
        cursor += 4 + length
        if kind != 1:
            continue
        values = []
        position = 0
        for value, missing, width in (
            (size, size == 0xFFFFFFFF, 8),
            (compressed, compressed == 0xFFFFFFFF, 8),
            (offset, offset == 0xFFFFFFFF, 8),
            (disk, disk == 0xFFFF, 4),
        ):
            if missing:
                if position + width > len(payload):
                    raise AuditError("truncated ZIP64 extra field")
                value = int.from_bytes(payload[position : position + width], "little")
                position += width
            values.append(value)
        return values[1], values[0], values[2], values[3]
    if 0xFFFFFFFF in (compressed, size, offset) or disk == 0xFFFF:
        raise AuditError("ZIP64 sentinel without ZIP64 extra field")
    return compressed, size, offset, disk


class LocalZip:
    """Read selected members of a pinned, possibly split, local ZIP."""

    def __init__(self, source: str, archive_root: Path):
        config = ARCHIVES[source]
        self.source = source
        self.files = config["files"]
        self.paths = [archive_root / source / name for name, _, _ in self.files]
        tail_size = min(1 << 20, self.files[-1][1])
        tail = _read_range(self.paths[-1], self.files[-1][1] - tail_size, tail_size)
        zip64_at = tail.rfind(b"PK\x06\x06")
        if zip64_at < 0:
            raise AuditError(f"{source}: ZIP64 end record missing")
        fields = struct.unpack_from("<4sQ2H2L4Q", tail, zip64_at)
        count, directory_size, directory_offset = fields[7], fields[8], fields[9]
        directory = _read_range(self.paths[-1], directory_offset, directory_size)
        self.members = {}
        cursor = 0
        while cursor < len(directory):
            if directory[cursor : cursor + 4] != b"PK\x01\x02":
                raise AuditError(f"{source}: malformed ZIP central directory at {cursor}")
            fields = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
            flags, method = fields[3], fields[4]
            crc32, compressed, size = fields[7:10]
            name_length, extra_length, comment_length, disk = fields[10:14]
            offset = fields[-1]
            start = cursor + 46
            raw_name = directory[start : start + name_length]
            extra = directory[start + name_length : start + name_length + extra_length]
            compressed, size, offset, disk = _zip64_values(extra, compressed, size, offset, disk)
            name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
            self.members[name] = {
                "archive": source,
                "name": name,
                "disk": disk,
                "offset": offset,
                "compressed_bytes": compressed,
                "bytes": size,
                "method": method,
                "crc32": f"{crc32:08x}",
            }
            cursor += 46 + name_length + extra_length + comment_length
        if len(self.members) != count:
            raise AuditError(f"{source}: central directory count {len(self.members)} != {count}")

    def _spanned(self, disk: int, offset: int, length: int) -> bytes:
        chunks = []
        while length:
            if disk >= len(self.files):
                raise AuditError(f"{self.source}: ZIP member exceeds final volume")
            available = self.files[disk][1] - offset
            take = min(length, available)
            chunks.append(_read_range(self.paths[disk], offset, take))
            length -= take
            disk += 1
            offset = 0
        return b"".join(chunks)

    def find(self, suffix: str) -> dict:
        matches = [row for name, row in self.members.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise AuditError(f"{self.source}: {suffix!r} matched {len(matches)} ZIP members")
        return matches[0]

    def read(self, member: dict) -> bytes:
        header = self._spanned(member["disk"], member["offset"], 30)
        fields = struct.unpack("<4s5H3L2H", header)
        if fields[0] != b"PK\x03\x04" or fields[3] != member["method"]:
            raise AuditError(f"{self.source}: invalid local header for {member['name']}")
        data_offset = member["offset"] + 30 + fields[-2] + fields[-1]
        raw = self._spanned(member["disk"], data_offset, member["compressed_bytes"])
        if member["method"] == 8:
            payload = zlib.decompress(raw, -zlib.MAX_WBITS)
        elif member["method"] == 0:
            payload = raw
        else:
            raise AuditError(f"{self.source}: unsupported ZIP method {member['method']}")
        if len(payload) != member["bytes"] or f"{binascii.crc32(payload) & 0xFFFFFFFF:08x}" != member["crc32"]:
            raise AuditError(f"{self.source}: size/CRC mismatch for {member['name']}")
        return payload

    def json(self, suffix: str):
        return json.loads(self.read(self.find(suffix)))

    def shard(self, index: int) -> tuple[dict, bytes]:
        suffix = f"train.tfrecord-{index:05d}-of-{ARCHIVES[self.source]['shards']:05d}"
        member = self.find(suffix)
        return member, self.read(member)


def _varint(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if cursor >= len(data):
            raise AuditError("truncated protobuf varint")
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
    raise AuditError("protobuf varint exceeds 64 bits")


def _protobuf_fields(data: bytes):
    cursor = 0
    while cursor < len(data):
        tag, cursor = _varint(data, cursor)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise AuditError("invalid protobuf field zero")
        if wire == 0:
            value, cursor = _varint(data, cursor)
        elif wire == 1:
            value = data[cursor : cursor + 8]
            cursor += 8
        elif wire == 2:
            length, cursor = _varint(data, cursor)
            value = data[cursor : cursor + length]
            cursor += length
        elif wire == 5:
            value = data[cursor : cursor + 4]
            cursor += 4
        else:
            raise AuditError(f"unsupported protobuf wire type {wire}")
        if cursor > len(data):
            raise AuditError("truncated protobuf field")
        yield number, wire, value


def parse_tf_example(data: bytes) -> dict[str, list]:
    features_messages = [value for number, wire, value in _protobuf_fields(data) if number == 1 and wire == 2]
    if len(features_messages) != 1:
        raise AuditError("TF Example must contain one Features message")
    result = {}
    for number, wire, entry in _protobuf_fields(features_messages[0]):
        if number != 1 or wire != 2:
            continue
        fields = list(_protobuf_fields(entry))
        keys = [value.decode() for field, kind, value in fields if field == 1 and kind == 2]
        values = [value for field, kind, value in fields if field == 2 and kind == 2]
        if len(keys) != 1 or len(values) != 1:
            raise AuditError("invalid TF Example feature map entry")
        feature = list(_protobuf_fields(values[0]))
        if len(feature) != 1 or feature[0][1] != 2:
            raise AuditError(f"invalid TF Feature for {keys[0]}")
        kind, payload = feature[0][0], feature[0][2]
        items = list(_protobuf_fields(payload))
        if kind == 1:
            result[keys[0]] = [value for field, _, value in items if field == 1]
        elif kind == 2:
            packed = b"".join(value for field, _, value in items if field == 1)
            if len(packed) % 4:
                raise AuditError(f"unaligned float_list for {keys[0]}")
            result[keys[0]] = list(struct.unpack(f"<{len(packed) // 4}f", packed))
        elif kind == 3:
            decoded = []
            for field, item_wire, value in items:
                if field != 1:
                    continue
                if item_wire == 0:
                    decoded.append(value)
                else:
                    cursor = 0
                    while cursor < len(value):
                        item, cursor = _varint(value, cursor)
                        decoded.append(item)
            result[keys[0]] = decoded
        else:
            raise AuditError(f"unknown TF Feature kind {kind}")
    return result


def _crc32c(data: bytes) -> int:
    if not hasattr(_crc32c, "table"):
        table = []
        for value in range(256):
            for _ in range(8):
                value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
            table.append(value)
        _crc32c.table = table
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _crc32c.table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return ((crc >> 15) | (crc << 17) & 0xFFFFFFFF) + 0xA282EAD8 & 0xFFFFFFFF


def tfrecord_examples(data: bytes, validate_crc: bool = True):
    cursor = 0
    ordinal = 0
    while cursor < len(data):
        start = cursor
        if cursor + 12 > len(data):
            raise AuditError("truncated TFRecord header")
        length_bytes = data[cursor : cursor + 8]
        length = struct.unpack("<Q", length_bytes)[0]
        length_crc = struct.unpack_from("<L", data, cursor + 8)[0]
        cursor += 12
        payload = data[cursor : cursor + length]
        cursor += length
        if cursor + 4 > len(data):
            raise AuditError("truncated TFRecord payload")
        payload_crc = struct.unpack_from("<L", data, cursor)[0]
        cursor += 4
        if validate_crc and (
            length_crc != _masked_crc32c(length_bytes) or payload_crc != _masked_crc32c(payload)
        ):
            raise AuditError(f"TFRecord CRC mismatch at ordinal {ordinal}")
        yield ordinal, start, payload, parse_tf_example(payload)
        ordinal += 1


def ordinal_to_shard(episode_index: int, shard_lengths: list[int]) -> tuple[int, int]:
    if episode_index < 0 or episode_index >= sum(shard_lengths):
        raise AuditError(f"episode index {episode_index} is outside TFDS split")
    ends = []
    total = 0
    for length in shard_lengths:
        total += int(length)
        ends.append(total)
    shard = bisect_right(ends, episode_index)
    return shard, episode_index - (ends[shard - 1] if shard else 0)


def hub_download(source: str, relative: str, local_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download

    item = REVISIONS[source]
    path = Path(
        hf_hub_download(
            repo_id=item["repo"],
            filename=relative,
            repo_type="model" if source == "qwen" else "dataset",
            revision=item["sha"],
            local_dir=local_dir,
        )
    )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def download_official_archives(archive_root: Path) -> list[dict]:
    from huggingface_hub import hf_hub_download

    archive_root.mkdir(parents=True, exist_ok=True)
    verified_path = archive_root / "verified.json"
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    jobs = [
        (source, name, size, digest)
        for source, config in ARCHIVES.items()
        for name, size, digest in config["files"]
    ]

    def download(job):
        source, name, size, digest = job
        revision = REVISIONS[source]
        path = Path(
            hf_hub_download(
                repo_id=revision["repo"],
                filename=name,
                repo_type="dataset",
                revision=revision["sha"],
                local_dir=archive_root / source,
            )
        )
        if path.stat().st_size != size:
            raise AuditError(f"{path}: size {path.stat().st_size} != pinned size {size}")
        return source, name, size, digest, path

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        downloaded = list(executor.map(download, jobs))

    def verify(item):
        source, name, size, digest, path = item
        stat = path.stat()
        cached = verified.get(f"{source}/{name}", {})
        actual = (
            digest
            if cached.get("bytes") == size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and cached.get("sha256") == digest
            else sha256_file(path)
        )
        if actual != digest:
            raise AuditError(f"{path}: SHA-256 {actual} != pinned LFS digest {digest}")
        return source, name, path, stat, actual

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        checked = list(executor.map(verify, downloaded))

    records = []
    for source, name, path, stat, digest in checked:
        verified[f"{source}/{name}"] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
        records.append(
            {
                "kind": "official_archive",
                "source": source,
                "repo": REVISIONS[source]["repo"],
                "revision": REVISIONS[source]["sha"],
                "name": name,
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "lfs_sha256": digest,
            }
        )
    atomic_json(verified_path, verified)
    return records


def task_stem(suite: str, task: str) -> str:
    normalized = task.lower().replace(" ", "_")
    folder = ROOT / "libero/libero/bddl_files" / suite
    direct = folder / f"{normalized}.bddl"
    if direct.exists():
        return direct.stem
    matches = [
        path.stem
        for path in folder.glob(f"*_{normalized}.bddl")
        if not re.search(r"_(?:table|tb|language|light|add|level|moved|view)_| copy", path.stem)
    ]
    if len(matches) != 1:
        raise AuditError(f"{suite}: cannot uniquely map task to canonical BDDL: {task}")
    return matches[0]


def canonical_bddl(suite: str, task: str) -> Path:
    return ROOT / "libero/libero/bddl_files" / suite / f"{task_stem(suite, task)}.bddl"


def task_suite_map(tasks: list[dict]) -> dict[str, str]:
    result = {}
    for row in tasks:
        hits = []
        for suite in SUITES:
            try:
                canonical_bddl(suite, row["task"])
                hits.append(suite)
            except AuditError:
                pass
        if len(hits) == 1:
            result[row["task"]] = hits[0]
    if Counter(result.values()) != Counter({suite: 10 for suite in SUITES}):
        raise AuditError("public tasks do not map to exactly 10 canonical tasks per suite")
    return result


def ranked_tasks(tasks: list[dict], episodes: list[dict]) -> dict[str, list[dict]]:
    suites = task_suite_map(tasks)
    lengths = defaultdict(list)
    for episode in episodes:
        lengths[episode["tasks"][0]].append(episode["length"])
    task_ids = {row["task"]: row["task_index"] for row in tasks}
    result = {}
    for suite in SUITES:
        suite_tasks = [task for task, owner in suites.items() if owner == suite]
        suite_median = median(length for task in suite_tasks for length in lengths[task])
        rows = [
            {
                "suite": suite,
                "task": task,
                "task_index": task_ids[task],
                "task_median_length": median(lengths[task]),
                "suite_episode_median_length": suite_median,
                "distance": abs(median(lengths[task]) - suite_median),
                "episode_count": len(lengths[task]),
                "task_stem": task_stem(suite, task),
            }
            for task in suite_tasks
        ]
        result[suite] = sorted(rows, key=lambda row: (row["distance"], row["task_index"]))
    return result


def source_filename(row: dict) -> str:
    return f"{row['suite']}/{row['task_stem']}_demo.hdf5"


def parquet_path(episode_index: int) -> str:
    return f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def unique_subsequence_mask(full, filtered, tolerance=1e-6):
    import numpy as np

    full = np.asarray(full)
    filtered = np.asarray(filtered)
    if full.ndim != 2 or filtered.ndim != 2 or full.shape[1:] != filtered.shape[1:]:
        return None

    def align(full_indices, filtered_indices):
        picked = []
        wanted = iter(filtered_indices)
        cursor = next(wanted, None)
        for index in full_indices:
            if cursor is not None and np.max(np.abs(full[index] - filtered[cursor])) <= tolerance:
                picked.append(index)
                cursor = next(wanted, None)
        return picked if cursor is None else None

    forward = align(range(len(full)), range(len(filtered)))
    backward = align(range(len(full) - 1, -1, -1), range(len(filtered) - 1, -1, -1))
    if backward is not None:
        backward = list(reversed(backward))
    if forward is None or forward != backward:
        return None
    mask = np.zeros(len(full), dtype=bool)
    mask[forward] = True
    return mask


def source_demos(path: Path, actions_only: bool = False) -> list[dict]:
    import h5py
    import numpy as np

    result = []
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        demos = sorted(data, key=lambda name: int(re.search(r"\d+", name).group()))
        for name in demos:
            group = data[name]
            row = {"id": name, "actions": np.asarray(group["actions"])}
            if not actions_only:
                row.update(
                    {
                        "states": np.asarray(group["states"]),
                        "model_xml": (
                            group.attrs["model_file"].decode()
                            if isinstance(group.attrs["model_file"], bytes)
                            else str(group.attrs["model_file"])
                        ),
                    }
                )
            result.append(row)
    return result


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
        inventory.append(
            {
                "kind": element.tag,
                "id": element.attrib.get("name"),
                "original_file": original,
                "resolved_file": str(path),
                "sha256": sha256_file(path),
            }
        )
    if missing:
        raise AuditError(f"source XML has {len(missing)} unresolved assets: {missing[:3]}")
    return ET.tostring(root, encoding="unicode"), sorted(inventory, key=canonical_json)


def source_xml_asset_evidence(source_path: Path, selected: list[dict]) -> list[dict]:
    import h5py

    result = []
    with h5py.File(source_path, "r") as handle:
        for row in selected:
            xml = handle[f"data/{row['source_demo']}"] .attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode()
            processed, assets = resolve_source_model_xml(str(xml))
            result.append(
                {
                    "source_demo": row["source_demo"],
                    "asset_count": len(assets),
                    "processed_xml_sha256": sha256_bytes(processed.encode()),
                }
            )
    return result


def parquet_actions(path: Path):
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["action"])
    return np.asarray(table["action"].to_pylist(), dtype=np.float64)


def choose_source_matches(
    source_path: Path, episode_rows: list[dict], parquet_root: Path, downloaded_paths: list[Path]
) -> dict:
    demos = source_demos(source_path)
    masks = {}
    matched_episodes = defaultdict(list)
    inconsistent = set()
    scanned = 0
    for episode in episode_rows:
        relative = parquet_path(episode["episode_index"])
        local = parquet_root / relative
        existed = local.exists()
        hub_download("lerobot", relative, parquet_root)
        if not existed:
            downloaded_paths.append(local)
        actions = parquet_actions(local)
        matches = []
        for demo in demos:
            mask = unique_subsequence_mask(demo["actions"], actions)
            if mask is not None:
                matches.append((demo["id"], mask))
        scanned += 1
        if len(matches) != 1:
            continue
        demo_id, mask = matches[0]
        mask_hash = sha256_bytes(mask.tobytes())
        if demo_id in masks and masks[demo_id]["mask_hash"] != mask_hash:
            inconsistent.add(demo_id)
            continue
        masks[demo_id] = {
            "mask": mask,
            "mask_hash": mask_hash,
            "episode_index": episode["episode_index"],
            "kept_length": int(mask.sum()),
        }
        matched_episodes[demo_id].append(episode["episode_index"])
        consistent = [name for name in masks if name not in inconsistent]
        if len(consistent) >= 3 and scanned >= 8:
            break
    valid = [name for name in masks if name not in inconsistent]
    if len(valid) < 3:
        raise AuditError(f"{source_path.name}: only {len(valid)} uniquely matched source demos")

    lengths = {name: masks[name]["kept_length"] for name in valid}
    ordered_lengths = sorted(lengths.values())

    def quantile(q):
        position = (len(ordered_lengths) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return ordered_lengths[low] + (ordered_lengths[high] - ordered_lengths[low]) * (position - low)

    selected = []
    for q in (0.25, 0.5, 0.75):
        target = quantile(q)
        name = min(
            (name for name in valid if name not in selected),
            key=lambda item: (abs(lengths[item] - target), item),
        )
        selected.append(name)
    return {
        "source_file": str(source_path),
        "source_sha256": sha256_file(source_path),
        "scanned_public_episodes": scanned,
        "selected": [
            {
                "source_demo": name,
                "public_episode": masks[name]["episode_index"],
                "keep_mask": masks[name]["mask"].astype(int).tolist(),
                "keep_mask_sha256": masks[name]["mask_hash"],
                "kept_length": masks[name]["kept_length"],
                "all_matching_public_episodes": matched_episodes[name],
            }
            for name in selected
        ],
        "inconsistent_source_demos": sorted(inconsistent),
        "max_abs_error": 1e-6,
    }


def parquet_episode(path: Path) -> tuple:
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["action", "observation.state"])
    return (
        np.asarray(table["action"].to_pylist(), dtype="<f4"),
        np.asarray(table["observation.state"].to_pylist(), dtype="<f4"),
    )


def action_key(actions) -> tuple[int, str]:
    import numpy as np

    values = np.ascontiguousarray(actions, dtype="<f4")
    return len(values), sha256_bytes(values.tobytes())


def _image_hashes(encoded: list[bytes]) -> list[str]:
    if not encoded:
        raise AuditError("official record has no main-camera images")
    return [sha256_bytes(encoded[index]) for index in (0, len(encoded) // 2, len(encoded) - 1)]


def _rgb_signature(image):
    import cv2
    import numpy as np

    return cv2.resize(np.asarray(image), (32, 32), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _encoded_signatures(encoded: list[bytes]):
    import cv2
    import numpy as np

    result = []
    for index in (0, len(encoded) // 2, len(encoded) - 1):
        image = cv2.imdecode(np.frombuffer(encoded[index], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise AuditError("official RGB JPEG cannot be decoded for disambiguation")
        result.append(_rgb_signature(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    return result


def official_record(features: dict, raw: bytes, member: dict, ordinal: int, record_offset: int) -> dict:
    import cv2
    import numpy as np

    required = (
        "steps/action",
        "steps/observation/state",
        "steps/observation/image",
        "steps/language_instruction",
        "episode_metadata/file_path",
    )
    missing = [name for name in required if name not in features]
    if missing:
        raise AuditError(f"official record is missing {missing}")
    actions = np.asarray(features["steps/action"], dtype="<f4").reshape(-1, 7)
    states = np.asarray(features["steps/observation/state"], dtype="<f4").reshape(-1, 8)
    if len(actions) != len(states) or len(actions) != len(features["steps/observation/image"]):
        raise AuditError("official record action/state/image lengths differ")
    rgb_bytes = features["steps/observation/image"][0]
    rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise AuditError("official RGB JPEG cannot be decoded")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    mask = None
    if "steps/observation/segmentation" in features:
        mask = cv2.imdecode(
            np.frombuffer(features["steps/observation/segmentation"][0], np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if mask is None:
            raise AuditError("official segmentation PNG cannot be decoded")
        mask = np.squeeze(mask).astype(np.uint8)
    object_map = {}
    if "steps/obj_name_to_seg_id" in features:
        object_map = json.loads(features["steps/obj_name_to_seg_id"][0])
    extrinsics = features.get("episode_metadata/camera_calibration/primary_cam_extrinsics")
    return {
        "actions": actions,
        "states": states,
        "first_rgb": rgb,
        "first_mask": mask,
        "language": features["steps/language_instruction"][0].decode(),
        "file_path": features["episode_metadata/file_path"][0].decode(),
        "extrinsics": np.asarray(extrinsics, dtype=np.float64).reshape(4, 4) if extrinsics else None,
        "object_map": object_map,
        "action_key": action_key(actions),
        "frame_sha256": _image_hashes(features["steps/observation/image"]),
        "rgb_signatures": _encoded_signatures(features["steps/observation/image"]),
        "raw_record_sha256": sha256_bytes(raw),
        "record_bytes": len(raw),
        "record_offset": record_offset,
        "record_ordinal": ordinal,
        "member": member,
    }


def _path_settings(path: str, record: dict) -> tuple[str, ...]:
    normalized = path.replace("\\", "/").lower()
    if "/camera_view/" in normalized or "/extrinsics_camera_view/" in normalized:
        return ("camera",)
    for setting in ("language", "noise", "light"):
        if f"/{setting}/" in normalized:
            return (setting,)
    if "/env/" not in normalized:
        return ()
    if record["first_mask"] is None:
        raise AuditError("Segmentation /env/ record has no instance mask")
    import numpy as np

    mapped = set(map(int, record["object_map"].values()))
    record["extra_objects"] = len(set(map(int, np.unique(record["first_mask"]))) - mapped - {0})
    return ("objects", "background") if record["extra_objects"] else ("background",)


def _task_path_matches(path: str, selected: dict) -> bool:
    name = Path(path).name.lower()
    return name.endswith(f"{selected['task_stem'].lower()}_demo.hdf5")


def _archive_evidence(source: str, member: dict) -> dict:
    filename, size, lfs_sha = ARCHIVES[source]["files"][member["disk"]]
    return {
        "archive_repo": REVISIONS[source]["repo"],
        "archive_revision": REVISIONS[source]["sha"],
        "archive_volume": filename,
        "archive_volume_bytes": size,
        "archive_lfs_sha256": lfs_sha,
        "member": member["name"],
        "member_disk": member["disk"],
        "member_offset": member["offset"],
        "member_bytes": member["bytes"],
        "member_compressed_bytes": member["compressed_bytes"],
        "member_crc32": member["crc32"],
    }


def _verify_record_mapping(record: dict, target: dict) -> tuple[float, float]:
    import numpy as np

    if record["actions"].shape != target["actions"].shape or record["states"].shape != target["states"].shape:
        return math.inf, math.inf
    action_error = float(np.max(np.abs(record["actions"] - target["actions"]), initial=0))
    state_error = float(np.max(np.abs(record["states"] - target["states"]), initial=0))
    return action_error, state_error


def _match_reordered(record: dict, targets: dict, selected: dict, direct_rgb) -> tuple[dict, str] | None:
    candidates = [
        row
        for row in targets.get(record["action_key"], [])
        if row["suite"] == selected["suite"] and row["task"] == selected["task"]
    ]
    exact = []
    for target in candidates:
        action_error, state_error = _verify_record_mapping(record, target)
        if action_error <= 1e-6 and state_error <= 1e-6:
            exact.append((target, action_error, state_error))
    if len(exact) == 1:
        target, record["max_action_error"], record["max_state_error"] = exact[0]
        return target, "task,length,action,state"
    if len(exact) > 1:
        import numpy as np

        ranked = []
        for item in exact:
            target_rgb = direct_rgb(item[0])
            loss = float(
                np.mean(
                    [
                        np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 255
                        for left, right in zip(record["rgb_signatures"], target_rgb)
                    ]
                )
            )
            ranked.append((loss, item[0]["episode_index"], item))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if ranked[0][0] <= 0.05 and ranked[0][0] + 0.005 < ranked[1][0]:
            target, record["max_action_error"], record["max_state_error"] = ranked[0][2]
            record["rgb_disambiguation_loss"] = ranked[0][0]
            return target, "task,length,action,state,first/middle/last RGB"
    return None


def _reference_row(record: dict, target: dict, requested: dict, setting: str, criterion: str) -> dict:
    logical = f"lerobot:{target['episode_index']}"
    reference_id = "ref-" + sha256_bytes(
        canonical_json((requested["suite"], setting, logical, record["raw_record_sha256"])).encode()
    )[:20]
    return {
        "reference_id": reference_id,
        "suite": requested["suite"],
        "setting": setting,
        "requested_task": requested["task"],
        "mapped_task": target["task"],
        "logical_episode": logical,
        "lerobot_episode_index": target["episode_index"],
        "lerobot_revision": REVISIONS["lerobot"]["sha"],
        "action_length": len(record["actions"]),
        "action_sha256": record["action_key"][1],
        "max_action_error": record["max_action_error"],
        "max_state_error": record["max_state_error"],
        "disambiguation": criterion,
        "confidence": "high" if requested["task"] == target["task"] else "medium",
        "file_path": record["file_path"],
        "raw_record_sha256": record["raw_record_sha256"],
        "record_bytes": record["record_bytes"],
        "record_offset": record["record_offset"],
        "record_ordinal": record["record_ordinal"],
        "frame_sha256": record["frame_sha256"],
        "extra_objects": record.get("extra_objects"),
        "tfrecord_crc_valid": record.get("tfrecord_crc_valid", False),
        **_archive_evidence(record["member"]["archive"], record["member"]),
        "_record": record,
    }


def stable_reference_split(rows: list[dict]) -> list[dict]:
    if len(rows) != 8 or len({row["logical_episode"] for row in rows}) != 8:
        raise AuditError("reference split requires eight unique logical episodes")
    result = sorted(
        (dict(row) for row in rows),
        key=lambda row: (stable_seed("reference-split", row["reference_id"]), row["reference_id"]),
    )
    for index, row in enumerate(result):
        row["split"] = "calibration" if index < 6 else "holdout"
    return result


def _write_official_references(output: Path, rows: list[dict]) -> None:
    import h5py

    path = output / "evidence/official_references.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".hdf5.partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w") as handle:
        handle.attrs["version"] = 2
        handle.attrs["count"] = len(rows)
        for row in rows:
            record = row["_record"]
            group = handle.create_group(row["reference_id"])
            group.create_dataset("actions", data=record["actions"], compression="lzf")
            group.create_dataset("state", data=record["states"], compression="lzf")
            group.create_dataset("first_rgb", data=record["first_rgb"], compression="lzf")
            if record["first_mask"] is not None:
                group.create_dataset("first_mask", data=record["first_mask"], compression="lzf")
            if record["extrinsics"] is not None:
                group.create_dataset("extrinsics", data=record["extrinsics"])
            text_dataset(group, "language", record["language"])
            text_dataset(group, "file_path", record["file_path"])
            text_dataset(group, "object_map_json", canonical_json(record["object_map"]))
            text_dataset(group, "mapping_json", canonical_json({k: v for k, v in row.items() if k != "_record"}))
        handle.flush()
    os.replace(partial, path)


def recover_official_references(
    output: Path, evidence_root: Path, archive_root: Path, episodes: list[dict], selections: dict
) -> tuple[list[dict], dict]:
    import numpy as np

    archives = {
        source: LocalZip(source, archive_root)
        for source in ("rlds", "camera", "segmentation")
    }
    rlds_info = archives["rlds"].json("dataset_info.json")
    shard_lengths = list(map(int, rlds_info["splits"][0]["shardLengths"]))
    if len(shard_lengths) != 1024 or sum(shard_lengths) != 14347:
        raise AuditError("pinned RLDS shardLengths no longer describe 14,347 episodes")

    targets_by_key = defaultdict(list)
    targets_by_suite = defaultdict(list)
    temporary_root = evidence_root / "reference_work"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
        parquet_root = Path(temporary)
        target_jobs = []
        for suite, selected in selections.items():
            candidates = sorted(
                (row for row in episodes if row["tasks"][0] == selected["task"]),
                key=lambda row: (stable_seed("official-reference", suite, row["episode_index"]), row["episode_index"]),
            )[:128]
            for episode in candidates:
                target_jobs.append((suite, selected, episode))

        def load_target(job):
            suite, selected, episode = job
            relative = parquet_path(episode["episode_index"])
            local = evidence_root / "lerobot" / relative
            if not local.exists():
                hub_download("lerobot", relative, evidence_root / "lerobot")
            actions, states = parquet_episode(local)
            return {
                "episode_index": episode["episode_index"],
                "suite": suite,
                "task": selected["task"],
                "actions": actions,
                "states": states,
                "action_key": action_key(actions),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            for target in executor.map(load_target, target_jobs):
                targets_by_key[target["action_key"]].append(target)
                targets_by_suite[target["suite"]].append(target)

        def rlds_record(target: dict, validate_crc: bool = False) -> dict:
            shard, wanted = ordinal_to_shard(target["episode_index"], shard_lengths)
            member, data = archives["rlds"].shard(shard)
            for ordinal, offset, raw, features in tfrecord_examples(data, validate_crc=False):
                if ordinal == wanted:
                    if validate_crc:
                        list(tfrecord_examples(data[offset : offset + 12 + len(raw) + 4], validate_crc=True))
                    return official_record(features, raw, member, ordinal, offset)
            raise AuditError(f"RLDS shard {shard} lacks ordinal {wanted}")

        rgb_cache = {}

        def direct_rgb(target: dict):
            index = target["episode_index"]
            if index not in rgb_cache:
                rgb_cache[index] = rlds_record(target)["rgb_signatures"]
            return rgb_cache[index]

        pools = defaultdict(list)
        for suite, selected in selections.items():
            counts = Counter()
            candidates = targets_by_suite[suite]
            for start in range(0, len(candidates), 4):
                batch = candidates[start : start + 4]
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    records = list(executor.map(lambda target: rlds_record(target, validate_crc=True), batch))
                for target, record in zip(batch, records):
                    record["tfrecord_crc_valid"] = True
                    normalized_path = record["file_path"].replace("\\", "/").lower()
                    setting = next(
                        (name for name in ("language", "noise") if f"/{name}/" in normalized_path),
                        None,
                    )
                    if setting not in {"language", "noise"} or counts[setting] >= 8:
                        continue
                    if not _task_path_matches(record["file_path"], selected):
                        raise AuditError(f"RLDS ordinal {target['episode_index']} task/file_path mismatch")
                    action_error, state_error = _verify_record_mapping(record, target)
                    if action_error > 1e-6 or state_error > 1e-6:
                        raise AuditError(f"RLDS ordinal {target['episode_index']} differs from LeRobot")
                    record["max_action_error"], record["max_state_error"] = action_error, state_error
                    pools[(suite, setting)].append((record, target, "ordinal,task,length,action,state"))
                    counts[setting] += 1
                if counts["language"] >= 8 and counts["noise"] >= 8:
                    break
            print(canonical_json({"reference_scan": "rlds", "suite": suite, **counts}), flush=True)

        scan_limits = {"camera": (256,), "segmentation": (1024,)}
        wanted_settings = {
            "camera": {"camera"},
            "segmentation": {"objects", "background", "light"},
        }
        scan_counts = {}
        for source in ("camera", "segmentation"):
            ordered_shards = sorted(
                range(ARCHIVES[source]["shards"]),
                key=lambda index: (stable_seed("official-shard", source, index), index),
            )
            scanned = 0
            seen = set()
            for limit in scan_limits[source]:
                shard_batch = ordered_shards[scanned:limit]

                def read_shard(shard):
                    member, data = archives[source].shard(shard)
                    return shard, member, data

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
                try:
                    loaded = executor.map(read_shard, shard_batch)
                    for batch_index, (shard, member, data) in enumerate(loaded, 1):
                        for ordinal, offset, raw, features in tfrecord_examples(data, validate_crc=False):
                            path = features.get("episode_metadata/file_path", [b""])[0].decode()
                            selected = next(
                                (item for item in selections.values() if _task_path_matches(path, item)), None
                            )
                            if selected is None:
                                continue
                            record = official_record(features, raw, member, ordinal, offset)
                            settings = tuple(
                                name for name in _path_settings(path, record) if name in wanted_settings[source]
                            )
                            if not settings:
                                continue
                            match = _match_reordered(record, targets_by_key, selected, direct_rgb)
                            if match is None:
                                continue
                            list(
                                tfrecord_examples(
                                    data[offset : offset + 12 + len(raw) + 4], validate_crc=True
                                )
                            )
                            record["tfrecord_crc_valid"] = True
                            target, criterion = match
                            for setting in settings:
                                unique = (source, setting, member["name"], ordinal, target["episode_index"])
                                if unique in seen:
                                    continue
                                seen.add(unique)
                                pools[(selected["suite"], setting)].append((record, target, criterion))
                        del data
                        if batch_index % 16 == 0 or batch_index == len(shard_batch):
                            print(
                                canonical_json(
                                    {
                                        "reference_scan": source,
                                        "shards": scanned + batch_index,
                                        "matches": sum(
                                            len(
                                                {
                                                    item[1]["episode_index"]
                                                    for item in pools[(suite, setting)]
                                                }
                                            )
                                            for suite in SUITES
                                            for setting in wanted_settings[source]
                                        ),
                                    }
                                ),
                                flush=True,
                            )
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
                scanned = limit
                if all(
                    len({item[1]["episode_index"] for item in pools[(suite, setting)]}) >= 8
                    for suite in SUITES
                    for setting in wanted_settings[source]
                ):
                    break
            scan_counts[source] = scanned

    language_allocations = {}
    for requested_index, requested_suite in enumerate(SUITES):
        allocation = []
        for source_suite in SUITES:
            source = sorted(
                pools[(source_suite, "language")],
                key=lambda item: (
                    stable_seed(
                        "reference-pick",
                        source_suite,
                        "language",
                        item[1]["episode_index"],
                        item[0]["raw_record_sha256"],
                    ),
                    item[1]["episode_index"],
                ),
            )
            unique_source = []
            logical = set()
            for item in source:
                if item[1]["episode_index"] not in logical:
                    logical.add(item[1]["episode_index"])
                    unique_source.append(item)
            if len(unique_source) < 8:
                raise AuditError(f"{source_suite}/language: only {len(unique_source)} unambiguous mappings")
            allocation.extend(unique_source[requested_index * 2 : requested_index * 2 + 2])
        language_allocations[requested_suite] = allocation

    rows = []
    for suite in SUITES:
        for setting in SETTINGS:
            candidates = language_allocations[suite] if setting == "language" else pools[(suite, setting)]
            candidates = sorted(
                candidates,
                key=lambda item: (
                    stable_seed("reference-pick", suite, setting, item[1]["episode_index"], item[0]["raw_record_sha256"]),
                    item[1]["episode_index"],
                ),
            )
            unique = []
            logical = set()
            for item in candidates:
                if item[1]["episode_index"] not in logical:
                    logical.add(item[1]["episode_index"])
                    unique.append(item)
            if len(unique) < 8:
                raise AuditError(f"{suite}/{setting}: only {len(unique)} unambiguous official mappings")
            group = [
                _reference_row(record, target, selections[suite], setting, criterion)
                for record, target, criterion in unique[:8]
            ]
            group = stable_reference_split(group)
            if setting == "language" and len(
                {row["_record"]["language"].lower() for row in group if row["split"] == "calibration"}
            ) < 3:
                raise AuditError(f"{suite}/language calibration has fewer than three distinct targets")
            rows.extend(group)

    public_rows = [{key: value for key, value in row.items() if key != "_record"} for row in rows]
    if len(rows) != 192 or Counter(row["setting"] for row in rows) != Counter({setting: 32 for setting in SETTINGS}):
        raise AuditError("official reference mapping cardinality is not 192 / 32 per setting")
    if any(row["max_action_error"] > 1e-6 or row["max_state_error"] > 1e-6 for row in rows):
        raise AuditError("official reference mapping exceeds 1e-6 action/state tolerance")
    if not all(row["tfrecord_crc_valid"] for row in rows):
        raise AuditError("selected official reference lacks a valid TFRecord CRC")
    _write_official_references(output, rows)
    atomic_jsonl(output / "evidence/reference_mapping.jsonl", public_rows)
    return public_rows, {
        "count": len(rows),
        "per_setting": dict(Counter(row["setting"] for row in rows)),
        "calibration": sum(row["split"] == "calibration" for row in rows),
        "holdout": sum(row["split"] == "holdout" for row in rows),
        "scan_shards": scan_counts,
        "smoke": {setting: any(row["setting"] == setting for row in rows) for setting in SETTINGS},
    }


def protocol(qwen_path: Path) -> dict:
    prompts_hash = sha256_bytes(canonical_json(QWEN_STYLES).encode())
    return {
        "name": "LIBERO-Plus RGB-D 72 representative samples",
        "version": 2,
        "research_type": "experiment_iteration",
        "hypothesis": "A finite, unambiguous join from pinned RLDS/Camera/Segmentation records to pinned LeRobot episodes can calibrate six perturbation probes without indexing all 14,347 episodes.",
        "master_seed": MASTER_SEED,
        "suites": list(SUITES),
        "settings": list(SETTINGS),
        "slots_per_task_setting": 3,
        "attempts_per_slot": 3,
        "expected_episodes": 72,
        "maximum_attempts": 216,
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "fps": 20,
        "source_revisions": REVISIONS,
        "qwen": {
            "path": str(qwen_path),
            "revision": REVISIONS["qwen"]["sha"],
            "decoding": {
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_new_tokens": 96,
            },
            "prompt_styles": list(QWEN_STYLES),
            "prompt_sha256": prompts_hash,
        },
        "replay": {
            "semantics": "set initial state once, execute every full source action, save post-step observations only where keep_mask is true",
            "action_match_max_abs_error": 1e-6,
        },
        "depth": {
            "unit": "meter",
            "conversion": "robosuite.utils.camera_utils.get_real_depth_map",
            "valid_layout": "[T,2,H,W], camera order front,wrist",
        },
        "coordinates": {
            "T_world_cam": "OpenCV camera coordinates to MuJoCo world; +x right, +y down, +z forward",
            "T_cam_world": "inverse of T_world_cam",
            "pixel": "(u,v), origin top-left",
        },
        "selection_bias": "Only successful trajectories are retained; attempts.jsonl is required for attempt accounting.",
        "excluded": [
            "target object pose variants",
            "robot initial state variants",
            "full 14,347-episode local index",
            "full archive extraction",
            "official depth truth",
            "RLDS/LeRobot export",
        ],
    }


def audit(args) -> None:
    output = args.output.resolve()
    evidence_root = output / "evidence"
    metadata_root = evidence_root / "lerobot/meta"
    evidence = []
    for relative in ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        item = hub_download("lerobot", f"meta/{relative}", evidence_root / "lerobot")
        evidence.append({"kind": "official_metadata", "revision": REVISIONS["lerobot"]["sha"], **item})
    item = hub_download("lerobot", "norm_stats.json", evidence_root / "lerobot")
    evidence.append({"kind": "official_metadata", "revision": REVISIONS["lerobot"]["sha"], **item})

    info = json.loads((metadata_root / "info.json").read_text())
    tasks = read_jsonl(metadata_root / "tasks.jsonl")
    episodes = read_jsonl(metadata_root / "episodes.jsonl")
    if (info["total_episodes"], info["total_tasks"], len(episodes), len(tasks)) != (14347, 40, 14347, 40):
        raise AuditError("pinned LeRobot metadata cardinality changed")

    qwen_path = args.qwen_path.expanduser().resolve()
    required_qwen = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    )
    missing = [name for name in required_qwen if not (qwen_path / name).is_file()]
    if missing:
        raise AuditError(f"immutable Qwen snapshot is incomplete at {qwen_path}: {missing}")
    qwen_hashes = {
        name: {"bytes": (qwen_path / name).stat().st_size, "sha256": sha256_file(qwen_path / name)}
        for name in required_qwen
    }
    evidence.append(
        {
            "kind": "language_model",
            "repo": REVISIONS["qwen"]["repo"],
            "revision": REVISIONS["qwen"]["sha"],
            "path": str(qwen_path),
            "files": qwen_hashes,
        }
    )

    rankings = ranked_tasks(tasks, episodes)
    episode_offsets = {}
    offset = 0
    for row in episodes:
        episode_offsets[row["episode_index"]] = offset
        offset += row["length"]
    parquet_root = evidence_root / "lerobot"
    source_root = evidence_root / "source_demos"
    selections = {}
    for suite in SUITES:
        last_error = None
        for candidate in rankings[suite]:
            relative = source_filename(candidate)
            source_path = source_root / relative
            downloaded_source = False
            downloaded_parquets = []
            source_evidence = None
            try:
                if not source_path.exists():
                    if args.no_download_sources:
                        raise AuditError(f"missing source demo: {source_path}")
                    source_evidence = hub_download("source", relative, source_root)
                    downloaded_source = True
                public = [row for row in episodes if row["tasks"][0] == candidate["task"]]
                public = sorted(
                    public,
                    key=lambda row: (
                        stable_seed("source_match", suite, row["episode_index"]),
                        row["episode_index"],
                    ),
                )[: args.max_match_episodes]
                match = choose_source_matches(source_path, public, parquet_root, downloaded_parquets)
                match["source_xml_assets"] = source_xml_asset_evidence(
                    source_path, match["selected"]
                )
                selections[suite] = {
                    **candidate,
                    "bddl": str(canonical_bddl(suite, candidate["task"]).resolve()),
                    "bddl_sha256": sha256_file(canonical_bddl(suite, candidate["task"])),
                    "source": match,
                }
                source_evidence = source_evidence or {
                    "path": str(source_path),
                    "bytes": source_path.stat().st_size,
                    "sha256": sha256_file(source_path),
                }
                evidence.append(
                    {
                        "kind": "source_hdf5",
                        "revision": REVISIONS["source"]["sha"],
                        **source_evidence,
                    }
                )
                break
            except Exception as error:
                last_error = str(error)
                if downloaded_source and source_path.exists():
                    source_path.unlink()
                for path in downloaded_parquets:
                    path.unlink(missing_ok=True)
        if suite not in selections:
            raise AuditError(f"{suite}: no ranked task yielded three source demos: {last_error}")

    archive_root = evidence_root / "archives"
    archive_evidence = download_official_archives(archive_root)
    reference_rows, reference_summary = recover_official_references(
        output, evidence_root, archive_root, episodes, selections
    )
    evidence.extend(archive_evidence)

    audit_result = {
        "complete": True,
        "task_rankings": rankings,
        "selected_tasks": selections,
        "episode_global_offsets": {
            str(selection["source"]["selected"][0]["public_episode"]): episode_offsets[
                selection["source"]["selected"][0]["public_episode"]
            ]
            for selection in selections.values()
        },
        "official_references": reference_summary,
        "official_setting_labels": {
            "available": True,
            "method": "pinned archive file_path plus object-mask cardinality, followed by task/length/action/state/RGB disambiguation",
            "mapping_path": str((output / "evidence/reference_mapping.jsonl").resolve()),
            "hdf5_path": str((output / "evidence/official_references.hdf5").resolve()),
        },
    }
    atomic_json(output / "protocol.json", protocol(qwen_path))
    atomic_json(output / "audit.json", audit_result)
    atomic_jsonl(output / "evidence.jsonl", evidence)
    print(canonical_json({"audit": "complete", "selected_tasks": {s: v["task"] for s, v in selections.items()}}))


def bddl_block(text: str, name: str) -> str:
    start = text.index(f"(:{name}")
    depth = 0
    for index in range(start, len(text)):
        depth += text[index] == "("
        depth -= text[index] == ")"
        if depth == 0:
            return text[start : index + 1]
    raise ValueError(f"unterminated BDDL block: {name}")


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


def benchmark_map() -> dict:
    path = ROOT / "libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(path.read_text())
    return ast.literal_eval(tree.body[0].value)


def evenly_spaced(items: list, count=8):
    if len(items) < count:
        raise AuditError(f"candidate pool has only {len(items)} items, need {count}")
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def camera_matrix(params: dict):
    import numpy as np

    def rx(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)))

    def ry(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))

    def rz(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    degrees = math.pi / 180
    return (
        ry(params["endpoint_vertical"] * degrees)
        @ rz(params["endpoint_horizontal"] * degrees)
        @ rz(params["horizontal"] * degrees)
        @ ry(params["vertical"] * degrees)
        @ rx(0)
    )


def rotation_distance(left: dict, right: dict) -> float:
    import numpy as np

    delta = camera_matrix(left).T @ camera_matrix(right)
    return math.degrees(math.acos(float(np.clip((np.trace(delta) - 1) / 2, -1, 1))))


def parse_camera_name(name: str):
    match = re.search(
        r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_(\d+)", name
    )
    if not match:
        return None
    h, v, scale, eh, ev, state = map(int, match.groups())
    return {
        "horizontal": h,
        "vertical": v,
        "scale": scale / 100,
        "endpoint_horizontal": eh,
        "endpoint_vertical": ev,
        "initstate": state,
    }


def camera_pool(test_names: list[str]) -> list[dict]:
    tests = [
        parsed
        for name in test_names
        if (parsed := parse_camera_name(name)) is not None
        and parsed["initstate"] == 0
        and any(parsed[key] for key in ("horizontal", "vertical", "endpoint_horizontal", "endpoint_vertical"))
    ]
    pool = []
    for horizontal in (-50, -40, -30, -20, -10, 10, 20, 30, 40, 50):
        for vertical in (-20, -10, 0, 10, 20):
            for scale in (0.9, 1.0, 1.1):
                item = {
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "scale": scale,
                    "endpoint_horizontal": 0,
                    "endpoint_vertical": 0,
                }
                distances = [rotation_distance(item, test) for test in tests]
                if not distances or min(distances) >= 5 - 1e-9:
                    item["min_test_angle_deg"] = min(distances, default=180.0)
                    item["default_angle_deg"] = rotation_distance(
                        item,
                        {
                            "horizontal": 0,
                            "vertical": 0,
                            "scale": 1,
                            "endpoint_horizontal": 0,
                            "endpoint_vertical": 0,
                        },
                    )
                    pool.append(item)
    return sorted(pool, key=lambda item: (item["default_angle_deg"], canonical_json(item)))


def protected_language_terms(canonical: str, bddl: Path) -> tuple[list[str], list[str]]:
    lower = canonical.lower()
    entities = []
    for _, category in bddl_objects(bddl):
        words = category.rstrip("0123456789_").replace("_", " ").split()
        bases = [" ".join(words[start:]) for start in range(len(words))]
        bases += [base[:-1] for base in bases if base.endswith("s")]
        matches = []
        for base in bases:
            match = re.search(
                rf"\b{re.escape(base)}(?:\s+(?:box|bottle|bowl|can|container))?\b", lower
            )
            if match:
                matches.append(match.group())
        term = max(matches, key=len, default=None)
        if term and term not in entities:
            entities.append(term)
    relations = [term for term in RELATIONS if term in lower]
    return entities, relations


def valid_rewrite(canonical: str, rewrite: str, bddl: Path) -> bool:
    normalized = " ".join(rewrite.lower().strip(" \"'\n\t.").split())
    original = " ".join(canonical.lower().strip(" .").split())
    entities, relations = protected_language_terms(canonical, bddl)
    actions = [phrase for phrase in ACTION_PHRASES if phrase in original]
    return bool(normalized and normalized != original) and all(
        term in normalized for term in (*entities, *relations, *actions)
    )


def qwen_rewrites(canonical: str, bddl: Path, model_path: Path, tokenizer, model) -> list[dict]:
    import torch

    entities, relations = protected_language_terms(canonical, bddl)
    actions = [phrase for phrase in ACTION_PHRASES if phrase in canonical.lower()]
    rows = []
    observed = []
    for index, style in enumerate(QWEN_STYLES):
        prompt = (
            "Rewrite one robot instruction. Preserve every named entity, action goal, and spatial "
            "relation exactly; add no objects or steps. Output only one rewritten instruction.\n"
            f"Required entities (copy verbatim): {entities}\n"
            f"Required relations (copy verbatim): {relations}\n"
            f"Required action phrases (copy verbatim): {actions}\n"
            f"Style: {style}\nInstruction: {canonical}"
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=96,
            )
        rewrite = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        rewrite = " ".join(rewrite.strip(" \"'\n\t").split())
        valid = valid_rewrite(canonical, rewrite, bddl)
        duplicate = rewrite.lower() in {row["instruction"].lower() for row in rows}
        observed.append({"style": index + 1, "rewrite": rewrite, "valid": valid, "duplicate": duplicate})
        if valid and not duplicate:
            rows.append(
                {
                    "candidate_id": f"language-{index + 1}",
                    "instruction": rewrite,
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode()),
                    "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
                }
            )
    if len(rows) != len(QWEN_STYLES):
        raise AuditError(
            f"Qwen produced {len(rows)}/{len(QWEN_STYLES)} distinct hard-valid rewrites for "
            f"{canonical}: {canonical_json(observed)}"
        )
    return rows


def noise_candidates(algorithm: str, slot: int, seed: int) -> list[dict]:
    import numpy as np

    rng = np.random.default_rng(seed)
    severity_center = (0.2, 0.5, 0.8)[slot - 1]
    result = []
    for index, severity in enumerate(np.clip(severity_center + np.linspace(-0.09, 0.09, 8), 0.05, 0.95)):
        if algorithm == "motion_blur":
            params = {
                "kernel": int(3 + 12 * severity) | 1,
                "sigma": round(0.65 + 6.1 * float(severity), 4),
                "angle": round(float(rng.uniform(-44.5, 44.5)), 4),
            }
        elif algorithm == "gaussian_blur":
            params = {"sigma": round(0.55 + 8.9 * float(severity), 4)}
        elif algorithm == "zoom_blur":
            params = {
                "maximum": round(1.055 + 0.47 * float(severity), 4),
                "steps": int(7 + 18 * severity),
            }
        elif algorithm == "fog":
            params = {
                "alpha": round(0.35 + 4.4 * float(severity), 4),
                "scale": int(10 + 36 * severity),
            }
        else:
            params = {
                "sigma": round(0.35 + 2.0 * float(severity), 4),
                "delta": int(1 + 4 * severity),
                "iterations": 2,
            }
        result.append(
            {
                "candidate_id": f"{algorithm}-{index + 1}",
                "algorithm": algorithm,
                "severity": round(float(severity), 6),
                "parameters": params,
                "benchmark_tuple_equal": False,
            }
        )
    return result


def choose_three(candidates: list[dict], score) -> tuple[list[dict], list[str]]:
    if len(candidates) < 3:
        raise AuditError(f"only {len(candidates)} candidates available")
    ordered = sorted(candidates, key=lambda item: (score(item), canonical_json(item)))
    return ordered[:3], [item["candidate_id"] for item in candidates]


def create_language_cache(audit_data: dict, output: Path, qwen_path: Path, gpu: int) -> dict:
    path = output / "language_candidates.json"
    if path.exists():
        return json.loads(path.read_text())
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": gpu},
        attn_implementation="eager",
    ).eval()
    result = {}
    try:
        for suite, selected in audit_data["selected_tasks"].items():
            bddl = Path(selected["bddl"])
            result[suite] = qwen_rewrites(selected["task"], bddl, qwen_path, tokenizer, model)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(path, result)
    return result


def build_candidates(setting: str, suite: str, slot: int, selected: dict, tests: list[str], languages):
    stem = selected["task_stem"]
    folder = Path(selected["bddl"]).parent
    canonical = Path(selected["bddl"])
    test_set = set(tests)
    if setting == "objects":
        base_count = len(bddl_objects(canonical))
        pool = []
        for path in folder.glob(f"{stem}_add_*.bddl"):
            if path.stem in test_set or len(bddl_objects(path)) - base_count != slot:
                continue
            pool.append(
                {
                    "candidate_id": path.stem,
                    "bddl": str(path.resolve()),
                    "bddl_sha256": sha256_file(path),
                    "extra_objects": slot,
                }
            )
        pool = sorted(pool, key=lambda item: item["candidate_id"])
        if not pool:
            raise AuditError("no object candidate survives benchmark isolation")
        candidates = []
        for index in range(8):
            base = pool[round(index * (len(pool) - 1) / 7)]
            placement_seed = stable_seed(suite, setting, slot, index, base["candidate_id"])
            candidates.append(
                {
                    **base,
                    "asset_candidate_id": base["candidate_id"],
                    "candidate_id": f"{base['candidate_id']}@placement-{index + 1}",
                    "placement_seed": placement_seed,
                    "pose_hash": sha256_bytes(
                        canonical_json((base["bddl_sha256"], placement_seed)).encode()
                    ),
                    "benchmark_object_pose_hash_equal": False,
                }
            )
        return candidates
    if setting == "background":
        pool = [
            {
                "candidate_id": path.stem,
                "bddl": str(path.resolve()),
                "bddl_sha256": sha256_file(path),
                "asset_id": bddl_problem(path),
            }
            for pattern in (f"{stem}_table_*.bddl", f"{stem}_tb_*.bddl")
            for path in folder.glob(pattern)
            if path.stem not in test_set
        ]
        pool = sorted(pool, key=lambda item: item["asset_id"])
        return evenly_spaced(pool)
    if setting == "light":
        pool = [
            {
                "candidate_id": path.stem,
                "bddl": str(path.resolve()),
                "bddl_sha256": sha256_file(path),
                "asset_id": bddl_problem(path),
            }
            for path in folder.glob(f"{stem}_light_*.bddl")
            if path.stem not in test_set
        ]
        pool = sorted(pool, key=lambda item: item["asset_id"])
        return evenly_spaced(pool)
    if setting == "camera":
        pool = camera_pool(tests)
        return [
            {"candidate_id": f"camera-{index + 1}", **item}
            for index, item in enumerate(
                evenly_spaced(pool)
            )
        ]
    if setting == "language":
        return sorted(languages[suite], key=lambda item: item["candidate_id"])
    algorithm = NOISE_ORDER[(SUITES.index(suite) * 3 + slot - 1) % len(NOISE_ORDER)]
    candidates = noise_candidates(algorithm, slot, stable_seed(suite, setting, slot))
    return candidates


def validate_manifest_rows(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["setting"], row["variant_slot"])].append(row)
    if len(rows) != 216 or len({row["attempt_id"] for row in rows}) != 216:
        raise AuditError("manifest must contain 216 unique attempts")
    if len(groups) != 72 or any(
        sorted(item["retry_rank"] for item in attempts) != [1, 2, 3]
        for attempts in groups.values()
    ):
        raise AuditError("manifest must contain three ordered retries for each of 72 slots")
    if any(len(row["candidate_ids"]) != 8 for row in rows):
        raise AuditError("every attempt must reference an eight-candidate pool")
    for row in rows:
        calibration = row.get("calibration", {})
        calibration_ids = calibration.get("reference_ids", [])
        holdout_ids = calibration.get("holdout_reference_ids", [])
        if len(calibration_ids) != 6 or len(holdout_ids) != 2 or set(calibration_ids) & set(holdout_ids):
            raise AuditError("manifest calibration/holdout references must be a disjoint 6/2 split")
        if calibration.get("official_reference_count") != 8:
            raise AuditError("manifest official_reference_count must be eight")
        if not math.isfinite(calibration.get("probe_loss", math.nan)):
            raise AuditError("manifest probe_loss must be finite")
        if calibration.get("target_id") not in calibration_ids:
            raise AuditError("manifest target must come from calibration references")
    noise = Counter(
        attempts[0]["randomization"]["algorithm"]
        for (_, setting, _), attempts in groups.items()
        if setting == "noise"
    )
    if any(noise[algorithm] < 2 for algorithm in NOISE_ORDER):
        raise AuditError("all five noise algorithms must cover at least two final slots")
    for suite in SUITES:
        primary = [
            min(groups[(suite, "language", slot)], key=lambda row: row["retry_rank"])["randomization"][
                "instruction"
            ].lower()
            for slot in (1, 2, 3)
        ]
        if len(set(primary)) != 3:
            raise AuditError(f"{suite}: language slot primary rewrites are not distinct")


def manifest(args) -> None:
    output = args.output.resolve()
    audit_path = output / "audit.json"
    if not audit_path.exists():
        raise AuditError("run audit before manifest")
    audit_data = json.loads(audit_path.read_text())
    if not audit_data.get("complete"):
        raise AuditError("audit is incomplete")
    configure_libero(output)
    qwen_path = args.qwen_path.expanduser().resolve()
    languages = create_language_cache(audit_data, output, qwen_path, args.gpu)
    references = load_reference_features(output, audit_data, args.gpu, languages)
    tests_by_suite = benchmark_map()
    rows = []
    probe_rows = []
    cache_path = output / "evidence/manifest_probe_cache.json"
    probe_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache_context = (PROBE_CACHE_VERSION, sha256_file(audit_path))
    candidate_cache_path = output / "evidence/manifest_candidate_cache.json"
    candidate_cache = (
        json.loads(candidate_cache_path.read_text()) if candidate_cache_path.exists() else {}
    )
    legality_path = output / "evidence/candidate_legality.json"
    legality = json.loads(legality_path.read_text()).get("slots", {}) if legality_path.exists() else {}
    for suite in SUITES:
        selected = audit_data["selected_tasks"][suite]
        tests = [name for name in tests_by_suite[suite] if name.startswith(selected["task_stem"])]
        sources = selected["source"]["selected"]
        language_primaries = set()
        for setting in SETTINGS:
            for slot in (1, 2, 3):
                candidates = build_candidates(setting, suite, slot, selected, tests, languages)
                if len(candidates) != 8:
                    raise AuditError(f"{suite}/{setting}/slot{slot}: candidate pool has {len(candidates)} != 8")
                cache_key = f"{suite}/{setting}/{slot}"
                slot_legality = legality.get(cache_key, {})
                fingerprint = sha256_bytes(
                    canonical_json((cache_context, candidates, slot_legality)).encode()
                )
                cached = probe_cache.get(cache_key)
                if cached and cached.get("fingerprint") == fingerprint:
                    chosen, calibration, probes = (
                        cached["chosen"], cached["calibration"], cached["probes"]
                    )
                else:
                    chosen, calibration, probes = calibrate_candidates(
                        output,
                        audit_data,
                        references,
                        suite,
                        setting,
                        slot,
                        candidates,
                        args.gpu,
                        candidate_cache,
                        candidate_cache_path,
                        cache_context,
                        slot_legality,
                    )
                    probe_cache[cache_key] = {
                        "fingerprint": fingerprint,
                        "chosen": chosen,
                        "calibration": calibration,
                        "probes": probes,
                    }
                    atomic_json(cache_path, probe_cache)
                if setting == "language":
                    primary = next(
                        item for item in chosen if item["candidate_id"] not in language_primaries
                    )
                    language_primaries.add(primary["candidate_id"])
                    chosen = [primary, *(item for item in chosen if item is not primary)]
                candidate_ids = [item["candidate_id"] for item in candidates]
                for probe in probes:
                    probe_rows.append(
                        {
                            "suite": suite,
                            "setting": setting,
                            "variant_slot": slot,
                            "target_id": calibration["target_id"],
                            **probe,
                        }
                    )
                for retry_rank, probe in enumerate(chosen, 1):
                    attempt_id = f"{suite}.{setting}.slot{slot}.try{retry_rank}"
                    row = dict(probe["row"])
                    row.update(
                        {
                            "attempt_id": attempt_id,
                            "retry_rank": retry_rank,
                            "candidate_ids": candidate_ids,
                            "calibration": {
                                **calibration,
                                "probe_feature": probe["feature"],
                                "probe_loss": probe["loss"],
                            },
                        }
                    )
                    rows.append(row)
    validate_manifest_rows(rows)
    atomic_jsonl(output / "evidence/calibration_probes.jsonl", probe_rows)
    path = output / "sample_manifest.jsonl"
    content = "".join(canonical_json(row) + "\n" for row in rows)
    if path.exists() and path.read_text() != content:
        raise AuditError("sample_manifest.jsonl is frozen and differs from the requested manifest")
    atomic_text(path, content)
    print(canonical_json({"manifest": "frozen", "attempts": 216, "slots": 72}))


class AttemptFailure(RuntimeError):
    pass


class WorkerFailure(AttemptFailure):
    def __init__(self, message: str, worker_traceback: str = ""):
        super().__init__(message)
        self.worker_traceback = worker_traceback


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def configure_libero(output: Path) -> None:
    config_dir = output / "runtime/libero_config"
    benchmark = (ROOT / "libero/libero").resolve()
    atomic_json(
        config_dir / "config.yaml",
        {
            "benchmark_root": str(benchmark),
            "bddl_files": str(benchmark / "bddl_files"),
            "init_states": str(benchmark / "init_files"),
            "datasets": str((ROOT / "libero/datasets").resolve()),
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
    import importlib.metadata
    import mujoco
    import robosuite
    import libero.libero as libero_package

    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    conda = command_output(["conda", "list", "--json", "-p", sys.prefix])
    return {
        "code_commit": command_output(["git", "rev-parse", "HEAD"]),
        "code_dirty": bool(command_output(["git", "status", "--porcelain", "--untracked-files=no"])),
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "libero_module": str(Path(libero_package.__file__).resolve()),
        "robosuite": {"version": robosuite.__version__, "module": str(Path(robosuite.__file__).resolve())},
        "mujoco": {"version": mujoco.__version__, "module": str(Path(mujoco.__file__).resolve())},
        "conda_prefix": sys.prefix,
        "conda_packages": json.loads(conda) if conda else None,
        "python_packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "gpu_index": gpu,
        "gpu_driver": command_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=name,driver_version,uuid",
                "--format=csv,noheader",
            ]
        ),
        "egl": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
    }


def load_source_episode(row: dict) -> dict:
    import h5py
    import numpy as np

    with h5py.File(row["source_file"], "r") as handle:
        group = handle[f"data/{row['source_demo']}"]
        xml = group.attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode()
        processed_xml, source_assets = resolve_source_model_xml(xml)
        return {
            "actions": np.asarray(group["actions"], dtype=np.float64),
            "states": np.asarray(group["states"], dtype=np.float64),
            "joint_states": np.asarray(group["obs/joint_states"], dtype=np.float64),
            "ee_pos": np.asarray(group["obs/ee_pos"], dtype=np.float64),
            "ee_ori": np.asarray(group["obs/ee_ori"], dtype=np.float64),
            "source_xml": xml,
            "source_xml_processed": processed_xml,
            "source_assets": source_assets,
        }


def execution_bddl(row: dict, output: Path) -> Path:
    setting = row["setting"]
    randomization = row["randomization"]
    base = Path(randomization.get("bddl", row["canonical_bddl"]))
    if setting == "language":
        text = Path(row["canonical_bddl"]).read_text()
        text, count = re.subn(
            r"(\(:language\s+)(.+?)(\s*\))",
            lambda match: match.group(1) + randomization["instruction"] + match.group(3),
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise AttemptFailure("could not replace BDDL language")
        base = output / "evidence/generated_bddl" / f"{row['attempt_id'].replace('.', '_')}.bddl"
        atomic_text(base, text)
    if setting != "camera":
        return base
    camera = randomization
    fields = (
        camera["horizontal"],
        camera["vertical"],
        round(camera["scale"] * 100),
        camera["endpoint_horizontal"],
        camera["endpoint_vertical"],
    )
    suffix = "_".join(str(value) for value in fields)
    return base.with_name(f"{base.stem}_view_{suffix}_initstate_0.bddl")


def make_env(bddl: Path, gpu: int, horizon: int, segmentation: bool = False):
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
        camera_segmentations="instance" if segmentation else None,
        render_gpu_device_id=gpu,
        control_freq=20,
        horizon=horizon,
        ignore_done=True,
        hard_reset=True,
    )


def make_replay_env(bddl: Path, horizon: int):
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=str(bddl),
        robots=["Panda"],
        initialization_noise=None,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=horizon,
        ignore_done=True,
        hard_reset=True,
    )


def verify_source_replay(audit_data: dict, output: Path) -> None:
    import numpy as np
    import robosuite.utils.transform_utils as transform

    report_path = output / "reports/replay_semantics.json"
    script_hash = sha256_file(Path(__file__))
    limits = {
        "max_joint_error": 0.1,
        "max_eef_position_error": 0.02,
        "max_eef_orientation_error": 0.1,
    }
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (
            report.get("accepted")
            and report.get("thresholds") == limits
            and len(report.get("episodes", [])) == 12
        ):
            if report.get("version") != 1:
                report["version"] = 1
                atomic_json(report_path, report)
            return
    results = []
    for suite in SUITES:
        selected = audit_data["selected_tasks"][suite]
        for source_row in selected["source"]["selected"]:
            row = {
                "source_file": selected["source"]["source_file"],
                "source_demo": source_row["source_demo"],
            }
            source = load_source_episode(row)
            env = make_replay_env(Path(selected["bddl"]), len(source["actions"]) + 5)
            try:
                seed_environment(env, stable_seed("replay_semantics", suite, row["source_demo"]))
                env.env.reset()
                env.reset_from_xml_string(source["source_xml_processed"])
                env.sim.reset()
                env.set_state(source["states"][0])
                env.sim.forward()
                metrics = {
                    "max_state_error": 0.0,
                    "max_joint_error": 0.0,
                    "max_eef_position_error": 0.0,
                    "max_eef_orientation_error": 0.0,
                }
                for timestep, action in enumerate(source["actions"]):
                    obs, _, _, _ = env.step(action)
                    metrics["max_joint_error"] = max(
                        metrics["max_joint_error"],
                        float(
                            np.max(
                                np.abs(obs["robot0_joint_pos"] - source["joint_states"][timestep]),
                                initial=0,
                            )
                        ),
                    )
                    metrics["max_eef_position_error"] = max(
                        metrics["max_eef_position_error"],
                        float(
                            np.max(
                                np.abs(obs["robot0_eef_pos"] - source["ee_pos"][timestep]),
                                initial=0,
                            )
                        ),
                    )
                    axis_angle = transform.quat2axisangle(obs["robot0_eef_quat"])
                    metrics["max_eef_orientation_error"] = max(
                        metrics["max_eef_orientation_error"],
                        float(
                            np.max(np.abs(axis_angle - source["ee_ori"][timestep]), initial=0)
                        ),
                    )
                    if timestep + 1 < len(source["states"]):
                        metrics["max_state_error"] = max(
                            metrics["max_state_error"],
                            float(
                                np.max(
                                    np.abs(env.sim.get_state().flatten() - source["states"][timestep + 1]),
                                    initial=0,
                                )
                            ),
                        )
                final_success = bool(env.check_success())
                accepted = final_success and all(metrics[name] <= limit for name, limit in limits.items())
                results.append(
                    {
                        "suite": suite,
                        "source_demo": row["source_demo"],
                        "actions_executed": len(source["actions"]),
                        "final_success": final_success,
                        "accepted": accepted,
                        **metrics,
                    }
                )
            finally:
                env.close()
    report = {
        "version": 1,
        "accepted": all(row["accepted"] for row in results),
        "semantics": "load source XML, set initial state once, step every action, compare post-step joint/EEF observations",
        "thresholds": limits,
        "state_error_role": "reported diagnostic only; it includes contact velocities and is not a replay acceptance metric",
        "script_sha256": script_hash,
        "episodes": results,
    }
    atomic_json(report_path, report)
    if not report["accepted"]:
        raise AuditError("existing source replay semantics failed; see reports/replay_semantics.json")


def seed_environment(env, seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)


def decoded_names(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def named_state_snapshot(sim) -> dict:
    import numpy as np

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


def source_named_state(source: dict) -> dict:
    from robosuite.utils.binding_utils import MjSim

    sim = MjSim.from_xml_string(source["source_xml_processed"])
    try:
        sim.set_state_from_flattened(source["states"][0])
        sim.forward()
        return named_state_snapshot(sim)
    finally:
        sim.free()


def apply_named_state(snapshot: dict, target) -> dict:
    import numpy as np

    source_joints = set(snapshot["joints"])
    target_joints = set(decoded_names(target.sim.model.joint_names))
    common_joints = sorted(source_joints & target_joints)
    for name in common_joints:
        target.sim.data.set_joint_qpos(name, np.asarray(snapshot["joints"][name]["qpos"]))
        target.sim.data.set_joint_qvel(name, np.asarray(snapshot["joints"][name]["qvel"]))

    target_bodies = set(decoded_names(target.sim.model.body_names))
    common_mocap = []
    for name in sorted(set(snapshot["mocap"]) & target_bodies):
        target_id = target.sim.model.body_name2id(name)
        if target.sim.model.body_mocapid[target_id] >= 0:
            target.sim.data.set_mocap_pos(name, np.asarray(snapshot["mocap"][name]["pos"]))
            target.sim.data.set_mocap_quat(name, np.asarray(snapshot["mocap"][name]["quat"]))
            common_mocap.append(name)
    target.sim.data.time = snapshot["time"]
    target.sim.forward()
    return {
        "method": "joint_and_mocap_body_names",
        "common_joints": common_joints,
        "common_mocap_bodies": common_mocap,
        "source_only_joints": sorted(source_joints - target_joints),
        "target_only_joints": sorted(target_joints - source_joints),
    }


def copy_named_state(source, target) -> dict:
    return apply_named_state(named_state_snapshot(source.sim), target)


def merge_agentview_camera(source_xml: str, target_xml: str) -> str:
    source_root = ET.fromstring(source_xml)
    target_root = ET.fromstring(target_xml)
    target = next(
        (element for element in target_root.iter("camera") if element.get("name") == "agentview"),
        None,
    )
    source = next(
        (element for element in source_root.iter("camera") if element.get("name") == "agentview"),
        None,
    )
    if source is None or target is None:
        raise AuditError("agentview camera missing while merging source XML")
    for attribute in ("pos", "quat", "mode", "fovy"):
        if target.get(attribute) is not None:
            source.set(attribute, target.get(attribute))
    return ET.tostring(source_root, encoding="unicode")


def set_source_initial_state(
    env, row: dict, source: dict, gpu: int, source_env=None, source_state: dict | None = None
) -> dict:
    import numpy as np

    initial = source["states"][0]
    if row["setting"] in {"language", "noise", "camera"}:
        xml = source["source_xml_processed"]
        if row["setting"] == "camera":
            xml = merge_agentview_camera(xml, env.sim.model.get_xml())
        env.reset_from_xml_string(xml)
        env.sim.reset()
        if env.sim.get_state().flatten().shape != initial.shape:
            raise AttemptFailure("source XML state layout differs from source initial state")
        env.set_state(initial)
        env.sim.forward()
        return {
            "method": "source_xml_flat_state",
            "state_size": int(initial.size),
            "camera_override": row["setting"] == "camera",
        }
    if row["setting"] != "objects":
        if env.sim.get_state().flatten().shape != initial.shape:
            raise AttemptFailure("non-object variant changed MuJoCo state layout")
        env.set_state(initial)
        env.sim.forward()
        return {"method": "flat_state_exact", "state_size": int(initial.size)}

    if source_state is not None:
        return apply_named_state(source_state, env)
    if source_env is not None:
        return copy_named_state(source_env, env)
    source_env = make_replay_env(Path(row["canonical_bddl"]), len(source["actions"]) + 5)
    try:
        seed_environment(source_env, row["seeds"]["environment"])
        source_env.env.reset()
        source_env.reset_from_xml_string(source["source_xml_processed"])
        source_env.sim.reset()
        source_env.set_state(initial)
        source_env.sim.forward()
        return copy_named_state(source_env, env)
    finally:
        source_env.close()


def as_rgb(image):
    import numpy as np

    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image * (255 if image.max(initial=0) <= 1 else 1), 0, 255).astype(np.uint8)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise AttemptFailure(f"unexpected RGB shape: {image.shape}")
    return image


def apply_noise(image, config: dict, seed: int):
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    algorithm = config["algorithm"]
    params = config["parameters"]
    if algorithm == "gaussian_blur":
        return cv2.GaussianBlur(image, (0, 0), params["sigma"]).astype(np.uint8)
    if algorithm == "motion_blur":
        size = params["kernel"]
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2] = 1
        matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), params["angle"], 1)
        kernel = cv2.warpAffine(kernel, matrix, (size, size))
        kernel /= max(kernel.sum(), 1e-9)
        return cv2.filter2D(image, -1, kernel).astype(np.uint8)
    if algorithm == "zoom_blur":
        height, width = image.shape[:2]
        layers = []
        for zoom in np.linspace(1, params["maximum"], params["steps"]):
            resized = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
            y, x = (resized.shape[0] - height) // 2, (resized.shape[1] - width) // 2
            layers.append(resized[y : y + height, x : x + width])
        return np.mean(layers, axis=0).astype(np.uint8)
    if algorithm == "fog":
        noise = rng.normal(size=image.shape[:2]).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), params["scale"])
        noise = (noise - noise.min()) / max(float(noise.max() - noise.min()), 1e-6)
        alpha = min(0.8, params["alpha"] / 6)
        fog = np.repeat((noise * 255)[..., None], 3, axis=2)
        return np.clip(image * (1 - alpha) + fog * alpha, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(image, (0, 0), params["sigma"])
    delta = params["delta"]
    dx = cv2.GaussianBlur(rng.uniform(-delta, delta, image.shape[:2]).astype(np.float32), (0, 0), 1)
    dy = cv2.GaussianBlur(rng.uniform(-delta, delta, image.shape[:2]).astype(np.float32), (0, 0), 1)
    grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    return cv2.remap(
        blurred,
        (grid_x + dx).astype(np.float32),
        (grid_y + dy).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.uint8)


def _object_categories(names) -> set[str]:
    result = set()
    for name in names:
        normalized = re.sub(r"_\d+$", "", str(name).lower())
        if "panda" not in normalized and "mount" not in normalized:
            result.add(normalized)
    return result


def _instance_statistics(mask) -> list[float]:
    import numpy as np

    mask = np.asarray(mask)
    ids = [value for value in np.unique(mask) if value]
    if not ids:
        return [0.0] * 8
    areas, xs, ys = [], [], []
    height, width = mask.shape
    for value in ids:
        y, x = np.nonzero(mask == value)
        areas.append(len(x) / mask.size)
        xs.append(float(x.mean()) / max(width - 1, 1))
        ys.append(float(y.mean()) / max(height - 1, 1))
    return [
        float(len(ids)),
        float(np.mean(areas)),
        float(np.std(areas)),
        float(np.max(areas)),
        float(np.mean(xs)),
        float(np.mean(ys)),
        float(np.std(xs)),
        float(np.std(ys)),
    ]


def visual_feature(setting: str, rgb, mask=None, extra_objects=0, categories=(), clean=None) -> dict:
    import cv2
    import numpy as np

    rgb = np.asarray(rgb, dtype=np.uint8)
    if setting == "objects":
        if mask is None:
            raise AuditError("objects metric requires an instance mask")
        return {
            "numeric": [float(extra_objects), *_instance_statistics(mask)],
            "categories": sorted(_object_categories(categories)),
        }
    if setting == "background":
        if mask is None:
            raise AuditError("background metric requires an instance mask")
        background = np.asarray(mask) == 0
        pixels = rgb[background] if background.any() else rgb.reshape(-1, 3)
        histogram = []
        for channel in range(3):
            values, _ = np.histogram(pixels[:, channel], bins=32, range=(0, 256), density=False)
            histogram.extend((values / max(values.sum(), 1)).tolist())
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = []
        for scale in (1, 2, 4):
            sampled = gray[::scale, ::scale]
            sx = cv2.Sobel(sampled, cv2.CV_32F, 1, 0)
            sy = cv2.Sobel(sampled, cv2.CV_32F, 0, 1)
            edges.append(float(np.mean(cv2.magnitude(sx, sy))) / 255)
        return {"numeric": [*histogram, float(pixels.mean()) / 255, *edges], "categories": []}
    if setting == "light":
        values = rgb.astype(np.float32) / 255
        mean_rgb = values.reshape(-1, 3).mean(axis=0)
        chroma = mean_rgb / max(float(mean_rgb.sum()), 1e-6)
        luminance = values @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        x, y = float(chroma[0]), float(chroma[1])
        n = (x - 0.3320) / max(0.1858 - y, 1e-6)
        cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
        return {
            "numeric": [
                float(luminance.mean()),
                *map(float, chroma),
                float(np.clip(cct, 1000, 40000) / 40000),
                float((luminance < 0.1).mean()),
                float((luminance > 0.9).mean()),
            ],
            "categories": [],
        }
    if setting == "noise":
        if clean is None:
            raise AuditError("noise metric requires its clean replay frame")
        clean = np.asarray(clean, dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        clean_gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F).var()
        clean_laplacian = cv2.Laplacian(clean_gray, cv2.CV_32F).var()
        spectrum = np.abs(np.fft.rfft2(gray - gray.mean()))
        clean_spectrum = np.abs(np.fft.rfft2(clean_gray - clean_gray.mean()))
        high = spectrum[spectrum.shape[0] // 3 :, spectrum.shape[1] // 3 :].mean()
        clean_high = clean_spectrum[clean_spectrum.shape[0] // 3 :, clean_spectrum.shape[1] // 3 :].mean()
        residual = rgb.astype(np.float32) - clean.astype(np.float32)
        return {
            "numeric": [
                float(laplacian / max(clean_laplacian, 1e-6)),
                float(high / max(clean_high, 1e-6)),
                float(gray.std() / max(clean_gray.std(), 1e-6)),
                float(np.sqrt(np.mean(residual**2)) / 255),
            ],
            "categories": [],
        }
    raise AuditError(f"no visual metric for {setting}")


def language_feature(text: str) -> dict:
    words = re.findall(r"[a-z0-9]+", text.lower())
    starters = ("please", "now", "first", "could", "for", "your", "would", "the")
    return {
        "numeric": [
            float(len(words)),
            float(text.count(",")),
            float(text.count(";")),
            float(text.rstrip().endswith("?")),
            *[float(bool(words) and words[0] == item) for item in starters],
        ],
        "categories": [],
    }


def pose_distance(left, right) -> float:
    import numpy as np

    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    delta = left[:3, :3].T @ right[:3, :3]
    angle = math.acos(float(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
    translation = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
    return (angle + translation) / 2


def feature_distance(left: dict, right: dict, calibration: list[dict]) -> float:
    import numpy as np

    left_values = np.asarray(left["numeric"], dtype=float)
    right_values = np.asarray(right["numeric"], dtype=float)
    pool = np.asarray([item["numeric"] for item in calibration], dtype=float)
    if left_values.shape != right_values.shape or pool.ndim != 2 or pool.shape[1:] != left_values.shape:
        raise AuditError("metric feature dimensions differ")
    scale = np.percentile(pool, 75, axis=0) - np.percentile(pool, 25, axis=0)
    std = pool.std(axis=0)
    scale = np.where(scale > 0, scale, np.maximum(std, 1e-6))
    numeric = float(np.mean(np.abs(left_values - right_values) / scale))
    left_categories, right_categories = set(left["categories"]), set(right["categories"])
    categorical = 0.0
    components = [numeric]
    if left_categories or right_categories:
        categorical = 1 - len(left_categories & right_categories) / max(len(left_categories | right_categories), 1)
        components.append(categorical)
    return float(sum(components) / len(components))


def probe_observation(
    row: dict,
    output: Path,
    gpu: int,
    segmentation: bool = False,
    camera_only: bool = False,
    recorded_post_state: bool = False,
    source: dict | None = None,
    source_env=None,
    source_state: dict | None = None,
):
    import mujoco
    import numpy as np
    from robosuite.utils import camera_utils

    source = source or load_source_episode(row)
    keep = np.asarray(row["keep_mask"], dtype=bool)
    bddl = execution_bddl(row, output)
    placement_seed = row["randomization"].get("placement_seed", row["seeds"]["placement"])
    env = make_env(bddl, gpu, len(source["actions"]) + 5, segmentation=segmentation)
    try:
        seed_environment(env, stable_seed(row["seeds"]["environment"], placement_seed))
        env.env.reset()
        set_source_initial_state(env, row, source, gpu, source_env, source_state)
        render_context = env.sim._render_context_offscreen
        render_context.gl_ctx.make_current()
        for texture_id in range(env.sim.model.ntex):
            mujoco.mjr_uploadTexture(env.sim.model._model, render_context.con, texture_id)
        extrinsics = camera_utils.get_camera_extrinsic_matrix(env.sim, "agentview")
        if camera_only:
            return {"extrinsics": extrinsics}
        wanted = int(np.flatnonzero(keep)[0])
        if recorded_post_state:
            if wanted + 1 >= len(source["states"]):
                raise AuditError("source demo lacks the requested post-step state")
            obs = env.regenerate_obs_from_state(source["states"][wanted + 1])
            mask = None
            if segmentation:
                key = next((name for name in obs if name.startswith("agentview_segmentation")), None)
                if key is None:
                    raise AuditError("probe environment did not return agentview instance segmentation")
                mask = np.squeeze(np.asarray(obs[key])).astype(np.uint8)
            return {"rgb": as_rgb(obs["agentview_image"]), "mask": mask, "extrinsics": extrinsics}
        for timestep, action in enumerate(source["actions"]):
            obs, _, _, _ = env.step(action)
            if timestep == wanted:
                mask = None
                if segmentation:
                    key = next(
                        (name for name in obs if name.startswith("agentview_segmentation")), None
                    )
                    if key is None:
                        raise AuditError("probe environment did not return agentview instance segmentation")
                    mask = np.squeeze(np.asarray(obs[key])).astype(np.uint8)
                return {"rgb": as_rgb(obs["agentview_image"]), "mask": mask, "extrinsics": extrinsics}
        raise AuditError("probe did not reach first keep_mask timestep")
    finally:
        env.close()


def _clean_reference_frame(selected: dict, actions, output: Path, gpu: int, cache: dict):
    key = action_key(actions)
    source_file = selected["source"]["source_file"]
    cache_key = ("clean_frame", source_file, key)
    if cache_key in cache:
        return cache[cache_key]
    matches = []
    demos_key = ("source_demos", source_file)
    if demos_key not in cache:
        cache[demos_key] = source_demos(Path(source_file), actions_only=True)
    for demo in cache[demos_key]:
        mask = unique_subsequence_mask(demo["actions"], actions)
        if mask is not None:
            matches.append((demo["id"], mask))
    if len(matches) != 1:
        raise AuditError(f"noise reference clean replay matched {len(matches)} source demos")
    demo_id, mask = matches[0]
    source = selected["source"]
    row = {
        "attempt_id": f"probe.clean.{selected['suite']}.{key[1][:12]}",
        "suite": selected["suite"],
        "setting": "noise",
        "source_demo": demo_id,
        "source_file": source_file,
        "source_file_sha256": source["source_sha256"],
        "keep_mask": mask.astype(int).tolist(),
        "keep_mask_sha256": sha256_bytes(mask.tobytes()),
        "canonical_bddl": selected["bddl"],
        "randomization": {},
        "seeds": {
            name: stable_seed("clean-reference", selected["suite"], key[1], name)
            for name in ("environment", "placement", "camera", "noise", "language")
        },
    }
    cache[cache_key] = probe_observation(row, output, gpu, recorded_post_state=True)["rgb"]
    return cache[cache_key]


def load_reference_features(output: Path, audit_data: dict, gpu: int, languages: dict | None = None) -> dict:
    import h5py

    mapping_path = output / "evidence/reference_mapping.jsonl"
    reference_path = output / "evidence/official_references.hdf5"
    cache_path = output / "evidence/reference_features_cache.json"
    cache_inputs = (
        sha256_file(mapping_path),
        sha256_file(reference_path),
        sha256_bytes(canonical_json(languages or {}).encode()),
    )
    fingerprint = sha256_bytes(
        canonical_json((CALIBRATION_CACHE_VERSION, *cache_inputs)).encode()
    )
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("fingerprint") == fingerprint:
            if cached.get("fingerprint") != fingerprint:
                cached["fingerprint"] = fingerprint
                atomic_json(cache_path, cached)
            return {
                tuple(key.split("/", 1)): items for key, items in cached["groups"].items()
            }

    partial_path = output / "evidence/reference_noise_feature_cache.json"
    partial = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    noise_items = partial.get("items", {}) if partial.get("fingerprint") == fingerprint else {}

    mappings = read_jsonl(mapping_path)
    language_texts = defaultdict(lambda: defaultdict(set))
    with h5py.File(reference_path, "r") as handle:
        for mapping in mappings:
            if mapping["setting"] == "language":
                text = decoded_dataset(handle[mapping["reference_id"]]["language"]).strip().lower()
                language_texts[mapping["suite"]][mapping["split"]].add(text)
    language_diversity = {
        suite: {
            "all": len(values["calibration"] | values["holdout"]),
            "calibration": len(values["calibration"]),
            "holdout": len(values["holdout"]),
        }
        for suite, values in language_texts.items()
    }
    deficient = {
        suite: counts for suite, counts in language_diversity.items() if counts["calibration"] < 3
    }
    language_fallback = {}
    if deficient and languages:
        for suite in deficient:
            candidates = sorted(languages.get(suite, []), key=lambda item: item["candidate_id"])
            references = sorted(
                (
                    mapping
                    for mapping in mappings
                    if mapping["suite"] == suite and mapping["setting"] == "language"
                ),
                key=lambda mapping: mapping["reference_id"],
            )
            if len(candidates) == 8 and len({item["instruction"].lower() for item in candidates}) == 8:
                language_fallback.update(
                    {mapping["reference_id"]: candidate for mapping, candidate in zip(references, candidates)}
                )
        deficient = {
            suite: counts
            for suite, counts in deficient.items()
            if sum(
                mapping["reference_id"] in language_fallback
                for mapping in mappings
                if mapping["suite"] == suite and mapping["setting"] == "language"
            )
            != 8
        }
    if deficient:
        report = {
            "decision": "NO-GO",
            "metric_version": "official-rgbd-v2-iqr-snn-1",
            "blocker": "official RLDS language references preserve fewer than three distinct calibration targets",
            "language_text_diversity": language_diversity,
            "official_references": len(mappings),
            "calibration_loss": None,
            "holdout_loss": None,
        }
        atomic_json(output / "reports/calibration.json", report)
        raise AuditError(f"official RLDS language diversity is insufficient: {canonical_json(deficient)}")
    result = defaultdict(list)
    clean_cache = {}
    with h5py.File(reference_path, "r") as handle:
        for mapping in mappings:
            setting = mapping["setting"]
            reference_id = mapping["reference_id"]
            if setting == "noise" and reference_id in noise_items:
                result[(mapping["suite"], setting)].append(noise_items[reference_id])
                continue
            group = handle[reference_id]
            rgb = group["first_rgb"][()]
            mask = group["first_mask"][()] if "first_mask" in group else None
            object_map = json.loads(decoded_dataset(group["object_map_json"]))
            item = {
                "reference_id": mapping["reference_id"],
                "split": mapping["split"],
                "confidence": mapping["confidence"],
                "rgb": rgb,
                "mask": mask,
                "language": decoded_dataset(group["language"]),
                "extrinsics": group["extrinsics"][()] if "extrinsics" in group else None,
                "reference_source": "official-archive",
                "reference_source_id": mapping["reference_id"],
            }
            if setting == "language":
                if mapping["reference_id"] in language_fallback:
                    fallback = language_fallback[mapping["reference_id"]]
                    item["language"] = fallback["instruction"]
                    item["confidence"] = "medium"
                    item["reference_source"] = f"qwen-fallback:{REVISIONS['qwen']['sha']}"
                    item["reference_source_id"] = fallback["candidate_id"]
                item["feature"] = language_feature(item["language"])
            elif setting == "camera":
                if item["extrinsics"] is None:
                    raise AuditError("camera reference lacks extrinsics")
            elif setting == "noise":
                selected = audit_data["selected_tasks"][mapping["suite"]]
                clean = _clean_reference_frame(
                    selected, group["actions"][()], output, gpu, clean_cache
                )
                item["feature"] = visual_feature("noise", rgb, clean=clean)
                noise_items[reference_id] = {
                    name: (value.tolist() if name == "extrinsics" and value is not None else value)
                    for name, value in item.items()
                    if name not in {"rgb", "mask"}
                }
                atomic_json(
                    partial_path,
                    {"fingerprint": fingerprint, "items": noise_items},
                )
            else:
                item["feature"] = visual_feature(
                    setting,
                    rgb,
                    mask=mask,
                    extra_objects=mapping.get("extra_objects", 0),
                    categories=object_map,
                )
            result[(mapping["suite"], setting)].append(item)
    for key, items in result.items():
        if Counter(item["split"] for item in items) != Counter({"calibration": 6, "holdout": 2}):
            raise AuditError(f"{key}: official reference split is not 6/2")
    compact = {
        key: [
            {
                name: (value.tolist() if name == "extrinsics" and value is not None else value)
                for name, value in item.items()
                if name not in {"rgb", "mask"}
            }
            for item in items
        ]
        for key, items in result.items()
    }
    atomic_json(
        cache_path,
        {
            "fingerprint": fingerprint,
            "groups": {f"{key[0]}/{key[1]}": items for key, items in compact.items()},
        },
    )
    return compact


def _pose_feature(matrix, default) -> dict:
    import numpy as np

    matrix, default = np.asarray(matrix), np.asarray(default)
    delta = default[:3, :3].T @ matrix[:3, :3]
    angle = math.acos(float(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
    translation = float(np.linalg.norm(matrix[:3, 3] - default[:3, 3]))
    return {"numeric": [angle, translation], "categories": []}


def _medoids(features: list[dict], count: int) -> list[int]:
    best = None
    for choice in itertools.combinations(range(len(features)), count):
        loss = sum(
            min(feature_distance(item, features[index], features) for index in choice)
            for item in features
        )
        candidate = (loss, choice)
        if best is None or candidate < best:
            best = candidate
    return list(best[1])


def _reference_targets(setting: str, references: list[dict], default_pose=None) -> tuple[list[dict], list[dict]]:
    calibration = [item for item in references if item["split"] == "calibration"]
    if setting == "camera":
        for item in references:
            item["feature"] = _pose_feature(item["extrinsics"], default_pose)
    features = [item["feature"] for item in calibration]
    if setting == "objects":
        targets = [
            min(
                calibration,
                key=lambda item: (abs(item["feature"]["numeric"][0] - slot), item["reference_id"]),
            )
            for slot in (1, 2, 3)
        ]
    elif setting == "background":
        targets = [calibration[index] for index in _medoids(features, 3)]
        targets.sort(key=lambda item: (item["feature"]["numeric"][96], item["reference_id"]))
    elif setting in {"light", "noise", "camera"}:
        component = {"light": 0, "noise": 3, "camera": 0}[setting]
        ordered = sorted(calibration, key=lambda item: (item["feature"]["numeric"][component], item["reference_id"]))
        targets = [ordered[round(q * (len(ordered) - 1))] for q in (0.25, 0.5, 0.75)]
    else:
        unique = sorted(
            {item["language"].lower(): item for item in calibration}.values(),
            key=lambda item: (item["feature"]["numeric"][0], item["reference_id"]),
        )
        if len(unique) < 3:
            raise AuditError("language calibration has fewer than three distinct targets")
        targets = evenly_spaced(unique, 3)
    return targets, features


def _candidate_row(selected: dict, source: dict, suite: str, setting: str, slot: int, randomization: dict):
    candidate_id = randomization["candidate_id"]
    attempt_id = f"probe.{suite}.{setting}.slot{slot}.{candidate_id}"
    return {
        "attempt_id": attempt_id,
        "suite": suite,
        "task": selected["task"],
        "task_index": selected["task_index"],
        "task_stem": selected["task_stem"],
        "setting": setting,
        "variant_slot": slot,
        "source_demo": source["source_demo"],
        "source_file": selected["source"]["source_file"],
        "source_file_sha256": selected["source"]["source_sha256"],
        "source_public_episode": source["public_episode"],
        "keep_mask": source["keep_mask"],
        "keep_mask_sha256": source["keep_mask_sha256"],
        "canonical_bddl": selected["bddl"],
        "canonical_bddl_sha256": selected["bddl_sha256"],
        "canonical_language": selected["task"],
        "randomization": randomization,
        "seeds": {
            kind: stable_seed(attempt_id, kind)
            for kind in ("environment", "placement", "camera", "noise", "language")
        },
        "master_seed": MASTER_SEED,
    }


def calibrate_candidates(
    output: Path,
    audit_data: dict,
    references: dict,
    suite: str,
    setting: str,
    slot: int,
    candidates: list[dict],
    gpu: int,
    candidate_cache: dict,
    candidate_cache_path: Path,
    cache_context: tuple,
    legality: dict,
) -> tuple[list[dict], dict, list[dict]]:
    selected = audit_data["selected_tasks"][suite]
    source = selected["source"]["selected"][slot - 1]
    rows = [_candidate_row(selected, source, suite, setting, slot, item) for item in candidates]
    source_episode = load_source_episode(rows[0]) if setting != "language" else None
    source_state = None
    if setting == "objects":
        state_path = output / "evidence/source_named_states.json"
        states = json.loads(state_path.read_text()) if state_path.exists() else {}
        state_key = f"{rows[0]['source_file_sha256']}/{rows[0]['source_demo']}"
        if state_key not in states:
            states[state_key] = source_named_state(source_episode)
            atomic_json(state_path, states)
        source_state = states[state_key]
    default_pose = None
    try:
        if setting == "camera":
            baseline = _candidate_row(
                selected,
                source,
                suite,
                "camera",
                slot,
                {
                    "candidate_id": "camera-default",
                    "horizontal": 0,
                    "vertical": 0,
                    "scale": 1.0,
                    "endpoint_horizontal": 0,
                    "endpoint_vertical": 0,
                },
            )
            default_pose = probe_observation(
                baseline, output, gpu, camera_only=True, source=source_episode
            )["extrinsics"]
        targets, calibration_features = _reference_targets(
            setting, references[(suite, setting)], default_pose
        )
        target = targets[slot - 1]
        probes = []
        clean = None
        if setting == "noise":
            clean = probe_observation(
                rows[0], output, gpu, recorded_post_state=True, source=source_episode
            )["rgb"]
        for row in rows:
            candidate_key = (
                f"{suite}/{setting}/{slot}/{row['randomization']['candidate_id']}"
            )
            candidate_fingerprint = sha256_bytes(
                canonical_json((cache_context, row)).encode()
            )
            cached_candidate = candidate_cache.get(candidate_key)
            if cached_candidate and cached_candidate.get("fingerprint") == candidate_fingerprint:
                probe = cached_candidate["probe"]
            else:
                try:
                    if setting == "language":
                        feature = language_feature(row["randomization"]["instruction"])
                    elif setting == "camera":
                        matrix = probe_observation(
                            row, output, gpu, camera_only=True, source=source_episode
                        )["extrinsics"]
                        feature = _pose_feature(matrix, default_pose)
                    elif setting == "noise":
                        timestep = int(next(index for index, value in enumerate(row["keep_mask"]) if value))
                        noisy = apply_noise(
                            clean, row["randomization"], stable_seed(row["seeds"]["noise"], timestep)
                        )
                        feature = visual_feature("noise", noisy, clean=clean)
                    else:
                        observation = probe_observation(
                            row,
                            output,
                            gpu,
                            segmentation=True,
                            recorded_post_state=setting != "objects",
                            source=source_episode,
                            source_state=source_state,
                        )
                        categories = [
                            category for _, category in bddl_objects(Path(row["randomization"]["bddl"]))
                        ]
                        feature = visual_feature(
                            setting,
                            observation["rgb"],
                            mask=observation["mask"],
                            extra_objects=row["randomization"].get("extra_objects", 0),
                            categories=categories,
                        )
                    loss = feature_distance(feature, target["feature"], calibration_features)
                    probe = {
                        "candidate_id": row["randomization"]["candidate_id"],
                        "status": "valid",
                        "feature": feature,
                        "loss": loss,
                        "row": row,
                    }
                except Exception as error:
                    probe = {
                        "candidate_id": row["randomization"]["candidate_id"],
                        "status": "invalid",
                        "error": f"{type(error).__name__}: {error}",
                    }
                if probe["status"] == "valid":
                    candidate_cache[candidate_key] = {
                        "fingerprint": candidate_fingerprint,
                        "probe": probe,
                    }
                    atomic_json(candidate_cache_path, candidate_cache)
            verdict = legality.get(probe["candidate_id"])
            if verdict and verdict.get("legal") is False:
                probe = {
                    "candidate_id": probe["candidate_id"],
                    "status": "invalid",
                    "error": "full replay legality check failed",
                    "legality_evidence": verdict,
                }
            probes.append(probe)
    finally:
        pass
    valid = rank_probes(probes)
    if len(valid) < 3:
        raise AuditError(f"{suite}/{setting}/slot{slot}: only {len(valid)} legal calibration probes")
    reference_rows = references[(suite, setting)]
    metadata = {
        "metric_version": "official-rgbd-v2-iqr-snn-1",
        "reference_ids": [item["reference_id"] for item in reference_rows if item["split"] == "calibration"],
        "holdout_reference_ids": [item["reference_id"] for item in reference_rows if item["split"] == "holdout"],
        "official_reference_count": len(reference_rows),
        "target_id": target["reference_id"],
        "confidence": "high" if all(item["confidence"] == "high" for item in reference_rows) else "medium",
        "reference_source": sorted({item["reference_source"] for item in reference_rows}),
        "reference_source_ids": {
            item["reference_id"]: item["reference_source_id"] for item in reference_rows
        },
        "reference_features": {
            item["reference_id"]: item["feature"] for item in reference_rows
        },
    }
    public_probes = [{key: value for key, value in item.items() if key != "row"} for item in probes]
    return valid[:3], metadata, public_probes


def rank_probes(probes: list[dict]) -> list[dict]:
    return sorted(
        (item for item in probes if item["status"] == "valid" and math.isfinite(item["loss"])),
        key=lambda item: (item["loss"], item["candidate_id"]),
    )


def metric_depth(sim, raw_depth):
    import numpy as np
    from robosuite.utils import camera_utils

    raw = np.squeeze(np.asarray(raw_depth, dtype=np.float32))
    valid = np.isfinite(raw) & (raw >= 0) & (raw <= 1)
    sanitized = np.where(valid, raw, 1).astype(np.float32)
    depth = camera_utils.get_real_depth_map(sim, sanitized).astype(np.float32)
    depth[~valid] = 0
    return depth, valid


def asset_inventory(model_xml: str) -> list[dict]:
    result = []
    for element in ET.fromstring(model_xml).iter():
        value = element.attrib.get("file")
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        result.append(
            {
                "kind": element.tag,
                "id": element.attrib.get("name"),
                "file": value,
                "resolved_file": str(path),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return sorted(result, key=canonical_json)


def frame_dataset(group, name: str, data) -> None:
    import numpy as np

    data = np.asarray(data)
    if not data.ndim or not len(data):
        group.create_dataset(name, data=data)
        return
    group.create_dataset(name, data=data, chunks=(1, *data.shape[1:]), compression="lzf")


def text_dataset(group, name: str, value: str) -> None:
    import h5py

    group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))


def write_episode(path: Path, payload: dict) -> None:
    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
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
        frame_dataset(source, "actions_full", payload["actions_full"])
        frame_dataset(source, "keep_mask", payload["keep_mask"])
        frame_dataset(source, "timestep", payload["timestep"])
        frame_dataset(source, "initial_state", payload["initial_state"])

        camera = handle.create_group("camera")
        for short in ("front", "wrist"):
            group = camera.create_group(short)
            for name in ("K", "T_world_cam", "T_cam_world"):
                frame_dataset(group, name, payload["camera"][short][name])
            group.attrs["near_m"] = payload["near_m"]
            group.attrs["far_m"] = payload["far_m"]

        language = handle.create_group("language")
        text_dataset(language, "canonical", payload["manifest"]["canonical_language"])
        text_dataset(language, "instruction", payload["instruction"])
        metadata = handle.create_group("metadata")
        for name, value in (
            ("bddl", payload["bddl"]),
            ("source_xml", payload["source_xml"]),
            ("source_xml_original", payload["source_xml_original"]),
            ("model_xml", payload["model_xml"]),
            ("assets_json", canonical_json(payload["assets"])),
            ("source_assets_json", canonical_json(payload["source_assets"])),
            ("manifest_json", canonical_json(payload["manifest"])),
            ("runtime_json", canonical_json(payload["runtime"])),
            ("state_mapping_json", canonical_json(payload["state_mapping"])),
        ):
            text_dataset(metadata, name, value)
        handle.attrs["format"] = "libero-plus-rgbd-sample-v1"
        handle.attrs["episode_id"] = payload["episode_id"]
        handle.attrs["attempt_id"] = payload["manifest"]["attempt_id"]
        handle.attrs["source_demo"] = payload["manifest"]["source_demo"]
        handle.attrs["executed_full_actions"] = len(payload["actions_full"])
        handle.attrs["saved_actions"] = len(payload["actions"])
        handle.attrs["final_success"] = bool(payload["success"][-1])
        handle.attrs["depth_unit"] = "meter"
        handle.attrs["camera_convention"] = "OpenCV x-right y-down z-forward"
        handle.flush()
    os.replace(partial, path)


def run_attempt(row: dict, output: Path, runtime: dict, gpu: int) -> dict:
    import mujoco
    import numpy as np
    from robosuite.utils import camera_utils

    source = load_source_episode(row)
    keep = np.asarray(row["keep_mask"], dtype=bool)
    if keep.shape != (len(source["actions"]),) or sha256_bytes(keep.tobytes()) != row["keep_mask_sha256"]:
        raise AttemptFailure("frozen keep_mask does not match source actions")
    if sha256_file(Path(row["source_file"])) != row["source_file_sha256"]:
        raise AttemptFailure("source HDF5 hash changed")

    bddl = execution_bddl(row, output)
    base_bddl = Path(row["randomization"].get("bddl", row["canonical_bddl"]))
    bddl_text = base_bddl.read_text() if row["setting"] != "language" else bddl.read_text()
    placement_seed = row["randomization"].get("placement_seed", row["seeds"]["placement"])
    effective_environment_seed = stable_seed(row["seeds"]["environment"], placement_seed)
    env = make_env(bddl, gpu, len(source["actions"]) + 5)
    try:
        seed_environment(env, effective_environment_seed)
        env.env.reset()
        state_mapping = set_source_initial_state(env, row, source, gpu)
        render_context = env.sim._render_context_offscreen
        render_context.gl_ctx.make_current()
        for texture_id in range(env.sim.model.ntex):
            mujoco.mjr_uploadTexture(env.sim.model._model, render_context.con, texture_id)
        model_xml = env.sim.model.get_xml()
        near_m = float(env.sim.model.vis.map.znear * env.sim.model.stat.extent)
        far_m = float(env.sim.model.vis.map.zfar * env.sim.model.stat.extent)
        saved = defaultdict(list)
        camera = {
            "front": defaultdict(list),
            "wrist": defaultdict(list),
        }
        timesteps = np.flatnonzero(keep)
        for timestep, action in enumerate(source["actions"]):
            obs, reward, _, _ = env.step(action)
            if not keep[timestep]:
                continue
            front_clean = as_rgb(obs["agentview_image"])
            wrist = as_rgb(obs["robot0_eye_in_hand_image"])
            front = (
                apply_noise(front_clean, row["randomization"], stable_seed(row["seeds"]["noise"], timestep))
                if row["setting"] == "noise"
                else front_clean
            )
            front_depth, front_raw_valid = metric_depth(env.sim, obs["agentview_depth"])
            wrist_depth, wrist_raw_valid = metric_depth(
                env.sim, obs["robot0_eye_in_hand_depth"]
            )
            saved["front_rgb"].append(front)
            saved["wrist_rgb"].append(wrist)
            if row["setting"] == "noise":
                saved["front_rgb_clean"].append(front_clean)
            saved["front_depth_m"].append(front_depth)
            saved["wrist_depth_m"].append(wrist_depth)
            saved["depth_valid"].append(
                np.stack(
                    [
                        front_raw_valid
                        & np.isfinite(front_depth)
                        & (front_depth >= near_m)
                        & (front_depth <= far_m),
                        wrist_raw_valid
                        & np.isfinite(wrist_depth)
                        & (wrist_depth >= near_m)
                        & (wrist_depth <= far_m),
                    ]
                )
            )
            saved["state"].append(env.env.get_robot_state_vector(obs))
            saved["joint_state"].append(
                np.concatenate([obs["robot0_joint_pos"], obs.get("robot0_gripper_qpos", [])])
            )
            saved["sim_state"].append(env.sim.get_state().flatten())
            saved["reward"].append(reward)
            saved["success"].append(env.check_success())
            for short, name in (("front", "agentview"), ("wrist", "robot0_eye_in_hand")):
                intrinsic = camera_utils.get_camera_intrinsic_matrix(
                    env.sim, name, IMAGE_SIZE, IMAGE_SIZE
                )
                world_cam = camera_utils.get_camera_extrinsic_matrix(env.sim, name)
                camera[short]["K"].append(intrinsic)
                camera[short]["T_world_cam"].append(world_cam)
                camera[short]["T_cam_world"].append(np.linalg.inv(world_cam))

        if len(saved["front_rgb"]) != int(keep.sum()):
            raise AttemptFailure("not every keep_mask=true step was saved")
        if not saved["success"][-1]:
            raise AttemptFailure("full replay did not end in task success")
        episode_id = f"{row['suite']}__{row['setting']}__slot{row['variant_slot']}"
        path = output / "episodes" / f"{episode_id}.hdf5"
        actions = source["actions"][keep]
        payload = {
            **{name: np.asarray(value) for name, value in saved.items()},
            "actions": actions,
            "terminal": np.arange(len(actions)) == len(actions) - 1,
            "actions_full": source["actions"],
            "keep_mask": keep,
            "timestep": timesteps,
            "initial_state": source["states"][0],
            "camera": {
                short: {name: np.asarray(value) for name, value in values.items()}
                for short, values in camera.items()
            },
            "near_m": near_m,
            "far_m": far_m,
            "instruction": row["randomization"].get("instruction", row["canonical_language"]),
            "bddl": bddl_text,
            "source_xml": source["source_xml_processed"],
            "source_xml_original": source["source_xml"],
            "model_xml": model_xml,
            "assets": asset_inventory(model_xml),
            "source_assets": source["source_assets"],
            "manifest": row,
            "runtime": runtime,
            "state_mapping": state_mapping,
            "episode_id": episode_id,
        }
        write_episode(path, payload)
        return {
            "episode_id": episode_id,
            "path": str(path.resolve()),
            "attempt_id": row["attempt_id"],
            "frames": len(actions),
            "sha256": sha256_file(path),
        }
    finally:
        env.close()


def attempt_worker(row: dict, output: str, runtime: dict, gpu: int, result_path: str) -> None:
    try:
        output_path = Path(output)
        configure_libero(output_path)
        result = run_attempt(row, output_path, runtime, gpu)
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
        # MuJoCo/EGL can abort during multiprocessing's interpreter teardown after a valid result.
        os._exit(0)


def run_attempt_isolated(row: dict, output: Path, runtime: dict, gpu: int) -> dict:
    import multiprocessing

    result_path = output / "runtime/attempt_worker_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    process = multiprocessing.get_context("spawn").Process(
        target=attempt_worker,
        args=(row, str(output), runtime, gpu, str(result_path)),
    )
    process.start()
    process.join()
    if not result_path.exists():
        raise WorkerFailure(f"attempt worker exited {process.exitcode} without a result")
    payload = json.loads(result_path.read_text())
    result_path.unlink()
    if process.exitcode != 0 or not payload.get("ok"):
        raise WorkerFailure(
            payload.get("exception", f"attempt worker exited {process.exitcode}"),
            payload.get("traceback", ""),
        )
    return payload["result"]


def existing_episodes(output: Path) -> dict[tuple[str, str, int], dict]:
    import h5py

    result = {}
    for path in sorted((output / "episodes").glob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            row = json.loads(decoded_dataset(handle["metadata/manifest_json"]))
            result[(row["suite"], row["setting"], row["variant_slot"])] = {
                "episode_id": handle.attrs["episode_id"],
                "path": str(path.resolve()),
                "attempt_id": handle.attrs["attempt_id"],
                "frames": int(handle.attrs["saved_actions"]),
                "sha256": sha256_file(path),
                "suite": row["suite"],
                "setting": row["setting"],
                "variant_slot": row["variant_slot"],
                "source_demo": row["source_demo"],
            }
    return result


def refresh_index(output: Path) -> list[dict]:
    rows = sorted(existing_episodes(output).values(), key=lambda row: row["episode_id"])
    atomic_jsonl(output / "episodes/index.jsonl", rows)
    return rows


def generate(args) -> None:
    output = args.output.resolve()
    rows = read_jsonl(output / "sample_manifest.jsonl")
    validate_manifest_rows(rows)
    configure_libero(output)
    verify_source_replay(json.loads((output / "audit.json").read_text()), output)
    runtime_path = output / "runtime.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        if runtime.get("script_sha256") != sha256_file(Path(__file__)):
            raise AuditError("runtime.json belongs to different script content; audit output is frozen")
    else:
        runtime = runtime_metadata(args.gpu)
        atomic_json(runtime_path, runtime)

    attempt_path = output / "attempts.jsonl"
    attempts = read_jsonl(attempt_path) if attempt_path.exists() else []
    failed = {row["attempt_id"] for row in attempts if row["status"] == "failed"}
    completed = existing_episodes(output)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["setting"], row["variant_slot"])].append(row)
    smoke = [(SUITES[0], setting, 1) for setting in SETTINGS]
    order = smoke + [key for key in grouped if key not in smoke]
    if args.smoke_only:
        order = smoke

    for key in order:
        if key in completed:
            continue
        success = None
        for row in sorted(grouped[key], key=lambda item: item["retry_rank"]):
            if row["attempt_id"] in failed:
                continue
            try:
                success = run_attempt_isolated(row, output, runtime, args.gpu)
                append_jsonl(
                    attempt_path,
                    {
                        "attempt_id": row["attempt_id"],
                        "status": "success",
                        "failure_stage": None,
                        "exception": None,
                        "final_success": True,
                        "episode": success,
                    },
                )
                completed[key] = success
                print(canonical_json({"generated": success["episode_id"], "attempt": row["attempt_id"]}))
                break
            except Exception as error:
                append_jsonl(
                    attempt_path,
                    {
                        "attempt_id": row["attempt_id"],
                        "status": "failed",
                        "failure_stage": "generation",
                        "exception": f"{type(error).__name__}: {error}",
                        "traceback": getattr(error, "worker_traceback", "")
                        or traceback.format_exc(limit=20),
                        "final_success": False,
                    },
                )
                failed.add(row["attempt_id"])
                print(canonical_json({"failed": row["attempt_id"], "error": str(error)}))
        if success is None:
            index = refresh_index(output)
            atomic_json(
                output / "generation_status.json",
                {"complete": False, "failed_slot": key, "episodes": len(index)},
            )
            raise AuditError(f"slot exhausted all three frozen attempts: {key}")

    index = refresh_index(output)
    complete = len(index) == 72
    atomic_json(
        output / "generation_status.json",
        {"complete": complete, "smoke_only": args.smoke_only, "episodes": len(index)},
    )
    if complete:
        create_previews(output)
    print(canonical_json({"generation": "complete" if complete else "smoke-complete", "episodes": len(index)}))


def depth_preview(depth, near: float, far: float):
    import cv2
    import numpy as np

    normalized = np.clip((np.log(depth) - math.log(near)) / (math.log(far) - math.log(near)), 0, 1)
    return cv2.applyColorMap(((1 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def create_previews(output: Path) -> None:
    import cv2
    import h5py
    import imageio.v2 as imageio

    paths = existing_episodes(output)
    preview_root = output / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    for suite in SUITES:
        for setting in SETTINGS:
            episode_paths = [Path(paths[(suite, setting, slot)]["path"]) for slot in (1, 2, 3)]
            handles = [h5py.File(path, "r") for path in episode_paths]
            final = preview_root / f"{suite}__{setting}.mp4"
            partial = final.with_name(final.stem + ".partial.mp4")
            writer = imageio.get_writer(
                partial,
                fps=20,
                codec="libx264",
                pixelformat="yuv420p",
                output_params=["-movflags", "+faststart"],
            )
            try:
                length = max(len(handle["actions"]) for handle in handles)
                for timestep in range(length):
                    rows = []
                    for slot, handle in enumerate(handles, 1):
                        index = min(timestep, len(handle["actions"]) - 1)
                        near = handle["camera/front"].attrs["near_m"]
                        far = handle["camera/front"].attrs["far_m"]
                        panels = [
                            cv2.cvtColor(handle["observations/front_rgb"][index], cv2.COLOR_RGB2BGR),
                            cv2.cvtColor(handle["observations/wrist_rgb"][index], cv2.COLOR_RGB2BGR),
                            depth_preview(handle["observations/front_depth_m"][index], near, far),
                            depth_preview(handle["observations/wrist_depth_m"][index], near, far),
                        ]
                        row_image = cv2.hconcat(panels)
                        cv2.putText(
                            row_image,
                            f"slot {slot}  t={index}",
                            (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        rows.append(row_image)
                    writer.append_data(cv2.cvtColor(cv2.vconcat(rows), cv2.COLOR_BGR2RGB))
            finally:
                writer.close()
                for handle in handles:
                    handle.close()
            os.replace(partial, final)


def decoded_dataset(dataset) -> str:
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def h5_content_hash(path: Path) -> str:
    import h5py
    import numpy as np

    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        def add_object(name, item):
            digest.update(name.encode())
            digest.update(type(item).__name__.encode())
            for key in sorted(item.attrs):
                digest.update(key.encode())
                digest.update(np.asarray(item.attrs[key]).tobytes())
            if isinstance(item, h5py.Dataset):
                value = item[()]
                digest.update(str(item.shape).encode())
                digest.update(str(item.dtype).encode())
                if isinstance(value, bytes):
                    digest.update(value)
                elif isinstance(value, str):
                    digest.update(value.encode())
                else:
                    digest.update(np.asarray(value).tobytes())

        for key in sorted(handle.attrs):
            digest.update(key.encode())
            digest.update(np.asarray(handle.attrs[key]).tobytes())
        handle.visititems(add_object)
    return digest.hexdigest()


def compare_arrays(left, right, names: tuple[str, ...], errors: list[str], label: str) -> None:
    import numpy as np

    for name in names:
        if left[name].shape != right[name].shape or not np.array_equal(left[name][()], right[name][()]):
            errors.append(f"{label}: array differs: {name}")


def validate_episode(path: Path) -> tuple[dict, list[str], dict]:
    import h5py
    import numpy as np

    errors = []
    metrics = {"max_action_error": 0.0, "max_inverse_error": 0.0, "max_reprojection_px": 0.0}
    with h5py.File(path, "r") as handle:
        row = json.loads(decoded_dataset(handle["metadata/manifest_json"]))
        required = (
            "observations/front_rgb",
            "observations/wrist_rgb",
            "observations/front_depth_m",
            "observations/wrist_depth_m",
            "observations/depth_valid",
            "observations/state",
            "observations/joint_state",
            "sim/state",
            "actions",
            "reward",
            "success",
            "terminal",
            "source/actions_full",
            "source/keep_mask",
            "source/timestep",
            "source/initial_state",
            "camera/front/K",
            "camera/front/T_world_cam",
            "camera/front/T_cam_world",
            "camera/wrist/K",
            "camera/wrist/T_world_cam",
            "camera/wrist/T_cam_world",
            "language/canonical",
            "language/instruction",
            "metadata/bddl",
            "metadata/source_xml",
            "metadata/model_xml",
            "metadata/assets_json",
            "metadata/manifest_json",
        )
        for name in required:
            if name not in handle:
                errors.append(f"{path.name}: missing {name}")
        if errors:
            return row, errors, metrics

        length = len(handle["actions"])
        expected_shapes = {
            "observations/front_rgb": (length, IMAGE_SIZE, IMAGE_SIZE, 3),
            "observations/wrist_rgb": (length, IMAGE_SIZE, IMAGE_SIZE, 3),
            "observations/front_depth_m": (length, IMAGE_SIZE, IMAGE_SIZE),
            "observations/wrist_depth_m": (length, IMAGE_SIZE, IMAGE_SIZE),
            "observations/depth_valid": (length, 2, IMAGE_SIZE, IMAGE_SIZE),
        }
        for name, shape in expected_shapes.items():
            if handle[name].shape != shape:
                errors.append(f"{path.name}: {name} shape {handle[name].shape} != {shape}")
        for name in ("observations/front_rgb", "observations/wrist_rgb"):
            if handle[name].dtype != np.uint8:
                errors.append(f"{path.name}: {name} is not uint8")
        for name in ("observations/front_depth_m", "observations/wrist_depth_m"):
            if handle[name].dtype != np.float32:
                errors.append(f"{path.name}: {name} is not float32")

        time_aligned = (
            "observations/front_rgb",
            "observations/wrist_rgb",
            "observations/front_depth_m",
            "observations/wrist_depth_m",
            "observations/depth_valid",
            "observations/state",
            "observations/joint_state",
            "sim/state",
            "reward",
            "success",
            "terminal",
        )
        for name in time_aligned:
            if len(handle[name]) != length:
                errors.append(f"{path.name}: {name} is not time-aligned")
        for name in (*time_aligned, "actions", "source/actions_full", "source/initial_state"):
            value = handle[name][()]
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                errors.append(f"{path.name}: {name} contains NaN/Inf")
        for name in (*time_aligned, "actions", "source/actions_full"):
            dataset = handle[name]
            if dataset.compression != "lzf" or not dataset.chunks or dataset.chunks[0] != 1:
                errors.append(f"{path.name}: {name} is not LZF frame-chunked")

        actions = handle["actions"][()]
        full = handle["source/actions_full"][()]
        timestep = handle["source/timestep"][()]
        mask = handle["source/keep_mask"][()].astype(bool)
        if len(full) != int(handle.attrs["executed_full_actions"]):
            errors.append(f"{path.name}: full source actions were not all executed")
        if not np.array_equal(np.flatnonzero(mask), timestep):
            errors.append(f"{path.name}: source timestep differs from keep_mask")
        if actions.shape == full[timestep].shape:
            metrics["max_action_error"] = float(np.max(np.abs(actions - full[timestep]), initial=0))
            if metrics["max_action_error"] > 1e-6:
                errors.append(f"{path.name}: saved action mismatch")
        else:
            errors.append(f"{path.name}: saved/source action shape mismatch")
        if not bool(handle["success"][-1]) or not bool(handle.attrs["final_success"]):
            errors.append(f"{path.name}: final success is false")
        terminal = handle["terminal"][()].astype(bool)
        if terminal.sum() != 1 or not terminal[-1]:
            errors.append(f"{path.name}: terminal must be true only on final frame")

        near = float(handle["camera/front"].attrs["near_m"])
        far = float(handle["camera/front"].attrs["far_m"])
        for index, name in enumerate(("observations/front_depth_m", "observations/wrist_depth_m")):
            depth = handle[name][()]
            valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
            if not np.array_equal(valid, handle["observations/depth_valid"][:, index].astype(bool)):
                errors.append(f"{path.name}: depth_valid mismatch for {name}")
            if float(valid.mean()) < 0.9:
                errors.append(f"{path.name}: valid depth coverage below 90% for {name}")
            if valid.any() and (depth[valid].min() < near - 1e-6 or depth[valid].max() > far + 1e-6):
                errors.append(f"{path.name}: valid depth outside near/far")

        for short in ("front", "wrist"):
            k = handle[f"camera/{short}/K"][()]
            world_cam = handle[f"camera/{short}/T_world_cam"][()]
            cam_world = handle[f"camera/{short}/T_cam_world"][()]
            identity_error = float(np.max(np.abs(world_cam @ cam_world - np.eye(4))))
            metrics["max_inverse_error"] = max(metrics["max_inverse_error"], identity_error)
            if identity_error > 1e-6:
                errors.append(f"{path.name}: {short} camera inverse error {identity_error}")
            for frame in (0, length - 1):
                pixel = np.array([IMAGE_SIZE * 0.37, IMAGE_SIZE * 0.61, 1.0])
                camera_point = np.linalg.inv(k[frame]) @ pixel * ((near + far) / 4)
                world_point = world_cam[frame] @ np.r_[camera_point, 1]
                projected = k[frame] @ (cam_world[frame] @ world_point)[:3]
                projected = projected[:2] / projected[2]
                error = float(np.linalg.norm(projected - pixel[:2]))
                metrics["max_reprojection_px"] = max(metrics["max_reprojection_px"], error)
                if error > 1:
                    errors.append(f"{path.name}: {short} reprojection error {error}")

        assets = json.loads(decoded_dataset(handle["metadata/assets_json"]))
        for asset in assets:
            asset_path = Path(asset["resolved_file"])
            if not asset_path.is_file() or asset["sha256"] != sha256_file(asset_path):
                errors.append(f"{path.name}: asset missing or hash changed: {asset['file']}")
        if row["setting"] == "objects":
            mapping = json.loads(decoded_dataset(handle["metadata/state_mapping_json"]))
            if mapping.get("method") != "joint_and_mocap_body_names":
                errors.append(f"{path.name}: compounding state was not mapped by names")
        if row["setting"] == "noise" and "observations/front_rgb_clean" not in handle:
            errors.append(f"{path.name}: noise sample lacks clean front RGB")
    return row, errors, metrics


def strip_language(text: str) -> str:
    return re.sub(r"\(:language\s+.+?\s*\)", "(:language <removed>)", text, count=1, flags=re.I)


def validate_isolation(output: Path, episodes: dict, errors: list[str]) -> None:
    import h5py
    import numpy as np

    tests = benchmark_map()
    for key, episode in episodes.items():
        path = Path(episode["path"])
        with h5py.File(path, "r") as handle:
            row = json.loads(decoded_dataset(handle["metadata/manifest_json"]))
            randomization = row["randomization"]
            test_set = set(name for name in tests[row["suite"]] if name.startswith(row["task_stem"]))
            if row["setting"] in {"objects", "background", "light"}:
                if Path(randomization["bddl"]).stem in test_set:
                    errors.append(f"{path.name}: selected benchmark BDDL")
            elif row["setting"] == "camera" and randomization["min_test_angle_deg"] < 5 - 1e-9:
                errors.append(f"{path.name}: camera is closer than 5 degrees to benchmark")
            elif row["setting"] == "language":
                instruction = decoded_dataset(handle["language/instruction"])
                if not valid_rewrite(row["canonical_language"], instruction, Path(row["canonical_bddl"])):
                    errors.append(f"{path.name}: language hard validation failed")
                benchmark_texts = {
                    bddl_language(candidate).lower()
                    for name in test_set
                    if (candidate := Path(row["canonical_bddl"]).with_name(name + ".bddl")).is_file()
                }
                if instruction.lower() in benchmark_texts:
                    errors.append(f"{path.name}: language equals benchmark text")
            elif row["setting"] == "noise" and randomization.get("benchmark_tuple_equal") is not False:
                errors.append(f"{path.name}: noise tuple isolation is not proven")

    for suite in SUITES:
        for slot in (1, 2, 3):
            language_path = Path(episodes[(suite, "language", slot)]["path"])
            noise_path = Path(episodes[(suite, "noise", slot)]["path"])
            camera_path = Path(episodes[(suite, "camera", slot)]["path"])
            with h5py.File(language_path, "r") as language, h5py.File(noise_path, "r") as noise:
                compare_arrays(
                    language,
                    noise,
                    (
                        "observations/wrist_rgb",
                        "observations/front_depth_m",
                        "observations/wrist_depth_m",
                        "observations/state",
                        "observations/joint_state",
                        "sim/state",
                        "actions",
                    ),
                    errors,
                    f"{suite}/slot{slot} language-noise",
                )
                if not np.array_equal(
                    language["observations/front_rgb"][()], noise["observations/front_rgb_clean"][()]
                ):
                    errors.append(f"{suite}/slot{slot}: noise clean front differs from language baseline")
                if np.array_equal(
                    noise["observations/front_rgb"][()], noise["observations/front_rgb_clean"][()]
                ):
                    errors.append(f"{suite}/slot{slot}: noise did not alter front RGB")
                if strip_language(decoded_dataset(language["metadata/bddl"])) != strip_language(
                    Path(json.loads(decoded_dataset(language["metadata/manifest_json"]))["canonical_bddl"]).read_text()
                ):
                    errors.append(f"{suite}/slot{slot}: language BDDL changed physical content")
            with h5py.File(language_path, "r") as language, h5py.File(camera_path, "r") as camera:
                compare_arrays(
                    language,
                    camera,
                    ("observations/state", "observations/joint_state", "sim/state", "actions"),
                    errors,
                    f"{suite}/slot{slot} language-camera",
                )


def _symmetric_nearest_neighbor(generated: list[dict], references: list[dict], scale_pool: list[dict]) -> float:
    if not generated or not references:
        return math.nan
    forward = sum(
        min(feature_distance(item, reference, scale_pool) for reference in references)
        for item in generated
    ) / len(generated)
    backward = sum(
        min(feature_distance(reference, item, scale_pool) for item in generated)
        for reference in references
    ) / len(references)
    return (forward + backward) / 2


def calibration_metrics(output: Path, episodes: dict, errors: list[str]) -> dict:
    import h5py

    mappings = read_jsonl(output / "evidence/reference_mapping.jsonl")
    groups = defaultdict(list)
    for row in mappings:
        groups[(row["suite"], row["setting"])].append(row)
    expected_groups = {(suite, setting) for suite in SUITES for setting in SETTINGS}
    if len(mappings) != 192 or set(groups) != expected_groups:
        errors.append("official mapping must contain exactly 192 references across 24 groups")
    for key in expected_groups:
        rows = groups[key]
        if len(rows) != 8 or Counter(row["split"] for row in rows) != Counter({"calibration": 6, "holdout": 2}):
            errors.append(f"{key}: official reference split is not eight / 6+2")
        if len({row["logical_episode"] for row in rows}) != len(rows):
            errors.append(f"{key}: calibration and holdout leak a logical episode")
        if any(
            row["max_action_error"] > 1e-6
            or row["max_state_error"] > 1e-6
            or not row.get("tfrecord_crc_valid")
            for row in rows
        ):
            errors.append(f"{key}: mapping tolerance or TFRecord CRC failed")
    hdf5_path = output / "evidence/official_references.hdf5"
    if not hdf5_path.exists():
        errors.append("official_references.hdf5 is missing")
    else:
        with h5py.File(hdf5_path, "r") as handle:
            if int(handle.attrs.get("count", -1)) != 192 or len(handle) != 192:
                errors.append("official_references.hdf5 does not contain 192 groups")

    generated = defaultdict(list)
    references = defaultdict(dict)
    confidence = defaultdict(list)
    for episode in episodes.values():
        with h5py.File(episode["path"], "r") as handle:
            row = json.loads(decoded_dataset(handle["metadata/manifest_json"]))
        calibration = row.get("calibration", {})
        feature = calibration.get("probe_feature")
        if feature is None:
            errors.append(f"{row['attempt_id']}: generated episode lacks probe feature")
            continue
        setting = row["setting"]
        generated[setting].append(feature)
        confidence[setting].append(calibration.get("confidence", "low"))
        split = {
            reference_id: "calibration" for reference_id in calibration.get("reference_ids", [])
        }
        split.update(
            {reference_id: "holdout" for reference_id in calibration.get("holdout_reference_ids", [])}
        )
        for reference_id, reference_feature in calibration.get("reference_features", {}).items():
            prior = references[setting].get(reference_id)
            value = {"feature": reference_feature, "split": split.get(reference_id)}
            if prior is not None and canonical_json(prior) != canonical_json(value):
                errors.append(f"{setting}: inconsistent feature for {reference_id}")
            references[setting][reference_id] = value

    report = {"metric_version": "official-rgbd-v2-iqr-snn-1", "settings": {}}
    for setting in SETTINGS:
        calibration = [
            item["feature"] for item in references[setting].values() if item["split"] == "calibration"
        ]
        holdout = [
            item["feature"] for item in references[setting].values() if item["split"] == "holdout"
        ]
        calibration_loss = _symmetric_nearest_neighbor(generated[setting], calibration, calibration)
        holdout_loss = _symmetric_nearest_neighbor(generated[setting], holdout, calibration)
        if len(references[setting]) != 32 or len(calibration) != 24 or len(holdout) != 8:
            errors.append(f"{setting}: reference features are not 32 / 24+8")
        if not math.isfinite(calibration_loss) or not math.isfinite(holdout_loss):
            errors.append(f"{setting}: calibration or holdout loss is not finite")
        report["settings"][setting] = {
            "generated": len(generated[setting]),
            "official_references": len(references[setting]),
            "calibration_references": len(calibration),
            "holdout_references": len(holdout),
            "calibration_loss": calibration_loss,
            "holdout_loss": holdout_loss,
            "confidence": "high" if confidence[setting] and all(value == "high" for value in confidence[setting]) else "medium",
        }
    atomic_json(output / "reports/calibration.json", report)
    return report


def write_gap_report(
    output: Path, errors: list[str], metrics: dict, deterministic: dict, calibration: dict
) -> None:
    counts = Counter(row["setting"] for row in existing_episodes(output).values())
    lines = [
        "# LIBERO-Plus RGB-D sample gap report",
        "",
        f"**Decision: {'NO-GO' if errors else 'GO'}**",
        "",
        "These 72 success-selected episodes validate this pipeline only; they do not represent the full training distribution or estimate generation success rate.",
        "",
        "| setting | samples | official references | calibration loss | holdout loss | confidence |",
        "|---|---:|---:|---|---|---|",
    ]
    for setting in SETTINGS:
        row = calibration["settings"][setting]
        lines.append(
            f"| {setting} | {counts[setting]} | {row['official_references']} ({row['calibration_references']}/{row['holdout_references']}) | "
            f"{row['calibration_loss']:.6g} | {row['holdout_loss']:.6g} | {row['confidence']} |"
        )
    lines.extend(
        [
            "",
            "## Geometry and replay",
            "",
            f"- Maximum action error: `{metrics['max_action_error']:.3g}`",
            f"- Maximum camera inverse error: `{metrics['max_inverse_error']:.3g}`",
            f"- Maximum reprojection error: `{metrics['max_reprojection_px']:.3g}` px",
            f"- Deterministic reruns: `{canonical_json(deterministic)}`",
            "",
            "## Evidence boundary",
            "",
            "Pinned archive records are joined to LeRobot with task/length/action/state and, only when needed, RGB disambiguation. Official depth truth is not published, so no depth distribution loss is claimed.",
        ]
    )
    if errors:
        lines.extend(["", "## Validation failures", ""] + [f"- {error}" for error in errors])
    atomic_text(output / "reports/gap_report.md", "\n".join(lines) + "\n")


def validate(args) -> None:
    import h5py

    output = args.output.resolve()
    episodes = existing_episodes(output)
    errors = []
    if len(episodes) != 72:
        errors.append(f"episode count {len(episodes)} != 72")
    if Counter(row["suite"] for row in episodes.values()) != Counter({suite: 18 for suite in SUITES}):
        errors.append("suite counts are not exactly 18 each")
    if Counter(row["setting"] for row in episodes.values()) != Counter({setting: 12 for setting in SETTINGS}):
        errors.append("setting counts are not exactly 12 each")
    expected_keys = {(suite, setting, slot) for suite in SUITES for setting in SETTINGS for slot in (1, 2, 3)}
    if set(episodes) != expected_keys:
        errors.append("suite/setting/slot coverage is not exact")

    metrics = {"max_action_error": 0.0, "max_inverse_error": 0.0, "max_reprojection_px": 0.0}
    for episode in episodes.values():
        _, episode_errors, episode_metrics = validate_episode(Path(episode["path"]))
        errors.extend(episode_errors)
        for key in metrics:
            metrics[key] = max(metrics[key], episode_metrics[key])
    if set(episodes) == expected_keys:
        validate_isolation(output, episodes, errors)

    calibration = calibration_metrics(output, episodes, errors)

    previews = list((output / "previews").glob("*.mp4"))
    if len(episodes) == 72 and len(previews) != 24:
        errors.append(f"preview count {len(previews)} != 24")

    deterministic = {"run": not args.skip_determinism, "matched": 0, "expected": 6}
    if not args.skip_determinism and set(episodes) == expected_keys:
        configure_libero(output)
        runtime = json.loads((output / "runtime.json").read_text())
        (output / "reports").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output / "reports") as temporary:
            temporary = Path(temporary)
            for setting in SETTINGS:
                original = Path(episodes[(SUITES[0], setting, 1)]["path"])
                with h5py.File(original, "r") as handle:
                    row = json.loads(decoded_dataset(handle["metadata/manifest_json"]))
                rerun = run_attempt_isolated(row, temporary, runtime, args.gpu)
                if h5_content_hash(original) == h5_content_hash(Path(rerun["path"])):
                    deterministic["matched"] += 1
                else:
                    errors.append(f"determinism content hash mismatch: {setting}")
    elif args.skip_determinism:
        errors.append("determinism reruns were explicitly skipped")

    write_gap_report(output, errors, metrics, deterministic, calibration)
    atomic_json(
        output / "reports/validation.json",
        {
            "go": not errors,
            "errors": errors,
            "metrics": metrics,
            "determinism": deterministic,
            "calibration": calibration,
        },
    )
    if errors:
        raise AuditError(f"validation failed with {len(errors)} error(s); see reports/gap_report.md")
    print(canonical_json({"validation": "GO", "episodes": 72, "previews": 24}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    audit_parser = commands.add_parser("audit", help="fetch finite evidence and select four tasks")
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    audit_parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN)
    audit_parser.add_argument("--max-match-episodes", type=int, default=8)
    audit_parser.add_argument("--no-download-sources", action="store_true")
    audit_parser.set_defaults(handler=audit)

    manifest_parser = commands.add_parser("manifest", help="freeze 216 deterministic attempts")
    manifest_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    manifest_parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN)
    manifest_parser.add_argument("--gpu", type=int, default=0)
    manifest_parser.set_defaults(handler=manifest)

    generate_parser = commands.add_parser("generate", help="execute smoke slots then the frozen manifest")
    generate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    generate_parser.add_argument("--gpu", type=int, default=0)
    generate_parser.add_argument("--smoke-only", action="store_true")
    generate_parser.set_defaults(handler=generate)

    validate_parser = commands.add_parser("validate", help="validate replay, RGB-D, isolation, and determinism")
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
