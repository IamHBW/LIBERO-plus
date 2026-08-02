#!/usr/bin/env python3
"""Create action-identical Mirror / LIBERO-Plus precollected comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import textwrap
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import create_rgbd_samples as base  # noqa: E402


SETTING_SOURCE = {
    "objects": "segmentation",
    "background": "segmentation",
    "light": "segmentation",
    "camera": "camera",
    "language": "rlds",
    "noise": "rlds",
}


def exact_action_error(mirror, official) -> float:
    import numpy as np

    mirror = np.ascontiguousarray(mirror, dtype="<f4")
    official = np.ascontiguousarray(official, dtype="<f4")
    if mirror.shape != official.shape or base.action_key(mirror) != base.action_key(official):
        raise base.AuditError("comparison actions do not have the same float32 fingerprint")
    if not np.array_equal(mirror, official):
        raise base.AuditError("comparison action SHA-256 matched but bytes differ")
    return float(np.max(np.abs(mirror - official), initial=0))


def load_mirrors(output: Path) -> list[dict]:
    import h5py
    import numpy as np

    mirrors = []
    for path in sorted((output / "episodes").glob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            manifest = json.loads(base.decoded_dataset(handle["metadata/manifest_json"]))
            actions = np.asarray(handle["actions"], dtype="<f4")
            mirrors.append(
                {
                    "path": path,
                    "episode_id": str(handle.attrs["episode_id"]),
                    "attempt_id": str(handle.attrs["attempt_id"]),
                    "suite": manifest["suite"],
                    "setting": manifest["setting"],
                    "variant_slot": manifest["variant_slot"],
                    "task": manifest["task"],
                    "task_stem": manifest["task_stem"],
                    "source_public_episode": manifest["source_public_episode"],
                    "actions": actions,
                    "action_key": base.action_key(actions),
                }
            )
    if len(mirrors) != 72:
        raise base.AuditError(f"comparison requires 72 Mirror episodes, found {len(mirrors)}")
    return mirrors


def select_reference_matches(mirrors: list[dict], mappings: list[dict]) -> dict[str, tuple[dict, dict]]:
    suite_rank = {suite: index for index, suite in enumerate(base.SUITES)}
    selected = {}
    for setting in base.SETTINGS:
        candidates = []
        for mirror in mirrors:
            if mirror["setting"] != setting:
                continue
            length, digest = mirror["action_key"]
            for mapping in mappings:
                if (
                    mapping["setting"] == setting
                    and mapping["suite"] == mirror["suite"]
                    and mapping["mapped_task"] == mirror["task"]
                    and mapping["action_length"] == length
                    and mapping["action_sha256"] == digest
                ):
                    candidates.append(
                        (
                            suite_rank[mirror["suite"]],
                            mirror["variant_slot"],
                            mapping["reference_id"],
                            mirror,
                            mapping,
                        )
                    )
        if candidates:
            *_, mirror, mapping = min(candidates, key=lambda item: item[:3])
            selected[setting] = mirror, mapping
    return selected


def archive_source(mapping: dict) -> str:
    matches = [
        source
        for source in SETTING_SOURCE.values()
        if base.REVISIONS[source]["repo"] == mapping["archive_repo"]
    ]
    if len(set(matches)) != 1:
        raise base.AuditError(f"unknown official archive repo {mapping['archive_repo']}")
    return matches[0]


def record_crc_valid(data: bytes, offset: int, raw: bytes) -> bool:
    length_bytes = data[offset : offset + 8]
    if len(length_bytes) != 8 or struct.unpack("<Q", length_bytes)[0] != len(raw):
        return False
    length_crc = struct.unpack_from("<L", data, offset + 8)[0]
    payload_crc = struct.unpack_from("<L", data, offset + 12 + len(raw))[0]
    return length_crc == base._masked_crc32c(length_bytes) and payload_crc == base._masked_crc32c(raw)


def load_mapped_record(archive: base.LocalZip, mapping: dict) -> dict:
    member = archive.members.get(mapping["member"])
    if member is None:
        raise base.AuditError(f"archive member is missing: {mapping['member']}")
    data = archive.read(member)
    for ordinal, offset, raw, features in base.tfrecord_examples(data, validate_crc=False):
        if ordinal != mapping["record_ordinal"]:
            continue
        if base.sha256_bytes(raw) != mapping["raw_record_sha256"]:
            raise base.AuditError(f"{mapping['reference_id']}: raw record hash changed")
        if not record_crc_valid(data, offset, raw):
            raise base.AuditError(f"{mapping['reference_id']}: TFRecord CRC failed")
        return {
            "source": archive.source,
            "member": member,
            "ordinal": ordinal,
            "offset": offset,
            "raw": raw,
            "features": features,
            "mapping": mapping,
            "selection": "audited reference mapping",
            "tfrecord_crc_valid": True,
        }
    raise base.AuditError(f"{mapping['reference_id']}: record ordinal is missing")


def _path_has_setting(path: str, setting: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    if setting == "camera":
        return "/camera_view/" in normalized or "/extrinsics_camera_view/" in normalized
    return f"/{setting}/" in normalized


def scan_exact_record(
    archive: base.LocalZip, setting: str, mirrors: list[dict]
) -> tuple[dict, dict]:
    targets = {}
    for mirror in mirrors:
        if mirror["setting"] == setting:
            targets.setdefault((mirror["task_stem"].lower(), mirror["action_key"]), []).append(mirror)
    if not targets:
        raise base.AuditError(f"no Mirror target exists for {setting}")

    for shard in range(base.ARCHIVES[archive.source]["shards"]):
        if shard % 32 == 0:
            print(f"scan {setting}: shard {shard}", file=sys.stderr, flush=True)
        member, data = archive.shard(shard)
        for ordinal, offset, raw, features in base.tfrecord_examples(data, validate_crc=False):
            paths = features.get("episode_metadata/file_path", [])
            if len(paths) != 1:
                continue
            path = paths[0].decode()
            if not _path_has_setting(path, setting):
                continue
            filename = Path(path).name.lower()
            stems = {stem for stem, _ in targets if filename.endswith(f"{stem}_demo.hdf5")}
            if len(stems) != 1:
                continue
            values = features.get("steps/action", [])
            if len(values) % 7:
                continue
            import numpy as np

            actions = np.asarray(values, dtype="<f4").reshape(-1, 7)
            key = base.action_key(actions)
            candidates = targets.get((next(iter(stems)), key), [])
            for mirror in sorted(
                candidates,
                key=lambda item: (base.SUITES.index(item["suite"]), item["variant_slot"]),
            ):
                exact_action_error(mirror["actions"], actions)
                if not record_crc_valid(data, offset, raw):
                    raise base.AuditError("exact-action official record failed TFRecord CRC")
                return mirror, {
                    "source": archive.source,
                    "member": member,
                    "ordinal": ordinal,
                    "offset": offset,
                    "raw": raw,
                    "features": features,
                    "mapping": None,
                    "selection": "local archive scan by task and exact action fingerprint",
                    "tfrecord_crc_valid": True,
                }
    raise base.AuditError(f"no exact-action official {setting} record was found")


def decode_rgb(encoded: bytes):
    return base.decode_official_rgb(encoded)


def panel(image, label: str):
    import cv2

    result = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)
    cv2.rectangle(result, (0, 0), (319, 27), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def draw_text(image, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    import cv2

    y = 34
    for value, color in lines:
        for line in textwrap.wrap(value, width=43) or [""]:
            cv2.putText(
                image,
                line,
                (656, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 19
        y += 3


def render_video(output: Path, mirror: dict, official: dict, fps: int) -> dict:
    import cv2
    import h5py
    import imageio.v2 as imageio
    import numpy as np

    features = official["features"]
    front = features.get("steps/observation/image", [])
    wrist = features.get("steps/observation/wrist_image", [])
    official_actions = np.asarray(features.get("steps/action", []), dtype="<f4").reshape(-1, 7)
    action_error = exact_action_error(mirror["actions"], official_actions)
    length = len(official_actions)
    if len(front) != length or len(wrist) != length:
        raise base.AuditError("official comparison images are not aligned with actions")

    mapping = official["mapping"] or {}
    path = features["episode_metadata/file_path"][0].decode()
    official_task = mapping.get("mapped_task", mirror["task"])
    digest = mirror["action_key"][1]
    comparison_root = output / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    final = comparison_root / (
        f"{mirror['suite']}__{mirror['setting']}__slot{mirror['variant_slot']}__same_action.mp4"
    )
    partial = final.with_name(final.stem + ".partial.mp4")
    camera_delta = None

    with h5py.File(mirror["path"], "r") as handle:
        if len(handle["actions"]) != length:
            raise base.AuditError("Mirror image/action lengths differ")
        if mirror["setting"] == "camera" and features.get(
            "episode_metadata/camera_calibration/primary_cam_extrinsics"
        ):
            archive_pose = np.asarray(
                features["episode_metadata/camera_calibration/primary_cam_extrinsics"], dtype=float
            ).reshape(4, 4)
            pose = base._pose_feature(
                np.asarray(handle["camera/front/T_world_cam"][0]),
                base.official_camera_pose(archive_pose),
            )["numeric"]
            camera_delta = {
                "rotation_deg": math.degrees(pose[0]),
                "translation_m": pose[1],
            }

        writer = imageio.get_writer(
            partial,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=["-crf", "20", "-movflags", "+faststart"],
        )
        try:
            for index in range(length):
                canvas = np.full((640, 1024, 3), 24, dtype=np.uint8)
                canvas[:320, :320] = panel(handle["observations/front_rgb"][index], "MIRROR front")
                canvas[:320, 320:640] = panel(decode_rgb(front[index]), "LIBERO-PLUS front")
                canvas[320:, :320] = panel(handle["observations/wrist_rgb"][index], "MIRROR wrist")
                canvas[320:, 320:640] = panel(decode_rgb(wrist[index]), "LIBERO-PLUS wrist")
                lines = [
                    (mirror["setting"].upper(), (98, 203, 255)),
                    (
                        f"Mirror: {mirror['suite']} / slot{mirror['variant_slot']} / {mirror['episode_id']}",
                        (235, 235, 235),
                    ),
                    (f"Official: {mapping.get('reference_id', 'archive scan')}", (235, 235, 235)),
                    (f"EXACT ACTION MATCH  t={index}/{length - 1}", (120, 235, 140)),
                    (f"float32 action SHA-256: {digest[:16]}...", (235, 235, 235)),
                    (f"max action error: {action_error:.1g}", (235, 235, 235)),
                    (f"Task: {mirror['task']}", (235, 235, 235)),
                ]
                if camera_delta:
                    lines.append(
                        (
                            "Front pose delta: "
                            f"{camera_delta['rotation_deg']:.2f} deg / "
                            f"{camera_delta['translation_m']:.3f} m",
                            (120, 235, 140),
                        )
                    )
                draw_text(canvas, lines)
                progress = round(344 * (index + 1) / length)
                cv2.rectangle(canvas, (656, 610), (1000, 622), (72, 72, 72), -1)
                cv2.rectangle(canvas, (656, 610), (656 + progress, 622), (98, 203, 255), -1)
                writer.append_data(canvas)
        finally:
            writer.close()
    os.replace(partial, final)

    member = official["member"]
    row = {
        "setting": mirror["setting"],
        "suite": mirror["suite"],
        "variant_slot": mirror["variant_slot"],
        "alignment": "same float32 action fingerprint; video frame i is action index i on both sides",
        "action_length": length,
        "action_sha256": digest,
        "max_action_error": action_error,
        "mirror_dataset": "libero_plus_rgbd_sample_v2",
        "mirror_episode": mirror["episode_id"],
        "mirror_attempt": mirror["attempt_id"],
        "mirror_source_public_episode": mirror["source_public_episode"],
        "mirror_hdf5_sha256": base.sha256_file(mirror["path"]),
        "mirror_task": mirror["task"],
        "official_dataset": "LIBERO-Plus precollected",
        "official_display_transform": "vertical flip from MuJoCo framebuffer order",
        "official_selection": official["selection"],
        "official_reference_id": mapping.get("reference_id"),
        "official_logical_episode": mapping.get("logical_episode"),
        "official_lerobot_episode_index": mapping.get("lerobot_episode_index"),
        "official_task": official_task,
        "official_file_path": path,
        "official_record_ordinal": official["ordinal"],
        "official_record_offset": official["offset"],
        "official_raw_record_sha256": base.sha256_bytes(official["raw"]),
        "official_tfrecord_crc_valid": official["tfrecord_crc_valid"],
        "official_frame_sha256": base._image_hashes(front),
        **base._archive_evidence(official["source"], member),
        "frames": length,
        "fps": fps,
        "resolution": [1024, 640],
        "video": str(final.resolve()),
        "video_bytes": final.stat().st_size,
        "video_sha256": base.sha256_file(final),
    }
    if camera_delta:
        row["front_camera_pose_delta"] = camera_delta
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=base.DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    output = args.output.resolve()
    mirrors = load_mirrors(output)
    mappings = base.read_jsonl(output / "evidence/reference_mapping.jsonl")
    selected = select_reference_matches(mirrors, mappings)
    archives = {}

    def get_archive(source: str):
        archives.setdefault(source, base.LocalZip(source, output / "evidence/archives"))
        return archives[source]

    descriptors = {}
    for setting, (mirror, mapping) in selected.items():
        source = archive_source(mapping)
        descriptors[setting] = mirror, load_mapped_record(get_archive(source), mapping)
    for setting in base.SETTINGS:
        if setting not in descriptors:
            source = SETTING_SOURCE[setting]
            descriptors[setting] = scan_exact_record(get_archive(source), setting, mirrors)

    rows = []
    for setting in base.SETTINGS:
        mirror, official = descriptors[setting]
        rows.append(render_video(output, mirror, official, args.fps))
        print(json.dumps({"comparison": setting, "video": rows[-1]["video"]}), flush=True)
    manifest = {
        "version": 2,
        "layout": "left Mirror dataset; right LIBERO-Plus precollected; front above wrist",
        "pairing": "task + setting + (action length, SHA-256 of contiguous little-endian float32 bytes)",
        "sampling": "every action index; no normalized-progress alignment",
        "official_display_transform": "vertical flip from MuJoCo framebuffer order",
        "rows": rows,
    }
    base.atomic_json(output / "comparisons/comparison_manifest.json", manifest)


if __name__ == "__main__":
    main()
