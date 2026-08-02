import importlib.util
import inspect
from pathlib import Path
import struct
import unittest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/create_rgbd_samples.py"
SPEC = importlib.util.spec_from_file_location("create_rgbd_samples", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SampleProtocolTest(unittest.TestCase):
    @staticmethod
    def _varint(value):
        result = bytearray()
        while value > 0x7F:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value)
        return bytes(result)

    @classmethod
    def _field(cls, number, value):
        return cls._varint(number << 3 | 2) + cls._varint(len(value)) + value

    @classmethod
    def _example(cls):
        def feature(name, kind, payload):
            value = cls._field(kind, cls._field(1, payload))
            return cls._field(1, cls._field(1, name.encode()) + cls._field(2, value))

        features = b"".join(
            (
                feature("bytes", 1, b"jpeg"),
                feature("floats", 2, struct.pack("<2f", 1.25, -2.5)),
                feature("ints", 3, cls._varint(3) + cls._varint(9)),
            )
        )
        return cls._field(1, features)

    def test_tf_example_tfrecord_and_ordinal_location(self):
        example = self._example()
        length = struct.pack("<Q", len(example))
        tfrecord = (
            length
            + struct.pack("<L", MODULE._masked_crc32c(length))
            + example
            + struct.pack("<L", MODULE._masked_crc32c(example))
        )
        rows = list(MODULE.tfrecord_examples(tfrecord))
        self.assertEqual(rows[0][3]["bytes"], [b"jpeg"])
        self.assertTrue(np.allclose(rows[0][3]["floats"], [1.25, -2.5]))
        self.assertEqual(rows[0][3]["ints"], [3, 9])
        self.assertEqual(MODULE.ordinal_to_shard(0, [2, 3]), (0, 0))
        self.assertEqual(MODULE.ordinal_to_shard(4, [2, 3]), (1, 2))

    def test_unique_action_alignment_and_noise_coverage(self):
        full = np.array([[0.0], [1.0], [2.0], [3.0]])
        mask = MODULE.unique_subsequence_mask(full, full[[1, 3]])
        self.assertEqual(mask.tolist(), [False, True, False, True])
        self.assertIsNone(
            MODULE.unique_subsequence_mask(np.array([[0.0], [1.0], [1.0], [2.0]]), np.array([[1.0], [2.0]]))
        )
        algorithms = [
            MODULE.NOISE_ORDER[(MODULE.SUITES.index(suite) * 3 + slot - 1) % 5]
            for suite in MODULE.SUITES
            for slot in (1, 2, 3)
        ]
        self.assertTrue(all(algorithms.count(name) >= 2 for name in MODULE.NOISE_ORDER))

    def test_duplicate_action_multimap_and_disambiguation(self):
        actions = np.array([[1.0] * 7, [2.0] * 7], dtype=np.float32)
        states = np.array([[3.0] * 8, [4.0] * 8], dtype=np.float32)
        key = MODULE.action_key(actions)
        targets = {
            key: [
                {"episode_index": 1, "suite": "s", "task": "t", "actions": actions, "states": states},
                {"episode_index": 2, "suite": "s", "task": "t", "actions": actions, "states": states},
            ]
        }
        self.assertEqual(len(targets[key]), 2)
        record = {
            "action_key": key,
            "actions": actions,
            "states": states,
            "frame_sha256": ["rgb"],
            "rgb_signatures": [np.zeros((32, 32, 3), dtype=np.uint8)] * 3,
        }
        selected = {"suite": "s", "task": "t"}
        matched = MODULE._match_reordered(
            record,
            targets,
            selected,
            lambda target: [
                np.zeros((32, 32, 3), dtype=np.uint8)
                if target["episode_index"] == 2
                else np.full((32, 32, 3), 100, dtype=np.uint8)
            ]
            * 3,
        )
        self.assertEqual(matched[0]["episode_index"], 2)
        self.assertIn("RGB", matched[1])
        self.assertIsNone(
            MODULE._match_reordered(
                record,
                targets,
                selected,
                lambda target: [np.zeros((32, 32, 3), dtype=np.uint8)] * 3,
            )
        )
        env_record = {
            "first_mask": np.array([[0, 1, 9, 10]], dtype=np.uint8),
            "object_map": {"task_object": 1},
        }
        self.assertEqual(MODULE._path_settings("/env/suite/demo.hdf5", env_record), ("objects", "background"))
        self.assertEqual(env_record["extra_objects"], 2)

    def test_stable_split_probe_ranking_and_numeric_report(self):
        rows = [
            {"reference_id": f"r{index}", "logical_episode": f"e{index}"}
            for index in range(8)
        ]
        first = MODULE.stable_reference_split(rows)
        second = MODULE.stable_reference_split(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual([row["split"] for row in first].count("calibration"), 6)
        self.assertFalse(
            {row["logical_episode"] for row in first if row["split"] == "calibration"}
            & {row["logical_episode"] for row in first if row["split"] == "holdout"}
        )
        probes = [
            {"candidate_id": "b", "status": "valid", "loss": 1.0},
            {"candidate_id": "a", "status": "valid", "loss": 1.0},
            {"candidate_id": "c", "status": "invalid", "loss": 0.0},
        ]
        self.assertEqual([row["candidate_id"] for row in MODULE.rank_probes(probes)], ["a", "b"])
        self.assertNotIn("unavailable", inspect.getsource(MODULE.write_gap_report))

    def test_manifest_rejects_non_diverse_official_language_before_replay(self):
        import h5py
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [
                {
                    "reference_id": f"ref-{index}",
                    "suite": "libero_spatial",
                    "setting": "language",
                    "split": "calibration" if index < 6 else "holdout",
                    "confidence": "high",
                }
                for index in range(8)
            ]
            MODULE.atomic_jsonl(output / "evidence/reference_mapping.jsonl", rows)
            with h5py.File(output / "evidence/official_references.hdf5", "w") as handle:
                for row in rows:
                    group = handle.create_group(row["reference_id"])
                    group.create_dataset("first_rgb", data=np.zeros((2, 2, 3), dtype=np.uint8))
                    MODULE.text_dataset(group, "object_map_json", "{}")
                    MODULE.text_dataset(group, "language", "same text")
            with self.assertRaisesRegex(MODULE.AuditError, "language diversity"):
                MODULE.load_reference_features(output, {}, 0)
            report = json.loads((output / "reports/calibration.json").read_text())
            self.assertEqual(report["decision"], "NO-GO")
            self.assertEqual(report["language_text_diversity"]["libero_spatial"]["calibration"], 1)
            languages = {
                "libero_spatial": [
                    {"candidate_id": f"language-{index + 1}", "instruction": f"rewrite {index + 1}"}
                    for index in range(8)
                ]
            }
            features = MODULE.load_reference_features(output, {}, 0, languages)
            self.assertEqual(
                len({item["language"] for item in features[("libero_spatial", "language")]}), 8
            )
            self.assertTrue(
                all(
                    item["reference_source"].startswith("qwen-fallback:")
                    for item in features[("libero_spatial", "language")]
                )
            )


if __name__ == "__main__":
    unittest.main()
