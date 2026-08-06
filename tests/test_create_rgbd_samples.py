import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/create_rgbd_samples.py"
SPEC = importlib.util.spec_from_file_location("create_rgbd_samples", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_audit():
    tasks = []
    for task_index in range(40):
        multi_step = task_index % 3 == 0
        stem = f"fake_task_{task_index:02d}"
        tasks.append(
            {
                "task_index": task_index,
                "task": stem,
                "suite": MODULE.SUITES[task_index // 10],
                "task_stem": stem,
                "canonical_language": "put x in y and close z" if multi_step else "put x in y",
                "multi_step": multi_step,
                "canonical_bddl": str(SCRIPT),
                "canonical_bddl_sha256": "bddl-hash",
                "source_file": f"/source/{stem}.hdf5",
                "demos": [
                    {
                        "source_demo": "demo_0",
                        "source_ordinal": task_index,
                        "actions": 2,
                        "saved_frames": 1,
                        "action_sha256": f"action-{task_index}",
                        "keep_mask_sha256": f"mask-{task_index}",
                        "source_model_xml_sha256": f"xml-{task_index}",
                    }
                ],
            }
        )
    return {"source_demo_count": 40, "tasks": tasks}


def fake_languages(audit):
    result = {"tasks": {}}
    for task in audit["tasks"]:
        subtypes = {}
        for subtype in MODULE.SUBDIMENSIONS["language"]:
            if subtype == "R3" and not task["multi_step"]:
                continue
            subtypes[subtype] = [
                {
                    "candidate_id": f"{subtype.lower()}-{index}",
                    "instruction": f"{subtype} rewrite {index} for {task['task_stem']}",
                    "prompt": f"prompt {subtype} {index}",
                    "prompt_hash": f"prompt-{subtype}-{index}",
                    "model_hash": "qwen-hash",
                }
                for index in range(1, 4)
            ]
        result["tasks"][task["task_stem"]] = subtypes
    return result


def fake_bddl_candidates(task, setting, subtype, source_ordinal):
    rows = []
    for index in range(1, 4):
        row = {
            "candidate_id": f"{setting}-{subtype}-{index}",
            "bddl": str(SCRIPT),
            "bddl_sha256": f"bddl-{index}",
            "problem": f"problem-{index}",
            "benchmark_bddl_excluded": True,
        }
        if setting == "objects":
            row["extra_objects"] = source_ordinal % 3 + 1
        elif setting == "background":
            row["variant"] = "scene_and_surface" if subtype == "B1" else "surface_only"
        else:
            row["required_xml_field"] = MODULE.LIGHT_FIELDS[subtype]
        rows.append(row)
    return rows


def build_rows():
    audit = fake_audit()
    with mock.patch.object(MODULE, "task_test_names", return_value=[]), mock.patch.object(
        MODULE, "language_test_texts", return_value=set()
    ), mock.patch.object(MODULE, "_bddl_candidates", side_effect=fake_bddl_candidates):
        return MODULE.build_manifest_rows(audit, fake_languages(audit))


class ReplicaProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_rows()

    def test_manifest_is_six_jobs_per_source_and_covers_all_mechanisms(self):
        MODULE.validate_manifest_rows(self.rows, 40)
        self.assertEqual(len(self.rows), 240)
        grouped = {}
        for row in self.rows:
            grouped.setdefault((row["task_index"], row["source_demo"]), set()).add(row["setting"])
        self.assertTrue(all(settings == set(MODULE.SETTINGS) for settings in grouped.values()))
        self.assertEqual(
            {code for row in self.rows for code in row["subdimensions"]},
            {code for values in MODULE.SUBDIMENSIONS.values() for code in values},
        )
        self.assertTrue(
            all(
                candidate["roll_deg"] != 0
                for row in self.rows
                if row["subdimensions"] == ["C3"]
                for candidate in row["candidates"]
            )
        )
        self.assertTrue(
            all(row["multi_step"] for row in self.rows if row["subdimensions"] == ["R3"])
        )

    def test_language_variants_remain_distinct_when_qwen_repeats_a_rewrite(self):
        repeated = "place the condiment container in the woven receptacle"
        rewrites = {
            MODULE.enforce_variant_prefix(repeated, variant)
            for variant in MODULE.LANGUAGE_VARIANTS["R2"][:3]
        }
        self.assertEqual(len(rewrites), 3)

    def test_manifest_excludes_benchmark_candidates_and_partitions_exactly(self):
        for row in self.rows:
            for candidate in row["candidates"]:
                if row["setting"] in {"objects", "background", "light"}:
                    self.assertTrue(candidate["benchmark_bddl_excluded"])
                elif row["setting"] == "camera":
                    self.assertFalse(candidate["benchmark_tuple_equal"])
                    self.assertGreaterEqual(candidate["min_test_distance"], 5)
                elif row["setting"] == "language":
                    self.assertTrue(candidate["benchmark_language_excluded"])
                elif row["setting"] == "noise":
                    self.assertFalse(candidate["benchmark_tuple_equal"])

        partitions = [
            {row["job_id"] for row in self.rows if row["task_index"] == task_index}
            for task_index in range(40)
        ]
        self.assertTrue(all(len(partition) == 6 for partition in partitions))
        self.assertEqual(sum(map(len, partitions)), len(set().union(*partitions)))
        self.assertEqual(set().union(*partitions), {row["job_id"] for row in self.rows})

    def test_smoke_selection_has_three_sources_per_suite_and_full_coverage(self):
        rows = []
        for index, ordinal in enumerate(MODULE.SMOKE_SOURCE_ORDINALS):
            suite = MODULE.SUITES[index // 3]
            for setting in MODULE.SETTINGS:
                values = MODULE.SUBDIMENSIONS[setting]
                rows.append(
                    {
                        "source_ordinal": ordinal,
                        "suite": suite,
                        "setting": setting,
                        "subdimensions": [values[index % len(values)]],
                    }
                )
        selected = MODULE.select_smoke_jobs(rows)
        self.assertEqual(len(selected), 72)
        self.assertEqual(
            {code for row in selected for code in row["subdimensions"]},
            {code for values in MODULE.SUBDIMENSIONS.values() for code in values},
        )

    def test_noop_mask_matches_the_twelve_v2_source_trajectories(self):
        audit_path = ROOT / "data/libero_plus_rgbd_sample_v2/audit.json"
        if not audit_path.is_file():
            self.skipTest("v2 no-op regression evidence is not present")
        audit = json.loads(audit_path.read_text())
        checked = 0
        for suite in audit["selected_tasks"].values():
            source = suite["source"]
            with h5py.File(source["source_file"], "r") as handle:
                for selected in source["selected"]:
                    actions = handle[f"data/{selected['source_demo']}/actions"][()]
                    mask = MODULE.keep_action_mask(actions)
                    self.assertEqual(mask.astype(int).tolist(), selected["keep_mask"])
                    self.assertEqual(MODULE.sha256_bytes(mask.tobytes()), selected["keep_mask_sha256"])
                    checked += 1
        self.assertEqual(checked, 12)

    def test_random_noise_is_reproducible_but_varies_by_timestep(self):
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, (MODULE.IMAGE_SIZE, MODULE.IMAGE_SIZE, 3), dtype=np.uint8)
        for subtype in ("N1", "N4", "N5"):
            candidate = MODULE.noise_candidates(subtype, 7, 11)[0]
            first_seed = MODULE.stable_seed(123, 4)
            second_seed = MODULE.stable_seed(123, 5)
            first = MODULE.apply_noise(image, candidate, first_seed)
            self.assertTrue(np.array_equal(first, MODULE.apply_noise(image, candidate, first_seed)))
            self.assertFalse(np.array_equal(first, MODULE.apply_noise(image, candidate, second_seed)))
            if subtype == "N5":
                one_iteration = json.loads(json.dumps(candidate))
                one_iteration["parameters"]["iterations"] = 1
                self.assertFalse(
                    np.array_equal(first, MODULE.apply_noise(image, one_iteration, first_seed))
                )

    def test_camera_pose_is_episode_static_and_c3_has_roll(self):
        class Model:
            cam_pos = np.array([[1.0, 0.0, 1.0]])
            cam_quat = np.array([[1.0, 0.0, 0.0, 0.0]])

            @staticmethod
            def camera_name2id(name):
                return 0

        class Sim:
            model = Model()

            @staticmethod
            def forward():
                pass

        class Env:
            sim = Sim()

        for subtype in MODULE.SUBDIMENSIONS["camera"]:
            env = Env()
            candidate = MODULE.camera_candidates(subtype, [], 13)[0]
            MODULE.apply_camera_perturbation(env, candidate)
            poses = [(env.sim.model.cam_pos.copy(), env.sim.model.cam_quat.copy()) for _ in range(4)]
            self.assertTrue(all(np.array_equal(poses[0][0], pose[0]) for pose in poses[1:]))
            self.assertTrue(all(np.array_equal(poses[0][1], pose[1]) for pose in poses[1:]))
            if subtype == "C3":
                self.assertGreaterEqual(abs(candidate["roll_deg"]), 2)

    def test_failed_episode_does_not_stop_later_jobs_and_resume_skips_terminals(self):
        rows = []
        for index in range(2):
            rows.append(
                {
                    "job_id": f"job-{index}",
                    "task_index": 0,
                    "output_path": f"episodes/task-00/job-{index}.hdf5",
                    "candidates": [
                        {"candidate_index": candidate, "candidate_id": f"c{candidate}"}
                        for candidate in range(1, 4)
                    ],
                }
            )
        successful = set()
        calls = []

        def valid(path, row):
            return row["job_id"] in successful

        def attempt(row, candidate, output, runtime, gpu):
            calls.append((row["job_id"], candidate["candidate_index"]))
            if row["job_id"] == "job-0":
                raise MODULE.AttemptFailure("expected failure")
            successful.add(row["job_id"])
            return {"job_id": row["job_id"]}

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            MODULE.atomic_jsonl(output / "manifest.jsonl", rows)
            args = argparse.Namespace(output=output, task_index=0, gpu=0, smoke_only=False)
            patches = (
                mock.patch.object(MODULE, "validate_manifest_rows"),
                mock.patch.object(MODULE, "configure_libero"),
                mock.patch.object(MODULE, "runtime_metadata", return_value={}),
                mock.patch.object(MODULE, "episode_structurally_valid", side_effect=valid),
                mock.patch.object(MODULE, "run_attempt_isolated", side_effect=attempt),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                MODULE.generate(args)
                first_calls = list(calls)
                MODULE.generate(args)
            self.assertEqual(first_calls, [("job-0", 1), ("job-0", 2), ("job-0", 3), ("job-1", 1)])
            self.assertEqual(calls, first_calls)
            ledger = MODULE.read_jsonl(output / "attempts/task-00.jsonl")
            self.assertEqual(sum(row["status"] == "failed" for row in ledger), 3)
            self.assertEqual(sum(row["status"] == "rejected" for row in ledger), 1)
            self.assertEqual(sum(row["status"] == "success" for row in ledger), 1)

    def test_generated_document_uses_coverage_and_retention_counts(self):
        coverage_rows = []
        for setting in MODULE.SETTINGS:
            for code in MODULE.SUBDIMENSIONS[setting]:
                coverage_rows.append(
                    {
                        "setting": setting,
                        "subdimension": code,
                        "jobs": 4,
                        "attempts": 5,
                        "success": 3,
                        "rejected": 1,
                        "retention_rate": 0.75,
                        "parameters": {"proof": code},
                        "example": f"episodes/{code}.hdf5",
                    }
                )
        coverage = {"generated_at": "2026-08-05T00:00:00Z", "subdimensions": coverage_rows}
        retention = {
            "jobs": 72,
            "candidate_attempts": 90,
            "success": 54,
            "rejected": 18,
            "pending": 0,
            "retention_rate": 0.75,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            MODULE.atomic_json(output / "protocol.json", MODULE.protocol())
            MODULE.atomic_jsonl(output / "manifest.jsonl", self.rows)
            MODULE.atomic_json(output / "coverage.json", coverage)
            MODULE.atomic_json(output / "retention.json", retention)
            MODULE.write_perturbations_doc(output, coverage, retention, "manifest-hash")
            document = (output / "PERTURBATIONS.md").read_text()
            self.assertIn("- Jobs: `72`", document)
            self.assertIn("- Successful episodes: `54`", document)
            self.assertIn("- Rejected episodes: `18`", document)
            for code in {item["subdimension"] for item in coverage_rows}:
                self.assertIn(f" / {code} |", document)


if __name__ == "__main__":
    unittest.main()
