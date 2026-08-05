import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_example", EXAMPLE_ROOT / "run_example.py"
)
RUN_EXAMPLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUN_EXAMPLE)


def write_json(handle: h5py.File, path: str, value: dict) -> None:
    handle.create_dataset(
        path,
        data=np.frombuffer(
            json.dumps(value, sort_keys=True).encode(), dtype=np.uint8
        ),
    )


class RunExampleTest(unittest.TestCase):
    def test_default_analytic_geometry_command_uses_both_profiles(self) -> None:
        command = RUN_EXAMPLE.analytic_geometry_command(
            "/locked/python",
            EXAMPLE_ROOT,
            Path("/run/geometry.h5"),
            "smoke",
        )
        self.assertEqual(
            command[command.index("--analytic-profile") + 1], "debug"
        )
        self.assertEqual(
            command[command.index("--validation-profile") + 1], "smoke"
        )
        self.assertEqual(RUN_EXAMPLE.lxe_generator_profile("smoke"), ["--test"])
        self.assertEqual(RUN_EXAMPLE.lxe_generator_profile("coarse"), [])

    def test_scene_uses_the_selected_lxe_block_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            block = directory / "block.h5"
            output = directory / "scene.yaml"
            with h5py.File(block, "w") as handle:
                write_json(
                    handle,
                    "/metadata/generator_json",
                    {
                        "explicit_collision_order": 7,
                        "lxe_config": {
                            "radius_mm": 1.0,
                            "depth_mm": 5.0,
                            "side_reflectivity": 0.0,
                        },
                    },
                )
                write_json(
                    handle,
                    "/metadata/phase_grid_json",
                    {
                        "position_radial_bins": 2,
                        "position_phi_bins": 2,
                        "direction_mu_bins": 2,
                        "direction_phi_bins": 2,
                        "direction_mu_minimum": 0.804,
                    },
                )
            config = RUN_EXAMPLE.render_scene(
                EXAMPLE_ROOT / "scene.yaml", output, block
            )
            text = output.read_text(encoding="utf-8")
            self.assertEqual(config["radius_mm"], 1.0)
            self.assertEqual(config["depth_mm"], 5.0)
            self.assertEqual(
                config["phase_grid"]["position_radial_bins"], 2
            )
            self.assertIn('"position_radial_bins":2', text)
            self.assertNotIn('"position_radial_bins":24', text)

    def test_lxe_block_must_match_geometry_radius_and_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            geometry = directory / "geometry.h5"
            block = directory / "block.h5"
            with h5py.File(geometry, "w") as handle:
                write_json(
                    handle,
                    "/metadata/generator_json",
                    {
                        "config": {
                            "active_radius_mm": 12.0,
                            "lxe_depth_mm": 34.0,
                        }
                    },
                )
                analytic = handle.create_group("analytic")
                analytic.create_dataset("kind", data=[1])
                analytic.create_dataset("surface_id", data=[1])
                analytic.create_dataset(
                    "parameters", data=[[12.0, 0.0, 0.0, 0.0]]
                )
            with h5py.File(block, "w") as handle:
                write_json(
                    handle,
                    "/metadata/generator_json",
                    {
                        "lxe_config": {
                            "radius_mm": 12.0,
                            "depth_mm": 34.0,
                        },
                        "geometry_contract_sha256": "contract",
                    },
                )
                write_json(
                    handle,
                    "/metadata/phase_grid_json",
                    {
                        "radius_mm": 12.0,
                        "position_radial_bins": 4,
                        "position_phi_bins": 8,
                        "direction_mu_bins": 2,
                        "direction_phi_bins": 4,
                    },
                )
            contract = RUN_EXAMPLE.verify_lxe_block_geometry(geometry, block)
            self.assertEqual(contract["radius_mm"], 12.0)
            self.assertEqual(contract["depth_mm"], 34.0)
            self.assertTrue(contract["test_profile"])
            with h5py.File(block, "r+") as handle:
                del handle["/metadata/generator_json"]
                write_json(
                    handle,
                    "/metadata/generator_json",
                    {
                        "lxe_config": {
                            "radius_mm": 13.0,
                            "depth_mm": 34.0,
                        }
                    },
                )
            with self.assertRaisesRegex(RuntimeError, "dimensions"):
                RUN_EXAMPLE.verify_lxe_block_geometry(geometry, block)


if __name__ == "__main__":
    unittest.main()
