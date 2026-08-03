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


if __name__ == "__main__":
    unittest.main()
