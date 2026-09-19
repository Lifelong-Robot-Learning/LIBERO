import json
import importlib.util
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "libero/libero/envs/object_density_config.py"
spec = importlib.util.spec_from_file_location("object_density_config", module_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
apply_density_overrides = config_module.apply_density_overrides
load_density_config = config_module.load_density_config


class FakeObject:
    category_name = "porcelain_mug"

    def __init__(self):
        self.tree = ET.fromstring('<body><geom group="0" density="100"/><geom group="1" density="100"/></body>')

    def get_obj(self):
        return self.tree


class DensityConfigTests(unittest.TestCase):
    def test_only_collision_geometry_changes(self):
        obj = FakeObject()
        apply_density_overrides({"porcelain_mug_1": obj}, {"porcelain_mug": 800})
        self.assertEqual([g.get("density") for g in obj.get_obj().iter("geom")], ["800", "100"])

    def test_instance_takes_priority(self):
        obj = FakeObject()
        apply_density_overrides({"porcelain_mug_1": obj}, {"porcelain_mug": 800, "porcelain_mug_1": 950})
        self.assertEqual(next(obj.get_obj().iter("geom")).get("density"), "950")

    def test_explicit_mass_is_rejected(self):
        obj = FakeObject()
        next(obj.get_obj().iter("geom")).set("mass", "0.2")
        with self.assertRaisesRegex(ValueError, "explicit mass"):
            apply_density_overrides({"porcelain_mug_1": obj}, {"porcelain_mug": 800})

    def test_invalid_density_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "density.json"
            path.write_text(json.dumps({"porcelain_mug": -1}))
            with self.assertRaisesRegex(ValueError, "positive"):
                load_density_config(path)


if __name__ == "__main__":
    unittest.main()
