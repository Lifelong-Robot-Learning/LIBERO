import importlib.util
from pathlib import Path

import numpy as np


_SITE_OBJECT_PATH = Path(__file__).parents[1] / "libero" / "libero" / "envs" / "objects" / "site_object.py"
_SPEC = importlib.util.spec_from_file_location("libero_site_object", _SITE_OBJECT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
SiteObject = _MODULE.SiteObject


def test_in_box_is_invariant_to_site_rotation():
    site = SiteObject("site", size=np.array([1.0, 0.2, 0.2]))
    angle = np.deg2rad(45.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = np.array([0.3, -0.2, 0.5])

    inside = center + rotation @ np.array([0.9, 0.0, 0.0])
    outside = center + rotation @ np.array([0.0, 0.3, 0.0])

    assert site.in_box(center, rotation, inside)
    assert not site.in_box(center, rotation, outside)


def test_in_box_preserves_lower_z_tolerance_without_mutating_inputs():
    site = SiteObject("site", size=np.array([0.5, 0.5, 0.1]))
    center = np.array([0.0, 0.0, 0.0])
    rotation = np.eye(3)
    lower_edge = np.array([0.0, 0.0, -0.105])

    assert site.in_box(center, rotation, lower_edge)
    np.testing.assert_array_equal(center, np.zeros(3))
