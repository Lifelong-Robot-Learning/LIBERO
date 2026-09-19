"""Regression tests for containment in a site's own coordinate frame."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_source(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def site_classes(monkeypatch):
    for name in (
        "robosuite",
        "robosuite.utils",
        "robosuite.models",
        "libero",
        "libero.libero",
        "libero.libero.envs",
        "libero.libero.envs.objects",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    mjcf = ModuleType("robosuite.utils.mjcf_utils")
    mjcf.string_to_array = lambda value: np.fromstring(value, sep=" ")
    mjcf.array_to_string = lambda value: " ".join(map(str, value))
    for name in ("xml_path_completion", "find_elements", "CustomMaterial", "add_to_dict", "RED", "GREEN", "BLUE"):
        setattr(mjcf, name, None)
    monkeypatch.setitem(sys.modules, mjcf.__name__, mjcf)
    monkeypatch.setitem(sys.modules, "robosuite.utils.transform_utils", ModuleType("transform_utils"))

    objects = ModuleType("robosuite.models.objects")
    objects.MujocoXMLObject = object
    monkeypatch.setitem(sys.modules, objects.__name__, objects)

    base_object = ModuleType("libero.libero.envs.base_object")
    base_object.register_object = lambda cls: cls
    base_object.register_visual_change_object = lambda cls: cls
    monkeypatch.setitem(sys.modules, base_object.__name__, base_object)

    site = load_source(
        "libero/libero/envs/objects/site_object.py",
        "libero.libero.envs.objects.site_object",
    )
    target = load_source(
        "libero/libero/envs/objects/target_zones.py",
        "libero.libero.envs.objects.target_zones",
    )
    return site.SiteObject, target.TargetZone


@pytest.mark.parametrize("which", [0, 1], ids=["site", "target-zone"])
@pytest.mark.parametrize(
    "local_point, expected",
    [((0.09, 0.0, 0.0), True), ((0.0, 0.08, 0.0), False)],
    ids=["inside", "outside"],
)
@pytest.mark.parametrize("yaw", [0, 45, 90])
def test_containment_is_rotation_invariant(site_classes, which, local_point, expected, yaw):
    SiteObject, TargetZone = site_classes
    site = (
        SiteObject(name="region", size=np.array([0.12, 0.03, 0.01]))
        if which == 0
        else TargetZone(name="zone", zone_size=(0.12, 0.03), zone_height=0.01)
    )
    angle = np.deg2rad(yaw)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    center = np.array([0.3, -0.2, 0.7])
    point = center + rotation @ np.asarray(local_point)

    assert bool(site.in_box(center, rotation, point)) is expected


@pytest.mark.parametrize("which", [0, 1], ids=["site", "target-zone"])
def test_lower_z_tolerance_is_in_site_frame(site_classes, which):
    SiteObject, TargetZone = site_classes
    site = (
        SiteObject(name="region", size=np.array([0.12, 0.03, 0.01]))
        if which == 0
        else TargetZone(name="zone", zone_size=(0.12, 0.03), zone_height=0.01)
    )
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    center = np.array([0.3, -0.2, 0.7])
    inside = center + rotation @ np.array([0, 0, -0.015])
    outside = center + rotation @ np.array([0, 0, -0.025])

    assert bool(site.in_box(center, rotation, inside))
    assert not bool(site.in_box(center, rotation, outside))
