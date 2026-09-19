"""Optional density overrides applied before MuJoCo compiles a task."""

import json
import math
import os
from functools import lru_cache
from pathlib import Path


def load_density_config(path):
    densities = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(densities, dict):
        raise ValueError("Object density config must be a JSON object")
    for name, density in densities.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Object density keys must be nonempty strings")
        if isinstance(density, bool) or not isinstance(density, (int, float)) or not math.isfinite(density) or density <= 0:
            raise ValueError(f"Density for {name} must be a positive finite number in kg/m^3")
    return densities


@lru_cache(maxsize=None)
def _config_for_path(path):
    return load_density_config(path)


def apply_density_overrides(objects, densities):
    """Set collision-geom density for matching object categories or instances."""
    changed = 0
    for instance_name, obj in objects.items():
        density = densities.get(instance_name, densities.get(obj.category_name))
        if density is None:
            continue
        subtree = obj.get_obj()
        if any(body.find("inertial") is not None for body in subtree.iter("body")):
            raise ValueError(f"Cannot override {instance_name}: explicit inertial element")
        collision_geoms = [geom for geom in subtree.iter("geom") if geom.get("group", "0") == "0"]
        if any(geom.get("mass") is not None for geom in collision_geoms):
            raise ValueError(f"Cannot override {instance_name}: explicit mass on a collision geom")
        if not collision_geoms:
            raise ValueError(f"Cannot override {instance_name}: no collision geoms")
        for geom in collision_geoms:
            geom.set("density", f"{density:g}")
        changed += 1
    return changed


def apply_configured_densities(objects):
    path = os.environ.get("LIBERO_OBJECT_DENSITY_CONFIG")
    if path:
        return apply_density_overrides(objects, _config_for_path(path))
    return 0
