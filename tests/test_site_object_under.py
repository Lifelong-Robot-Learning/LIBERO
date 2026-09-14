"""Coordinate-frame conformance for the geometric part of the On predicate.

Loads the actual source modules without importing LIBERO's training stack or
interactive package configuration. MuJoCo supplies world poses and contacts;
no renderer, robot policy, demonstration dataset, or mocked rotation is used.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_source(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SiteObject = load_source(
    "libero/libero/envs/objects/site_object.py", "under_test_site_object"
).SiteObject
SiteObjectState = load_source(
    "libero/libero/envs/object_states/base_object_states.py", "under_test_states"
).SiteObjectState
On = load_source(
    "libero/libero/envs/predicates/base_predicates.py", "under_test_predicates"
).On


def scene(yaw, local_point, translation=(0.3, -0.2, 0.7)):
    """Two bodies under one frame keep the relative geometry fixed under yaw.

    The sphere overlaps the support slab slightly at z=.019, so mj_forward
    generates contact without stepping or relying on a settling trajectory.
    The task's narrow region covers only part of the support slab.
    """
    pos = " ".join(map(str, local_point))
    offset = " ".join(map(str, translation))
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco>
          <compiler angle="degree"/>
          <worldbody>
            <body pos="{offset}" euler="0 0 {yaw}">
              <body name="support">
                <geom name="support" type="box" size=".2 .2 .01"/>
                <site name="region" type="box" size=".12 .03 .01"/>
              </body>
              <body name="object" pos="{pos}">
                <joint type="slide" axis="0 0 1"/>
                <geom name="object" type="sphere" size=".01" mass=".01"/>
              </body>
            </body>
          </worldbody>
        </mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = SiteObject(name="region", parent_name="support", size=np.array([.12, .03, .01]))
    return model, data, site


@pytest.mark.parametrize("yaw", [0, 30, 45, 60, 90, -30, -45])
@pytest.mark.parametrize(
    "local_point, expected",
    [
        ((.09, 0, .019), True),  # inside the long x extent
        ((0, .09, .019), False),  # outside the short y extent
    ],
    ids=["inside", "outside"],
)
def test_under_preserves_membership_when_scene_yaws(yaw, local_point, expected):
    model, data, site = scene(yaw, local_point)
    actual = site.under(
        data.site("region").xpos,
        data.site("region").xmat.reshape(3, 3),
        data.body("object").xpos,
    )
    assert bool(actual) is expected


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0, 0, .004), False),  # below lower tolerance
        ((0, 0, .006), True),   # within lower tolerance
        ((0, 0, .109), True),   # below existing height ceiling
        ((0, 0, .111), False),  # above existing height ceiling
        ((.119, 0, .02), True),
        ((.121, 0, .02), False),
    ],
)
def test_under_keeps_existing_axis_aligned_thresholds(point, expected):
    _, data, site = scene(0, point)
    assert bool(site.under(
        data.site("region").xpos,
        data.site("region").xmat.reshape(3, 3),
        data.body("object").xpos,
    )) is expected


def on_predicate(model, data, site):
    """Adapt MuJoCo's official Python API to LIBERO's robosuite sim API."""
    objects = {name: SimpleNamespace(name=name) for name in ["support", "object"]}

    def check_contact(first, second):
        ids = {model.geom(first.name).id, model.geom(second.name).id}
        return any({c.geom1, c.geom2} == ids for c in data.contact)

    env = SimpleNamespace(
        fixtures_dict={"support": objects["support"]},
        objects_dict={"object": objects["object"]},
        object_sites_dict={"region": site},
        obj_body_id={"object": model.body("object").id},
        get_object=objects.get,
        check_contact=check_contact,
        sim=SimpleNamespace(data=SimpleNamespace(
            get_site_xpos=lambda name: data.site(name).xpos,
            get_site_xmat=lambda name: data.site(name).xmat.reshape(3, 3),
            body_xpos=data.xpos,
        )),
    )
    state = SiteObjectState(env, "region", "support")
    return bool(On()(SimpleNamespace(object_name="object"), state)), check_contact(
        objects["support"], objects["object"]
    )


@pytest.mark.parametrize("yaw", [0, 45])
@pytest.mark.parametrize(
    "local_point, expected", [((.09, 0, .019), True), ((0, .09, .019), False)],
    ids=["inside", "outside"],
)
def test_official_on_predicate_with_physical_contact(yaw, local_point, expected):
    model, data, site = scene(yaw, local_point)
    actual, contact = on_predicate(model, data, site)
    assert contact, "Control: the object must actually contact the parent support"
    assert actual is expected


def test_parent_contact_is_still_required():
    model, data, site = scene(0, (.09, 0, .05))
    actual, contact = on_predicate(model, data, site)
    assert not contact
    assert not actual
