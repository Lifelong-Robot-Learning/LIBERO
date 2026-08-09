from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": FixedBaseRobot,
        "OnTheGroundPanda": FixedBaseRobot,
    }
)
