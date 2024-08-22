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


# CUSTOM_XPOS_OFFSETS = {
#     "Panda": {
#         "bins": (-0.5, -0.1, 0),
#         "empty": (-0.6, 0, 0),
#         "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
#         "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
#         "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
#     },
#     "OnTheGroundPanda": {
#         "bins": (-0.5, -0.1, 0),
#         "empty": (-0.6, 0, 0),
#         "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
#         "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
#         "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
#     }
# }

