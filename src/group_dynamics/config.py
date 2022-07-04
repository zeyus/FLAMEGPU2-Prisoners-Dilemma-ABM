import math
import pyflamegpu
import random

###
# general
###

# Just an ID.
MODEL_NAME: str = 'uncert_ext'

# total number of agents to create
AGENT_COUNT: int = 10000

# how long the simulation should run
STEP_COUNT: int = 100000

# manually specify the random seed
RANDOM_SEED: int = 69420

###
# Debugging
###
VERBOSE_OUTPUT: bool = False

###
# Environment
###

# The x, y max for the environment
ENV_MAX: int = 1920
# env props are read only for agents, but can be updated with host functions (CPU // slow)
BASELINE_UNCERTAINTY: float = 0.2


###
# Communication
###

# how far messages will propagate (for message spatial)
MESSAGE_RADIUS: float = ENV_MAX / 16



###
# groups
###
# how many groups to model
GROUP_COUNT: int = 1000
# various properties attached to groups
GROUP_PROPERTIES: dict = {
    # how open / closed is the group
    'openness': {
        'func': random.normalvariate,
        'args': (0.0, 1.0),
    },
    'prototye_attributes': {
        'func': random.normalvariate,
        'args': (0.0, 1.0),
    },
}
# we create "group" agents to model the group dynamics
GROUP_AGENT_NAME: str = 'group'
GROUP_SIZE_MIN: int = 3
GROUP_SIZE_MAX: int = math.floor(AGENT_COUNT / 2)

###
# agents
###
INITIAL_GROUPS_MIN: int = 1
INITIAL_GROUPS_MAX: int = 5
AGENT_GROUPS_MAX: int = AGENT_COUNT // GROUP_COUNT
LOCAL_MESSAGE_FUNC: str = 'output_message_local'
MOVE_FUNC: str = 'move'
AGENT_ATTRIBUTE_COUNT: int = 3
# various properties attached to groups
AGENT_PROPERTIES: dict = {
    # how open / closed is the group
    'openness': {
        'func': random.normalvariate,
        'args': (0.0, 1.0),
    },
    'attributes': {
        'func': random.normalvariate,
        'args': (0.0, 1.0),
    },
}
AGENT_NAME: str = 'individual'
# agent size in units, x, y, z
AGENT_SCALE: float = [5] #(2.0, 2.0, 2.0)
# cubes are only 12 polygons
AGENT_3D_MODEL: str = './src/resources/models/primitive_pyramid_arrow.obj'
# agent starting colors - can be static, from a palette, or based on a variable
AGENT_COLOR = pyflamegpu.ViridisInterpolation("attribute1")


###
# visualisation
###
# should the visualisation be shown? or just save the data
USE_VIZ: bool = True
# interactive movement speed of the camera
VIZ_MOVE_SPEED: float = 0.1
# when shift / LT is pressed
VIZ_TURBO_MULT: float = 5.0
VIZ_START_PAUSED: bool = True
VIZ_SHOW_ENV_BOUNDARY: bool = True
# show messages grid
VIZ_COMM_GRID: bool = False


def set_sim_defaults(sim: pyflamegpu.CUDASimulation) -> None:
    sim.SimulationConfig().random_seed = RANDOM_SEED
    sim.SimulationConfig().steps = STEP_COUNT
    sim.SimulationConfig().verbose = VERBOSE_OUTPUT