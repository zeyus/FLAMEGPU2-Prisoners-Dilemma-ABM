# FLAMEGPU2 Based Prisoner's Dilemma ABM

## What is it?

A (3D) 2D ABM simulation executed on the GPU. This ABM models interactions of "games" between agents, specifically the Prisoner's Dilemma game, in which participants can either cooperate or defect, resulting in a payoff, depending on the combination of decisions.

Using [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2), agent counts can go into the millions without a problem, especially on a decent GPU.

## Details

One of the earlier descriptions of the game interactions:

<img width="326" alt="image" src="https://user-images.githubusercontent.com/75656/184105191-1f7af765-add8-4161-9998-062c39f65c18.png">

_Figure from Axelrod, R., & Hamilton, W. D. (1981). The Evolution of Cooperation. Science, 211(4489), 1390–1396. [https://doi.org/10.1126/science.7466396](https://doi.org/10.1126/science.7466396)_


In this setup the default payoff matrix for interactions is

|   | They Cooperate | They Defect |
|:---------:|:------------:|:---------:|
| I Cooperate | 3.0 | -1.0 |
| I Defect | 5.0 | 0.0 |

Agents play all of their neighbours and if they cannot, they move, and pay a cost of travel.

After play or movement each agent has the opportunity to reproduce if they have sufficient energy and an available space around them, if they reproduce, they pay the energy cost of reproduction.

Each round there is a cost of living imposed on all agents except agents that were just born in the reproduction phase.

Any agents that drop below zero energy die.

## Other features

- Agent strategy distributions can be configured (more than 4 strategies broken at the moment, but strategy probability can be set to 0.0)
- Agents have a random trait assigned to them (default 1 of 4 possible traits) (more than 4 traits broken at the moment)
- Strategy mutation can be configured at a specific mutation rate which applies during reproduction
- Agents can employ a global strategy (i.e. always cooperate, with any agent) or a strategy for agents with the same trait (kin) or others, OR (reporting broken for this but it works) a strategy per unique trait)
- Environmental noise can be configured for a chance of miscommunication (i.e. if i choose cooperate, it becomes defect)
- Can run in a CUDAEnsemble for a whole suite of simulation runs
- logging is configured for both single and multi runs, currently it collects the agent counts by their strategies, but it should also not bother doing any counts (for performance) if logging is disabled, which it still does

## Model description

<img width="1238" alt="Prisoner's Dilemma ABM model flow" src="https://user-images.githubusercontent.com/75656/184108979-10fbb3d9-32f0-4610-9941-a67593097527.png">

## Running the simulation

### Prerequisites

- python (tested on 3.11)
- CUDA Capable GPU
- Windows or Linux (not sure about FLAMEGPU2 mac support, might be possible to compile it)
- NVIDIA CUDA
- [pyflamegpu](https://github.com/FLAMEGPU/FLAMEGPU2/releases), [version 2.0.0-rc](https://github.com/FLAMEGPU/FLAMEGPU2/releases/tag/v2.0.0-rc) (or higher), either built from source with whichever CUDA version you like, or download a pip wheel that matches your system
- [numpy](https://numpy.org/) (for initial agent matrix positioning, I'll see if I can remove this requirement later because it's clearly adding huge overhead)

Numpy is included in [requirements.txt](requirements.txt):

```python
git clone https://github.com/zeyus/FLAMEGPU2-Prisoners-Dilemma-ABM.git
cd FLAMEGPU2-Prisoners-Dilemma-ABM
python3 -m pip install -r requirements.txt
```

### Try it out

from the root directory of the repository run:

`python3 src/model.py`

The first section in `model.py` contains most of the variables you might want to change.

The default settings are defined as follows:

```python
# upper agent limit ... please make it a square number for sanity
# this is essentially the size of the grid
MAX_AGENT_SPACES: int = 2**18
# starting agent limit
INIT_AGENT_COUNT: int = int(MAX_AGENT_SPACES * 0.16)

# you can set this anywhere between INIT_AGENT_COUNT and MAX_AGENT_COUNT inclusive
# carrying capacity
AGENT_HARD_LIMIT: int = int(MAX_AGENT_SPACES * 0.5)

# how long to run the sim for
STEP_COUNT: int = 100
# TODO: logging / Debugging
WRITE_LOG: bool = True
LOG_FILE: str = f"data/{strftime('%Y-%m-%d %H-%M-%S')}_{RANDOM_SEED}.json"
VERBOSE_OUTPUT: bool = False
DEBUG_OUTPUT: bool = False
OUTPUT_EVERY_N_STEPS: int = 1

# rate limit simulation?
SIMULATION_SPS_LIMIT: int = 0  # 0 = unlimited

# Show agent visualisation
USE_VISUALISATION: bool = True and pyflamegpu.VISUALISATION

# visualisation camera speed
VISUALISATION_CAMERA_SPEED: float = 0.1
# pause the simulation at start
PAUSE_AT_START: bool = False
VISUALISATION_BG_RGB: List[float] = [0.1, 0.1, 0.1]

# should agents rotate to face the direction of their last action?
VISUALISATION_ORIENT_AGENTS: bool = False
# radius of message search grid (broken now from hardcoded x,y offset map)
MAX_PLAY_DISTANCE: int = 1

# Energy cost per step
COST_OF_LIVING: float = 1.0

# Reproduce if energy is above this threshold
REPRODUCE_MIN_ENERGY: float = 100.0
# Cost of reproduction
REPRODUCE_COST: float = 50.0
# Can reproduce in dead agent's space?
# @TODO: if time, actually implement this, for now. no effect (always True)
ALLOW_IMMEDIATE_SPACE_OCCUPATION: bool = True
# Inheritence: (0, 1]. If 0.0, start with default energy, if 0.5, start with half of parent, etc.
REPRODUCTION_INHERITENCE: float = 0.0
# how many children max per step
MAX_CHILDREN_PER_STEP: int = 1

# Payoff for both cooperating
PAYOFF_CC: float = 3.0
# Payoff for the defector
PAYOFF_DC: float = 5.0
# Payoff for cooperating against a defector
PAYOFF_CD: float = -1.0
# Payoff for defecting against a defector
PAYOFF_DD: float = 0.0

# How agents move
AGENT_TRAVEL_STRATEGIES: List[str] = ["random"]
AGENT_TRAVEL_STRATEGY: int = AGENT_TRAVEL_STRATEGIES.index("random")

# Cost of movement / migration
AGENT_TRAVEL_COST: float = 0.5 * COST_OF_LIVING

# Upper energy limit (do we need this?)
MAX_ENERGY: float = 150.0
# How much energy an agent can start with (max)
INIT_ENERGY_MU: float = 50.0
INIT_ENERGY_SIGMA: float = 10.0
# of cours this can be a specific value
# but this allows for 5 moves before death.
INIT_ENERGY_MIN: float = 5.0
# Noise will invert the agent's decision
ENV_NOISE: float = 0.0

# Agent strategies for the PD game
# "proportion" let's you say how likely agents spawn with a particular strategy
AGENT_STRATEGY_COOP: int = 0
AGENT_STRATEGY_DEFECT: int = 1
AGENT_STRATEGY_TIT_FOR_TAT: int = 2
AGENT_STRATEGY_RANDOM: int = 3

# @TODO: fix if number of strategies is not 4 (logging var...)
AGENT_STRATEGIES: dict = {
    "always_coop": {
        "name": "always_coop",
        "id": AGENT_STRATEGY_COOP,
        "proportion": 1 / 4,
    },
    "always_defect": {
        "name": "always_defect",
        "id": AGENT_STRATEGY_DEFECT,
        "proportion": 1 / 4,
    },
    # defaults to coop if no previous play recorded
    "tit_for_tat": {
        "name": "tit_for_tat",
        "id": AGENT_STRATEGY_TIT_FOR_TAT,
        "proportion": 1 / 4,
    },
    "random": {
        "name": "random",
        "id": AGENT_STRATEGY_RANDOM,
        "proportion": 1 / 4,
    },
}

# How many variants of agents are there?, more wil result in more agent colors
AGENT_TRAIT_COUNT: int = 4
# @TODO: allow for 1 trait (implies no strategy per trait)
# AGENT_TRAIT_COUNT: int = 1

# if this is true, agents will just have ONE strategy for all
# regardless of AGENT_STRATEGY_PER_TRAIT setting.
AGENT_STRATEGY_PURE: bool = False
# Should an agent deal differently per variant? (max strategies = number of variants)
# or, should they have a strategy for same vs different (max strategies = 2)
AGENT_STRATEGY_PER_TRAIT: bool = False

# Mutation frequency
AGENT_TRAIT_MUTATION_RATE: float = 0.0


MULTI_RUN = False
MULTI_RUN_STEPS = 10000
MULTI_RUN_COUNT = 1
```

## Screenshot

![Screenshots from ABM simulation](https://user-images.githubusercontent.com/75656/184108676-8f6821eb-f792-484c-b4a8-ba02de789a1f.png)

## References

ABM concepts for tags/traits adapted from:
- Hammond, R. A., & Axelrod, R. (2006). Evolution of contingent altruism when cooperation is expensive. Theoretical Population Biology, 69(3), 333–338. https://doi.org/10.1016/j.tpb.2005.12.002

Environmental pressure/cost of living concepts adapted from:
- Smaldino, P., Schank, J., & Mcelreath, R. (2013). Increased Costs of Cooperation Help Cooperators in the Long Run. The American Naturalist. https://doi.org/10.1086/669615
