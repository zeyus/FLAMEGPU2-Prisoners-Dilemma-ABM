###
# pyflamegpu Prisoner's Dilemma Agent Based Model
###

# Import pyflamegpu
from typing import List
import pyflamegpu
# Import standard python libs that are used
import sys, random, math


##########################################
# SIMULATION CONFIGURATION               #
##########################################

# Define some constants
RANDOM_SEED: int = 69420

# upper agent limit ... please make it a square number for sanity
MAX_AGENT_COUNT: int = 16384 # if you change this please change the value in interact.cu
# starting agent limit
INIT_AGENT_COUNT: int = MAX_AGENT_COUNT // 4
# how long to run the sim for
STEP_COUNT: int = 10000
# TODO: logging
VERBOSE_OUTPUT: bool = False

# Show agent visualisation
USE_VISUALISATION: bool = True

MAX_PLAY_DISTANCE: int = 1 # radius of message search grid

# Energy cost per step
COST_OF_LIVING: float = 0.5

# Reproduce if energy is above this threshold
REPRODUCE_MIN_ENERGY: float = 100.0
# Cost of reproduction
REPRODUCE_COST: float = 50.0

# Payoff for both cooperating
PAYOFF_CC: float = 3.0
# Payoff for the defector
PAYOFF_DC: float = 5.0
# Payoff for cooperating against a defector
PAYOFF_CD: float = -1.0
# Upper energy limit (do we need this?)
MAX_ENERGY: float = 150.0
# How much energy an agent can start with (max)
MAX_INIT_ENERGY: float = 50.0
# Noise will invert the agent's decision
ENV_NOISE: float = 0.0

# How agents move
AGENT_TRAVEL_STRATEGIES: List[str] = ["random"]
AGENT_TRAVEL_STRATEGY: int = AGENT_TRAVEL_STRATEGIES.index("random")

# Cost of movement / migration
AGENT_TRAVEL_COST: float = 0.0

# Agent strategies for the PD game
# "proportion" let's you say how likely agents spawn with a particular strategy
AGENT_STRATEGIES: dict = {
  "always_coop": {
    "name": "always_coop",
    "id": 0,
    "proportion": 0.50,
  },
  "always_cheat": {
    "name": "always_cheat",
    "id": 1,
    "proportion": 0.25,
  },
  "tit_for_tat": {
    "name": "tit_for_tat",
    "id": 2,
    "proportion": 0.15,
  },
  "random": {
    "name": "random",
    "id": 3,
    "proportion": 0.10,
  },
}

# How many variants of agents are there?
AGENT_TRAITS: List[int] = list(range(4))

# Should an agent deal differently per variant? (max strategies = number of variants)
# or, should they have a strategy for same vs different (max strategies = 2)
AGENT_STRATEGY_PER_TRAIT: bool = False

# Mutation frequency
AGENT_TRAIT_MUTATION_RATE: float = 0.05


##########################################
# Main script                            #
##########################################
# You should not need to change anything #
# below this line                        #
##########################################

# grid dimensions x = y
ENV_MAX: int = math.ceil(math.sqrt(MAX_AGENT_COUNT))

# Generate weights based on strategy configuration
AGENT_WEIGHTS: List[float] = [AGENT_STRATEGIES[strategy]["proportion"] for strategy in AGENT_STRATEGIES]
# generate strategy IDs based on strategy configuration
AGENT_STRATEGY_IDS: List[int] = [AGENT_STRATEGIES[strategy]["id"] for strategy in AGENT_STRATEGIES]

# definie color pallete for each agent strategy, with fallback to white
AGENT_COLOR_SCHEME: pyflamegpu.uDiscreteColor = pyflamegpu.uDiscreteColor("agent_trait", pyflamegpu.SET1, pyflamegpu.WHITE)
AGENT_DEFAULT_SHAPE: str = './src/resources/models/primitive_pyramid.obj'
AGENT_DEFAULT_SCALE: float = 1 / 2.0
# Roll if we need to rotate the agents 270 degrees
ROLL_RADS_270: float = 3 * math.pi / 2

# where cuda scripts are stored
CUDA_SRC_PATH: str = "src/pd/cudasrc"

# agent functions
CUDA_SEARCH_FUNC_NAME: str = "search"
CUDA_SEARCH_FUNC: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_SEARCH_FUNC_NAME}, flamegpu::MessageNone, flamegpu::MessageArray2D) {{
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    const unsigned int my_grid_index = FLAMEGPU->getVariable<unsigned int>("grid_index");
    FLAMEGPU->message_out.setVariable<unsigned int>("grid_index", my_grid_index);
    FLAMEGPU->message_out.setVariable<float>("energy", FLAMEGPU->getVariable<float>("energy"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int>("x"), FLAMEGPU->getVariable<unsigned int>("y"));
    auto playspace = FLAMEGPU->environment.getProperty<unsigned int, {MAX_AGENT_COUNT}, 16384>("playspace");
    // set playspace at my position to my current energy
    playspace[my_grid_index][my_grid_index] = FLAMEGPU->getVariable<float>("energy");
    return flamegpu::ALIVE;
}}
"""
CUDA_INTERACT_FUNC_NAME: str = "interact"
CUDA_INTERACT_FUNC: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_INTERACT_FUNC_NAME}, flamegpu::MessageArray2D, flamegpu::MessageNone) {{
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int my_grid_index = FLAMEGPU->getVariable<unsigned int>("grid_index");

    // const unsigned int max_agents = FLAMEGPU->environment.getProperty<unsigned int>("max_agents");
    // const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    
    const unsigned int interaction_radius = FLAMEGPU->environment.getProperty<unsigned int>("max_play_distance");
    
    const float reproduction_threshold = FLAMEGPU->environment.getProperty<float>("reproduce_min_energy");
    const float reproduction_cost = FLAMEGPU->environment.getProperty<float>("reproduce_cost");

    float my_energy = FLAMEGPU->getVariable<float>("energy");
    // replace dimensions with python string formatting so agent count can vary
    auto playspace = FLAMEGPU->environment.getProperty<unsigned int, {MAX_AGENT_COUNT}>("playspace");
    // iterate over all cells in the neighbourhood
    // this also wraps across env boundaries.
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y, interaction_radius)) {{
        flamegpu::id_t local_competitor = message.getVariable<flamegpu::id_t>("id");
        unsigned int opponent_grid_index = message.getVariable<unsigned int>("grid_index");
        // play with the competitor if competitor grid index is lower
        if (opponent_grid_index < my_grid_index) {{
            float opponent_energy = message.getVariable<float>("energy");
            if (my_energy <= 0.0) {{
                return flamegpu::DEAD;
            }}
        }}
    }}
    
    if (my_energy > reproduction_threshold) {{
        // spawn child in a free adjacent cell
        FLAMEGPU->setVariable<float>("energy", my_energy - reproduction_cost);
        return flamegpu::ALIVE;
    }}
    return flamegpu::ALIVE;
}}
"""


# Define a method which when called will define the model, Create the simulation object and execute it.
def main():
  print(ENV_MAX)
  # Define the FLAME GPU model
  model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("prisoners_dilemma")
  # Define the location message list
  message: pyflamegpu.MessageArray2D_Description = model.newMessageArray2D("player_search_msg")
  message.newVariableID("id")
  message.newVariableUInt("grid_index")
  message.newVariableFloat("energy")
  # create array to fit all agents
  message.setDimensions(ENV_MAX, ENV_MAX)

  agent: pyflamegpu.AgentDescription = model.newAgent("prisoner")
  agent.newVariableID("id")
  # this is to hold a strategy per opponent trait
  agent.newVariableArrayUInt("agent_strategies", len(AGENT_TRAITS))
  agent.newVariableUInt("agent_trait")
  agent.newVariableUInt("x_a")
  agent.newVariableUInt("y_a")
  agent.newVariableUInt("grid_index")
  agent.newVariableFloat("energy")
  if USE_VISUALISATION:
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
  # load agent-specific interactions
  agent_search_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(CUDA_SEARCH_FUNC_NAME, CUDA_SEARCH_FUNC)
  agent_search_fn.setMessageOutput("player_search_msg")
  agent_move_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(CUDA_INTERACT_FUNC_NAME, CUDA_INTERACT_FUNC)
  agent_move_fn.setMessageInput("player_search_msg")
  agent_move_fn.setAllowAgentDeath(True)
  
  # Environment properties
  env: pyflamegpu.EnvironmentDescription = model.Environment()
  env.newPropertyUInt("env_max", ENV_MAX, isConst=True)
  env.newPropertyUInt("max_agents", MAX_AGENT_COUNT, isConst=True)
  env.newPropertyFloat("max_energy", MAX_ENERGY, isConst=True)
  env.newPropertyUInt("max_play_distance", MAX_PLAY_DISTANCE, isConst=True)
  env.newPropertyFloat("cost_of_living", COST_OF_LIVING, isConst=True)
  env.newPropertyFloat("payoff_cd", PAYOFF_CD, isConst=True)
  env.newPropertyFloat("payoff_cc", PAYOFF_CC, isConst=True)
  env.newPropertyFloat("payoff_dc", PAYOFF_DC, isConst=True)
  env.newPropertyFloat("reproduce_min_energy", REPRODUCE_MIN_ENERGY, isConst=True)
  env.newPropertyFloat("reproduce_cost", REPRODUCE_COST, isConst=True)
  env.newPropertyFloat("travel_strategy", AGENT_TRAVEL_STRATEGY, isConst=True)
  env.newPropertyFloat("travel_cost", AGENT_TRAVEL_COST, isConst=True)
  env.newPropertyFloat("trait_mutation_rate", AGENT_TRAIT_MUTATION_RATE, isConst=True)
  # An array to hold the energy of each agent
  env.newPropertyArrayFloat("playspace", [0] * MAX_AGENT_COUNT)

  # define playspace
  # env.newMacroPropertyUInt("playspace", MAX_AGENT_COUNT, MAX_AGENT_COUNT)

  # Layer #1
  layer1: pyflamegpu.LayerDescription = model.newLayer()
  layer1.addAgentFunction("prisoner", CUDA_SEARCH_FUNC_NAME)
  # Layer #2
  layer2: pyflamegpu.LayerDescription = model.newLayer()
  layer2.addAgentFunction("prisoner", CUDA_INTERACT_FUNC_NAME)


  simulation: pyflamegpu.CUDASimulation = pyflamegpu.CUDASimulation(model)

  if pyflamegpu.VISUALISATION:
    visualisation: pyflamegpu.ModelVis  = simulation.getVisualisation()
    # Configure the visualiastion.
    INIT_CAM = ENV_MAX / 2.0
    visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, ENV_MAX)
    visualisation.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0.0)
    visualisation.setCameraSpeed(0.1)
    # do not limit speed
    visualisation.setSimulationSpeed(0)
    
    vis_agent: pyflamegpu.AgentVis = visualisation.addAgent("prisoner")

    # Set the model to use, and scale it.
    vis_agent.setModel(AGENT_DEFAULT_SHAPE)
    vis_agent.setModelScale(AGENT_DEFAULT_SCALE)
    vis_agent.setColor(AGENT_COLOR_SCHEME)
    
    # Activate the visualisation.
    visualisation.activate()

  # set some simulation defaults
  if RANDOM_SEED is not None:
    simulation.SimulationConfig().random_seed = RANDOM_SEED
  simulation.SimulationConfig().steps = STEP_COUNT
  simulation.SimulationConfig().verbose = VERBOSE_OUTPUT

   # Initialise the simulation
  simulation.initialise(sys.argv)

  # Generate a population if an initial states file is not provided
  if not simulation.SimulationConfig().input_file:
    # Seed the host RNG using the cuda simulations' RNG
    if RANDOM_SEED is not None:
      random.seed(simulation.SimulationConfig().random_seed)
    # Generate a vector of agents
    population = pyflamegpu.AgentVector(agent, INIT_AGENT_COUNT)
    # Iterate the population, initialising per-agent values
    instance: pyflamegpu.AgentVector_Agent
    # randomly create starting position for agents
    import numpy as np
    if RANDOM_SEED is not None:
      np.random.RandomState(RANDOM_SEED)
    # initialise grid with id for all possible agents
    grid = np.arange(MAX_AGENT_COUNT, dtype=np.uint32)
    # shuffle grid
    np.random.shuffle(grid)
    # reshape it to match the environment size
    grid = np.reshape(grid, (ENV_MAX, ENV_MAX))
    # initialise agents
    for i, instance in enumerate(population):
      # find agent position in grid
      pos = np.where(grid == i)
      x = pos[0][0].item()
      y = pos[1][0].item()
      instance.setVariableUInt("x_a", int(x))
      instance.setVariableUInt("y_a", int(y))
      instance.setVariableUInt("grid_index", int(x * ENV_MAX + y))
      if USE_VISUALISATION:
        instance.setVariableFloat("x", float(x))
        instance.setVariableFloat("y", float(y))
      instance.setVariableFloat("energy", random.uniform(1, MAX_INIT_ENERGY))
      # select agent strategy
      agent_trait: int = random.choice(AGENT_TRAITS)
      instance.setVariableUInt("agent_trait", agent_trait)
      # select agent strategy
      if AGENT_STRATEGY_PER_TRAIT:
        # if we are using a per-trait strategy, then pick random weighted strategies
        instance.setVariableArrayUInt('agent_strategies', random.choices(AGENT_STRATEGY_IDS, weights=AGENT_WEIGHTS, k=len(AGENT_TRAITS)))
      else:
        # otherwise, we need a strategy for agents with matching traits
        # and a second for agents with different traits
        strategy_my: int
        strategy_other: int
        strategy_my, strategy_other = random.choices(AGENT_STRATEGY_IDS, weights=AGENT_WEIGHTS, k=2)
        agent_strategies: List[int] = []
        trait: int
        for i, trait in enumerate(AGENT_TRAITS):
          if trait == agent_trait:
            agent_strategies.append(strategy_my)
          else:
            agent_strategies.append(strategy_other)
        instance.setVariableArrayUInt('agent_strategies', agent_strategies)
    del x, y, grid, np
    # Set the population for the simulation object
    simulation.setPopulationData(population)

  simulation.simulate()
  # Potentially export the population to disk
  # simulation.exportData("end.xml")
  # If visualisation is enabled, end the visualisation
  if pyflamegpu.VISUALISATION:
      visualisation.join()
  
if __name__ == "__main__":
    main()