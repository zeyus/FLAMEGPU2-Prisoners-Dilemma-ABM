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
MAX_AGENT_COUNT: int = 2**16
# starting agent limit
INIT_AGENT_COUNT: int = MAX_AGENT_COUNT // 16
# how long to run the sim for
STEP_COUNT: int = 100
# TODO: logging / Debugging

VERBOSE_OUTPUT: bool = True

# rate limit simulation?
SIMULATION_SPS_LIMIT: int = 1 # 0 = unlimited

# Show agent visualisation
USE_VISUALISATION: bool = True

# visualisation camera speed
VISUALISATION_CAMERA_SPEED: float = 0.1
# pause the simulation at start
PAUSE_AT_START: bool = True

# radius of message search grid
MAX_PLAY_DISTANCE: int = 1
# get max number of surrounding agents within this radius
SPACES_WITHIN_RADIUS: int = ((1 + 2 * MAX_PLAY_DISTANCE)**2) - 1

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
AGENT_TRAVEL_COST: float = 1.0

# Agent status
AGENT_STATUS_READY: int = 0
AGENT_STATUS_READY_TO_PLAY: int = 1
AGENT_STATUS_PLAYING: int = 2
AGENT_STATUS_PLAY_COMPLETED: int = 3
AGENT_STATUS_MOVEMENT_UNRESOLVED: int = 4
AGENT_STATUS_MOVING: int = 5
AGENT_STATUS_MOVEMENT_COMPLETED: int = 6

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
AGENT_DEFAULT_SCALE: float = 0.9
# Roll if we need to rotate the agents 270 degrees
ROLL_RADS_270: float = 3 * math.pi / 2


# agent functions
CUDA_SEARCH_FUNC_NAME: str = "search"
CUDA_SEARCH_FUNC: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_SEARCH_FUNC_NAME}, flamegpu::MessageNone, flamegpu::MessageArray2D) {{
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    const unsigned int my_grid_index = FLAMEGPU->getVariable<unsigned int>("grid_index");
    FLAMEGPU->message_out.setVariable<unsigned int>("grid_index", my_grid_index);
    FLAMEGPU->message_out.setVariable<float>("energy", FLAMEGPU->getVariable<float>("energy"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int>("x_a"), FLAMEGPU->getVariable<unsigned int>("y_a"));
    // auto playspace = FLAMEGPU->environment.getProperty<float, {MAX_AGENT_COUNT}>("playspace");
    // set playspace at my position to my current energy
    // playspace[my_grid_index][my_grid_index] = FLAMEGPU->getVariable<float>("energy");
    return flamegpu::ALIVE;
}}
"""
CUDA_GAME_LIST_FUNC_NAME: str = "get_game_list"
CUDA_GAME_LIST_FUNC: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_GAME_LIST_FUNC_NAME}, flamegpu::MessageArray2D, flamegpu::MessageNone) {{
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int my_grid_index = FLAMEGPU->getVariable<unsigned int>("grid_index");

    // const unsigned int max_agents = FLAMEGPU->environment.getProperty<unsigned int>("max_agents");
    // const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    
    //const float reproduction_threshold = FLAMEGPU->environment.getProperty<float>("reproduce_min_energy");
    //const float reproduction_cost = FLAMEGPU->environment.getProperty<float>("reproduce_cost");

    float my_energy = FLAMEGPU->getVariable<float>("energy");
    // auto playspace = FLAMEGPU->environment.getProperty<float, {MAX_AGENT_COUNT}>("playspace");
    // iterate over all cells in the neighbourhood
    // this also wraps across env boundaries.
    unsigned int num_neighbours = 0;
    unsigned int neighbour_id = 0;
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y, {MAX_PLAY_DISTANCE})) {{
        flamegpu::id_t local_competitor = message.getVariable<flamegpu::id_t>("id");
        // always set, so it is also reset per round
        // my_game_list[neighbour_id] = local_competitor;
        // this is probably inefficient
        FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("game_list", neighbour_id, local_competitor);
        if (local_competitor == flamegpu::ID_NOT_SET) {{
            continue;
        }}
        unsigned int opponent_grid_index = message.getVariable<unsigned int>("grid_index");
        // play with the competitor if competitor grid index is lower
        if (opponent_grid_index < my_grid_index) {{
            
            float opponent_energy = message.getVariable<float>("energy");
            if (my_energy <= 0.0) {{
                return flamegpu::DEAD;
            }}
        }}
        ++num_neighbours;
        ++neighbour_id;
    }}
    if (num_neighbours == 0) {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVEMENT_UNRESOLVED});
    }} else {{
        
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_PLAY});
    }}
    
    // if (my_energy > reproduction_threshold) {{
    //    // spawn child in a free adjacent cell
    //    FLAMEGPU->setVariable<float>("energy", my_energy - reproduction_cost);
    //    return flamegpu::ALIVE;
    //}}
    return flamegpu::ALIVE;
}}
"""

CUDA_AGENT_PLAY_CONDITION_NAME: str = "move_condition"
CUDA_AGENT_PLAY_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_PLAY_CONDITION_NAME}) {{
    return FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_READY_TO_PLAY};
}}
"""

CUDA_AGENT_MOVE_CONDITION_NAME: str = "move_condition"
CUDA_AGENT_MOVE_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_MOVE_CONDITION_NAME}) {{
    return FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_MOVEMENT_UNRESOLVED};
}}
"""

CUDA_AGENT_MOVE_FUNCTION_NAME: str = "move"
CUDA_AGENT_MOVE_UPDATE_VIZ: str = "true" if USE_VISUALISATION else "false"
CUDA_AGENT_MOVE_FUNCTION: str = rf"""
// getting here means that there are no neighbours, so, free movement
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_MOVE_FUNCTION_NAME}, flamegpu::MessageNone, flamegpu::MessageNone) {{
    const float travel_cost = FLAMEGPU->environment.getProperty<float>("travel_cost");
    float my_energy = FLAMEGPU->getVariable<float>("energy");

    // try and deduct travel cost, die if below zero
    my_energy -= travel_cost;
    if (my_energy < 0.0) {{
        //JUST A TEST
        FLAMEGPU->setVariable<unsigned int>("agent_trait", 9);
        return flamegpu::DEAD;
    }}
    FLAMEGPU->setVariable<float>("energy", my_energy);

    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int my_grid_index = FLAMEGPU->getVariable<unsigned int>("grid_index");
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");

    // uniform int represents the direction to move,
    // i.e. 1 = northwest, 2 = north, 3 = northeast, 4 = west,
    // (5 = no movement), 6 = east, 7 = southwest, 8 = south, 9 = southeast
    unsigned int uniform_int = FLAMEGPU->random.uniform<int>(1, 8);
    if (uniform_int == 5) {{
      ++uniform_int;
    }}

    // Convert to x,y offsets
    const int new_x_offset = my_x + uniform_int % 3 - 1;
    const int new_y_offset = my_y + uniform_int / 3 - 1;

    // set location to new x,y and wrap around env boundaries
    const unsigned int new_x = (new_x_offset + env_max) % env_max;
    const unsigned int new_y = (new_y_offset + env_max) % env_max;
    
    FLAMEGPU->setVariable<unsigned int>("x_a", new_x);
    FLAMEGPU->setVariable<unsigned int>("y_a", new_y);

    // also update visualisation float values if required
    if({CUDA_AGENT_MOVE_UPDATE_VIZ}) {{
      FLAMEGPU->setVariable<float>("x", (float) new_x);
      FLAMEGPU->setVariable<float>("y", (float) new_y);
    }}

    // @TODO: remove? set new grid index
    FLAMEGPU->setVariable<unsigned int>("grid_index", new_x + new_y * env_max);
    return flamegpu::ALIVE;
}}
"""
def _print_prisoner_states(prisoner: pyflamegpu.HostAgentAPI) -> None:
  n_ready: int = prisoner.countUInt("agent_status", AGENT_STATUS_READY)
  n_playing: int = prisoner.countUInt("agent_status", AGENT_STATUS_PLAYING)
  n_ready_to_play: int = prisoner.countUInt("agent_status", AGENT_STATUS_READY_TO_PLAY)
  n_play_completed: int = prisoner.countUInt("agent_status", AGENT_STATUS_PLAY_COMPLETED)
  n_moving: int = prisoner.countUInt("agent_status", AGENT_STATUS_MOVING)
  n_move_unresolved: int = prisoner.countUInt("agent_status", AGENT_STATUS_MOVEMENT_UNRESOLVED)
  n_move_completed: int = prisoner.countUInt("agent_status", AGENT_STATUS_MOVEMENT_COMPLETED)
  print(f"n_ready: {n_ready}, n_playing: {n_playing}, n_ready_to_play: {n_ready_to_play}, n_play_completed: {n_play_completed}, n_moving: {n_moving}, n_move_unresolved: {n_move_unresolved}, n_move_completed: {n_move_completed}")

if VERBOSE_OUTPUT:
  class step_fn(pyflamegpu.HostFunctionCallback):
    def __init__(self):
      super().__init__()

    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
      prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
      _print_prisoner_states(prisoner)

class exit_always_fn(pyflamegpu.HostFunctionConditionCallback):
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    return pyflamegpu.EXIT
class exit_search_fn(pyflamegpu.HostFunctionConditionCallback):
  iterations: int = 0
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    self.iterations += 1
    if self.iterations < 9:
      # Agent movements still unresolved
      prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
      _print_prisoner_states(prisoner)
      if prisoner.countUInt("agent_status", AGENT_STATUS_READY):
        return pyflamegpu.CONTINUE
    
    self.iterations = 0
    return pyflamegpu.EXIT

class exit_move_fn(pyflamegpu.HostFunctionConditionCallback):
  iterations: int = 0
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    self.iterations += 1
    if self.iterations < 9:
      # Agent movements still unresolved
      prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
      if prisoner.countUInt("agent_status", AGENT_STATUS_MOVEMENT_UNRESOLVED):
        return pyflamegpu.CONTINUE
    
    self.iterations = 0
    return pyflamegpu.EXIT

def make_core_agent(model: pyflamegpu.ModelDescription) -> pyflamegpu.AgentDescription:
  agent: pyflamegpu.AgentDescription = model.newAgent("prisoner")
  agent.newVariableID("id")
  # this is to hold a strategy per opponent trait
  agent.newVariableArrayUInt("agent_strategies", len(AGENT_TRAITS))
  agent.newVariableUInt("agent_trait")
  agent.newVariableUInt("x_a")
  agent.newVariableUInt("y_a")
  agent.newVariableUInt("grid_index")
  agent.newVariableFloat("energy")
  agent.newVariableUInt("agent_status", AGENT_STATUS_READY)
  if USE_VISUALISATION:
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
  
  return agent

def _print_environment_properties() -> None:
  print(f"env_max (grid width): {ENV_MAX}")
  print(f"max agent count: {MAX_AGENT_COUNT}")

# Define a method which when called will define the model, Create the simulation object and execute it.
def main():
  if VERBOSE_OUTPUT:
    _print_environment_properties()
  if pyflamegpu.SEATBELTS:
    print("Seatbelts are enabled, this will significantly impact performance.")
    print("Ignore this if you are developing the model. Otherwise consider using a build without seatbelts.")
  # Define the FLAME GPU model
  model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("prisoners_dilemma")
  # Environment properties
  # These should be limited as much as possible
  # to only the variables which are to be simulated
  # across a number of values, otherwise they can be
  # defined as constants in the model / baked into the
  # CUDA code. This will limit the operations on GPU memory.
  # env: pyflamegpu.EnvironmentDescription = model.Environment()
  # env.newPropertyUInt("env_max", ENV_MAX, isConst=True)
  # env.newPropertyUInt("max_agents", MAX_AGENT_COUNT, isConst=True)
  # env.newPropertyFloat("max_energy", MAX_ENERGY, isConst=True)
  # env.newPropertyFloat("cost_of_living", COST_OF_LIVING, isConst=True)
  # env.newPropertyFloat("payoff_cd", PAYOFF_CD, isConst=True)
  # env.newPropertyFloat("payoff_cc", PAYOFF_CC, isConst=True)
  # env.newPropertyFloat("payoff_dc", PAYOFF_DC, isConst=True)
  # env.newPropertyFloat("reproduce_min_energy", REPRODUCE_MIN_ENERGY, isConst=True)
  # env.newPropertyFloat("reproduce_cost", REPRODUCE_COST, isConst=True)
  # env.newPropertyFloat("travel_strategy", AGENT_TRAVEL_STRATEGY, isConst=True)
  # env.newPropertyFloat("travel_cost", AGENT_TRAVEL_COST, isConst=True)
  # env.newPropertyFloat("trait_mutation_rate", AGENT_TRAIT_MUTATION_RATE, isConst=True)
  if VERBOSE_OUTPUT:
    model.addStepFunctionCallback(step_fn().__disown__())

  agent = make_core_agent(model)
  
  # An array to hold the energy of each agent
  # env.newPropertyArrayFloat("playspace", [0] * MAX_AGENT_COUNT)

  # define playspace
  # env.newMacroPropertyUInt("playspace", MAX_AGENT_COUNT, MAX_AGENT_COUNT)

  # load agent-specific interactions
  
  # play resolution submodel
  pdgame_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("pdgame_model")
  pdgame_model.addExitConditionCallback(exit_always_fn().__disown__())
  # Define the location message list
  message: pyflamegpu.MessageArray2D_Description = pdgame_model.newMessageArray2D("player_search_msg")
  message.newVariableID("id")
  message.newVariableUInt("grid_index")
  message.newVariableFloat("energy")
  pdgame_submodel: pyflamegpu.SubModelDescription = model.newSubModel("pdgame_model", pdgame_model)
  pdgame_subagent: pyflamegpu.AgentDescription = make_core_agent(pdgame_model)
  # the surrounding list of playable agents
  pdgame_subagent.newVariableArrayID("game_list", SPACES_WITHIN_RADIUS, [pyflamegpu.ID_NOT_SET] * SPACES_WITHIN_RADIUS)
  # create array to fit all agents
  message.setDimensions(ENV_MAX, ENV_MAX)
  agent_search_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(CUDA_SEARCH_FUNC_NAME, CUDA_SEARCH_FUNC)
  agent_search_fn.setMessageOutput("player_search_msg")
  agent_game_list_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(CUDA_GAME_LIST_FUNC_NAME, CUDA_GAME_LIST_FUNC)
  agent_game_list_fn.setMessageInput("player_search_msg")
  # the following condition is for playing, not for searching.
  # agent_game_list_fn.setRTCFunctionCondition(CUDA_AGENT_PLAY_CONDITION)
  # agent_game_list_fn.setAllowAgentDeath(True)
  pdgame_submodel.bindAgent("prisoner", "prisoner", auto_map_vars=True)
  submodel_pdgame_layer1: pyflamegpu.LayerDescription = pdgame_model.newLayer()
  submodel_pdgame_layer1.addAgentFunction(agent_search_fn)
  submodel_pdgame_layer1: pyflamegpu.LayerDescription = pdgame_model.newLayer()
  submodel_pdgame_layer1.addAgentFunction(agent_game_list_fn)
  
  # movement resolution submodel
  movement_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("movement_model")
  movement_model.addExitConditionCallback(exit_move_fn().__disown__())
  movement_env: pyflamegpu.EnvironmentDescription = movement_model.Environment()
  movement_env.newPropertyFloat("travel_cost", AGENT_TRAVEL_COST, isConst=True)
  movement_env.newPropertyUInt("env_max", ENV_MAX, isConst=True)
  movement_submodel: pyflamegpu.SubModelDescription = model.newSubModel("movement_model", movement_model)
  movement_subagent: pyflamegpu.AgentDescription = make_core_agent(movement_model)
  agent_move_fn: pyflamegpu.AgentFunctionDescription = movement_subagent.newRTCFunction(CUDA_AGENT_MOVE_FUNCTION_NAME, CUDA_AGENT_MOVE_FUNCTION)
  agent_move_fn.setRTCFunctionCondition(CUDA_AGENT_MOVE_CONDITION)
  agent_move_fn.setAllowAgentDeath(True)
  movement_submodel.bindAgent("prisoner", "prisoner", auto_map_vars=True)
  submodel_movement_layer1: pyflamegpu.LayerDescription = movement_model.newLayer()
  submodel_movement_layer1.addAgentFunction(agent_move_fn)


  # Layer #2: play submodel
  layer1: pyflamegpu.LayerDescription = model.newLayer()
  layer1.addSubModel("pdgame_model")
  #layer2.addAgentFunction("prisoner", CUDA_INTERACT_FUNC_NAME)
  # Layer #3: movement submodel
  layer2: pyflamegpu.LayerDescription = model.newLayer()
  layer2.addSubModel("movement_model")
  #layer3.addAgentFunction("prisoner", CUDA_AGENT_MOVE_FUNCTION_NAME)

  
  simulation: pyflamegpu.CUDASimulation = pyflamegpu.CUDASimulation(model)

  if pyflamegpu.VISUALISATION:
    visualisation: pyflamegpu.ModelVis  = simulation.getVisualisation()
    visualisation.setBeginPaused(PAUSE_AT_START)
    # Configure the visualiastion.
    INIT_CAM = ENV_MAX / 2.0
    visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, ENV_MAX)
    visualisation.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0.0)
    visualisation.setCameraSpeed(VISUALISATION_CAMERA_SPEED)
    # do not limit speed
    visualisation.setSimulationSpeed(SIMULATION_SPS_LIMIT)
    
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
    # Iterate the population, initialising per-agent values
    instance: pyflamegpu.AgentVector_Agent
    for i, instance in enumerate(population):
      # find agent position in grid
      pos = np.where(grid == i)
      x = pos[0][0].item()
      y = pos[1][0].item()
      instance.setVariableUInt("x_a", int(x))
      instance.setVariableUInt("y_a", int(y))
      instance.setVariableUInt("grid_index", int(x + y * ENV_MAX))
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