###
# pyflamegpu Prisoner's Dilemma Agent Based Model
###

# Import pyflamegpu
import pyflamegpu
# Import standard python libs that are used
import sys, random, math

# Define some constants
RANDOM_SEED: int = 69420
MAX_AGENT_COUNT: int = 16384
INIT_AGENT_COUNT: int = 16384
STEP_COUNT: int = 10000
VERBOSE_OUTPUT: bool = False
ENV_MAX: int = math.floor(math.sqrt(MAX_AGENT_COUNT))
RADIUS: float = 1.0
USE_VISUALISATION: bool = True
VISUALISE_COMMUNICATION_GRID = False

# Define the environment
INIT_COOP_FREQ: float = 0.5
COST_OF_LIVING: float = 0.5
PAYOFF_CD: float = -1.0

REPRODUCE_MIN_ENERGY: float = 100.0
REPRODUCE_COST: float = 50.0
PAYOFF_CC: float = 3.0
PAYOFF_DC: float = 5.0
MAX_ENERGY: float = 150.0

CUDA_SRC_PATH: str = "src/pd/model/"
CUDA_SEARCH_FUNC: str = "search"
CUDA_INTERACT_FUNC: str = "interact"

ROLL_RADS_270: float = 3 * math.pi / 2
AGENT_DEFAULT_SHAPE: str = './src/resources/models/primitive_pyramid.obj'
AGENT_DEFAULT_SCALE: float = 1 / 10.0
AGENT_STRATEGIES: list = [
  "always_coop",
  "always_cheat",
]
AGENT_PROPORTIONS: dict = {
  "always_coop": 0.5,
  "always_cheat": 0.5,
}

# definie color pallete for each agent strategy, with fallback to white
AGENT_COLOR_SCHEME: pyflamegpu.iDiscreteColor = pyflamegpu.iDiscreteColor("agent_strategy", pyflamegpu.WHITE)
AGENT_COLOR_SCHEME[0] = pyflamegpu.BLUE
AGENT_COLOR_SCHEME[1] = pyflamegpu.RED


# Define a method which when called will define the model, Create the simulation object and execute it.
def main():
  # Define the FLAME GPU model
  model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("prisoners_dilemma")
  # Define the location message list
  message: pyflamegpu.MessageArray2D_Description = model.newMessageArray2D("player_search")
  message.newVariableID("id")
  message.newVariableInt("lets_play")

  agent: pyflamegpu.AgentDescription = model.newAgent("prisoner")
  agent.newVariableInt("id")
  agent.newVariableInt("agent_strategy")
  agent.newVariableFloat("x")
  agent.newVariableFloat("y")
  agent.newVariableFloat("roll")
  # load agent-specific interactions
  agent_search_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunctionFile(CUDA_SEARCH_FUNC, '/'.join([CUDA_SRC_PATH, CUDA_SEARCH_FUNC]))
  agent_search_fn.setMessageOutput("player_search")
  agent_move_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunctionFile(CUDA_INTERACT_FUNC, '/'.join([CUDA_SRC_PATH, CUDA_INTERACT_FUNC]))
  agent_move_fn.setMessageInput("player_search")
  agent_move_fn.setAllowAgentDeath(True)
  
  
  # Environment properties
  env: pyflamegpu.EnvironmentDescription = model.Environment()
  env.newPropertyInt("max_agents", MAX_AGENT_COUNT)
  env.newPropertyFloat("max_energy", MAX_ENERGY)
  env.newPropertyFloat("init_coop_freq", INIT_COOP_FREQ)
  env.newPropertyFloat("cost_of_living", COST_OF_LIVING)
  env.newPropertyFloat("payoff_cd", PAYOFF_CD)
  env.newPropertyFloat("payoff_cc", PAYOFF_CC)
  env.newPropertyFloat("payoff_dc", PAYOFF_DC)
  env.newPropertyFloat("reproduce_min_energy", REPRODUCE_MIN_ENERGY)
  env.newPropertyFloat("reproduce_cost", REPRODUCE_COST)
  env.newPropertyFloat("max_energy", MAX_ENERGY)



  # Layer #1
  layer1 = model.newLayer()
  # Layer #2
  layer2 = model.newLayer()

  # add agent-specific layer functions

  # the interact funcion always has the same name, even if it varies by type (for now)
  layer1.addAgentFunction("prisoner", CUDA_INTERACT_FUNC)
  layer2.addAgentFunction("prisoner", CUDA_MOVE_FUNC)

  simulation: pyflamegpu.CUDASimulation = pyflamegpu.CUDASimulation(model)

  if pyflamegpu.VISUALISATION:
    visualisation: pyflamegpu.ModelVis  = simulation.getVisualisation()
    # Configure the visualiastion.
    INIT_CAM = ENV_MAX / 2.0
    visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, 450.0)
    visualisation.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0.0)
    visualisation.setCameraSpeed(0.1)
    # do not limit speed
    visualisation.setSimulationSpeed(0)
    
    vis_agent: pyflamegpu.AgentVis = visualisation.addAgent("prisoner")
    # Position vars are named x, y, z so they are used by default
    # Set the model to use, and scale it.
    vis_agent.setModel(AGENT_DEFAULT_SHAPE)
    vis_agent.setModelScale(AGENT_DEFAULT_SCALE)
    vis_agent.setColor(AGENT_COLOR_SCHEME)
    # vis_agent.setRollVariable("roll")
    
    # Activate the visualisation.
    visualisation.activate()

  # set some simulation defaults
  simulation.SimulationConfig().random_seed = RANDOM_SEED
  simulation.SimulationConfig().steps = STEP_COUNT
  simulation.SimulationConfig().verbose = VERBOSE_OUTPUT

   # Initialise the simulation
  simulation.initialise(sys.argv)

  # Generate a population if an initial states file is not provided
  if not simulation.SimulationConfig().input_file:
    # Seed the host RNG using the cuda simulations' RNG
    random.seed(simulation.SimulationConfig().random_seed)
    # Generate a vector of agents
    population = pyflamegpu.AgentVector(agent, INIT_AGENT_COUNT)
    # Iterate the population, initialising per-agent values
    instance: pyflamegpu.AgentVector_Agent
    for i, instance in enumerate(population):
      instance.setVariableFloat("x", random.uniform(0, ENV_MAX))
      instance.setVariableFloat("y", random.uniform(0, ENV_MAX))
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