import pyflamegpu
import math
import sys

MAX_AGENT_COUNT: int = 4096 #make sure this is a square number
# starting agent limit
INIT_AGENT_COUNT: int = MAX_AGENT_COUNT // 16

ENV_MAX: int = math.ceil(math.sqrt(MAX_AGENT_COUNT))

fn1 = """
FLAMEGPU_AGENT_FUNCTION(search, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int>("x_a"), FLAMEGPU->getVariable<unsigned int>("y_a"));

    return flamegpu::ALIVE;
}
"""

fn2 = """
FLAMEGPU_AGENT_FUNCTION(interact, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");

    unsigned int num_neighbours = 0;
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y, 1)) {
        ++num_neighbours;
    }
    if (num_neighbours == 0) {
        FLAMEGPU->setVariable<unsigned int>("agent_status", 0);
    } else {
        FLAMEGPU->setVariable<unsigned int>("agent_status", 1);
    }
    return flamegpu::ALIVE;
}
"""

class step_fn(pyflamegpu.HostFunctionCallback):
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
    n_ready: int = prisoner.countUInt("agent_status", 0)
    n_playing: int = prisoner.countUInt("agent_status", 1)
    print(f"step: {FLAMEGPU.getStepCounter()}, n_ready: {n_ready}, n_playing: {n_playing}")

def main():
  print(ENV_MAX)
  model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("prisoners_dilemma")
  model.addStepFunctionCallback(step_fn().__disown__())
  message: pyflamegpu.MessageArray2D_Description = model.newMessageArray2D("player_search_msg")
  message.newVariableID("id")
  
  message.setDimensions(ENV_MAX, ENV_MAX)

  agent: pyflamegpu.AgentDescription = model.newAgent("prisoner")
  agent.newVariableID("id")
  agent.newVariableUInt("x_a")
  agent.newVariableUInt("y_a")
  agent.newVariableUInt("agent_status", 0)
  agent.newVariableFloat("x")
  agent.newVariableFloat("y")
  
  env: pyflamegpu.EnvironmentDescription = model.Environment()
  env.newPropertyUInt("env_max", ENV_MAX, isConst=True)

  agent_search_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction("search", fn1)
  agent_search_fn.setMessageOutput("player_search_msg")

  agent_interact_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction("interact", fn2)
  agent_interact_fn.setMessageInput("player_search_msg")
  agent_interact_fn.setAllowAgentDeath(True)

    # Layer #1
  layer1: pyflamegpu.LayerDescription = model.newLayer()
  layer1.addAgentFunction("prisoner", "search")
  # Layer #2
  layer2: pyflamegpu.LayerDescription = model.newLayer()
  layer2.addAgentFunction("prisoner", "interact")

  simulation: pyflamegpu.CUDASimulation = pyflamegpu.CUDASimulation(model)

  if pyflamegpu.VISUALISATION:
    visualisation: pyflamegpu.ModelVis  = simulation.getVisualisation()

    INIT_CAM = ENV_MAX / 2.0
    visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, ENV_MAX)
    visualisation.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0.0)
    visualisation.setCameraSpeed(0.1)
    # do not limit speed
    visualisation.setSimulationSpeed(0)
    
    vis_agent: pyflamegpu.AgentVis = visualisation.addAgent("prisoner")

    # Set the model to use, and scale it.
    vis_agent.setModel(pyflamegpu.CUBE)
    vis_agent.setModelScale(0.8)
    
    # Activate the visualisation.
    visualisation.activate()

  simulation.SimulationConfig().steps = 10
  simulation.SimulationConfig().verbose = True
  simulation.initialise(sys.argv)

  # Generate a population if an initial states file is not provided
  if not simulation.SimulationConfig().input_file:
    population = pyflamegpu.AgentVector(agent, INIT_AGENT_COUNT)
    import numpy as np
    # initialise grid with id for all possible agents
    grid = np.arange(MAX_AGENT_COUNT, dtype=np.uint32)
    # shuffle grid
    np.random.shuffle(grid)
    # reshape it to match the environment size
    grid = np.reshape(grid, (ENV_MAX, ENV_MAX))
    instance: pyflamegpu.AgentVector_Agent
    for i, instance in enumerate(population):
      # find agent position in grid
      pos = np.where(grid == i)
      x = pos[0][0].item()
      y = pos[1][0].item()
      instance.setVariableUInt("x_a", int(x))
      instance.setVariableUInt("y_a", int(y))
      instance.setVariableFloat("x", float(x))
      instance.setVariableFloat("y", float(y))
    del x, y, grid, np

  # Set the population for the simulation object
  simulation.setPopulationData(population)

  simulation.simulate()
  if pyflamegpu.VISUALISATION:
      visualisation.join()

if __name__ == "__main__":
    main()