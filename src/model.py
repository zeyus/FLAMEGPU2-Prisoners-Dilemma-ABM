"""
Implementation of the Circles FLAME GPU model in python, as an example.
"""

import pyflamegpu

# Import standard python libs that are used
import sys, random, math

from group_dynamics import config
from group_dynamics.agent import step_validation, create_agent, create_environment, populate_simulation

# Define a method which when called will define the model, Create the simulation object and execute it.
def main():
    # Define the FLAME GPU model
    model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription(config.MODEL_NAME)

    # Define the location message list
    message = model.newMessageSpatial2D("location")
    # A message to hold the location of an agent.
    message.newVariableID("id")
    # X Y Z are implicit for spatial3D messages
    # Set Spatial2D message list parameters
    message.setRadius(config.MESSAGE_RADIUS)
    message.setMin(0, 0)
    message.setMax(config.ENV_MAX, config.ENV_MAX)

    # Define the Circle agent type including variables and messages
    agent = create_agent(model)

    # Define environment properties
    env = create_environment(model)

    # Layer #1
    layer1 = model.newLayer()
    layer1.addAgentFunction(config.AGENT_NAME, config.LOCAL_MESSAGE_FUNC)
    # Layer #2
    layer2 = model.newLayer()
    layer2.addAgentFunction(config.AGENT_NAME, config.MOVE_FUNC)

    # Add the callback step function to the model.
    step_validation_fn = step_validation()
    model.addStepFunctionCallback(step_validation_fn)

    # Create the simulation object.
    simulation = pyflamegpu.CUDASimulation(model)
    
    # If visualisation is enabled, use it.

    if pyflamegpu.VISUALISATION and config.USE_VIZ:
        visualisation: pyflamegpu.ModelVis = simulation.getVisualisation()
        # Configure the visualiastion.
        INIT_CAM = config.ENV_MAX / 2
        visualisation.setWindowDimensions(config.ENV_MAX, config.ENV_MAX)
        visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, config.ENV_MAX * 0.865)
        visualisation.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0)
        visualisation.setCameraSpeed(config.VIZ_MOVE_SPEED, config.VIZ_TURBO_MULT)
        visualisation.setBeginPaused(config.VIZ_START_PAUSED)

        vis_agent: pyflamegpu.AgentVis = visualisation.addAgent(config.AGENT_NAME)
        # Position vars are named x, y, z so they are used by default
        # Set the model to use, and scale it.
        vis_agent.setModel(config.AGENT_3D_MODEL)
        vis_agent.setModelScale(*config.AGENT_SCALE)
        vis_agent.setColor(config.AGENT_COLOR)
        # Optionally render the Subdivision of spatial messaging
        ENV_MIN = 0
        if config.VIZ_COMM_GRID:
            DIM = int(math.ceil((config.ENV_MAX - ENV_MIN) / config.MESSAGE_RADIUS))  # Spatial partitioning scales up to fit non exact environments
            DIM_MAX = DIM * config.MESSAGE_RADIUS
            pen: pyflamegpu.LineVis = visualisation.newLineSketch(1, 1, 0.2)  # yellow
            # X lines
            for y in range(0, DIM + 1):
                pen.addVertex(ENV_MIN, y * config.MESSAGE_RADIUS, 0)
                pen.addVertex(DIM_MAX, y * config.MESSAGE_RADIUS, 0)
            # Y axis
            for x in range(0, DIM + 1):
                pen.addVertex(x * config.MESSAGE_RADIUS, ENV_MIN, 0)
                pen.addVertex(x * config.MESSAGE_RADIUS, DIM_MAX, 0)
            
            # draw environment boundary
        if config.VIZ_SHOW_ENV_BOUNDARY:
            pen2: pyflamegpu.LineVis = visualisation.newPolylineSketch(0.0, 0.0, 1.0) # blue
            pen2.addVertex(ENV_MIN, ENV_MIN, 0.0)
            pen2.addVertex(config.ENV_MAX, ENV_MIN, 0.0)
            pen2.addVertex(config.ENV_MAX, config.ENV_MAX, 0.0)
            pen2.addVertex(ENV_MIN, config.ENV_MAX, 0.0)
            pen2.addVertex(ENV_MIN, ENV_MIN, 0.0)

        # Activate the visualisation.
        visualisation.activate()

    # Initialise the simulation with default values
    config.set_sim_defaults(simulation)
    # now update with any command line arguments
    simulation.initialise(sys.argv)
    
    populate_simulation(simulation, agent)

    # Execute the simulation
    simulation.simulate()

    # Potentially export the population to disk
    # simulation.exportData("end.xml")

    # If visualisation is enabled, end the visualisation
    if pyflamegpu.VISUALISATION and config.USE_VIZ:
        visualisation.join()


# If this python script is the entry point, execute the main method
if __name__ == "__main__":
    main()