# Define FLAME GPU Agent functions as strings
import pyflamegpu
import random
from .config import *
from logging import info

# Agent Function to output the agents ID and position in to a 2D spatial message list
output_message_local: str = rf'''
FLAMEGPU_AGENT_FUNCTION({LOCAL_MESSAGE_FUNC}, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {{
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y")
    );
    return flamegpu::ALIVE;
}}
'''

# Agent function to iterate messages, and move according to the rules of the circle model
move: str = rf'''
FLAMEGPU_AGENT_FUNCTION({MOVE_FUNC}, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {{
    const flamegpu::id_t ID = FLAMEGPU->getID();
    const float REPULSE_FACTOR = FLAMEGPU->environment.getProperty<float>("repulse");
    const float RADIUS = FLAMEGPU->message_in.radius();
    float fx = 0.0;
    float fy = 0.0;
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    int count = 0;
    for (const auto &message : FLAMEGPU->message_in(x1, y1)) {{
        if (message.getVariable<flamegpu::id_t>("id") != ID) {{
            const float x2 = message.getVariable<float>("x");
            const float y2 = message.getVariable<float>("y");
            float x21 = x2 - x1;
            float y21 = y2 - y1;
            const float separation = sqrt(x21*x21 + y21*y21);
            if (separation < RADIUS && separation > 0.0f) {{
                float k = sinf((separation / RADIUS)*3.141*-2)*REPULSE_FACTOR;
                // Normalise without recalculating separation
                x21 /= separation;
                y21 /= separation;
                fx += k * x21;
                fy += k * y21;
                count++;
            }}
        }}
    }}
    fx /= count > 0 ? count : 1;
    fy /= count > 0 ? count : 1;
    FLAMEGPU->setVariable<float>("x", x1 + fx);
    FLAMEGPU->setVariable<float>("y", y1 + fy);
    FLAMEGPU->setVariable<float>("drift", sqrt(fx*fx + fy*fy));
    return flamegpu::ALIVE;
}}
'''

# A Callback host function, to check the progress of the model / validate the model.

class step_validation(pyflamegpu.HostFunctionCallback):
    def __init__(self) -> None:
        super().__init__()
        # Static variables?
        self.prevTotalDrift = 3.402823e+38 # @todo - static
        self.driftDropped = 0 # @todo - static
        self.driftIncreased = 0 # @todo - static

    def run(self, FLAMEGPU) -> None:
        # This value should decline? as the model moves towards a steady equilibrium state
        # Once an equilibrium state is reached, it is likely to oscillate between 2-4? values
        totalDrift = FLAMEGPU.agent(AGENT_NAME).sumFloat('drift')
        if totalDrift <= self.prevTotalDrift:
            self.driftDropped += 1
        else:
            self.driftIncreased += 1
        self.prevTotalDrift = totalDrift
        # print('{:.2f} Drift correct'.format(100 * self.driftDropped / float(self.driftDropped + self.driftIncreased)))


def create_agent(model: pyflamegpu.ModelDescription) -> pyflamegpu.AgentDescription:
    agent = model.newAgent(AGENT_NAME)
    agent.newVariableInt('id')
    agent.newVariableFloat('x')
    agent.newVariableFloat('y')
    # add z to jitter agents
    agent.newVariableFloat('z')
    for i in range(AGENT_ATTRIBUTE_COUNT):
        agent.newVariableFloat(f'attribute{i}')
    agent.newVariableFloat('drift')  # Store the distance moved here, for validation
    agent.newRTCFunction(LOCAL_MESSAGE_FUNC, output_message_local).setMessageOutput('location')
    agent.newRTCFunction(MOVE_FUNC, move).setMessageInput('location')
    return agent

def create_group(model: pyflamegpu.ModelDescription) -> pyflamegpu.AgentDescription:
    group = model.newAgent(GROUP_AGENT_NAME)
    group.newVariableInt('id')
    group.newVariableFloat('x')
    group.newVariableFloat('y')
    group.newVariableFloat('z')
    for i in range(AGENT_ATTRIBUTE_COUNT):
        group.newVariableFloat(f'group_proto_attribute{i}')
    group.newPropertyArrayFloat('group_openness')
    

def create_environment(model: pyflamegpu.ModelDescription) -> pyflamegpu.EnvironmentDescription:
    env = model.Environment()
    env.newPropertyFloat('uncertainty', BASELINE_UNCERTAINTY)
    env.newPropertyFloat('repulse', 0.5)
    # for i in range(AGENT_ATTRIBUTE_COUNT):
    #    # for each attribute, use the prototye attribute function to create a new property
    #    # this gives us a range of groups with various prototypical attributes.
    #     env.newPropertyArrayFloat(f'group_proto_attribute{i}',
    #         [GROUP_PROPERTIES['prototye_attributes']['func'](*GROUP_PROPERTIES['prototye_attributes']['args']) for _ in range(GROUP_COUNT)])
    # env.newPropertyArrayFloat('group_openness', [GROUP_PROPERTIES['openness']['func'](*GROUP_PROPERTIES['openness']['args']) for _ in range(GROUP_COUNT)])
    return env

def populate_simulation(simulation: pyflamegpu.CUDASimulation, agent: pyflamegpu.AgentDescription):
    import numpy as np
    import sklearn.cluster as cluster
    def cluster_agents_by_attributes(attributes: np.ndarray) -> np.ndarray:
        info('Clustering agents by attributes')
        return cluster.KMeans(n_clusters=GROUP_COUNT).fit(attributes).cluster_centers_, cluster.KMeans(n_clusters=GROUP_COUNT).fit(attributes).labels_
        
    # Generate a population if an initial states file is not provided
    if not simulation.SimulationConfig().input_file:
        # Seed the host RNG using the cuda simulations' RNG
        random.seed(simulation.SimulationConfig().random_seed)
        # Generate a vector of agents
        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        # Iterate the population, initialising per-agent values
        for i, instance in enumerate(population):
            # uniformly distribute agents across the environment
            instance.setVariableFloat('x', random.uniform(0.0, ENV_MAX))
            instance.setVariableFloat('y', random.uniform(0.0, ENV_MAX))
            # jitter agents above z
            instance.setVariableFloat('z', random.uniform(0.0, 5.0))
            for i in range(AGENT_ATTRIBUTE_COUNT):
                instance.setVariableFloat(f'attribute{i}',
                    AGENT_PROPERTIES['attributes']['func'](*AGENT_PROPERTIES['attributes']['args']))
        # Set the population for the simulation object
        simulation.setPopulationData(population)
    del cluster