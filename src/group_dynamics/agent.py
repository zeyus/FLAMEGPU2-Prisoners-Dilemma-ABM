# Define FLAME GPU Agent functions as strings
import logging
import pyflamegpu
import random
from .config import *

# # A Callback host function, to check the progress of the model / validate the model.
# class step_validation(pyflamegpu.HostFunctionCallback):
#     def __init__(self) -> None:
#         super().__init__()
#         # Static variables?
#         self.prevTotalDrift = 3.402823e+38 # @todo - static
#         self.driftDropped = 0 # @todo - static
#         self.driftIncreased = 0 # @todo - static

#     def run(self, FLAMEGPU) -> None:
#         # This value should decline? as the model moves towards a steady equilibrium state
#         # Once an equilibrium state is reached, it is likely to oscillate between 2-4? values
#         # totalDrift = FLAMEGPU.agent(AGENT_NAME).sumFloat('drift')
#         # if totalDrift <= self.prevTotalDrift:
#         #     self.driftDropped += 1
#         # else:
#         #     self.driftIncreased += 1
#         # self.prevTotalDrift = totalDrift
#         # print('{:.2f} Drift correct'.format(100 * self.driftDropped / float(self.driftDropped + self.driftIncreased)))
#         pass


def create_agent(model: pyflamegpu.ModelDescription) -> pyflamegpu.AgentDescription:
    agent: pyflamegpu.AgentDescription = model.newAgent(AGENT_NAME)
    agent.newVariableInt('id')
    agent.newVariableFloat('x')
    agent.newVariableFloat('y')
    # add z to jitter agents
    agent.newVariableFloat('z')
    agent.newVariableFloat('roll')
    # this should be int but it is a float so it can be used for agent color
    agent.newVariableFloat('group_id')
    for i in range(AGENT_ATTRIBUTE_COUNT):
        agent.newVariableFloat(f'attribute{i}')
    agent.newVariableFloat('drift')  # Store the distance moved here, for validation
    logging.info(get_cuda_file_path(LOCAL_MESSAGE_FUNC))
    agent.newRTCFunctionFile(LOCAL_MESSAGE_FUNC, get_cuda_file_path(LOCAL_MESSAGE_FUNC)).setMessageOutput('location')
    agent.newRTCFunctionFile(MOVE_FUNC, get_cuda_file_path(MOVE_FUNC)).setMessageInput('location')
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
    env.newPropertyFloat('repulse', 25.0)
    # for i in range(AGENT_ATTRIBUTE_COUNT):
    #    # for each attribute, use the prototye attribute function to create a new property
    #    # this gives us a range of groups with various prototypical attributes.
    #     env.newPropertyArrayFloat(f'group_proto_attribute{i}',
    #         [GROUP_PROPERTIES['prototye_attributes']['func'](*GROUP_PROPERTIES['prototye_attributes']['args']) for _ in range(GROUP_COUNT)])
    # env.newPropertyArrayFloat('group_openness', [GROUP_PROPERTIES['openness']['func'](*GROUP_PROPERTIES['openness']['args']) for _ in range(GROUP_COUNT)])
    return env

def populate_simulation(simulation: pyflamegpu.CUDASimulation, agent: pyflamegpu.AgentDescription):
    import numpy as np
    
    def cluster_agents_by_attributes(attributes: np.ndarray) -> np.ndarray:
        import sklearn.cluster as cluster
        logging.info('Clustering agents by attributes')
        clusters = cluster.MiniBatchKMeans(n_clusters=GROUP_COUNT).fit(attributes)
        logging.info(f'Clustered agents into {GROUP_COUNT} groups')
        return clusters.cluster_centers_, clusters.labels_
        
    # Generate a population if an initial states file is not provided
    if not simulation.SimulationConfig().input_file:
        # Seed the host RNG using the cuda simulations' RNG
        random.seed(simulation.SimulationConfig().random_seed)
        # Generate a vector of agents
        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        # temporarily save attributes for clustering
        agent_attributes = np.zeros((AGENT_COUNT, AGENT_ATTRIBUTE_COUNT))
        for i in range(AGENT_COUNT):
            for j in range(AGENT_ATTRIBUTE_COUNT):
                agent_attributes[i, j] = AGENT_PROPERTIES['attributes']['func'](*AGENT_PROPERTIES['attributes']['args'])
        # group_centers, agent_group_labels = cluster_agents_by_attributes(agent_attributes)
        # logging.info('Group centers: {}, n_labels: {}'.format(len(group_centers), len(agent_group_labels)))
        # Iterate the population, initialising per-agent values
        instance: pyflamegpu.AgentVector_Agent
        for i, instance in enumerate(population):
            # uniformly distribute agents across the environment
            instance.setVariableFloat('x', random.uniform(0.0, ENV_MAX))
            instance.setVariableFloat('y', random.uniform(0.0, ENV_MAX))
            # jitter agents above z
            instance.setVariableFloat('z', random.uniform(0.0, 5.0))
            instance.setVariableFloat('roll', AGENT_ROLL)
            for j in range(AGENT_ATTRIBUTE_COUNT):
                instance.setVariableFloat(f'attribute{j}', agent_attributes[i, j])
            # get the group id from the labels
            # instance.setVariableFloat('group_id', float(agent_group_labels[i].item()))
            instance.setVariableFloat('group_id', 0.0)

        
        del agent_attributes
        
        # Set the population for the simulation object
        simulation.setPopulationData(population)
    del np