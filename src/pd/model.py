###
# pyflamegpu Prisoner's Dilemma Agent Based Model
###

# Order of execution is as follows:
# 1. Look for neighbouring agents
# 2.a. If no neighbours, move to a random location one space away, deduct travel cost
# 2.b. If neighbours, play a PD game with one neighbour
# 2.b.1. If no energy left after game, die
# 2.b.2. If energy left after game, go back to 2.b.
# 3. If enough energy to reproduce, do so and deduct cost
# 4. deduct environment energy, if below 0, die
# 5. return to step 1

# @TODO: resolv condition where an agent plays a neighbour and dies
# but another neighbour has no games to play (they should move).

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
MAX_AGENT_COUNT: int = 2**8
# starting agent limit
INIT_AGENT_COUNT: int = MAX_AGENT_COUNT // 16
# how long to run the sim for
STEP_COUNT: int = 1000
# TODO: logging / Debugging

VERBOSE_OUTPUT: bool = True

# rate limit simulation?
SIMULATION_SPS_LIMIT: int = 0 # 0 = unlimited

# Show agent visualisation
USE_VISUALISATION: bool = True and pyflamegpu.VISUALISATION

# visualisation camera speed
VISUALISATION_CAMERA_SPEED: float = 0.1
# pause the simulation at start
PAUSE_AT_START: bool = True

# radius of message search grid
MAX_PLAY_DISTANCE: int = 1

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
AGENT_STATUS_READY: int = 1
AGENT_STATUS_READY_TO_CHALLENGE: int = 2
AGENT_STATUS_SKIP_CHALLENGE: int = 4
AGENT_STATUS_READY_TO_RESPOND: int = 8
AGENT_STATUS_SKIP_RESPONSE: int = 16
AGENT_STATUS_PLAY_COMPLETED: int = 32
AGENT_STATUS_MOVEMENT_UNRESOLVED: int = 64
AGENT_STATUS_MOVING: int = 128
AGENT_STATUS_MOVEMENT_COMPLETED: int = 256

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

# get max number of surrounding agents within this radius
# use these as constanst for the CUDA functions
SEARCH_GRID_SIZE: int = 1 + 2 * MAX_PLAY_DISTANCE
SEARCH_GRID_OFFSET: int = SEARCH_GRID_SIZE // 2

SPACES_WITHIN_RADIUS_INCL: int = SEARCH_GRID_SIZE**2
SPACES_WITHIN_RADIUS: int = SPACES_WITHIN_RADIUS_INCL - 1
SPACES_WITHIN_RADIUS_ZERO_INDEXED: int = SPACES_WITHIN_RADIUS - 1
CENTER_SPACE: int = SPACES_WITHIN_RADIUS // 2



# general function that returns the new position based on the index/sequence of a wrapped moore neighborhood iterator.
CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME: str = "pos_from_moore_seq"
CUDA_POS_FROM_MOORE_SEQ_FUNCTION: str = rf"""
FLAMEGPU_HOST_DEVICE_FUNCTION void {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(const unsigned int x, const unsigned int y, unsigned int &move_index, unsigned int &new_x, unsigned int &new_y, const unsigned int env_max) {{
    // uniform int represents the direction to move,
    // e.g. for radius 1, 0 = northwest, 1 = west, 2 = southwest, 3 = north
    // (4 = no movement), 5 = south, 6 = northeast, 7 = east, 8 = southeast
    if (move_index >= {CENTER_SPACE}) {{
      ++move_index;
    }} else if (move_index > {SPACES_WITHIN_RADIUS_INCL}) {{
      // wrap around the space, e.g. with radius 1, if move_index is 10, then move_index = 1.
      move_index = move_index % {SPACES_WITHIN_RADIUS_INCL};
    }}
    // Convert to x,y offsets
    const int new_x_offset = move_index / {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};
    const int new_y_offset = move_index % {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};
    // const int new_x_offset = move_index % {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};
    // const int new_y_offset = move_index / {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};

    // set location to new x,y and wrap around env boundaries
    new_x = (x + new_x_offset) % env_max;
    new_y = (y + new_y_offset) % env_max;
}}
"""

# agent functions
CUDA_SEARCH_FUNC_NAME: str = "search_for_neighbours"
CUDA_SEARCH_FUNC: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_SEARCH_FUNC_NAME}, flamegpu::MessageNone, flamegpu::MessageArray2D) {{
    const float die_roll =  FLAMEGPU->random.uniform<float>();
    FLAMEGPU->message_out.setVariable<float>("die_roll", die_roll);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int>("x_a"), FLAMEGPU->getVariable<unsigned int>("y_a"));
    return flamegpu::ALIVE;
}}
"""
CUDA_GAME_LIST_FUNC_NAME: str = "get_game_list"
CUDA_GAME_LIST_FUNC: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_GAME_LIST_FUNC_NAME}, flamegpu::MessageArray2D, flamegpu::MessageNone) {{
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int my_id = FLAMEGPU->getID();
    const auto my_message = FLAMEGPU->message_in.at(my_x, my_y);
    const float my_roll = my_message.getVariable<float>("die_roll");
    // iterate over all cells in the neighbourhood
    // this also wraps across env boundaries.
    unsigned int num_neighbours = 0;
    unsigned int neighbour_id = 0;
    unsigned int num_responders = 0;
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y, {MAX_PLAY_DISTANCE})) {{
        bool challenge = false;
        flamegpu::id_t competitor_id = message.getVariable<flamegpu::id_t>("id");
        
        if (competitor_id != flamegpu::ID_NOT_SET) {{
          // valid neighbour
          ++num_neighbours;
          const float competitor_roll = message.getVariable<float>("die_roll");
          // if I rolled higher, I initiate the challenge
          // if we rolled the same, the lower ID initiates the challenge
          // otherwise, the opponent will challenge me.
          if (my_roll > competitor_roll || (competitor_roll == my_roll && my_id < competitor_id)) {{
              challenge = true;
          }}
        }}
        if (challenge) {{
            // we will challenge them
            ++num_responders;
            FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("game_list", neighbour_id, competitor_id);
        }} else {{
          FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("game_list", neighbour_id, flamegpu::ID_NOT_SET);
        }}
        ++neighbour_id;
    }}
    // If there are no neighbours, it's time to move, otherwise let's play a game.
    if (num_neighbours == 0) {{
        float my_energy = FLAMEGPU->getVariable<float>("energy");
        float travel_cost = FLAMEGPU->environment.getProperty<float>("travel_cost");
        // try and deduct travel cost, die if below zero, this will prevent
        // unnecessary movement requests
        my_energy -= travel_cost;
        if (my_energy <= 0.0) {{
            return flamegpu::DEAD;
        }}
        FLAMEGPU->setVariable<float>("energy", my_energy);
        // we have to move
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVEMENT_UNRESOLVED});
        return flamegpu::ALIVE;
    }}
    
    const int8_t num_challengers = num_neighbours - num_responders;
    FLAMEGPU->setVariable<int8_t>("challengers", num_challengers); 
    FLAMEGPU->setVariable<int8_t>("responders", num_responders);
    if (num_responders > 0) {{
        // we have to broadcast a challenge
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});
    }} else {{
        // we only have to respond to challenges
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_SKIP_CHALLENGE});
    }}
    return flamegpu::ALIVE;
}}
"""

CUDA_AGENT_PLAY_CHALLENGE_CONDITION_NAME: str = "play_challenge_condition"
CUDA_AGENT_PLAY_CHALLENGE_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_PLAY_CHALLENGE_CONDITION_NAME}) {{
    const unsigned int agent_status = FLAMEGPU->getVariable<unsigned int>("agent_status");
    return agent_status == {AGENT_STATUS_READY_TO_CHALLENGE} || agent_status == {AGENT_STATUS_SKIP_CHALLENGE};
}}
"""

CUDA_AGENT_PLAY_CHALLENGE_FUNC_NAME: str = "play_challenge"
CUDA_AGENT_PLAY_CHALLENGE_FUNC: str = rf"""

FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_PLAY_CHALLENGE_FUNC_NAME}, flamegpu::MessageNone, flamegpu::MessageArray2D) {{
    // if I don't have any responders, I don't need to continue
    if (FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_SKIP_CHALLENGE}) {{
        // if we get here, it's because we don't have to challenge anyone else
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_RESPOND});
        return flamegpu::ALIVE;
    }}
    unsigned int game_sequence = FLAMEGPU->getVariable<unsigned int>("game_sequence");
    flamegpu::id_t opponent = FLAMEGPU->getVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("game_list", game_sequence);
    ++game_sequence;
    FLAMEGPU->setVariable<unsigned int>("game_sequence", game_sequence);
    if (opponent == flamegpu::ID_NOT_SET) {{
      // no opponent in this sequence, move along
      return flamegpu::ALIVE;
    }}

    // set message to opponent    
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("responder_id", opponent);
    // no other agent should have the same opponent this round
    // so no issue with multiple messages out
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int>("x_a"), FLAMEGPU->getVariable<unsigned int>("y_a"));

    const int8_t num_challengers = FLAMEGPU->getVariable<int8_t>("challengers");     
    int8_t num_responders = FLAMEGPU->getVariable<int8_t>("responders");
    // decrement responders
    --num_responders;
    
    // update agent
    if (num_responders + num_challengers <= 0) {{
        // nothing else to do this step
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
        return flamegpu::ALIVE;
    }}
    // if we have challengers, we need to respond
    if (num_challengers > 0) {{
      // we have to accept responses
      FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_RESPOND});
    }} else {{
      // otherwise we indicate that we are still playing (but we can skip challenge responses)
      FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_SKIP_RESPONSE});
    }}
    
    FLAMEGPU->setVariable<int8_t>("responders", num_responders);
    
    return flamegpu::ALIVE;
}}
"""


CUDA_AGENT_PLAY_RESPONSE_CONDITION_NAME: str = "play_response_condition"
CUDA_AGENT_PLAY_RESPONSE_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_PLAY_RESPONSE_CONDITION_NAME}) {{
    const unsigned int agent_status = FLAMEGPU->getVariable<unsigned int>("agent_status");
    return agent_status == {AGENT_STATUS_READY_TO_RESPOND} || agent_status == {AGENT_STATUS_SKIP_RESPONSE};
}}
"""
CUDA_AGENT_PLAY_RESPONSE_FUNC_NAME: str = "play_response"
CUDA_AGENT_PLAY_RESPONSE_FUNC: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_PLAY_RESPONSE_FUNC_NAME}, flamegpu::MessageArray2D, flamegpu::MessageNone) {{
    // if I don't have any challengers, I don't need to continue
    if (FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_SKIP_RESPONSE}) {{
        // if we get here, it's because a sent one challenge, but has no challengers themselves
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});
        return flamegpu::ALIVE;
    }}
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const flamegpu::id_t my_id = FLAMEGPU->getID();
    const int8_t num_responders = FLAMEGPU->getVariable<int8_t>("responders");
    int8_t num_challengers = FLAMEGPU->getVariable<int8_t>("challengers");
    // see if there are any challengers
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y, {MAX_PLAY_DISTANCE})) {{
        const flamegpu::id_t challenger_id = message.getVariable<flamegpu::id_t>("id");
        if (challenger_id == flamegpu::ID_NOT_SET) {{
            continue;
        }}
        // we have a challenger, maybe
        if (my_id != message.getVariable<flamegpu::id_t>("responder_id")) {{
            continue;
        }}

        // challenger found, play a game
        // decrement challenger count
        
        FLAMEGPU->setVariable<int8_t>("challengers", --num_challengers);
        if (num_responders + num_challengers <= 0) {{
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
        }} else if (num_responders > 0) {{
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});
        }} else {{
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_SKIP_CHALLENGE});
        }}

        // there should only be one per round so
        // let's exit and not waste ops.
        return flamegpu::ALIVE;
    }}
    return flamegpu::ALIVE;
}}
"""

CUDA_AGENT_MOVE_REQUEST_CONDITION_NAME: str = "move_request_condition"
CUDA_AGENT_MOVE_REQUEST_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_MOVE_REQUEST_CONDITION_NAME}) {{
    return FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_MOVEMENT_UNRESOLVED};
}}
"""



# @TODO: figure out why sometimes agents DO NOT MOVE
CUDA_AGENT_MOVE_REQUEST_FUNCTION_NAME: str = "move_request"
CUDA_AGENT_MOVE_REQUEST_FUNCTION: str = rf"""
{CUDA_POS_FROM_MOORE_SEQ_FUNCTION}

// getting here means that there are no neighbours, so, free movement
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_MOVE_REQUEST_FUNCTION_NAME}, flamegpu::MessageNone, flamegpu::MessageArray2D) {{
    const flamegpu::id_t my_id = FLAMEGPU->getID();
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    unsigned int last_move_attempt = FLAMEGPU->getVariable<unsigned int>("last_move_attempt");

    // set location to new x,y and wrap around env boundaries
    unsigned int new_x;
    unsigned int new_y;
    // try to limit the need for calling random.
    if (last_move_attempt >= {SPACES_WITHIN_RADIUS_INCL}) {{
      // this will give us 0 to 7
      last_move_attempt = FLAMEGPU->random.uniform<unsigned int>(0, {SPACES_WITHIN_RADIUS_ZERO_INDEXED});
    }}
    // get a new x,y location for the agent based on the move index.
    {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, last_move_attempt, new_x, new_y, {ENV_MAX});
    ++last_move_attempt;
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", my_id);
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_x", new_x);
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_y", new_y);
    auto move_requests = FLAMEGPU->environment.getMacroProperty<unsigned int, {ENV_MAX}, {ENV_MAX}>("move_requests");
    // update requests to move to the new location, only if nobody has requested it (id = 0).
    move_requests[new_x][new_y].CAS(0, (unsigned int) my_id);
    // move_requests[new_x][new_y].max((unsigned int) my_id);

    // set it to my index
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int>("x_a"), FLAMEGPU->getVariable<unsigned int>("y_a"));

    // set me as moving
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVING});

    return flamegpu::ALIVE;
}}
"""

CUDA_AGENT_MOVE_UPDATE_VIZ: str = "true" if USE_VISUALISATION else "false"
CUDA_AGENT_MOVE_RESPONSE_CONDITION_NAME: str = "move_response_condition"
CUDA_AGENT_MOVE_RESPONSE_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_MOVE_RESPONSE_CONDITION_NAME}) {{
    return FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_MOVING};
}}
"""

CUDA_AGENT_MOVE_RESPONSE_FUNCTION_NAME: str = "move_response"
CUDA_AGENT_MOVE_RESPONSE_FUNCTION: str = rf"""
// getting here means that there are no neighbours, so, free movement
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_MOVE_RESPONSE_FUNCTION_NAME}, flamegpu::MessageArray2D, flamegpu::MessageNone) {{
    const flamegpu::id_t my_id = FLAMEGPU->getID();
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");

    // get my message to myself
    const auto message = FLAMEGPU->message_in.at(my_x, my_y);
    const flamegpu::id_t sender_id = message.getVariable<flamegpu::id_t>("id");
    if(my_id != sender_id) {{
      // this should NEVER happen, as move requests are sent to agent's current pos
      // but just in case
      return flamegpu::ALIVE;
    }}
    
    // get requested x,y
    const unsigned int requested_x = message.getVariable<unsigned int>("requested_x");
    const unsigned int requested_y = message.getVariable<unsigned int>("requested_y");

    // get move requests
    auto move_requests = FLAMEGPU->environment.getMacroProperty<unsigned int, {ENV_MAX}, {ENV_MAX}>("move_requests");

    // check if my ID is the one that is allowed to move (max ID)
    if (move_requests[requested_x][requested_y] == (unsigned int) my_id) {{
        // set location to new x, y
        FLAMEGPU->setVariable<unsigned int>("x_a", requested_x);
        FLAMEGPU->setVariable<unsigned int>("y_a", requested_y);

        // also update visualisation float values if required
        if({CUDA_AGENT_MOVE_UPDATE_VIZ}) {{
          FLAMEGPU->setVariable<float>("x", (float) requested_x);
          FLAMEGPU->setVariable<float>("y", (float) requested_y);
        }}
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
    }} else {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVEMENT_UNRESOLVED});
    }}

    return flamegpu::ALIVE;
}}
"""


def _print_prisoner_states(prisoner: pyflamegpu.HostAgentAPI) -> None:
  n_ready: int = prisoner.countUInt("agent_status", AGENT_STATUS_READY)
  n_ready_to_challenge: int = prisoner.countUInt("agent_status", AGENT_STATUS_READY_TO_CHALLENGE)
  n_ready_to_respond: int = prisoner.countUInt("agent_status", AGENT_STATUS_READY_TO_RESPOND)
  n_play_completed: int = prisoner.countUInt("agent_status", AGENT_STATUS_PLAY_COMPLETED)
  n_moving: int = prisoner.countUInt("agent_status", AGENT_STATUS_MOVING)
  n_move_unresolved: int = prisoner.countUInt("agent_status", AGENT_STATUS_MOVEMENT_UNRESOLVED)
  n_move_completed: int = prisoner.countUInt("agent_status", AGENT_STATUS_MOVEMENT_COMPLETED)
  print(f"n_ready: {n_ready}, n_ready_to_challenge: {n_ready_to_challenge}, n_ready_to_respond: {n_ready_to_respond} n_play_completed: {n_play_completed}, n_moving: {n_moving}, n_move_unresolved: {n_move_unresolved}, n_move_completed: {n_move_completed}")
  n_challengers: int = prisoner.sumInt8("challengers")
  n_responders: int = prisoner.sumInt8("responders")
  print(f"total challenges: {n_challengers}, total responses: {n_responders}")

if VERBOSE_OUTPUT:
  class step_fn(pyflamegpu.HostFunctionCallback):
    def __init__(self):
      super().__init__()

    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
      prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
      _print_prisoner_states(prisoner)
      n_at_x0: int = prisoner.countUInt("x_a", 0)
      n_at_y0: int = prisoner.countUInt("y_a", 0)
      print(f"n_at_x0: {n_at_x0}, n_at_y0: {n_at_y0}")


class exit_play_fn(pyflamegpu.HostFunctionConditionCallback):
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
    if prisoner.countUInt("agent_status", AGENT_STATUS_READY) < prisoner.count() - prisoner.countUInt("agent_status", AGENT_STATUS_MOVEMENT_UNRESOLVED):
      return pyflamegpu.CONTINUE
    return pyflamegpu.EXIT

class exit_move_fn(pyflamegpu.HostFunctionConditionCallback):
  iterations: int = 0
  max_iterations: int = SPACES_WITHIN_RADIUS + 1
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    self.iterations += 1
    if self.iterations < self.max_iterations:
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
  agent.newVariableUInt("x_a")
  agent.newVariableUInt("y_a")
  agent.newVariableFloat("energy")
  agent.newVariableUInt("agent_status", AGENT_STATUS_READY)
  agent.newState("playing")
  agent.newState("moving")
  agent.newState("ready")
  agent.setInitialState("ready")
  if USE_VISUALISATION:
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
  
  return agent

def add_game_vars(agent: pyflamegpu.AgentDescription) -> None:
  agent.newVariableArrayUInt("agent_strategies", len(AGENT_TRAITS))
  agent.newVariableUInt("agent_trait")
  agent.newVariableInt8("challengers", 0)
  agent.newVariableInt8("responders", 0)
  agent.newVariableArrayID("game_list", SPACES_WITHIN_RADIUS, [pyflamegpu.ID_NOT_SET] * SPACES_WITHIN_RADIUS)

def add_movement_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
  env.newMacroPropertyUInt("move_requests", ENV_MAX, ENV_MAX)

def _print_environment_properties() -> None:
  print(f"env_max (grid width): {ENV_MAX}")
  print(f"max agent count: {MAX_AGENT_COUNT}")

# Define a method which when called will define the model, Create the simulation object and execute it.
def main():
  if VERBOSE_OUTPUT:
    _print_environment_properties()
  if pyflamegpu.SEATBELTS:
    print("Seatbelts are enabled, this will significantly impact performance.")
    print("Buckle up if you are developing the model. Otherwise throw caution to the wind and use a pyflamegpu build without seatbelts.")
  # Define the FLAME GPU model
  model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("prisoners_dilemma")
  env: pyflamegpu.EnvironmentDescription = model.Environment()
  env.newPropertyFloat("travel_cost", AGENT_TRAVEL_COST, isConst=True)
  if VERBOSE_OUTPUT:
    model.addStepFunctionCallback(step_fn().__disown__())

  agent = make_core_agent(model)
  add_game_vars(agent)

  search_message: pyflamegpu.MessageArray2D_Description = model.newMessageArray2D("player_search_msg")
  search_message.newVariableID("id")
  search_message.newVariableFloat("die_roll")
  search_message.setDimensions(ENV_MAX, ENV_MAX)

  agent_search_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(CUDA_SEARCH_FUNC_NAME, CUDA_SEARCH_FUNC)
  agent_search_fn.setMessageOutput("player_search_msg")

  agent_game_list_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(CUDA_GAME_LIST_FUNC_NAME, CUDA_GAME_LIST_FUNC)
  agent_game_list_fn.setMessageInput("player_search_msg")
  # Agents can die if they should travel, but don't have enough energy to do so
  agent_game_list_fn.setAllowAgentDeath(True)

  # load agent-specific interactions
  
  # play resolution submodel
  pdgame_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("pdgame_model")
  pdgame_model.addExitConditionCallback(exit_play_fn().__disown__())

  # add message for game challenges
  challenge_message: pyflamegpu.MessageArray2D_Description = pdgame_model.newMessageArray2D("player_challenge_msg")
  challenge_message.newVariableID("id")
  challenge_message.newVariableID("responder_id")
  challenge_message.setDimensions(ENV_MAX, ENV_MAX)
  
  # create the submodel
  pdgame_submodel: pyflamegpu.SubModelDescription = model.newSubModel("pdgame_model", pdgame_model)
  pdgame_subagent: pyflamegpu.AgentDescription = make_core_agent(pdgame_model)
  add_game_vars(pdgame_subagent)
  # add variable for tracking which neighbour is the target
  pdgame_subagent.newVariableUInt("game_sequence", 0)

  agent_challenge_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(CUDA_AGENT_PLAY_CHALLENGE_FUNC_NAME, CUDA_AGENT_PLAY_CHALLENGE_FUNC)
  agent_challenge_fn.setMessageOutput("player_challenge_msg")
  agent_challenge_fn.setRTCFunctionCondition(CUDA_AGENT_PLAY_CHALLENGE_CONDITION)

  agent_response_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(CUDA_AGENT_PLAY_RESPONSE_FUNC_NAME, CUDA_AGENT_PLAY_RESPONSE_FUNC)
  agent_response_fn.setMessageInput("player_challenge_msg")
  agent_response_fn.setRTCFunctionCondition(CUDA_AGENT_PLAY_RESPONSE_CONDITION)

  # the following condition is for playing, not for searching.
  pdgame_submodel.bindAgent("prisoner", "prisoner", auto_map_vars=True)

  pdgame_submodel_layer1: pyflamegpu.LayerDescription = pdgame_model.newLayer()
  pdgame_submodel_layer1.addAgentFunction(agent_challenge_fn)

  pdgame_submodel_layer2: pyflamegpu.LayerDescription = pdgame_model.newLayer()
  pdgame_submodel_layer2.addAgentFunction(agent_response_fn)
  
  
  
  # movement resolution submodel
  movement_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("movement_model")
  movement_model.addExitConditionCallback(exit_move_fn().__disown__())
  
  move_request_msg: pyflamegpu.MessageArray2D_Description = movement_model.newMessageArray2D("agent_move_request_msg")
  move_request_msg.newVariableID("id")
  move_request_msg.newVariableUInt("requested_x")
  move_request_msg.newVariableUInt("requested_y")
  move_request_msg.setDimensions(ENV_MAX, ENV_MAX)
  
  movement_env: pyflamegpu.EnvironmentDescription = movement_model.Environment()
  
  add_movement_env_vars(movement_env)
  movement_submodel: pyflamegpu.SubModelDescription = model.newSubModel("movement_model", movement_model)
  movement_subagent: pyflamegpu.AgentDescription = make_core_agent(movement_model)
  movement_subagent.newVariableUInt("last_move_attempt", SPACES_WITHIN_RADIUS_INCL)
  
  agent_move_request_fn: pyflamegpu.AgentFunctionDescription = movement_subagent.newRTCFunction(CUDA_AGENT_MOVE_REQUEST_FUNCTION_NAME, CUDA_AGENT_MOVE_REQUEST_FUNCTION)
  agent_move_request_fn.setMessageOutput("agent_move_request_msg")
  agent_move_request_fn.setRTCFunctionCondition(CUDA_AGENT_MOVE_REQUEST_CONDITION)

  agent_move_response_fn: pyflamegpu.AgentFunctionDescription = movement_subagent.newRTCFunction(CUDA_AGENT_MOVE_RESPONSE_FUNCTION_NAME, CUDA_AGENT_MOVE_RESPONSE_FUNCTION)
  agent_move_response_fn.setMessageInput("agent_move_request_msg")
  agent_move_response_fn.setRTCFunctionCondition(CUDA_AGENT_MOVE_RESPONSE_CONDITION)

  movement_submodel.bindAgent("prisoner", "prisoner", auto_map_vars=True)

  movement_submodel_layer1: pyflamegpu.LayerDescription = movement_model.newLayer()
  movement_submodel_layer1.addAgentFunction(agent_move_request_fn)

  movement_submodel_layer2: pyflamegpu.LayerDescription = movement_model.newLayer()
  movement_submodel_layer2.addAgentFunction(agent_move_response_fn)


  # main broadcast location, find neighbours functions
  main_layer1: pyflamegpu.LayerDescription = model.newLayer()
  main_layer1.addAgentFunction(agent_search_fn)

  main_layer2: pyflamegpu.LayerDescription = model.newLayer()
  main_layer2.addAgentFunction(agent_game_list_fn)
  # Layer #2: play a game submodel (only matching ready to play agents)
  main_layer3: pyflamegpu.LayerDescription = model.newLayer()
  main_layer3.addSubModel("pdgame_model")
  
  # Layer #3: movement submodel
  main_layer4: pyflamegpu.LayerDescription = model.newLayer()
  main_layer4.addSubModel("movement_model")


  
  simulation: pyflamegpu.CUDASimulation = pyflamegpu.CUDASimulation(model)

  if USE_VISUALISATION:
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
      # instance.setVariableUInt("grid_index", int(x + y * ENV_MAX))
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
    simulation.setPopulationData(population, "ready")

  simulation.simulate()
  # Potentially export the population to disk
  # simulation.exportData("end.xml")
  # If visualisation is enabled, end the visualisation
  if USE_VISUALISATION:
      visualisation.join()
  
if __name__ == "__main__":
    main()