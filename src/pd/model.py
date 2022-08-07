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
# this is essentially the size of the grid
MAX_AGENT_SPACES: int = 2**16
# starting agent limit
INIT_AGENT_COUNT: int = MAX_AGENT_SPACES // 16

# you can set this anywhere between INIT_AGENT_COUNT and MAX_AGENT_COUNT inclusive
AGENT_HARD_LIMIT: int = MAX_AGENT_SPACES // 2

# how long to run the sim for
STEP_COUNT: int = 10000
# TODO: logging / Debugging

VERBOSE_OUTPUT: bool = False
DEBUG_OUTPUT: bool = False
OUTPUT_EVERY_N_STEPS: int = 10

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
COST_OF_LIVING: float = 2

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
# Upper energy limit (do we need this?)
MAX_ENERGY: float = 150.0
# How much energy an agent can start with (max)
INIT_ENERGY_MU: float = 50.0
INIT_ENERGY_SIGMA: float = 10.0
INIT_ENERGY_MIN: float = 5.0
# Noise will invert the agent's decision
ENV_NOISE: float = 0.1

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
AGENT_STATUS_ATTEMPTING_REPRODUCTION: int = 512
AGENT_STATUS_REPRODUCTION_IMPOSSIBLE: int = 1024
AGENT_STATUS_REPRODUCTION_COMPLETE: int = 2048
AGENT_STATUS_NEW_AGENT: int = 4096

# Agent strategies for the PD game
# "proportion" let's you say how likely agents spawn with a particular strategy
AGENT_STRATEGY_COOP = 0
AGENT_STRATEGY_DEFECT = 1
AGENT_STRATEGIES: dict = {
  "always_coop": {
    "name": "always_coop",
    "id": AGENT_STRATEGY_COOP,
    "proportion": 0.50,
  },
  "always_defect": {
    "name": "always_defect",
    "id": AGENT_STRATEGY_DEFECT,
    "proportion": 0.50,
  },
  # "tit_for_tat": {
  #   "name": "tit_for_tat",
  #   "id": 2,
  #   "proportion": 0.15,
  # },
  # "random": {
  #   "name": "random",
  #   "id": 3,
  #   "proportion": 0.10,
  # },
}

# How many variants of agents are there?
AGENT_TRAIT_COUNT: int = 4
AGENT_TRAITS: List[int] = list(range(AGENT_TRAIT_COUNT))

# Should an agent deal differently per variant? (max strategies = number of variants)
# or, should they have a strategy for same vs different (max strategies = 2)
AGENT_STRATEGY_PER_TRAIT: bool = False

AGENT_RESULT_COOP: int = 0
AGENT_RESULT_DEFECT: int = 1

# Mutation frequency
AGENT_TRAIT_MUTATION_RATE: float = 0.05


##########################################
# Main script                            #
##########################################
# You should not need to change anything #
# below this line                        #
##########################################

# grid dimensions x = y
ENV_MAX: int = math.ceil(math.sqrt(MAX_AGENT_SPACES))
# this is intentially one more than the max (when zero indexing)
# that way we have a spare "trash" bucket for No-comm.
BUCKET_SIZE: int = ENV_MAX**2 

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


# CUDA_FY_SHUFFLE_FUNCTION_NAME: str = "fy_shuffle"
# CUDA_FY_SHUFFLE_FUNCTION: str = rf"""
# #ifndef SPACES_WITHIN
# #define SPACES_WITHIN {SPACES_WITHIN_RADIUS}
# #include <random>
# FLAMEGPU_HOST_DEVICE_FUNCTION void {CUDA_FY_SHUFFLE_FUNCTION_NAME}() {{
#   uint8_t arr[SPACES_WITHIN];
#   uint8_t idx_arr[SPACES_WITHIN];
#   for (uint8_t i = 0; i < SPACES_WITHIN; i++) {{
#     arr[i] = i;
#     idx_arr[i] = 0;
#   }}
#   uint8_t idx;
#   for (uint8_t i = 0; i < SPACES_WITHIN; i++) {{
#     do {{
#       uint8_t idx = rand() % SPACES_WITHIN;
#     }} while (idx_arr[idx] != 0);
#     idx_arr[idx] = 1;
#     arr[i] = arr[idx];
#   }}
# }}
# #endif
# """
# general function that returns the new position based on the index/sequence of a wrapped moore neighborhood iterator.
CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME: str = "pos_from_moore_seq"
CUDA_POS_FROM_MOORE_SEQ_FUNCTION: str = rf"""
#ifndef POS_FROM_MOORE_SEQ_
#define POS_FROM_MOORE_SEQ_
FLAMEGPU_HOST_DEVICE_FUNCTION void {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(const unsigned int x, const unsigned int y, const unsigned int sequence_index, const unsigned int env_max, unsigned int &new_x, unsigned int &new_y) {{
    // uniform int represents the direction to move,
    // e.g. for radius 1, 0 = northwest, 1 = west, 2 = southwest, 3 = north
    // (4 = no movement), 5 = south, 6 = northeast, 7 = east, 8 = southeast
    unsigned int index = sequence_index;
    if (index >= {CENTER_SPACE}) {{
      ++index;
    }} else if (index > {SPACES_WITHIN_RADIUS_INCL}) {{
      // wrap around the space, e.g. with radius 1, if index is 10, then index = 1.
      index = index % {SPACES_WITHIN_RADIUS_INCL};
    }}
    // Convert to x,y offsets
    const int new_x_offset = index / {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};
    const int new_y_offset = index % {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};
    // const int new_x_offset = index % {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};
    // const int new_y_offset = index / {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};

    // set location to new x,y and wrap around env boundaries
    new_x = (x + new_x_offset) % env_max;
    new_y = (y + new_y_offset) % env_max;
}}
#endif
"""

CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME: str = "pos_to_bucket_id"
CUDA_POS_TO_BUCKET_ID_FUNCTION: str = rf"""
#ifndef POS_TO_BUCKET_ID_
#define POS_TO_BUCKET_ID_
FLAMEGPU_HOST_DEVICE_FUNCTION unsigned int {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(const unsigned int x, const unsigned int y, const unsigned int env_max) {{
    return x + (y * env_max);
}}
#endif
"""

# agent functions
CUDA_SEARCH_FUNC_NAME: str = "search_for_neighbours"
CUDA_SEARCH_FUNC: str = rf"""
{CUDA_POS_TO_BUCKET_ID_FUNCTION}
FLAMEGPU_AGENT_FUNCTION({CUDA_SEARCH_FUNC_NAME}, flamegpu::MessageNone, flamegpu::MessageBucket) {{
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    // this needs to be reset each step, in case agent has moved, for now
    // @TODO: update on agent move instead
    const unsigned int my_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(my_x, my_y, env_max);
    FLAMEGPU->setVariable<unsigned int>("my_bucket", my_bucket);

    const float die_roll =  FLAMEGPU->random.uniform<float>();
    FLAMEGPU->setVariable("die_roll", die_roll);
    
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("die_roll", die_roll);
    FLAMEGPU->message_out.setKey(my_bucket);
    return flamegpu::ALIVE;
}}
"""
CUDA_GAME_LIST_FUNC_NAME: str = "get_game_list"
CUDA_GAME_LIST_FUNC: str = rf"""
{CUDA_POS_FROM_MOORE_SEQ_FUNCTION}
{CUDA_POS_TO_BUCKET_ID_FUNCTION}

FLAMEGPU_AGENT_FUNCTION({CUDA_GAME_LIST_FUNC_NAME}, flamegpu::MessageBucket, flamegpu::MessageNone) {{
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int my_id = FLAMEGPU->getID();
    const float my_roll = FLAMEGPU->getVariable<float>("die_roll");
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    // iterate over all cells in the neighbourhood
    // this also wraps across env boundaries.
    unsigned int num_neighbours = 0;
    //bool challenge;
    unsigned int neighbour_x = my_x;
    unsigned int neighbour_y = my_y;
    float neighbour_roll;
    flamegpu::id_t neighbour_id;
    int8_t my_action;
    for (unsigned int i = 0; i < {SPACES_WITHIN_RADIUS}; ++i) {{
        {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, i, env_max, neighbour_x, neighbour_y);
        const unsigned int neighbour_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(neighbour_x, neighbour_y, env_max);
        // reset neighbour info.
        neighbour_roll = 0.0;
        neighbour_id = flamegpu::ID_NOT_SET;
        my_action = -1;
        // we can safely assume one message per bucket, because agents
        // only output a message at their current location.
        for (const auto& message : FLAMEGPU->message_in(neighbour_bucket)) {{
            neighbour_id = message.getVariable<flamegpu::id_t>("id");
            if (neighbour_id == flamegpu::ID_NOT_SET) {{
              break;
            }}
            ++num_neighbours;
            neighbour_roll = message.getVariable<float>("die_roll");
            
            // if I rolled higher, I initiate the challenge
            // if we rolled the same, the lower ID initiates the challenge
            // otherwise, the opponent will challenge me.
            if (my_roll > neighbour_roll || (neighbour_roll == my_roll && my_id > neighbour_id)) {{
                // we will challenge this neighbour (probably)
                my_action = 1;
            }} else {{
                // else we will have to respond to a challenge (probably)
                my_action = 0;
            }}
            break;
        }}
        // if no message was found, it will default to ID_NOT_SET and 0.0
        FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", i, neighbour_id);
        FLAMEGPU->setVariable<float, {SPACES_WITHIN_RADIUS}>("neighbour_rolls", i, neighbour_roll);
        FLAMEGPU->setVariable<int8_t, {SPACES_WITHIN_RADIUS}>("my_actions", i, my_action);
    }}

    // If there are no neighbours, it's time to move, otherwise let's play a game.
    if (num_neighbours == 0) {{
        
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVEMENT_UNRESOLVED});
        return flamegpu::ALIVE;
    }}
    
    // we have to broadcast a challenge (if needed)
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});

    return flamegpu::ALIVE;
}}
"""

CUDA_AGENT_PLAY_CHALLENGE_CONDITION_NAME: str = "play_challenge_condition"
CUDA_AGENT_PLAY_CHALLENGE_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_PLAY_CHALLENGE_CONDITION_NAME}) {{
    const unsigned int agent_status = FLAMEGPU->getVariable<unsigned int>("agent_status");
    return agent_status == {AGENT_STATUS_READY_TO_CHALLENGE};
}}
"""

CUDA_AGENT_PLAY_CHALLENGE_FUNC_NAME: str = "play_challenge"
CUDA_AGENT_PLAY_CHALLENGE_FUNC: str = rf"""
{CUDA_POS_FROM_MOORE_SEQ_FUNCTION}
{CUDA_POS_TO_BUCKET_ID_FUNCTION}

FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_PLAY_CHALLENGE_FUNC_NAME}, flamegpu::MessageNone, flamegpu::MessageBucket) {{
    FLAMEGPU->setVariable<uint8_t>("round_resolved", 0);
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");

    unsigned int challenge_sequence = FLAMEGPU->getVariable<unsigned int>("challenge_sequence");
    const unsigned int response_sequence = {SPACES_WITHIN_RADIUS} - challenge_sequence - 1;

    FLAMEGPU->setVariable<unsigned int>("response_sequence", response_sequence);

    const int8_t my_challenge_action = FLAMEGPU->getVariable<int8_t, {SPACES_WITHIN_RADIUS}>("my_actions", challenge_sequence);
    const int8_t my_response_action = FLAMEGPU->getVariable<int8_t, {SPACES_WITHIN_RADIUS}>("my_actions", response_sequence);
    const bool my_challenge = my_challenge_action == 1;
    const bool my_response = my_response_action == 0;
    // if my action is -1, it means I have no action to take
    // if it's 1, I challenge, if it's 0, I respond
    if (!my_challenge && !my_response) {{
        FLAMEGPU->setVariable<uint8_t>("round_resolved", 1);
        FLAMEGPU->setVariable<unsigned int>("challenge_sequence", ++challenge_sequence);
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});
        // just send the communication to a bucket that wont be read
        const unsigned int trash_bin = FLAMEGPU->environment.getProperty<unsigned int>("trash_bin");
        FLAMEGPU->message_out.setKey(trash_bin);
        return flamegpu::ALIVE;
    }} else if (!my_challenge && my_response) {{
        // we don't need to send out a challenge, so just leave here
        FLAMEGPU->setVariable<unsigned int>("challenge_sequence", ++challenge_sequence);
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_RESPOND});
        // just send the communication to a bucket that wont be read
        const unsigned int trash_bin = FLAMEGPU->environment.getProperty<unsigned int>("trash_bin");
        FLAMEGPU->message_out.setKey(trash_bin);
        return flamegpu::ALIVE;
    }}

    // we need to send out a challenge
    // we /may/ need to respond as well.
    if (my_challenge) {{
        const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
        const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
        const flamegpu::id_t my_id = FLAMEGPU->getVariable<flamegpu::id_t>("id");
        const flamegpu::id_t responder_id = FLAMEGPU->getVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", challenge_sequence);
        unsigned int neighbour_x;
        unsigned int neighbour_y;
        {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, challenge_sequence, env_max, neighbour_x, neighbour_y);
        const unsigned int neighbour_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(neighbour_x, neighbour_y, env_max);
        FLAMEGPU->message_out.setKey(neighbour_bucket);
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("challenger_id", my_id);
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("responder_id", responder_id);
        
        for (unsigned int i = 0; i < {AGENT_TRAIT_COUNT}; ++i) {{
            FLAMEGPU->message_out.setVariable<unsigned int, {AGENT_TRAIT_COUNT}>("challenger_strategies", i, FLAMEGPU->getVariable<unsigned int, {AGENT_TRAIT_COUNT}>("agent_strategies", i));
        }}

        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("challenger_trait", FLAMEGPU->getVariable<unsigned int>("agent_trait"));
        FLAMEGPU->message_out.setVariable<unsigned int>("challenger_energy", FLAMEGPU->getVariable<float>("energy"));
        FLAMEGPU->message_out.setVariable<unsigned int>("challenger_x", my_x);
        FLAMEGPU->message_out.setVariable<unsigned int>("challenger_y", my_y);
        FLAMEGPU->message_out.setVariable<float>("challenger_roll", FLAMEGPU->getVariable<float>("die_roll"));
        FLAMEGPU->message_out.setVariable<float>("challenger_bucket", FLAMEGPU->getVariable<float>("my_bucket"));

    }}

    if (my_response) {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_RESPOND});
    }} else {{
        if (challenge_sequence < {SPACES_WITHIN_RADIUS}) {{
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});
        }} else {{
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
        }}
    }}

    FLAMEGPU->setVariable<unsigned int>("challenge_sequence", ++challenge_sequence);
      
    
    
    return flamegpu::ALIVE;
}}
"""


CUDA_AGENT_PLAY_RESPONSE_CONDITION_NAME: str = "play_response_condition"
CUDA_AGENT_PLAY_RESPONSE_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_PLAY_RESPONSE_CONDITION_NAME}) {{
    return FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_READY_TO_RESPOND};
}}
"""
CUDA_AGENT_PLAY_RESPONSE_FUNC_NAME: str = "play_response"
CUDA_AGENT_PLAY_RESPONSE_FUNC: str = rf"""
// if we get here, we're kind of pretty sure we have to respond.
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_PLAY_RESPONSE_FUNC_NAME}, flamegpu::MessageBucket, flamegpu::MessageBucket) {{
    const flamegpu::id_t my_id = FLAMEGPU->getID();

    const unsigned int my_bucket = FLAMEGPU->getVariable<unsigned int>("my_bucket");
    
    for (const auto& message : FLAMEGPU->message_in(my_bucket)) {{
        const flamegpu::id_t responder_id = message.getVariable<flamegpu::id_t>("responder_id");
        if (responder_id == my_id) {{
            const flamegpu::id_t challenger_id = message.getVariable<flamegpu::id_t>("challenger_id");
            
            const unsigned int challenger_trait = message.getVariable<unsigned int>("challenger_trait");
            const unsigned int my_trait = FLAMEGPU->getVariable<unsigned int>("agent_trait");
            
            const unsigned int my_strategy = FLAMEGPU->getVariable<unsigned int, {AGENT_TRAIT_COUNT}>("agent_strategies", challenger_trait);
            const unsigned int challenger_strategy = message.getVariable<unsigned int, {AGENT_TRAIT_COUNT}>("challenger_strategies", my_trait);
            
            float challenger_energy = message.getVariable<float>("challenger_energy");
            float my_energy = FLAMEGPU->getVariable<float>("energy");

            bool i_coop;
            bool challenger_coop;

            if (my_strategy == {AGENT_STRATEGIES["always_coop"]["id"]}) {{
                i_coop = true;
            }} else if (my_strategy == {AGENT_STRATEGIES["always_defect"]["id"]}) {{
                i_coop = false;
            }}
            if (challenger_strategy == {AGENT_STRATEGIES["always_coop"]["id"]}) {{
                challenger_coop = true;
            }} else if (challenger_strategy == {AGENT_STRATEGIES["always_defect"]["id"]}) {{
                challenger_coop = false;
            }}
            
            const float payoff_cc = FLAMEGPU->environment.getProperty<float>("payoff_cc");
            const float payoff_cd = FLAMEGPU->environment.getProperty<float>("payoff_cd");
            const float payoff_dc = FLAMEGPU->environment.getProperty<float>("payoff_dc");
            const float payoff_dd = FLAMEGPU->environment.getProperty<float>("payoff_dd");
            const float env_noise = FLAMEGPU->environment.getProperty<float>("env_noise");

            const float my_roll = FLAMEGPU->getVariable<float>("die_roll");
            const float challenger_roll = message.getVariable<float>("challenger_roll");

            // flip my choice if my roll below noise
            if (my_roll < env_noise) {{
                i_coop = !i_coop;
            }}
            // flip challenger choice if challenger roll below noise
            if (challenger_roll < env_noise) {{
                challenger_coop = !challenger_coop;
            }}

            // 4 possible outcomes
            if (i_coop && challenger_coop) {{
                challenger_energy += payoff_cc;
                my_energy += payoff_cc;
            }} else if (!i_coop && !challenger_coop) {{
                challenger_energy += payoff_dd;
                my_energy += payoff_dd;
            }} else if (i_coop && !challenger_coop) {{
                challenger_energy += payoff_dc;
                my_energy += payoff_cd;
            }} else if (!i_coop && challenger_coop) {{
                challenger_energy += payoff_cd;
                my_energy += payoff_dc;
            }}

            FLAMEGPU->message_out.setKey(message.getVariable<unsigned int>("challenger_bucket"));
            FLAMEGPU->message_out.setVariable<flamegpu::id_t>("responder_id", my_id);
            FLAMEGPU->message_out.setVariable<flamegpu::id_t>("challenger_id", challenger_id);
            FLAMEGPU->message_out.setVariable<float>("challenger_energy", challenger_energy);
            if (my_energy <= 0)  {{
                return flamegpu::DEAD;
            }}
            float max_energy = FLAMEGPU->environment.getProperty<float>("max_energy");
            if (my_energy > max_energy) {{
                my_energy = max_energy;
            }}
            FLAMEGPU->setVariable<float>("energy", my_energy);
            uint8_t games_played = FLAMEGPU->setVariable<uint8_t>("games_played");
            FLAMEGPU->setVariable<uint8_t>("games_played", ++games_played);
            break;
        }}
    }}

    const unsigned int challenge_sequence = FLAMEGPU->getVariable<unsigned int>("challenge_sequence");
    if (challenge_sequence < {SPACES_WITHIN_RADIUS}) {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});
    }} else {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
    }}
    return flamegpu::ALIVE;
}}
"""

CUDA_AGENT_PLAY_RESOLVE_CONDITION_NAME: str = "play_resolve_condition"
CUDA_AGENT_PLAY_RESOLVE_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_PLAY_RESOLVE_CONDITION_NAME}) {{
    // any that haven't resolved this round, AND have responders
    return FLAMEGPU->getVariable<uint8_t>("round_resolved") == 0;
}}
"""


CUDA_AGENT_PLAY_RESOLVE_FUNC_NAME: str = "play_resolve"
CUDA_AGENT_PLAY_RESOLVE_FUNC: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_PLAY_RESOLVE_FUNC_NAME}, flamegpu::MessageBucket, flamegpu::MessageNone) {{
    const flamegpu::id_t my_id = FLAMEGPU->getID();

    const unsigned int my_bucket = FLAMEGPU->getVariable<unsigned int>("my_bucket");

    for (const auto& message : FLAMEGPU->message_in(my_bucket)) {{
        const flamegpu::id_t challenger_id = message.getVariable<flamegpu::id_t>("challenger_id");
        if (challenger_id == my_id) {{
            const float my_energy = message.getVariable<float>("challenger_energy");
            if (my_energy <= 0) {{
                return flamegpu::DEAD;
            }}
            FLAMEGPU->setVariable<float>("energy", my_energy);
            uint8_t games_played = FLAMEGPU->setVariable<uint8_t>("games_played");
            FLAMEGPU->setVariable<uint8_t>("games_played", ++games_played);
            break;
        }}
    }}
    FLAMEGPU->setVariable<uint8_t>("round_resolved", 1);
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
// @TODO: change to message bucket...2d won't let a const env_max dimension wtf
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_MOVE_REQUEST_FUNCTION_NAME}, flamegpu::MessageNone, flamegpu::MessageArray2D) {{
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    const flamegpu::id_t my_id = FLAMEGPU->getID();
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    unsigned int last_move_attempt = FLAMEGPU->getVariable<unsigned int>("last_move_attempt");

    // set location to new x,y and wrap around env boundaries
    unsigned int new_x;
    unsigned int new_y;
    // try to limit the need for calling random.
    // @TODO: FIX HACK
    if (last_move_attempt >= {SPACES_WITHIN_RADIUS}) {{
        float my_energy = FLAMEGPU->getVariable<float>("energy");
        float travel_cost = FLAMEGPU->environment.getProperty<float>("travel_cost");
        // try and deduct travel cost, die if below zero, this will prevent
        // unnecessary movement requests
        my_energy -= travel_cost;
        if (my_energy <= 0.0) {{
            return flamegpu::DEAD;
        }}
        FLAMEGPU->setVariable<float>("energy", my_energy);
        
      // this will give us 0 to 7
      last_move_attempt = FLAMEGPU->random.uniform<unsigned int>(0, {SPACES_WITHIN_RADIUS_ZERO_INDEXED});
    }}
    // get a new x,y location for the agent based on the move index.
    {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, last_move_attempt, env_max, new_x, new_y);
    ++last_move_attempt;
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", my_id);
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_x", new_x);
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_y", new_y);
    // @TODO: change to message bucket...2d won't let a const env_max dimension wtf
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
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
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

CUDA_AGENT_NEIGHBOURHOOD_BROADCAST_FUNCTION_NAME: str = "neighbourhood_broadcast"
CUDA_AGENT_NEIGHBOURHOOD_BROADCAST_FUNCTION: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_NEIGHBOURHOOD_BROADCAST_FUNCTION_NAME}, flamegpu::MessageNone, flamegpu::MessageBucket) {{
    // we could be smart here and iterate the neighbour list and only check the existing neighbours,
    // because none spawn, but movement could affect this...so for now just check all.
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setKey(FLAMEGPU->getVariable<unsigned int>("my_bucket"));
    return flamegpu::ALIVE;
}}
"""
CUDA_AGENT_NEIGHBOURHOOD_UPDATE_CONDITION_NAME: str = "neighbourhood_update_condition"
CUDA_AGENT_NEIGHBOURHOOD_UPDATE_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_NEIGHBOURHOOD_UPDATE_CONDITION_NAME}) {{
    const float reproduce_min_energy = FLAMEGPU->environment.getProperty<float>("reproduce_min_energy");
    return FLAMEGPU->getVariable<float>("energy") >= reproduce_min_energy;
}}
"""

CUDA_AGENT_NEIGHBOURHOOD_UPDATE_FUNCTION_NAME: str = "neighbourhood_update"
CUDA_AGENT_NEIGHBOURHOOD_UPDATE_FUNCTION: str = rf"""
{CUDA_POS_FROM_MOORE_SEQ_FUNCTION}
{CUDA_POS_TO_BUCKET_ID_FUNCTION}
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_NEIGHBOURHOOD_UPDATE_FUNCTION_NAME}, flamegpu::MessageBucket, flamegpu::MessageNone) {{
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int my_id = FLAMEGPU->getID();
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    // iterate over all cells in the neighbourhood
    // this also wraps across env boundaries.
    unsigned int num_neighbours = 0;
    unsigned int neighbour_x = my_x;
    unsigned int neighbour_y = my_y;
    flamegpu::id_t neighbour_id;
    for (unsigned int i = 0; i < {SPACES_WITHIN_RADIUS}; ++i) {{
        {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, i, env_max, neighbour_x, neighbour_y);
        const unsigned int neighbour_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(neighbour_x, neighbour_y, env_max);
        // reset neighbour info.
        neighbour_id = flamegpu::ID_NOT_SET;
        // we can safely assume one message per bucket, because agents
        // only output a message at their current location.
        for (const auto& message : FLAMEGPU->message_in(neighbour_bucket)) {{
            neighbour_id = message.getVariable<flamegpu::id_t>("id");
            if (neighbour_id == flamegpu::ID_NOT_SET) {{
              break;
            }}
            ++num_neighbours;
            break;
        }}
        // if no message was found, it will default to ID_NOT_SET
        FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", i, neighbour_id);
    }}

    if (num_neighbours < {SPACES_WITHIN_RADIUS}) {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_ATTEMPTING_REPRODUCTION});
        return flamegpu::ALIVE;
    }}
    
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_IMPOSSIBLE});
    return flamegpu::ALIVE;
}}
"""

CUDA_AGENT_GOD_GO_FORTH_CONDITION_NAME: str = "god_go_forth_condition"
CUDA_AGENT_GOD_GO_FORTH_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_GOD_GO_FORTH_CONDITION_NAME}) {{
    const unsigned int agent_count = FLAMEGPU->environment.getProperty<unsigned int>("agent_count");
    const unsigned int max_agents = FLAMEGPU->environment.getProperty<unsigned int>("max_agents");
    return agent_count < max_agents && FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_ATTEMPTING_REPRODUCTION};
}}
"""
# CUDA_AGENT_GOD_GO_FORTH_CONDITION: str = rf"""
# FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_GOD_GO_FORTH_CONDITION_NAME}) {{
#     const uint8_t max_children_per_step = FLAMEGPU->environment.getProperty<uint8_t>("max_children_per_step");
#     const uint8_t agents_spawned = FLAMEGPU->getVariable<uint8_t>("agents_spawned");
#     const float reproduce_min_energy = FLAMEGPU->environment.getProperty<float>("reproduce_min_energy");
#     return FLAMEGPU->getVariable<uint8_t>("newborn") == 0 && FLAMEGPU->getVariable<float>("energy") >= reproduce_min_energy
#       && FLAMEGPU->getVariable<unsigned int>("reproduce_sequence") < {SPACES_WITHIN_RADIUS}
#       && agents_spawned < max_children_per_step
#       && FLAMEGPU->getVariable<unsigned int>("agent_status") != {AGENT_STATUS_REPRODUCTION_COMPLETE};
# }}
# """

REQUIRES_EMPTY_SPACE_AT_START = "false" if ALLOW_IMMEDIATE_SPACE_OCCUPATION else "true"
CUDA_AGENT_GOD_GO_FORTH_FUNCTION_NAME: str = "god_go_forth"
CUDA_AGENT_GOD_GO_FORTH_FUNCTION: str = rf"""
{CUDA_POS_FROM_MOORE_SEQ_FUNCTION}
{CUDA_POS_TO_BUCKET_ID_FUNCTION}

FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_GOD_GO_FORTH_FUNCTION_NAME}, flamegpu::MessageNone, flamegpu::MessageBucket) {{

    const uint8_t max_children_per_step = FLAMEGPU->environment.getProperty<uint8_t>("max_children_per_step");
    const uint8_t agents_spawned = FLAMEGPU->getVariable<uint8_t>("agents_spawned");

    // if we have already spawned enough children, we can't go forth.
    if (agents_spawned >= max_children_per_step) {{
        FLAMEGPU->message_out.setKey(FLAMEGPU->environment.getProperty<unsigned int>("trash_bin"));
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_COMPLETE});
        return flamegpu::ALIVE;
    }}

    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int my_id = FLAMEGPU->getID();
    
    // const float max_energy = FLAMEGPU->environment.getProperty<float>("max_energy");
    unsigned int reproduce_sequence = FLAMEGPU->getVariable<unsigned int>("reproduce_sequence");
    
    unsigned int last_reproduction_attempt = FLAMEGPU->getVariable<unsigned int>("last_reproduction_attempt");

    // set location to new x,y and wrap around env boundaries
    unsigned int new_x = env_max + 1;
    unsigned int new_y = env_max + 1;
    // try to limit the need for calling random.
    if (last_reproduction_attempt >= {SPACES_WITHIN_RADIUS}) {{
      // this will give us 0 to 7
      last_reproduction_attempt = FLAMEGPU->random.uniform<unsigned int>(0, {SPACES_WITHIN_RADIUS_ZERO_INDEXED});
    }}

    flamegpu::id_t space_is_free = flamegpu::ID_NOT_SET;
    for (unsigned int i = 0; i < {SPACES_WITHIN_RADIUS}; i++) {{
      space_is_free = FLAMEGPU->getVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", i);
      ++reproduce_sequence;
      if(space_is_free != flamegpu::ID_NOT_SET) {{
        continue;
      }}
      // get a new x,y location.
      {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, i, env_max, new_x, new_y);
      
      last_reproduction_attempt = i + 1;
      last_reproduction_attempt %= {SPACES_WITHIN_RADIUS};
      
    }}

    FLAMEGPU->setVariable<unsigned int>("reproduce_sequence", reproduce_sequence);
    
    // check if we found a free space
    if (new_x > env_max || new_y > env_max) {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_COMPLETE});
        return flamegpu::ALIVE;
    }}
    
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_ATTEMPTING_REPRODUCTION});

    FLAMEGPU->setVariable<unsigned int>("last_reproduction_attempt", last_reproduction_attempt);

    const unsigned int request_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(new_x, new_y, env_max);
    FLAMEGPU->message_out.setKey(request_bucket);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", my_id);
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_x", new_x);
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_y", new_y);
    FLAMEGPU->message_out.setVariable<float>("die_roll", FLAMEGPU->getVariable<float>("die_roll"));
    FLAMEGPU->setVariable<unsigned int>("request_bucket", request_bucket);

    return flamegpu::ALIVE;
}}
"""
CUDA_AGENT_GOD_MULTIPLY_CONDITION_NAME: str = "god_multiply_condition"
CUDA_AGENT_GOD_MULTIPLY_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_GOD_MULTIPLY_CONDITION_NAME}) {{
    
    return FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_ATTEMPTING_REPRODUCTION};
}}
"""
CUDA_AGENT_GOD_MULTIPLY_FUNCTION_NAME: str = "god_multiply"
CUDA_AGENT_GOD_MULTIPLY_FUNCTION: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_GOD_MULTIPLY_FUNCTION_NAME}, flamegpu::MessageBucket, flamegpu::MessageNone) {{
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int my_id = FLAMEGPU->getID();
    // first resolve the message where I want to reproduce
    unsigned int requested_x;
    unsigned int requested_y;
    const unsigned int request_bucket = FLAMEGPU->getVariable<unsigned int>("request_bucket");
    flamegpu::id_t highest_roller_id = flamegpu::ID_NOT_SET;
    float highest_roll = 0;
    
    for (const auto& message : FLAMEGPU->message_in(request_bucket)) {{
        const flamegpu::id_t requester_id = message.getVariable<flamegpu::id_t>("id");
        const float die_roll = message.getVariable<float>("die_roll");
        if (die_roll > highest_roll) {{
            highest_roll = die_roll;
            highest_roller_id = requester_id;
            requested_x = message.getVariable<unsigned int>("requested_x");
            requested_y = message.getVariable<unsigned int>("requested_y");
        }} else if (die_roll == highest_roll) {{
            // if we have a tie, pick the higher id
            if (requester_id > highest_roller_id) {{
                highest_roll = die_roll;
                highest_roller_id = requester_id;
                requested_x = message.getVariable<unsigned int>("requested_x");
                requested_y = message.getVariable<unsigned int>("requested_y");
            }}
        }}
    }}
    // if I can claim the space, do so.
    if (highest_roller_id == my_id) {{
      // deduct reproduction cost
      float my_energy = FLAMEGPU->getVariable<float>("energy");
      const float reproduce_cost = FLAMEGPU->environment.getProperty<float>("reproduce_cost");
      const float reproduction_inheritence = FLAMEGPU->environment.getProperty<float>("reproduction_inheritence");
      my_energy -= reproduce_cost;
      FLAMEGPU->setVariable<float>("energy", my_energy);
      FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_COMPLETE});

      // now spawn a child
      if ({CUDA_AGENT_MOVE_UPDATE_VIZ}) {{
        FLAMEGPU->agent_out.setVariable<float>("x", (float)requested_x);
        FLAMEGPU->agent_out.setVariable<float>("y", (float)requested_y);
      }}
      FLAMEGPU->agent_out.setVariable<unsigned int>("x_a", requested_x);
      FLAMEGPU->agent_out.setVariable<unsigned int>("y_a", requested_y);

      FLAMEGPU->agent_out.setVariable<unsigned int>("agent_trait", FLAMEGPU->getVariable<unsigned int>("agent_trait"));
      const float init_energy_min = FLAMEGPU->environment.getProperty<float>("init_energy_min");
      const float max_energy = FLAMEGPU->environment.getProperty<float>("max_energy");
      float child_energy = 0.0;
      if (reproduction_inheritence <= 0.0 || reproduction_inheritence > 1.0) {{
        const float init_energy_mu = FLAMEGPU->environment.getProperty<float>("init_energy_mu");
        const float init_energy_sigma = FLAMEGPU->environment.getProperty<float>("init_energy_sigma");
        
        // use default strategy
        child_energy = FLAMEGPU->random.normal<float>();
        child_energy *= init_energy_sigma;
        child_energy += init_energy_mu;
        
      }} else {{
        // use inheritence strategy
        child_energy = reproduction_inheritence * my_energy;
        
      }}
      if (child_energy < init_energy_min) {{
        child_energy = init_energy_min;
      }} else if (child_energy > max_energy) {{
        child_energy = max_energy;
      }}
      FLAMEGPU->agent_out.setVariable<float>("energy", child_energy);

      for (int i = 0; i < {AGENT_TRAIT_COUNT}; i++) {{
        FLAMEGPU->agent_out.setVariable<unsigned int, {AGENT_TRAIT_COUNT}>("agent_strategies", i, FLAMEGPU->getVariable<unsigned int, {AGENT_TRAIT_COUNT}>("agent_strategies", i));
      }}
      FLAMEGPU->agent_out.setVariable<unsigned int>("agent_status", {AGENT_STATUS_NEW_AGENT});
      uint8_t agents_spawned = FLAMEGPU->getVariable<uint8_t>("agents_spawned");
      FLAMEGPU->setVariable<uint8_t>("agents_spawned", ++agents_spawned);
      FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_COMPLETE});
    }}
    return flamegpu::ALIVE;
}}
"""


# @TODO: change to it's own layer
CUDA_AGENT_GOD_THEN_DIE_CONDITION_NAME: str = "environmental_punishment_condition"
CUDA_ENVIRONMENTAL_PUNISHMENT_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_GOD_THEN_DIE_CONDITION_NAME}) {{
    const unsigned int max_agents = FLAMEGPU->environment.getProperty<unsigned int>("max_agents");
    return FLAMEGPU->getVariable<unsigned int>("agent_status") != {AGENT_STATUS_NEW_AGENT} || FLAMEGPU->getThreadIndex() >= max_agents;
}}
"""
CUDA_ENVIRONMENTAL_PUNISHMENT_NAME: str = "environmental_punishment"
CUDA_ENVIRONMENTAL_PUNISHMENT_FUNCTION: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_ENVIRONMENTAL_PUNISHMENT_NAME}, flamegpu::MessageNone, flamegpu::MessageNone) {{
    // begin the cull
    const unsigned int max_agents = FLAMEGPU->environment.getProperty<unsigned int>("max_agents");
    if (FLAMEGPU->getThreadIndex() >= max_agents) {{
        return flamegpu::DEAD;
    }}
    float my_energy = FLAMEGPU->getVariable<float>("energy");
    const float cost_of_living = FLAMEGPU->environment.getProperty<float>("cost_of_living");
    const float max_energy = FLAMEGPU->environment.getProperty<float>("max_energy");
    if (my_energy > max_energy) {{
        my_energy = max_energy;
    }}
    my_energy -= cost_of_living;
    if (my_energy <= 0) {{
        return flamegpu::DEAD;
    }}
    FLAMEGPU->setVariable<float>("energy", my_energy);
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
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



class step_fn(pyflamegpu.HostFunctionCallback):
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    if VERBOSE_OUTPUT:
      prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
      if FLAMEGPU.getStepCounter() % OUTPUT_EVERY_N_STEPS == 0:
        _print_prisoner_states(prisoner)
        mean, sd = prisoner.meanStandardDeviationFloat("energy")
        print(f"mean energy: {mean}, sd: {sd}")
      
class init_fn(pyflamegpu.HostFunctionCallback):
  def __init__(self):
    super().__init__()
  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    FLAMEGPU.environment.setPropertyUInt("agent_count", INIT_AGENT_COUNT)


class exit_play_fn(pyflamegpu.HostFunctionConditionCallback):
  iterations: int = 0
  max_iterations: int = SPACES_WITHIN_RADIUS
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    # print("play")
    self.iterations += 1
    if self.iterations < self.max_iterations:
      prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
      if VERBOSE_OUTPUT:
        if FLAMEGPU.getStepCounter() % OUTPUT_EVERY_N_STEPS == 0:
          print("ready: ", prisoner.countUInt("agent_status", AGENT_STATUS_READY), "ready_respond: ", prisoner.countUInt("agent_status", AGENT_STATUS_READY_TO_RESPOND))
      if prisoner.count() > AGENT_HARD_LIMIT:
        return pyflamegpu.EXIT
      #print(prisoner.countUInt("agent_status", AGENT_STATUS_SKIP_RESPONSE))
      if prisoner.countUInt("agent_status", AGENT_STATUS_READY) < prisoner.count() - prisoner.countUInt("agent_status", AGENT_STATUS_MOVEMENT_UNRESOLVED):
        return pyflamegpu.CONTINUE
    self.iterations = 0
    return pyflamegpu.EXIT

class exit_move_fn(pyflamegpu.HostFunctionConditionCallback):
  iterations: int = 0
  max_iterations: int = SPACES_WITHIN_RADIUS
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    # print("move")
    self.iterations += 1
    if self.iterations < self.max_iterations:
      # Agent movements still unresolved
      prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
      if prisoner.countUInt("agent_status", AGENT_STATUS_MOVEMENT_UNRESOLVED):
        return pyflamegpu.CONTINUE
    
    self.iterations = 0
    return pyflamegpu.EXIT

class exit_neighbourhood_fn(pyflamegpu.HostFunctionConditionCallback):
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    return pyflamegpu.EXIT

class init_god_fn(pyflamegpu.HostFunctionCallback):
  def __init__(self):
    super().__init__()
  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
    FLAMEGPU.environment.setPropertyUInt("agent_count", prisoner.count())


class exit_god_fn(pyflamegpu.HostFunctionConditionCallback):
  iterations: int = 0
  max_iterations: int = SPACES_WITHIN_RADIUS
  def __init__(self):
    super().__init__()

  def run(self, FLAMEGPU: pyflamegpu.HostAPI):
    prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
    FLAMEGPU.environment.setPropertyUInt("agent_count", prisoner.count())
    self.iterations += 1
    if self.iterations < self.max_iterations:
      prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
      print(prisoner.count())
      if prisoner.countUInt("agent_status", AGENT_STATUS_ATTEMPTING_REPRODUCTION) and prisoner.count() < AGENT_HARD_LIMIT:
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
  agent.newVariableUInt("agent_trait")
  agent.newVariableArrayUInt("agent_strategies", AGENT_TRAIT_COUNT)
  agent.newVariableArrayID("neighbour_list", SPACES_WITHIN_RADIUS, [pyflamegpu.ID_NOT_SET] * SPACES_WITHIN_RADIUS)
  agent.newVariableArrayFloat("neighbour_rolls", SPACES_WITHIN_RADIUS, [0.0] * SPACES_WITHIN_RADIUS)
  # @TODO: flesh out? for now -1 = no neighbour, 0 = I respond, 1 = I challenge
  agent.newVariableArrayInt8("my_actions", SPACES_WITHIN_RADIUS, [-1] * SPACES_WITHIN_RADIUS)
  agent.newVariableFloat("die_roll", 0.0)
  agent.newVariableUInt("my_bucket", 0)
  

  if USE_VISUALISATION:
    agent.newVariableFloat("x")
    agent.newVariableFloat("y")
  
  return agent


def add_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
  env.newPropertyUInt("env_max", ENV_MAX, isConst=True)
  env.newPropertyUInt("trash_bin", BUCKET_SIZE, isConst=True)
  env.newPropertyUInt("agent_count", 0)

def add_pdgame_vars(agent: pyflamegpu.AgentDescription) -> None:
  # add variable for tracking which neighbour is the target
  agent.newVariableUInt("challenge_sequence", 0)
  agent.newVariableUInt("response_sequence", SPACES_WITHIN_RADIUS)
  agent.newVariableUInt8("round_resolved", 0)
  agent.newVariableUInt8("games_played", 0)

def add_pdgame_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
  env.newPropertyFloat("payoff_cc", PAYOFF_CC, isConst=True)
  env.newPropertyFloat("payoff_cd", PAYOFF_CD, isConst=True)
  env.newPropertyFloat("payoff_dc", PAYOFF_DC, isConst=True)
  env.newPropertyFloat("payoff_dd", PAYOFF_DD, isConst=True)
  env.newPropertyFloat("env_noise", ENV_NOISE, isConst=True)
  env.newPropertyFloat("max_energy", MAX_ENERGY, isConst=True)

def add_movement_vars(agent: pyflamegpu.AgentDescription) -> None:
  agent.newVariableUInt("last_move_attempt", SPACES_WITHIN_RADIUS_INCL)
  

def add_movement_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
  env.newMacroPropertyUInt("move_requests", ENV_MAX, ENV_MAX)
  env.newPropertyFloat("travel_cost", AGENT_TRAVEL_COST, isConst=True)
  

def add_god_vars(agent: pyflamegpu.AgentDescription) -> None:
  agent.newVariableUInt("request_bucket", 0) # is this needed?
  agent.newVariableUInt("last_reproduction_attempt", SPACES_WITHIN_RADIUS_INCL)
  agent.newVariableUInt("reproduce_sequence", 0)
  #agent.newVariableUInt8("newborn", 0)
  agent.newVariableUInt8("agents_spawned", 0)
  
def add_neighbourhood_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
  env.newPropertyFloat("reproduce_min_energy", REPRODUCE_MIN_ENERGY, isConst=True)

def add_god_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
  env.newPropertyFloat("reproduce_cost", REPRODUCE_COST, isConst=True)
  env.newPropertyFloat("reproduce_min_energy", REPRODUCE_MIN_ENERGY, isConst=True)
  env.newPropertyFloat("max_energy", MAX_ENERGY, isConst=True)
  env.newPropertyFloat("cost_of_living", COST_OF_LIVING, isConst=True)
  env.newPropertyFloat("init_energy_mu", INIT_ENERGY_MU, isConst=True)
  env.newPropertyFloat("init_energy_sigma", INIT_ENERGY_SIGMA, isConst=True)
  env.newPropertyFloat("init_energy_min", INIT_ENERGY_MIN, isConst=True)
  env.newPropertyFloat("mutation_rate", AGENT_TRAIT_MUTATION_RATE, isConst=True)
  env.newPropertyFloat("reproduction_inheritence", REPRODUCTION_INHERITENCE, isConst=True)
  env.newPropertyUInt8("max_children_per_step", MAX_CHILDREN_PER_STEP, isConst=True)
  env.newPropertyUInt("max_agents", AGENT_HARD_LIMIT, isConst=True)

def _print_environment_properties() -> None:
  print(f"env_max (grid width): {ENV_MAX}")
  print(f"max agent count: {MAX_AGENT_SPACES}")

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
  add_env_vars(env)
  env.newPropertyFloat("cost_of_living", COST_OF_LIVING, isConst=True)
  env.newPropertyUInt("max_agents", AGENT_HARD_LIMIT, isConst=True)
  env.newPropertyFloat("max_energy", MAX_ENERGY, isConst=True)
  
  model.addStepFunctionCallback(step_fn().__disown__())

  agent = make_core_agent(model)

  search_message: pyflamegpu.MessageBucket_Description = model.newMessageBucket("player_search_msg")
  search_message.newVariableID("id")
  search_message.newVariableFloat("die_roll")
  search_message.setBounds(0, BUCKET_SIZE)

  agent_search_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(CUDA_SEARCH_FUNC_NAME, CUDA_SEARCH_FUNC)
  agent_search_fn.setMessageOutput("player_search_msg")

  agent_game_list_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(CUDA_GAME_LIST_FUNC_NAME, CUDA_GAME_LIST_FUNC)
  agent_game_list_fn.setMessageInput("player_search_msg")
  

  agent_environmental_punishment_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(CUDA_ENVIRONMENTAL_PUNISHMENT_NAME, CUDA_ENVIRONMENTAL_PUNISHMENT_FUNCTION)
  agent_environmental_punishment_fn.setAllowAgentDeath(True)
  agent_environmental_punishment_fn.setRTCFunctionCondition(CUDA_ENVIRONMENTAL_PUNISHMENT_CONDITION)

  # load agent-specific interactions
  
  # play resolution submodel
  pdgame_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("pdgame_model")
  pdgame_model.addExitConditionCallback(exit_play_fn().__disown__())

  # add message for game challenges
  challenge_message: pyflamegpu.MessageBucket_Description = pdgame_model.newMessageBucket("player_challenge_msg")
  challenge_message.newVariableID("challenger_id")
  challenge_message.newVariableID("responder_id")
  challenge_message.newVariableArrayUInt("challenger_strategies", AGENT_TRAIT_COUNT)
  challenge_message.newVariableUInt("challenger_trait")
  challenge_message.newVariableFloat("challenger_energy")
  challenge_message.newVariableFloat("challenger_roll")
  challenge_message.newVariableUInt("challenger_x")
  challenge_message.newVariableUInt("challenger_bucket")
  challenge_message.newVariableUInt("challenger_y")
  challenge_message.setBounds(0, BUCKET_SIZE)

  resolve_message: pyflamegpu.MessageBucket_Description = pdgame_model.newMessageBucket("play_resolve_msg")
  resolve_message.newVariableID("challenger_id")
  resolve_message.newVariableID("responder_id")
  resolve_message.newVariableFloat("challenger_energy")
  resolve_message.setBounds(0, BUCKET_SIZE)

  pdgame_env: pyflamegpu.EnvironmentDescription = pdgame_model.Environment()
  add_env_vars(pdgame_env)
  add_pdgame_env_vars(pdgame_env)
  
  # create the submodel
  pdgame_submodel: pyflamegpu.SubModelDescription = model.newSubModel("pdgame_model", pdgame_model)
  pdgame_subagent: pyflamegpu.AgentDescription = make_core_agent(pdgame_model)
  add_pdgame_vars(pdgame_subagent)
  

  agent_challenge_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(CUDA_AGENT_PLAY_CHALLENGE_FUNC_NAME, CUDA_AGENT_PLAY_CHALLENGE_FUNC)
  agent_challenge_fn.setMessageOutput("player_challenge_msg")
  agent_challenge_fn.setRTCFunctionCondition(CUDA_AGENT_PLAY_CHALLENGE_CONDITION)

  agent_response_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(CUDA_AGENT_PLAY_RESPONSE_FUNC_NAME, CUDA_AGENT_PLAY_RESPONSE_FUNC)
  agent_response_fn.setMessageInput("player_challenge_msg")
  agent_response_fn.setRTCFunctionCondition(CUDA_AGENT_PLAY_RESPONSE_CONDITION)
  agent_response_fn.setMessageOutput("play_resolve_msg")
  # it shouldn't be possible for an agent to enter the function if they can't respond to the challenge
  # agent_response_fn.setMessageOutputOptional(True)
  agent_response_fn.setAllowAgentDeath(True)

  agent_resolve_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(CUDA_AGENT_PLAY_RESOLVE_FUNC_NAME, CUDA_AGENT_PLAY_RESOLVE_FUNC)
  agent_resolve_fn.setMessageInput("play_resolve_msg")
  agent_resolve_fn.setRTCFunctionCondition(CUDA_AGENT_PLAY_RESOLVE_CONDITION)
  agent_resolve_fn.setAllowAgentDeath(True)
  

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
  add_env_vars(movement_env)
  add_movement_env_vars(movement_env)

  movement_submodel: pyflamegpu.SubModelDescription = model.newSubModel("movement_model", movement_model)
  movement_subagent: pyflamegpu.AgentDescription = make_core_agent(movement_model)
  add_movement_vars(movement_subagent)
  
  agent_move_request_fn: pyflamegpu.AgentFunctionDescription = movement_subagent.newRTCFunction(CUDA_AGENT_MOVE_REQUEST_FUNCTION_NAME, CUDA_AGENT_MOVE_REQUEST_FUNCTION)
  agent_move_request_fn.setMessageOutput("agent_move_request_msg")
  agent_move_request_fn.setRTCFunctionCondition(CUDA_AGENT_MOVE_REQUEST_CONDITION)
  # Agents can die if they should travel, but don't have enough energy to do so
  agent_move_request_fn.setAllowAgentDeath(True)

  agent_move_response_fn: pyflamegpu.AgentFunctionDescription = movement_subagent.newRTCFunction(CUDA_AGENT_MOVE_RESPONSE_FUNCTION_NAME, CUDA_AGENT_MOVE_RESPONSE_FUNCTION)
  agent_move_response_fn.setMessageInput("agent_move_request_msg")
  agent_move_response_fn.setRTCFunctionCondition(CUDA_AGENT_MOVE_RESPONSE_CONDITION)

  movement_submodel.bindAgent("prisoner", "prisoner", auto_map_vars=True)

  movement_submodel_layer1: pyflamegpu.LayerDescription = movement_model.newLayer()
  movement_submodel_layer1.addAgentFunction(agent_move_request_fn)

  movement_submodel_layer2: pyflamegpu.LayerDescription = movement_model.newLayer()
  movement_submodel_layer2.addAgentFunction(agent_move_response_fn)

  # update neighbours submodel
  neighbourhood_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("neighbourhood_model")
  neighbourhood_model.addExitConditionCallback(exit_neighbourhood_fn().__disown__())
  

  neighbourhood_broadcast_msg: pyflamegpu.MessageBucket_Description = neighbourhood_model.newMessageBucket("neighbourhood_broadcast_msg")
  neighbourhood_broadcast_msg.newVariableID("id")
  neighbourhood_broadcast_msg.setBounds(0, BUCKET_SIZE)

  
  neighbourhood_env: pyflamegpu.EnvironmentDescription = neighbourhood_model.Environment()
  add_env_vars(neighbourhood_env)
  add_neighbourhood_env_vars(neighbourhood_env)
  neighbourhood_submodel: pyflamegpu.SubModelDescription = model.newSubModel("neighbourhood_model", neighbourhood_model)
  neighbourhood_subagent: pyflamegpu.AgentDescription = make_core_agent(neighbourhood_model)

  agent_neighbourhood_broadcast_fn: pyflamegpu.AgentFunctionDescription = neighbourhood_subagent.newRTCFunction(CUDA_AGENT_NEIGHBOURHOOD_BROADCAST_FUNCTION_NAME, CUDA_AGENT_NEIGHBOURHOOD_BROADCAST_FUNCTION)
  agent_neighbourhood_broadcast_fn.setMessageOutput("neighbourhood_broadcast_msg")

  agent_neighbourhood_update_fn: pyflamegpu.AgentFunctionDescription = neighbourhood_subagent.newRTCFunction(CUDA_AGENT_NEIGHBOURHOOD_UPDATE_FUNCTION_NAME, CUDA_AGENT_NEIGHBOURHOOD_UPDATE_FUNCTION)
  agent_neighbourhood_update_fn.setMessageInput("neighbourhood_broadcast_msg")
  # only need to update agents who could reproduce, hence the condition
  agent_neighbourhood_update_fn.setRTCFunctionCondition(CUDA_AGENT_NEIGHBOURHOOD_UPDATE_CONDITION)

  neighbourhood_submodel.bindAgent("prisoner", "prisoner", auto_map_vars=True)

  neighbourhood_submodel_layer1: pyflamegpu.LayerDescription = neighbourhood_model.newLayer()
  neighbourhood_submodel_layer1.addAgentFunction(agent_neighbourhood_broadcast_fn)

  neighbourhood_submodel_layer2: pyflamegpu.LayerDescription = neighbourhood_model.newLayer()
  neighbourhood_submodel_layer2.addAgentFunction(agent_neighbourhood_update_fn)

  # god submodel, asexual reproduction, and environmental slaughter
  god_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription("god_model")
  # only attempt reproduction if there are agents with the right status
  # AND the current count of agents is less than the maximum
  god_model.addInitFunctionCallback(init_god_fn().__disown__())
  god_model.addExitConditionCallback(exit_god_fn().__disown__())
  
  god_go_forth_msg: pyflamegpu.MessageBucket_Description = god_model.newMessageBucket("god_go_forth_msg")
  god_go_forth_msg.newVariableID("id")
  god_go_forth_msg.newVariableUInt("requested_x")
  god_go_forth_msg.newVariableUInt("requested_y")
  god_go_forth_msg.newVariableFloat("die_roll")
  god_go_forth_msg.setBounds(0, BUCKET_SIZE)
  
  god_env: pyflamegpu.EnvironmentDescription = god_model.Environment()
  add_env_vars(god_env)
  add_god_env_vars(god_env)
  god_submodel: pyflamegpu.SubModelDescription = model.newSubModel("god_model", god_model)
  god_subagent: pyflamegpu.AgentDescription = make_core_agent(god_model)
  add_god_vars(god_subagent)

  agent_god_go_forth_fn: pyflamegpu.AgentFunctionDescription = god_subagent.newRTCFunction(CUDA_AGENT_GOD_GO_FORTH_FUNCTION_NAME, CUDA_AGENT_GOD_GO_FORTH_FUNCTION)

  agent_god_go_forth_fn.setMessageOutput("god_go_forth_msg")
  agent_god_go_forth_fn.setRTCFunctionCondition(CUDA_AGENT_GOD_GO_FORTH_CONDITION)

  agent_god_multiply_fn: pyflamegpu.AgentFunctionDescription = god_subagent.newRTCFunction(CUDA_AGENT_GOD_MULTIPLY_FUNCTION_NAME, CUDA_AGENT_GOD_MULTIPLY_FUNCTION)
  agent_god_multiply_fn.setMessageInput("god_go_forth_msg")
  agent_god_multiply_fn.setRTCFunctionCondition(CUDA_AGENT_GOD_MULTIPLY_CONDITION)
  agent_god_multiply_fn.setAgentOutput(god_subagent)

  god_submodel.bindAgent("prisoner", "prisoner", auto_map_vars=True)

  god_submodel_layer1: pyflamegpu.LayerDescription = god_model.newLayer()
  god_submodel_layer1.addAgentFunction(agent_god_go_forth_fn)

  god_submodel_layer2: pyflamegpu.LayerDescription = god_model.newLayer()
  god_submodel_layer2.addAgentFunction(agent_god_multiply_fn)


  # main broadcast location, find neighbours functions
  main_layer1: pyflamegpu.LayerDescription = model.newLayer()
  main_layer1.addAgentFunction(agent_search_fn)

  main_layer2: pyflamegpu.LayerDescription = model.newLayer()
  main_layer2.addAgentFunction(agent_game_list_fn)
  # Layer #3: play a game submodel (only matching ready to play agents)
  main_layer3: pyflamegpu.LayerDescription = model.newLayer()
  main_layer3.addSubModel("pdgame_model")
  
  # Layer #4: movement submodel
  main_layer4: pyflamegpu.LayerDescription = model.newLayer()
  main_layer4.addSubModel("movement_model")

  # Layer #5: neighbourhood submodel
  main_layer5: pyflamegpu.LayerDescription = model.newLayer()
  main_layer5.addSubModel("neighbourhood_model")

  # Layer #6: god submodel
  main_layer6: pyflamegpu.LayerDescription = model.newLayer()
  main_layer6.addSubModel("god_model")

  main_layer7: pyflamegpu.LayerDescription = model.newLayer()
  main_layer7.addAgentFunction(agent_environmental_punishment_fn)

  # this doesn't work because it only shows defined dependencies
  # of which there are none.
  # graph: pyflamegpu.DependencyGraph = model.getDependencyGraph()
  # graph.addRoot(agent_search_fn)
  # graph.addRoot(agent_game_list_fn)
  # graph.addRoot(pdgame_submodel)
  # graph.addRoot(agent_challenge_fn)
  # graph.addRoot(agent_response_fn)
  # graph.addRoot(agent_resolve_fn)
  # graph.addRoot(movement_submodel)
  # graph.addRoot(agent_move_request_fn)
  # graph.addRoot(agent_move_response_fn)

  # graph.addRoot(god_submodel)
  # graph.addRoot(agent_god_go_forth_fn)
  # graph.addRoot(agent_god_multiply_fn)
  # graph.addRoot(agent_god_then_die_fn)
  # graph.generateDOTDiagram("graphdiagram.gv")
  
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
  simulation.SimulationConfig().verbose = DEBUG_OUTPUT

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
    grid = np.arange(MAX_AGENT_SPACES, dtype=np.uint32)
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
      energy = max(random.normalvariate(INIT_ENERGY_MU, INIT_ENERGY_SIGMA), INIT_ENERGY_MIN)
      if MAX_ENERGY > 0.0:
        energy = min(energy, MAX_ENERGY)
      instance.setVariableFloat("energy", energy)
      # select agent strategy
      agent_trait: int = random.choice(AGENT_TRAITS)
      instance.setVariableUInt("agent_trait", agent_trait)
      # select agent strategy
      if AGENT_STRATEGY_PER_TRAIT:
        # if we are using a per-trait strategy, then pick random weighted strategies
        instance.setVariableArrayUInt('agent_strategies', random.choices(AGENT_STRATEGY_IDS, weights=AGENT_WEIGHTS, k=AGENT_TRAIT_COUNT))
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
  if USE_VISUALISATION:
      visualisation.join()
  
if __name__ == "__main__":
    main()