###
# pyflamegpu Prisoner's Dilemma Agent Based Model
# @author: @zeyus and @EwBew
# @date: 2020-08-09
# @version: v0.0.3
###
__VERSION__ = "v0.0.3"
__VERSION_STR__ = f"{__name__} v{__VERSION__}"
# @TODO: update
# Order of execution is as follows: (outdated...)
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

from distutils.command.config import config
from time import strftime
from typing import List
import pyflamegpu

# Import standard python libs that are used
import sys
import random
import math


##########################################
# SIMULATION CONFIGURATION               #
##########################################

# Define some constants
RANDOM_SEED: int = random.randint(0, 2 ** 32 / 2 - 1)

# upper agent limit ... please make it a square number for sanity
# this is essentially the size of the grid
MAX_AGENT_SPACES: int = 2**18
# starting agent limit
INIT_AGENT_COUNT: int = int(MAX_AGENT_SPACES * 0.16)

# you can set this anywhere between INIT_AGENT_COUNT and MAX_AGENT_COUNT inclusive
# carrying capacity
AGENT_HARD_LIMIT: int = int(MAX_AGENT_SPACES * 0.5)

# how long to run the sim for
STEP_COUNT: int = 10000
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
PAUSE_AT_START: bool = True
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
        "proportion": 1/4,
    },
    "always_defect": {
        "name": "always_defect",
        "id": AGENT_STRATEGY_DEFECT,
        "proportion": 1/4,
    },
    # defaults to coop if no previous play recorded
    "tit_for_tat": {
        "name": "tit_for_tat",
        "id": AGENT_STRATEGY_TIT_FOR_TAT,
        "proportion": 1/4,
    },
    "random": {
      "name": "random",
      "id": AGENT_STRATEGY_RANDOM,
      "proportion": 1/4,
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

##########################################
# Main script                            #
##########################################
# You should not need to change anything #
# below this line                        #
##########################################

# set up logging
def configure_logging(model: pyflamegpu.ModelDescription) -> pyflamegpu.StepLoggingConfig: 
    step_log_cfg = pyflamegpu.StepLoggingConfig(model)
    step_log_cfg.setFrequency(OUTPUT_EVERY_N_STEPS)
    step_log_cfg.agent("prisoner").logCount()
    step_log_cfg.logEnvironment("population_strat_count")
    return step_log_cfg
    #step_log_cfg



AGENT_RESULT_COOP: int = 0
AGENT_RESULT_DEFECT: int = 1

AGENT_TRAITS: List[int] = list(range(AGENT_TRAIT_COUNT))
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

# grid dimensions x = y
ENV_MAX: int = math.ceil(math.sqrt(MAX_AGENT_SPACES))
# this is intentially one more than the max (when zero indexing)
# that way we have a spare "trash" bucket for No-comm.
BUCKET_SIZE: int = ENV_MAX**2


# Generate weights based on strategy configuration
AGENT_WEIGHTS: List[float] = [AGENT_STRATEGIES[strategy]
                              ["proportion"] for strategy in AGENT_STRATEGIES]  # type: ignore
# generate strategy IDs based on strategy configuration
AGENT_STRATEGY_IDS: List[int] = [
    AGENT_STRATEGIES[strategy]["id"] for strategy in AGENT_STRATEGIES]  # type: ignore

AGENT_STRATEGY_COUNT: int = len(AGENT_STRATEGY_IDS)

POPULATION_COUNT_BINS: int = AGENT_STRATEGY_COUNT ** 2
# definie color pallete for each agent strategy, with fallback to white
AGENT_COLOR_SCHEME: pyflamegpu.uDiscreteColor = pyflamegpu.uDiscreteColor(
    "agent_color", pyflamegpu.SET1, pyflamegpu.WHITE)
AGENT_DEFAULT_SHAPE: str = './src/resources/models/primitive_pyramid_arrow.obj'
AGENT_DEFAULT_SCALE: float = 0.9

ROLL_INCREMENT: float = math.pi / 4
# RAD ANGLES
ROLL_RADS: List[float] = [
  -3 * ROLL_INCREMENT,
  -4 * ROLL_INCREMENT,
  3 * ROLL_INCREMENT,
  -2 * ROLL_INCREMENT,
  4 * ROLL_INCREMENT, # i know :P
  -1 * ROLL_INCREMENT,
  0 * ROLL_INCREMENT,
  1 * ROLL_INCREMENT,
]

# get max number of surrounding agents within this radius
# use these as constanst for the CUDA functions
SEARCH_GRID_SIZE: int = 1 + 2 * MAX_PLAY_DISTANCE
SEARCH_GRID_OFFSET: int = SEARCH_GRID_SIZE // 2

SPACES_WITHIN_RADIUS_INCL: int = SEARCH_GRID_SIZE**2
SPACES_WITHIN_RADIUS: int = SPACES_WITHIN_RADIUS_INCL - 1
SPACES_WITHIN_RADIUS_ZERO_INDEXED: int = SPACES_WITHIN_RADIUS - 1
CENTER_SPACE: int = SPACES_WITHIN_RADIUS // 2

# if we use visualisation, update agent position and direction.
CUDA_AGENT_MOVE_UPDATE_VIZ: str = "true" if USE_VISUALISATION else "false"
CUDA_ORIENT_AGENTS: str = "true" if USE_VISUALISATION and VISUALISATION_ORIENT_AGENTS else "false"


CUDA_GET_POP_INDEX_FUNCTION_NAME: str = "get_pop_index"
CUDA_GET_POP_INDEX_FUNCTION: str = rf"""
#ifndef GET_POP_INDEX_
#define GET_POP_INDEX_
// @TODO: @NOTE: ONLY FOR FOUR STRATEGIES ...UGH, TIME
FLAMEGPU_HOST_DEVICE_FUNCTION unsigned int {CUDA_GET_POP_INDEX_FUNCTION_NAME}(const uint8_t strat_my, const uint8_t strat_other) {{
    return strat_my * 4 + strat_other;
}}
#endif
""" 


CUDA_SEQ_TO_ANGLE_FUNCTION_NAME: str = "seq_to_angle"
CUDA_SEQ_TO_ANGLE_FUNCTION: str = rf"""
#ifndef SEQ_TO_ANGLE_
#define SEQ_TO_ANGLE_
// @TODO: @NOTE: this does not work with spaces_within_radius != 8 at the moment.
FLAMEGPU_HOST_DEVICE_FUNCTION float {CUDA_SEQ_TO_ANGLE_FUNCTION_NAME}(const unsigned int seq) {{
    static const float seq_map[{SPACES_WITHIN_RADIUS}] = {{
        {ROLL_RADS[0]}, {ROLL_RADS[1]}, {ROLL_RADS[2]}, {ROLL_RADS[3]},
        {ROLL_RADS[4]}, {ROLL_RADS[5]}, {ROLL_RADS[6]}, {ROLL_RADS[7]}
    }};

    return seq_map[seq % {SPACES_WITHIN_RADIUS}];
}}
#endif
""" # if VISUALISATION_ORIENT_AGENTS else ""
# general function that returns the new position based on the index/sequence of a wrapped moore neighborhood iterator.
CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME: str = "pos_from_moore_seq"
CUDA_POS_FROM_MOORE_SEQ_FUNCTION: str = rf"""
#ifndef POS_FROM_MOORE_SEQ_
#define POS_FROM_MOORE_SEQ_
FLAMEGPU_HOST_DEVICE_FUNCTION void {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(const unsigned int x, const unsigned int y, const unsigned int sequence_index, const unsigned int env_max, unsigned int &new_x, unsigned int &new_y) {{
    // uniform int represents the direction to move,
    // e.g. for radius 1, 0 = northwest, 1 = west, 2 = southwest, 3 = north
    // (4 = no movement), 5 = south, 6 = northeast, 7 = east, 8 = southeast
    const int8_t x_offset[{SPACES_WITHIN_RADIUS}] = {{
      -1,-1,-1, 0, 0, 1, 1, 1
    }};
    const int8_t y_offset[{SPACES_WITHIN_RADIUS}] = {{
      -1, 0, 1,-1, 1,-1, 0, 1
    }};

    const int8_t new_x_offset = x_offset[sequence_index];
    const int8_t new_y_offset = y_offset[sequence_index];
    new_x = (x + new_x_offset) % env_max;
    new_y = (y + new_y_offset) % env_max;

    //unsigned int index = sequence_index;
    //if (index >= {CENTER_SPACE}) {{
    //  ++index;
    //}} else if (index > {SPACES_WITHIN_RADIUS_INCL}) {{
    //  // wrap around the space, e.g. with radius 1, if index is 10, then index = 1.
    //  index = index % {SPACES_WITHIN_RADIUS_INCL};
    //}}
    // Convert to x,y offsets
    //const int new_x_offset = index / {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};
    //const int new_y_offset = index % {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};
    // const int new_x_offset = index % {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};
    // const int new_y_offset = index / {SEARCH_GRID_SIZE} - {SEARCH_GRID_OFFSET};

    // set location to new x,y and wrap around env boundaries
    // new_x = (x + new_x_offset) % env_max;
    // new_y = (y + new_y_offset) % env_max;
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

def debug_set_color(index: int = -1) -> str:
  if index == -1:
    return ''
  return f'FLAMEGPU->setVariable<unsigned int>("agent_color", {index});'

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
        // we can safely?? assume one message per bucket, because agents
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
{CUDA_SEQ_TO_ANGLE_FUNCTION}
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_PLAY_CHALLENGE_FUNC_NAME}, flamegpu::MessageNone, flamegpu::MessageBucket) {{
    FLAMEGPU->setVariable<uint8_t>("round_resolved", 0);
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");

    uint8_t challenge_sequence = FLAMEGPU->getVariable<uint8_t>("challenge_sequence");

    // quick check to exit if we got too past the max
    if (challenge_sequence >= {SPACES_WITHIN_RADIUS}) {{
        const unsigned int trash_bin = FLAMEGPU->environment.getProperty<unsigned int>("trash_bin");
        FLAMEGPU->message_out.setKey(trash_bin);
        const uint8_t games_played = FLAMEGPU->getVariable<uint8_t>("games_played");
        if (games_played < 1) {{
            // we've run out of spaces, and no games have been played.
            // that means that the agent(s) we were to play against have
            // died and we can instead do a movement action this turn.
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVEMENT_UNRESOLVED});
        }} else {{
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
        }}
        return flamegpu::ALIVE;
    }}

    const uint8_t response_sequence = {SPACES_WITHIN_RADIUS} - challenge_sequence - 1;
    FLAMEGPU->setVariable<uint8_t>("response_sequence", response_sequence);

    const int8_t my_challenge_action = FLAMEGPU->getVariable<int8_t, {SPACES_WITHIN_RADIUS}>("my_actions", challenge_sequence);
    const int8_t my_response_action = FLAMEGPU->getVariable<int8_t, {SPACES_WITHIN_RADIUS}>("my_actions", response_sequence);
    const bool my_challenge = my_challenge_action == 1;
    const bool my_response = my_response_action == 0;
    // if my action is -1, it means I have no action to take
    // if it's 1, I challenge, if it's 0, I respond
    if (!my_challenge && !my_response) {{
        FLAMEGPU->setVariable<uint8_t>("round_resolved", 1);
        if ({CUDA_AGENT_MOVE_UPDATE_VIZ} && {CUDA_ORIENT_AGENTS}) {{
          FLAMEGPU->setVariable<float>("pitch", {CUDA_SEQ_TO_ANGLE_FUNCTION_NAME}(challenge_sequence));
        }}
        FLAMEGPU->setVariable<uint8_t>("challenge_sequence", ++challenge_sequence);
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});
        // just send the communication to a bucket that wont be read
        const unsigned int trash_bin = FLAMEGPU->environment.getProperty<unsigned int>("trash_bin");
        FLAMEGPU->message_out.setKey(trash_bin);
        return flamegpu::ALIVE;
    }} else if (!my_challenge && my_response) {{
        // we don't need to send out a challenge, so just leave here
        FLAMEGPU->setVariable<uint8_t>("challenge_sequence", ++challenge_sequence);
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_RESPOND});
        // just send the communication to a bucket that wont be read
        const unsigned int trash_bin = FLAMEGPU->environment.getProperty<unsigned int>("trash_bin");
        // only challengers need to resolve the results.
        FLAMEGPU->setVariable<uint8_t>("round_resolved", 1);
        FLAMEGPU->message_out.setKey(trash_bin);
        return flamegpu::ALIVE;
    }}

    // we need to send out a challenge
    // we /may/ need to respond as well.
    if (my_challenge) {{
        const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
        const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
        const flamegpu::id_t my_id = FLAMEGPU->getID();
        const flamegpu::id_t responder_id = FLAMEGPU->getVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", challenge_sequence);
        unsigned int neighbour_x;
        unsigned int neighbour_y;
        {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, challenge_sequence, env_max, neighbour_x, neighbour_y);
        const unsigned int neighbour_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(neighbour_x, neighbour_y, env_max);
        FLAMEGPU->message_out.setKey(neighbour_bucket);
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("challenger_id", my_id);
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("responder_id", responder_id);
        
        for (unsigned int i = 0; i < {AGENT_TRAIT_COUNT}; ++i) {{
            FLAMEGPU->message_out.setVariable<uint8_t, {AGENT_TRAIT_COUNT}>("challenger_strategies", i, FLAMEGPU->getVariable<uint8_t, {AGENT_TRAIT_COUNT}>("agent_strategies", i));
        }}

        FLAMEGPU->message_out.setVariable<uint8_t>("challenger_trait", FLAMEGPU->getVariable<uint8_t>("agent_trait"));
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("challenger_game_memory_id", FLAMEGPU->getVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("game_memory", challenge_sequence));
        FLAMEGPU->message_out.setVariable<uint8_t>("challenger_game_memory_choice", FLAMEGPU->getVariable<uint8_t, {SPACES_WITHIN_RADIUS}>("game_memory_choices", challenge_sequence));
        FLAMEGPU->message_out.setVariable<unsigned int>("challenger_energy", FLAMEGPU->getVariable<float>("energy"));
        FLAMEGPU->message_out.setVariable<unsigned int>("challenger_x", my_x);
        FLAMEGPU->message_out.setVariable<unsigned int>("challenger_y", my_y);
        FLAMEGPU->message_out.setVariable<float>("challenger_roll", FLAMEGPU->getVariable<float>("die_roll"));
        FLAMEGPU->message_out.setVariable<float>("challenger_bucket", FLAMEGPU->getVariable<float>("my_bucket"));

    }}

    FLAMEGPU->setVariable<uint8_t>("challenge_sequence", ++challenge_sequence);

    if (my_response) {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_RESPOND});
    }} else {{
        if (challenge_sequence < {SPACES_WITHIN_RADIUS}) {{
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});
        }} else {{
            const uint8_t games_played = FLAMEGPU->getVariable<uint8_t>("games_played");
            if (games_played < 1) {{
                // we've run out of spaces, and no games have been played.
                // that means that the agent(s) we were to play against have
                // died and we can instead do a movement action this turn.
                FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVEMENT_UNRESOLVED});
                //FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
            }} else {{
                FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
            }}
        }}
    }}

    
      
    
    
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
{CUDA_SEQ_TO_ANGLE_FUNCTION}
// if we get here, we're kind of pretty sure we have to respond.
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_PLAY_RESPONSE_FUNC_NAME}, flamegpu::MessageBucket, flamegpu::MessageBucket) {{
    const flamegpu::id_t my_id = FLAMEGPU->getID();
    
    const unsigned int my_bucket = FLAMEGPU->getVariable<unsigned int>("my_bucket");
    uint8_t games_played = FLAMEGPU->getVariable<uint8_t>("games_played");

    for (const auto& message : FLAMEGPU->message_in(my_bucket)) {{
        const flamegpu::id_t responder_id = message.getVariable<flamegpu::id_t>("responder_id");
        if (responder_id == my_id) {{
            
            const uint8_t response_sequence = FLAMEGPU->getVariable<uint8_t>("response_sequence");

            if ({CUDA_AGENT_MOVE_UPDATE_VIZ} && {CUDA_ORIENT_AGENTS}) {{
              FLAMEGPU->setVariable<float>("pitch", {CUDA_SEQ_TO_ANGLE_FUNCTION_NAME}(response_sequence));
            }}
            const flamegpu::id_t challenger_id = message.getVariable<flamegpu::id_t>("challenger_id");
            
            const uint8_t challenger_trait = message.getVariable<uint8_t>("challenger_trait");
            const uint8_t my_trait = FLAMEGPU->getVariable<uint8_t>("agent_trait");
            
            const uint8_t my_strategy = FLAMEGPU->getVariable<uint8_t, {AGENT_TRAIT_COUNT}>("agent_strategies", challenger_trait);
            const uint8_t challenger_strategy = message.getVariable<uint8_t, {AGENT_TRAIT_COUNT}>("challenger_strategies", my_trait);
            
            float challenger_energy = message.getVariable<float>("challenger_energy");
            float my_energy = FLAMEGPU->getVariable<float>("energy");
            
            const float payoff_cc = FLAMEGPU->environment.getProperty<float>("payoff_cc");
            const float payoff_cd = FLAMEGPU->environment.getProperty<float>("payoff_cd");
            const float payoff_dc = FLAMEGPU->environment.getProperty<float>("payoff_dc");
            const float payoff_dd = FLAMEGPU->environment.getProperty<float>("payoff_dd");
            const float env_noise = FLAMEGPU->environment.getProperty<float>("env_noise");

            const float my_roll = FLAMEGPU->random.uniform<float>();
            const float challenger_roll = FLAMEGPU->random.uniform<float>();
            
            bool i_coop;
            bool challenger_coop;

            if (challenger_strategy == {AGENT_STRATEGY_COOP}) {{
                challenger_coop = true;
            }} else if (challenger_strategy == {AGENT_STRATEGY_DEFECT}) {{
                challenger_coop = false;
            }} else if (challenger_strategy == {AGENT_STRATEGY_TIT_FOR_TAT}) {{
                const flamegpu::id_t challenger_last_opponent = message.getVariable<flamegpu::id_t>("challenger_game_memory_id");
                if (challenger_last_opponent == my_id) {{
                    const uint8_t challenger_last_opponent_choice = message.getVariable<uint8_t>("challenger_game_memory_choice");
                    challenger_coop = challenger_last_opponent_choice == {AGENT_RESULT_COOP};
                }} else {{
                    // default to coop
                    challenger_coop = true;
                }}
            }} else if (challenger_strategy == {AGENT_STRATEGY_RANDOM}) {{
                if (FLAMEGPU->random.uniform<float>() > 0.5) {{
                    challenger_coop = true;
                }} else {{
                    challenger_coop = false;
                }}
            }}
            // flip challenger choice if challenger roll below noise
            if (challenger_roll < env_noise) {{
                challenger_coop = !challenger_coop;
            }}

            if (my_strategy == {AGENT_STRATEGY_COOP}) {{
                i_coop = true;
            }} else if (my_strategy == {AGENT_STRATEGY_DEFECT}) {{
                i_coop = false;
            }} else if (my_strategy == {AGENT_STRATEGY_TIT_FOR_TAT}) {{
                flamegpu::id_t previous_opponent = FLAMEGPU->getVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("game_memory", response_sequence);
                if (previous_opponent == challenger_id) {{
                    const uint8_t previous_opponent_choice = FLAMEGPU->getVariable<uint8_t, {SPACES_WITHIN_RADIUS}>("game_memory_choices", response_sequence);
                    i_coop = previous_opponent_choice == {AGENT_RESULT_COOP};
                }} else {{
                    i_coop = true;
                    previous_opponent = challenger_id;
                    FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("game_memory", response_sequence, previous_opponent);
                    
                }}
                if (challenger_coop) {{
                    FLAMEGPU->setVariable<uint8_t, {SPACES_WITHIN_RADIUS}>("game_memory_choices", response_sequence, {AGENT_RESULT_COOP});
                }} else {{
                    FLAMEGPU->setVariable<uint8_t, {SPACES_WITHIN_RADIUS}>("game_memory_choices", response_sequence, {AGENT_RESULT_DEFECT});
                }}
                
            }} else if (my_strategy == {AGENT_STRATEGY_RANDOM}) {{
                if (FLAMEGPU->random.uniform<float>() > 0.5) {{
                    i_coop = true;
                }} else {{
                    i_coop = false;
                }}
            }}
            // @TODO: add new random number here and above otherwise the conditions always come together
            // flip my choice if my roll below noise
            if (my_roll < env_noise) {{
                i_coop = !i_coop;
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

            if (i_coop) {{
                FLAMEGPU->message_out.setVariable<uint8_t>("responder_response", {AGENT_RESULT_COOP});
            }} else {{
                FLAMEGPU->message_out.setVariable<uint8_t>("responder_response", {AGENT_RESULT_DEFECT});
            }}
            
            FLAMEGPU->setVariable<uint8_t>("games_played", ++games_played);
            break;
        }}
    }}

    const uint8_t challenge_sequence = FLAMEGPU->getVariable<uint8_t>("challenge_sequence");
    if (challenge_sequence < {SPACES_WITHIN_RADIUS}) {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY_TO_CHALLENGE});
    }} else {{
        if (games_played < 1) {{
            // we've run out of spaces, and no games have been played.
            // that means that the agent(s) we were to play against have
            // died and we can instead do a movement action this turn.
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVEMENT_UNRESOLVED});
            // FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});

            // also nothing to resolve
            FLAMEGPU->setVariable<uint8_t>("round_resolved", 1);
        }} else {{
            FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
        }}
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
            uint8_t games_played = FLAMEGPU->getVariable<uint8_t>("games_played");
            FLAMEGPU->setVariable<uint8_t>("games_played", ++games_played);
            const uint8_t my_strategy = message.getVariable<uint8_t>("challenger_strategy");
            if (my_strategy == {AGENT_STRATEGY_TIT_FOR_TAT}) {{
                const uint8_t challenge_sequence = FLAMEGPU->getVariable<uint8_t>("challenge_sequence") - 1;
                FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("game_memory", challenge_sequence, message.getVariable<flamegpu::id_t>("responder_id"));
                FLAMEGPU->setVariable<uint8_t, {SPACES_WITHIN_RADIUS}>("game_memory_choices", challenge_sequence, message.getVariable<uint8_t>("responder_response"));
            }}
            
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

CUDA_AGENT_MOVE_REQUEST_FUNCTION_NAME: str = "move_request"
CUDA_AGENT_MOVE_REQUEST_FUNCTION: str = rf"""
{CUDA_POS_FROM_MOORE_SEQ_FUNCTION}
{CUDA_POS_TO_BUCKET_ID_FUNCTION}
{CUDA_SEQ_TO_ANGLE_FUNCTION}
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_MOVE_REQUEST_FUNCTION_NAME}, flamegpu::MessageNone, flamegpu::MessageBucket) {{
    unsigned int last_move_attempt = FLAMEGPU->getVariable<unsigned int>("last_move_attempt");

    // try to limit the need for calling random.
    // with spaces_within_radius + 1.
    if (last_move_attempt > {SPACES_WITHIN_RADIUS}) {{
        float my_energy = FLAMEGPU->getVariable<float>("energy");
        
        float travel_cost = FLAMEGPU->environment.getProperty<float>("travel_cost");
        // try and deduct travel cost, die if below zero, this will prevent
        // unnecessary movement requests
        my_energy -= travel_cost;
        if (my_energy <= 0.0) {{
            FLAMEGPU->message_out.setKey(FLAMEGPU->environment.getProperty<unsigned int>("trash_bin"));
            return flamegpu::DEAD;
        }}
        FLAMEGPU->setVariable<float>("energy", my_energy);
        float die_roll = FLAMEGPU->random.uniform<float>();
        FLAMEGPU->setVariable<float>("die_roll", die_roll);
        
      // this will give us 0 to 7 as a random start point
      last_move_attempt = FLAMEGPU->random.uniform<unsigned int>(0, {SPACES_WITHIN_RADIUS_ZERO_INDEXED});
    }}

    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const flamegpu::id_t my_id = FLAMEGPU->getID();

    unsigned int move_sequence = FLAMEGPU->getVariable<unsigned int>("move_sequence");

    // set default outside of grid to check
    unsigned int new_x = env_max + 1;
    unsigned int new_y = env_max + 1;

    bool space_is_free = false;

    for (unsigned int i = 0; i < {SPACES_WITHIN_RADIUS}; ++i) {{
      ++move_sequence;
      if (move_sequence > {SPACES_WITHIN_RADIUS}) {{
          break;
      }}
      last_move_attempt = (last_move_attempt + i) % {SPACES_WITHIN_RADIUS};
      space_is_free = (FLAMEGPU->getVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", last_move_attempt) == flamegpu::ID_NOT_SET);
      
      if(space_is_free) {{
        // get the new x,y location.
        {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, last_move_attempt, env_max, new_x, new_y);
        break;
      }}
    }}

    // check if we found a free space
    if (new_x > env_max || new_y > env_max) {{
        FLAMEGPU->message_out.setKey(FLAMEGPU->environment.getProperty<unsigned int>("trash_bin"));
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});
        return flamegpu::ALIVE;
    }}

    last_move_attempt = (last_move_attempt + 1) % {SPACES_WITHIN_RADIUS};
    FLAMEGPU->setVariable<unsigned int>("move_sequence", move_sequence);

    // we have a free space so attempt to move there
    if ({CUDA_AGENT_MOVE_UPDATE_VIZ} && {CUDA_ORIENT_AGENTS}) {{
      FLAMEGPU->setVariable<float>("pitch", {CUDA_SEQ_TO_ANGLE_FUNCTION_NAME}(last_move_attempt - 1));
    }}

    // set me as moving
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVING});

    FLAMEGPU->setVariable<unsigned int>("last_move_attempt", last_move_attempt);

    const unsigned int request_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(new_x, new_y, env_max);
    FLAMEGPU->setVariable<unsigned int>("request_bucket", request_bucket);

    FLAMEGPU->message_out.setKey(request_bucket);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("requester_id", my_id);
    FLAMEGPU->message_out.setVariable<float>("requester_roll", FLAMEGPU->getVariable<float>("die_roll"));
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_x", new_x);
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_y", new_y);

    return flamegpu::ALIVE;
}}
"""

CUDA_AGENT_MOVE_RESPONSE_CONDITION_NAME: str = "move_response_condition"
CUDA_AGENT_MOVE_RESPONSE_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_MOVE_RESPONSE_CONDITION_NAME}) {{
    return FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_MOVING};
}}
"""

CUDA_AGENT_MOVE_RESPONSE_FUNCTION_NAME: str = "move_response"
CUDA_AGENT_MOVE_RESPONSE_FUNCTION: str = rf"""
{CUDA_POS_FROM_MOORE_SEQ_FUNCTION}
{CUDA_POS_TO_BUCKET_ID_FUNCTION}
// getting here means that there are no neighbours, so, free movement
FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_MOVE_RESPONSE_FUNCTION_NAME}, flamegpu::MessageBucket, flamegpu::MessageNone) {{
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const flamegpu::id_t my_id = FLAMEGPU->getID();

    const unsigned int request_bucket = FLAMEGPU->getVariable<unsigned int>("request_bucket");

    flamegpu::id_t requester_id;
    float requester_roll;
    
    unsigned int requested_x;
    unsigned int requested_y;
    unsigned int wrap_x;
    unsigned int wrap_y;
    unsigned int neighbour_bucket;

    flamegpu::id_t highest_roller_id = flamegpu::ID_NOT_SET;
    float highest_roll = 0.0;

    // we MUST loop around to keep track of all claimed spaces
    for (unsigned int i = 0; i < {SPACES_WITHIN_RADIUS}; ++i) {{
      {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, i, env_max, wrap_x, wrap_y);
      neighbour_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(wrap_x, wrap_y, env_max);

      
      requester_id = flamegpu::ID_NOT_SET;
      requester_roll = 0.0;
      for (const auto& message : FLAMEGPU->message_in(neighbour_bucket)) {{
          requester_id = message.getVariable<flamegpu::id_t>("requester_id");
          
          if (requester_id == flamegpu::ID_NOT_SET) {{
              continue;
          }}

          // if it's not where we want to move, then we just update if we have a new neighbour.
          if (neighbour_bucket != request_bucket) {{
              FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", i, requester_id);
              break;
          }}
          
          // this is our target space, so let's see if we can claim it
          requester_roll = message.getVariable<float>("requester_roll");
          if (requester_roll > highest_roll || (requester_roll == highest_roll && requester_id > highest_roller_id)) {{
              highest_roll = requester_roll;
              highest_roller_id = requester_id;
              requested_x = message.getVariable<unsigned int>("requested_x");
              requested_y = message.getVariable<unsigned int>("requested_y");
              FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", i, requester_id);
          }}
          // otherwise the die roll is lower and they have no claim.
      }}
    }}
    // if the space isn't ours, we have to try again
    if (highest_roller_id != my_id) {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_MOVEMENT_UNRESOLVED});
        return flamegpu::ALIVE;
    }}

    // set location to new x, y
    FLAMEGPU->setVariable<unsigned int>("x_a", requested_x);
    FLAMEGPU->setVariable<unsigned int>("y_a", requested_y);

    // also update visualisation float values if required
    if({CUDA_AGENT_MOVE_UPDATE_VIZ}) {{
      FLAMEGPU->setVariable<float>("x", (float) requested_x);
      FLAMEGPU->setVariable<float>("y", (float) requested_y);
    }}
    // update message bucket to new grid space
    FLAMEGPU->setVariable<unsigned int>("my_bucket", request_bucket);
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_READY});

    return flamegpu::ALIVE;
}}
"""

# everyone broadcasts
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

# only agents that can reproduce care about their neibours now 
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
    unsigned int neighbour_x;
    unsigned int neighbour_y;
    flamegpu::id_t neighbour_id;
    unsigned int neighbour_bucket;

    for (unsigned int i = 0; i < {SPACES_WITHIN_RADIUS}; ++i) {{
        {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, i, env_max, neighbour_x, neighbour_y);
        neighbour_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(neighbour_x, neighbour_y, env_max);
        // reset neighbour info.
        neighbour_id = flamegpu::ID_NOT_SET;
        
        // if there's a message, then we have a neighbour.
        for (const auto& message : FLAMEGPU->message_in(neighbour_bucket)) {{
            
            // but we only know for sure, if the neighbour has an ID
            neighbour_id = message.getVariable<flamegpu::id_t>("id");
            // if the neighbour has an ID, then the space is occupied
            if (neighbour_id != flamegpu::ID_NOT_SET) {{
              ++num_neighbours;
              break;
            }}
        }}

        // if no message was found, it will default to ID_NOT_SET, otherwise the ID from the message
        FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", i, neighbour_id);
    }}

    // if there is at least one space available, then we can reproduce.
    if (num_neighbours < {SPACES_WITHIN_RADIUS}) {{
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_ATTEMPTING_REPRODUCTION});
        return flamegpu::ALIVE;
    }}
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_IMPOSSIBLE});
    return flamegpu::ALIVE;
}}
"""
# @TODO: update this, can we refactor the need for agent couns per step?
CUDA_AGENT_GOD_GO_FORTH_CONDITION_NAME: str = "god_go_forth_condition"
CUDA_AGENT_GOD_GO_FORTH_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_GOD_GO_FORTH_CONDITION_NAME}) {{
    const uint8_t overpopuated = FLAMEGPU->environment.getProperty<uint8_t>("overpopulated");
    return overpopuated < 1 && FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_ATTEMPTING_REPRODUCTION};
}}
"""


CUDA_AGENT_GOD_GO_FORTH_FUNCTION_NAME: str = "god_go_forth"
CUDA_AGENT_GOD_GO_FORTH_FUNCTION: str = rf"""
{CUDA_POS_FROM_MOORE_SEQ_FUNCTION}
{CUDA_POS_TO_BUCKET_ID_FUNCTION}
{CUDA_SEQ_TO_ANGLE_FUNCTION}

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
    
    unsigned int reproduce_sequence = FLAMEGPU->getVariable<unsigned int>("reproduce_sequence");

    if (reproduce_sequence >= {SPACES_WITHIN_RADIUS}) {{
        FLAMEGPU->message_out.setKey(FLAMEGPU->environment.getProperty<unsigned int>("trash_bin"));
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_IMPOSSIBLE});
        return flamegpu::ALIVE;
    }}

    unsigned int last_reproduction_attempt = FLAMEGPU->getVariable<unsigned int>("last_reproduction_attempt");
    // try to limit the need for calling random.
    if (last_reproduction_attempt > {SPACES_WITHIN_RADIUS}) {{
      float die_roll = FLAMEGPU->random.uniform<float>();
      FLAMEGPU->setVariable<float>("die_roll", die_roll);
      // this will give us 0 to 7
      last_reproduction_attempt = FLAMEGPU->random.uniform<unsigned int>(0, {SPACES_WITHIN_RADIUS_ZERO_INDEXED});
    }}

    // set default outside of grid to check
    unsigned int new_x = env_max + 1;
    unsigned int new_y = env_max + 1;

    bool space_is_free = false;

    for (unsigned int i = 0; i < {SPACES_WITHIN_RADIUS}; ++i) {{
      ++reproduce_sequence;
      if (reproduce_sequence > {SPACES_WITHIN_RADIUS}) {{
          break;
      }}
      last_reproduction_attempt = (last_reproduction_attempt + i) % {SPACES_WITHIN_RADIUS};
      space_is_free = (FLAMEGPU->getVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", last_reproduction_attempt) == flamegpu::ID_NOT_SET);
      
      if(space_is_free) {{
        // get the new x,y location.
        {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, last_reproduction_attempt, env_max, new_x, new_y);
        break;
      }}
    }}
    
    // check if we found a free space
    if (new_x > env_max || new_y > env_max) {{
        FLAMEGPU->message_out.setKey(FLAMEGPU->environment.getProperty<unsigned int>("trash_bin"));
        FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_IMPOSSIBLE});
        return flamegpu::ALIVE;
    }}

    last_reproduction_attempt = (last_reproduction_attempt + 1) % {SPACES_WITHIN_RADIUS};
    FLAMEGPU->setVariable<unsigned int>("reproduce_sequence", reproduce_sequence);

    // we have a free space so attempt to reproduce
    if ({CUDA_AGENT_MOVE_UPDATE_VIZ} && {CUDA_ORIENT_AGENTS}) {{
      FLAMEGPU->setVariable<float>("pitch", {CUDA_SEQ_TO_ANGLE_FUNCTION_NAME}(last_reproduction_attempt - 1));
    }}

    // set me as attempting reproduction
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_ATTEMPTING_REPRODUCTION});

    FLAMEGPU->setVariable<unsigned int>("last_reproduction_attempt", last_reproduction_attempt);

    const unsigned int request_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(new_x, new_y, env_max);
    FLAMEGPU->setVariable<unsigned int>("request_bucket", request_bucket);

    FLAMEGPU->message_out.setKey(request_bucket);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", my_id);
    FLAMEGPU->message_out.setVariable<float>("die_roll", FLAMEGPU->getVariable<float>("die_roll"));
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_x", new_x);
    FLAMEGPU->message_out.setVariable<unsigned int>("requested_y", new_y);
    
    return flamegpu::ALIVE;
}}
"""
CUDA_AGENT_GOD_MULTIPLY_CONDITION_NAME: str = "god_multiply_condition"
CUDA_AGENT_GOD_MULTIPLY_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_AGENT_GOD_MULTIPLY_CONDITION_NAME}) {{
    const uint8_t overpopuated = FLAMEGPU->environment.getProperty<uint8_t>("overpopulated");
    return overpopuated < 1 && FLAMEGPU->getVariable<unsigned int>("agent_status") == {AGENT_STATUS_ATTEMPTING_REPRODUCTION};
}}
"""

STRAT_PER_TRAIT = "true" if AGENT_STRATEGY_PER_TRAIT else "false"
CUDA_AGENT_GOD_MULTIPLY_FUNCTION_NAME: str = "god_multiply"
CUDA_AGENT_GOD_MULTIPLY_FUNCTION: str = rf"""
{CUDA_POS_FROM_MOORE_SEQ_FUNCTION}
{CUDA_POS_TO_BUCKET_ID_FUNCTION}

FLAMEGPU_AGENT_FUNCTION({CUDA_AGENT_GOD_MULTIPLY_FUNCTION_NAME}, flamegpu::MessageBucket, flamegpu::MessageNone) {{
    const unsigned int env_max = FLAMEGPU->environment.getProperty<unsigned int>("env_max");
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int my_id = FLAMEGPU->getID();

    const unsigned int request_bucket = FLAMEGPU->getVariable<unsigned int>("request_bucket");

    flamegpu::id_t requester_id;
    float requester_roll;

    unsigned int requested_x;
    unsigned int requested_y;
    unsigned int wrap_x;
    unsigned int wrap_y;
    unsigned int neighbour_bucket;
    
    flamegpu::id_t highest_roller_id = flamegpu::ID_NOT_SET;
    float highest_roll = 0.0;
    
    // we MUST loop around to keep track of all claimed spaces
    for (unsigned int i = 0; i < {SPACES_WITHIN_RADIUS}; ++i) {{
      {CUDA_POS_FROM_MOORE_SEQ_FUNCTION_NAME}(my_x, my_y, i, env_max, wrap_x, wrap_y);
      neighbour_bucket = {CUDA_POS_TO_BUCKET_ID_FUNCTION_NAME}(wrap_x, wrap_y, env_max);

      requester_id = flamegpu::ID_NOT_SET;
      requester_roll = 0.0;
      for (const auto& message : FLAMEGPU->message_in(neighbour_bucket)) {{
          requester_id = message.getVariable<flamegpu::id_t>("id");

          if (requester_id == flamegpu::ID_NOT_SET) {{
              continue;
          }}

          // if it's not where we want to move, then we just update if we have a new neighbour.
          if (neighbour_bucket != request_bucket) {{
              // if it is an agent, then we are safe to update the neibour list
              FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", i, requester_id);
              break;
          }}

          // this is our target space, so let's see if we can claim it
          requester_roll = message.getVariable<float>("die_roll");
          if (requester_roll > highest_roll || (requester_roll == highest_roll && requester_id > highest_roller_id)) {{
              highest_roll = requester_roll;
              highest_roller_id = requester_id;
              requested_x = message.getVariable<unsigned int>("requested_x");
              requested_y = message.getVariable<unsigned int>("requested_y");
              FLAMEGPU->setVariable<flamegpu::id_t, {SPACES_WITHIN_RADIUS}>("neighbour_list", i, requester_id);
          }}
          // otherwise the die roll is lower and they have no claim.
      }}
    }}
    // if it's not me, then we can't reproduce
    if (highest_roller_id != my_id) {{
        return flamegpu::ALIVE;
    }}

    // deduct reproduction cost
    float my_energy = FLAMEGPU->getVariable<float>("energy");
    const float reproduce_cost = FLAMEGPU->environment.getProperty<float>("reproduce_cost");
    const float reproduction_inheritence = FLAMEGPU->environment.getProperty<float>("reproduction_inheritence");
    my_energy -= reproduce_cost;
    FLAMEGPU->setVariable<float>("energy", my_energy);
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_COMPLETE});

    //set child location
    FLAMEGPU->agent_out.setVariable<unsigned int>("x_a", requested_x);
    FLAMEGPU->agent_out.setVariable<unsigned int>("y_a", requested_y);

    if ({CUDA_AGENT_MOVE_UPDATE_VIZ}) {{
      FLAMEGPU->agent_out.setVariable<float>("x", (float)requested_x);
      FLAMEGPU->agent_out.setVariable<float>("y", (float)requested_y);
    }}

    const uint8_t my_trait = FLAMEGPU->getVariable<uint8_t>("agent_trait");
    FLAMEGPU->agent_out.setVariable<uint8_t>("agent_trait", my_trait);
    FLAMEGPU->agent_out.setVariable<unsigned int>("agent_color", my_trait);
    
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
    // update message bucket of the child to new agent's grid space
    FLAMEGPU->agent_out.setVariable<unsigned int>("my_bucket", request_bucket);

    const float mutation_rate = FLAMEGPU->environment.getProperty<float>("mutation_rate");
    uint8_t my_strat;
    uint8_t child_strat;
    float mutation_roll;
    // @TODO: refactor, this is GROSSSSSSS HACK
    if (FLAMEGPU->environment.getProperty<uint8_t>("strategy_pure") == 1) {{
        my_strat = FLAMEGPU->getVariable<uint8_t, {AGENT_TRAIT_COUNT}>("agent_strategies", 0);
        child_strat = my_strat;
        if (mutation_rate > 0.0) {{
            mutation_roll = FLAMEGPU->random.uniform<float>();
            while (child_strat == my_strat) {{
                child_strat = FLAMEGPU->random.uniform<int>(0, {AGENT_STRATEGY_COUNT} - 1);
            }}
        }}
        for (int i = 0; i < {AGENT_TRAIT_COUNT}; ++i) {{
            FLAMEGPU->agent_out.setVariable<uint8_t, {AGENT_TRAIT_COUNT}>("agent_strategies", i, child_strat);
        }}
        FLAMEGPU->agent_out.setVariable<uint8_t>("agent_strategy_id", (child_strat * 10) + child_strat);

    }} else if (FLAMEGPU->environment.getProperty<uint8_t>("strategy_per_trait") == 1) {{
        for (int i = 0; i < {AGENT_TRAIT_COUNT}; ++i) {{
            my_strat = FLAMEGPU->getVariable<uint8_t, {AGENT_TRAIT_COUNT}>("agent_strategies", i);
            child_strat = my_strat;
            if (mutation_rate > 0.0) {{
                mutation_roll = FLAMEGPU->random.uniform<float>();
                if(mutation_roll < mutation_rate) {{
                    while (child_strat == my_strat) {{
                        child_strat = FLAMEGPU->random.uniform<int>(0, {AGENT_STRATEGY_COUNT} - 1);
                    }}
                }}
            }}
            FLAMEGPU->agent_out.setVariable<uint8_t, {AGENT_TRAIT_COUNT}>("agent_strategies", i, child_strat);
        }}
    }} else {{
        float mutation_roll_other = 1.0;
        mutation_roll = 1.0;
        if (mutation_rate > 0.0) {{
            mutation_roll = FLAMEGPU->random.uniform<float>();
            mutation_roll_other = FLAMEGPU->random.uniform<float>();
        }}
        uint8_t child_strat_other = {AGENT_STRATEGY_COUNT} + 1;
        uint8_t child_strat_my = {AGENT_STRATEGY_COUNT} + 1;
        for (int i = 0; i < {AGENT_TRAIT_COUNT}; ++i) {{
            my_strat = FLAMEGPU->getVariable<uint8_t, {AGENT_TRAIT_COUNT}>("agent_strategies", i);
            child_strat = my_strat;
            if (i == my_trait) {{
                if (child_strat_my > {AGENT_STRATEGY_COUNT}) {{
                    if (mutation_roll < mutation_rate) {{
                        while (child_strat == my_strat) {{
                            child_strat = FLAMEGPU->random.uniform<int>(0, {AGENT_STRATEGY_COUNT} - 1);
                        }}
                    }}
                    child_strat_my = child_strat;
                }} else {{
                    child_strat = child_strat_my;
                }}
            }} else if (i != my_trait) {{
                if (child_strat_other > {AGENT_STRATEGY_COUNT}) {{
                    if (mutation_roll_other < mutation_rate) {{
                        while (child_strat == my_strat) {{
                            child_strat = FLAMEGPU->random.uniform<int>(0, {AGENT_STRATEGY_COUNT} - 1);
                        }}
                    }}
                    child_strat_other = child_strat;
                }} else {{
                    child_strat = child_strat_other;
                }}
            }}
            
            FLAMEGPU->agent_out.setVariable<uint8_t, {AGENT_TRAIT_COUNT}>("agent_strategies", i, child_strat);
        }}
        FLAMEGPU->agent_out.setVariable<uint8_t>("agent_strategy_id", (child_strat_my * 10) + child_strat_other);
    }}

    FLAMEGPU->agent_out.setVariable<unsigned int>("agent_status", {AGENT_STATUS_NEW_AGENT});
    uint8_t agents_spawned = FLAMEGPU->getVariable<uint8_t>("agents_spawned");
    FLAMEGPU->setVariable<uint8_t>("agents_spawned", ++agents_spawned);
    FLAMEGPU->setVariable<unsigned int>("agent_status", {AGENT_STATUS_REPRODUCTION_COMPLETE});
    
    return flamegpu::ALIVE;
}}
"""


# @TODO: change to it's own layer
CUDA_ENVIRONMENTAL_PUNISHMENT_CONDITION_NAME: str = "environmental_punishment_condition"
CUDA_ENVIRONMENTAL_PUNISHMENT_CONDITION: str = rf"""
FLAMEGPU_AGENT_FUNCTION_CONDITION({CUDA_ENVIRONMENTAL_PUNISHMENT_CONDITION_NAME}) {{
    const unsigned int max_agents = FLAMEGPU->environment.getProperty<unsigned int>("max_agents");
    return FLAMEGPU->getVariable<unsigned int>("agent_status") != {AGENT_STATUS_NEW_AGENT} || FLAMEGPU->getThreadIndex() >= max_agents;
}}
"""
CUDA_ENVIRONMENTAL_PUNISHMENT_NAME: str = "environmental_punishment"
CUDA_ENVIRONMENTAL_PUNISHMENT_FUNCTION: str = rf"""
FLAMEGPU_AGENT_FUNCTION({CUDA_ENVIRONMENTAL_PUNISHMENT_NAME}, flamegpu::MessageNone, flamegpu::MessageNone) {{
    // begin the cull
    const unsigned int max_agents = FLAMEGPU->environment.getProperty<unsigned int>("max_agents");
    
    // @TODO FUCK THIS CODE OFF
    const unsigned int agent_status = FLAMEGPU->getVariable<unsigned int>("agent_status");
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
    n_ready_to_challenge: int = prisoner.countUInt(
        "agent_status", AGENT_STATUS_READY_TO_CHALLENGE)
    n_ready_to_respond: int = prisoner.countUInt(
        "agent_status", AGENT_STATUS_READY_TO_RESPOND)
    n_play_completed: int = prisoner.countUInt(
        "agent_status", AGENT_STATUS_PLAY_COMPLETED)
    n_moving: int = prisoner.countUInt("agent_status", AGENT_STATUS_MOVING)
    n_move_unresolved: int = prisoner.countUInt(
        "agent_status", AGENT_STATUS_MOVEMENT_UNRESOLVED)
    n_move_completed: int = prisoner.countUInt(
        "agent_status", AGENT_STATUS_MOVEMENT_COMPLETED)
    print(f"n_ready: {n_ready}, n_ready_to_challenge: {n_ready_to_challenge}, n_ready_to_respond: {n_ready_to_respond} n_play_completed: {n_play_completed}, n_moving: {n_moving}, n_move_unresolved: {n_move_unresolved}, n_move_completed: {n_move_completed}")


def _update_agent_count(FLAMEGPU, prisoner: pyflamegpu.HostAgentAPI) -> int:
    agent_count = prisoner.count()
    overpopulated = 1 if agent_count > AGENT_HARD_LIMIT else 0
    FLAMEGPU.environment.setPropertyUInt("agent_count", agent_count)
    FLAMEGPU.environment.setPropertyUInt8("overpopulated", overpopulated)
    return overpopulated

class step_fn(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
        if WRITE_LOG:
            prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
            k = 0
            for i in range(0, 40, 10):
                for j in range(0, 4):
                    strat_count = prisoner.countUInt8("agent_strategy_id", i + j)
                    FLAMEGPU.environment.setPropertyUInt("population_strat_count", k, strat_count)
                    k += 1
            



# set up population
class init_fn(pyflamegpu.HostFunctionCallback):
    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
        agent_strat_per_trait = FLAMEGPU.environment.getPropertyUInt8("strategy_per_trait")
        agent_strat_pure = FLAMEGPU.environment.getPropertyUInt8("strategy_pure")

        # FLAMEGPU.environment.setPropertyUInt("agent_count", INIT_AGENT_COUNT)
        agent: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
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
        instance: pyflamegpu.AgentInstance
        for i in range(INIT_AGENT_COUNT):  # type: ignore
            instance = agent.newAgent()
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
                if VISUALISATION_ORIENT_AGENTS:
                    instance.setVariableFloat("pitch", 0.0)
                
            energy = max(random.normalvariate(INIT_ENERGY_MU,
                         INIT_ENERGY_SIGMA), INIT_ENERGY_MIN)
            if MAX_ENERGY > 0.0:
                energy = min(energy, MAX_ENERGY)
            instance.setVariableFloat("energy", energy)
            # select agent strategy
            agent_trait: int = random.choice(AGENT_TRAITS)
            instance.setVariableUInt8("agent_trait", agent_trait)
            # this could be based on strategy, or change during runtime!
            instance.setVariableUInt("agent_color", agent_trait)
            
            # select agent strategy
            if agent_strat_pure == 1:
                agent_strategies = random.choices(AGENT_STRATEGY_IDS, weights=AGENT_WEIGHTS, k=1)*AGENT_TRAIT_COUNT
            elif agent_strat_per_trait == 1:
                # if we are using a per-trait strategy, then pick random weighted strategies
                agent_strategies = random.choices(AGENT_STRATEGY_IDS, weights=AGENT_WEIGHTS, k=AGENT_TRAIT_COUNT)
            else:
                # otherwise, we need a strategy for agents with matching traits
                # and a second for agents with different traits
                strategy_my: int
                strategy_other: int
                strategy_my, strategy_other = random.choices(
                    AGENT_STRATEGY_IDS, weights=AGENT_WEIGHTS, k=2)
                agent_strategies: List[int] = []
                trait: int
                for i, trait in enumerate(AGENT_TRAITS):
                    if trait == agent_trait:
                        agent_strategies.append(strategy_my)
                    else:
                        agent_strategies.append(strategy_other)
            instance.setVariableArrayUInt8('agent_strategies', agent_strategies)
            strat_my: int = -1
            strat_other: int = -1
            for i, strat in enumerate(agent_strategies):
                # update population counts of strategies
                if strat_other < 0 and agent_trait != i:
                    strat_other = strat
                if strat_my < 0 and agent_trait == i:
                    strat_my = strat
                if strat_my >= 0 and strat_other >= 0:
                    break
            # convert "base 4" to base 10 for indexing
            strategy_id = (strat_my * 10) + strat_other
            idx = (strat_my * 4) + strat_other
            # print(strat_my, strat_other, idx)
            # print(type(idx))
            #agent_pop_counts[idx] = int(agent_pop_counts[idx]) + 1
            instance.setVariableUInt8('agent_strategy_id', strategy_id)
            
            strat_count = FLAMEGPU.environment.getPropertyUInt("population_strat_count", idx)
            FLAMEGPU.environment.setPropertyUInt("population_strat_count", idx, strat_count + 1)
            
        del grid, np


class exit_play_fn(pyflamegpu.HostFunctionConditionCallback):
    iterations: int = 0
    max_iterations: int = SPACES_WITHIN_RADIUS

    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
        # print("play")
        self.iterations += 1
        prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
        if self.iterations < self.max_iterations:
            if VERBOSE_OUTPUT:
                if FLAMEGPU.getStepCounter() % OUTPUT_EVERY_N_STEPS == 0:
                    print("ready: ", prisoner.countUInt("agent_status", AGENT_STATUS_READY),
                          "ready_respond: ", prisoner.countUInt("agent_status", AGENT_STATUS_READY_TO_RESPOND))
            #print(prisoner.countUInt("agent_status", AGENT_STATUS_SKIP_RESPONSE))
            if prisoner.countUInt("agent_status", AGENT_STATUS_READY_TO_CHALLENGE) or prisoner.countUInt("agent_status", AGENT_STATUS_READY_TO_RESPOND):
                return pyflamegpu.CONTINUE
        self.iterations = 0
        _update_agent_count(FLAMEGPU, prisoner)
        return pyflamegpu.EXIT


class exit_move_fn(pyflamegpu.HostFunctionConditionCallback):
    iterations: int = 0
    max_iterations: int = SPACES_WITHIN_RADIUS

    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
        # print("move")
        self.iterations += 1
        prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
        if self.iterations < self.max_iterations:
            # Agent movements still unresolved
            if prisoner.countUInt("agent_status", AGENT_STATUS_MOVEMENT_UNRESOLVED):
                return pyflamegpu.CONTINUE
        _update_agent_count(FLAMEGPU, prisoner)
        self.iterations = 0
        return pyflamegpu.EXIT

class exit_condition_fn(pyflamegpu.HostFunctionConditionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
        prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
        n_agents = prisoner.count()
        if n_agents <= 0:
            return pyflamegpu.EXIT
        return pyflamegpu.CONTINUE

class exit_neighbourhood_fn(pyflamegpu.HostFunctionConditionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
        return pyflamegpu.EXIT


class init_god_fn(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
        pass


class exit_god_fn(pyflamegpu.HostFunctionConditionCallback):
    iterations: int = 0
    max_iterations: int = SPACES_WITHIN_RADIUS

    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU: pyflamegpu.HostAPI):
        prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
        overpopulated = _update_agent_count(FLAMEGPU, prisoner)
        self.iterations += 1
        if self.iterations < self.max_iterations:
            prisoner: pyflamegpu.HostAgentAPI = FLAMEGPU.agent("prisoner")
            # print(prisoner.count())
            if prisoner.countUInt("agent_status", AGENT_STATUS_ATTEMPTING_REPRODUCTION) and overpopulated < 1:
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
    agent.newVariableUInt8("agent_trait")
    # this allows flexible setting of agent colors
    agent.newVariableUInt("agent_color")
    agent.newVariableArrayUInt8("agent_strategies", AGENT_TRAIT_COUNT)
    agent.newVariableUInt8("agent_strategy_id", 0)
    agent.newVariableArrayID("neighbour_list", SPACES_WITHIN_RADIUS, [
                             pyflamegpu.ID_NOT_SET] * SPACES_WITHIN_RADIUS)
    agent.newVariableArrayFloat("neighbour_rolls", SPACES_WITHIN_RADIUS, [
                                0.0] * SPACES_WITHIN_RADIUS)
    # @TODO: flesh out? for now -1 = no neighbour, 0 = I respond, 1 = I challenge
    agent.newVariableArrayInt8(
        "my_actions", SPACES_WITHIN_RADIUS, [-1] * SPACES_WITHIN_RADIUS)
    agent.newVariableFloat("die_roll", 0.0)  # type: ignore
    agent.newVariableUInt("my_bucket", 0)

    if USE_VISUALISATION:
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
        if VISUALISATION_ORIENT_AGENTS:
            agent.newVariableFloat("pitch")

    return agent


def add_agent_memory(agent: pyflamegpu.AgentDescription):
    agent.newVariableArrayID("game_memory", SPACES_WITHIN_RADIUS, [
                             pyflamegpu.ID_NOT_SET] * SPACES_WITHIN_RADIUS)
    agent.newVariableArrayUInt8("game_memory_choices", SPACES_WITHIN_RADIUS, [
                                0] * SPACES_WITHIN_RADIUS)


def add_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
    env.newPropertyUInt("env_max", ENV_MAX, isConst=True)
    env.newPropertyUInt("trash_bin", BUCKET_SIZE, isConst=True)
    env.newPropertyUInt("agent_count", 0)
    env.newPropertyUInt8("overpopulated", 0)
    env.newPropertyFloat("env_noise", ENV_NOISE, isConst=True)
    env.newPropertyUInt8("strategy_per_trait", 1 if AGENT_STRATEGY_PER_TRAIT else 0, isConst=True)
    env.newPropertyUInt8("strategy_pure", 1 if AGENT_STRATEGY_PURE else 0, isConst=True)


def add_pdgame_vars(agent: pyflamegpu.AgentDescription) -> None:
    # add variable for tracking which neighbour is the target
    agent.newVariableUInt8("challenge_sequence", 0)
    agent.newVariableUInt8("response_sequence", SPACES_WITHIN_RADIUS)
    agent.newVariableUInt8("round_resolved", 0)
    agent.newVariableUInt8("games_played", 0)
    add_agent_memory(agent)


def add_pdgame_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
    env.newPropertyFloat("payoff_cc", PAYOFF_CC, isConst=True)
    env.newPropertyFloat("payoff_cd", PAYOFF_CD, isConst=True)
    env.newPropertyFloat("payoff_dc", PAYOFF_DC, isConst=True)
    env.newPropertyFloat("payoff_dd", PAYOFF_DD, isConst=True)
    
    env.newPropertyFloat("max_energy", MAX_ENERGY, isConst=True)


def add_movement_vars(agent: pyflamegpu.AgentDescription) -> None:
    agent.newVariableUInt("last_move_attempt", SPACES_WITHIN_RADIUS_INCL)
    agent.newVariableUInt("request_bucket", 0)
    agent.newVariableUInt("move_sequence", 0)


def add_movement_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
    # env.newMacroPropertyUInt("move_requests", ENV_MAX, ENV_MAX)
    env.newPropertyFloat("travel_cost", AGENT_TRAVEL_COST, isConst=True)


def add_god_vars(agent: pyflamegpu.AgentDescription) -> None:
    agent.newVariableUInt("request_bucket", 0)
    agent.newVariableUInt("last_reproduction_attempt",
                          SPACES_WITHIN_RADIUS_INCL)
    agent.newVariableUInt("reproduce_sequence", 0)
    agent.newVariableUInt8("agents_spawned", 0)


def add_neighbourhood_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
    env.newPropertyFloat("reproduce_min_energy",
                         REPRODUCE_MIN_ENERGY, isConst=True)


def add_god_env_vars(env: pyflamegpu.EnvironmentDescription) -> None:
    env.newPropertyFloat("reproduce_cost", REPRODUCE_COST, isConst=True)
    env.newPropertyFloat("reproduce_min_energy",
                         REPRODUCE_MIN_ENERGY, isConst=True)
    env.newPropertyFloat("max_energy", MAX_ENERGY, isConst=True)
    env.newPropertyFloat("cost_of_living", COST_OF_LIVING, isConst=True)
    env.newPropertyFloat("init_energy_mu", INIT_ENERGY_MU, isConst=True)
    env.newPropertyFloat("init_energy_sigma", INIT_ENERGY_SIGMA, isConst=True)
    env.newPropertyFloat("init_energy_min", INIT_ENERGY_MIN, isConst=True)
    env.newPropertyFloat(
        "mutation_rate", AGENT_TRAIT_MUTATION_RATE, isConst=True)
    env.newPropertyFloat("reproduction_inheritence",
                         REPRODUCTION_INHERITENCE, isConst=True)
    env.newPropertyUInt8("max_children_per_step",
                         MAX_CHILDREN_PER_STEP, isConst=True)
    env.newPropertyUInt("max_agents", AGENT_HARD_LIMIT, isConst=True)


def _print_environment_properties() -> None:
    print(f"env_max (grid width): {ENV_MAX}")
    print(f"max agent count: {MAX_AGENT_SPACES}")
    print(f"random seed: {RANDOM_SEED}")

# Define a method which when called will define the model, Create the simulation object and execute it.
def configure_visualisation(simulation: pyflamegpu.CUDASimulation) -> pyflamegpu.ModelVis:
    visualisation: pyflamegpu.ModelVis = simulation.getVisualisation()
    visualisation.setBeginPaused(PAUSE_AT_START)
    # Configure the visualiastion.
    INIT_CAM = ENV_MAX / 2.0
    visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, ENV_MAX)
    visualisation.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0.0)
    visualisation.setCameraSpeed(VISUALISATION_CAMERA_SPEED)
    visualisation.setClearColor(*VISUALISATION_BG_RGB)
    
    visualisation.setSimulationSpeed(SIMULATION_SPS_LIMIT)

    vis_agent: pyflamegpu.AgentVis = visualisation.addAgent("prisoner")

    # Set the model to use, and scale it.
    vis_agent.setModel(AGENT_DEFAULT_SHAPE)
    vis_agent.setModelScale(AGENT_DEFAULT_SCALE)
    vis_agent.setColor(AGENT_COLOR_SCHEME)
    if VISUALISATION_ORIENT_AGENTS:
        vis_agent.setPitchVariable("pitch")

    # Activate the visualisation.
    return visualisation

def configure_simulation_single(model: pyflamegpu.ModelDescription, argv: list[str]) -> pyflamegpu.CUDASimulation:
    simulation: pyflamegpu.CUDASimulation = pyflamegpu.CUDASimulation(model)
    # set some simulation defaults
    if RANDOM_SEED is not None:
        simulation.SimulationConfig().random_seed = RANDOM_SEED
    simulation.SimulationConfig().steps = STEP_COUNT
    simulation.SimulationConfig().verbose = DEBUG_OUTPUT
    simulation.SimulationConfig().common_log_file = LOG_FILE
    # Initialise the simulation
    simulation.initialise(argv)
    # Generate a population if an initial states file is not provided
    if not simulation.SimulationConfig().input_file:
        # Seed the host RNG using the cuda simulations' RNG
        if RANDOM_SEED is not None:
            random.seed(simulation.SimulationConfig().random_seed)
    
    return simulation

def configure_ensemble(model: pyflamegpu.ModelDescription, argv: list[str]) -> pyflamegpu.CUDAEnsemble:
    ensemble: pyflamegpu.CUDAEnsemble = pyflamegpu.CUDAEnsemble(model)
    ensemble.Config().out_directory = "data"
    ensemble.Config().out_format = "json"

    ensemble.initialise(argv)
    
    return ensemble

def configure_runplan(model: pyflamegpu.ModelDescription) -> pyflamegpu.RunPlanVector:
    # How man initial runs for each
    runs_control: pyflamegpu.RunPlanVector = pyflamegpu.RunPlanVector(model, MULTI_RUN_COUNT)
    if RANDOM_SEED:
        # increment random seed by one each time
        runs_control.setRandomSimulationSeed(RANDOM_SEED, 1)
        # so all props for this run get the same seed if dists etc.
        runs_control.setRandomPropertySeed(RANDOM_SEED)
    runs_control.setSteps(MULTI_RUN_STEPS)
    runs: pyflamegpu.RunPlanVector = pyflamegpu.RunPlanVector(model, 0)
    # Initialise environment property 'lerp_float' with values uniformly distributed between 1 and 128
    # runs_control.setPropertyUniformDistributionFloat("lerp_float", 1.0, 128.0)
    for pure_stategy in [0, 1]:
        # [0, 0.1, 1, 2, 5]
        for cost_of_living in [0.1, 0.3, 1, 2/3, 1.5, 1.666]:
            runs_control.setOutputSubdirectory("pure%g_env_cost%g_%g_steps"%(pure_stategy, cost_of_living, MULTI_RUN_STEPS))
            runs_control.setPropertyUInt8("strategy_pure", pure_stategy)
            runs_control.setPropertyFloat("cost_of_living", cost_of_living)
            runs_control.setPropertyFloat("travel_cost", cost_of_living / 2)
            runs += runs_control
    return runs

def main():
    _print_environment_properties()
    if pyflamegpu.SEATBELTS:
        print("Seatbelts are enabled, this will significantly impact performance.")
        print("Buckle up if you are developing the model. Otherwise throw caution to the wind and use a pyflamegpu build without seatbelts.")
    # Define the FLAME GPU model
    model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription(
        "prisoners_dilemma")

    # Exit sim early if all agents die
    model.addExitConditionCallback(exit_condition_fn().__disown__())
    env: pyflamegpu.EnvironmentDescription = model.Environment()
    add_env_vars(env)
    env.newPropertyFloat("cost_of_living", COST_OF_LIVING, isConst=True)
    env.newPropertyUInt("max_agents", AGENT_HARD_LIMIT, isConst=True)
    env.newPropertyFloat("max_energy", MAX_ENERGY, isConst=True)
    #env.newPropertyArrayUInt("population_counts_step", [0] * POPULATION_COUNT_BINS)
    env.newPropertyArrayUInt("population_strat_count", [0] * POPULATION_COUNT_BINS)
    env.newPropertyFloat("travel_cost", AGENT_TRAVEL_COST, isConst=True)

    model.addStepFunctionCallback(step_fn().__disown__())
    # create all agents here
    model.addInitFunctionCallback(init_fn().__disown__())

    agent = make_core_agent(model)

    search_message: pyflamegpu.MessageBucket_Description = model.newMessageBucket(
        "player_search_msg")
    search_message.newVariableID("id")
    search_message.newVariableFloat("die_roll")
    search_message.setBounds(0, BUCKET_SIZE)

    agent_search_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(
        CUDA_SEARCH_FUNC_NAME, CUDA_SEARCH_FUNC)
    agent_search_fn.setMessageOutput("player_search_msg")

    agent_game_list_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(
        CUDA_GAME_LIST_FUNC_NAME, CUDA_GAME_LIST_FUNC)
    agent_game_list_fn.setMessageInput("player_search_msg")

    agent_environmental_punishment_fn: pyflamegpu.AgentFunctionDescription = agent.newRTCFunction(
        CUDA_ENVIRONMENTAL_PUNISHMENT_NAME, CUDA_ENVIRONMENTAL_PUNISHMENT_FUNCTION)
    agent_environmental_punishment_fn.setAllowAgentDeath(True)
    agent_environmental_punishment_fn.setRTCFunctionCondition(
        CUDA_ENVIRONMENTAL_PUNISHMENT_CONDITION)

    # load agent-specific interactions

    # play resolution submodel
    pdgame_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription(
        "pdgame_model")
    pdgame_model.addExitConditionCallback(exit_play_fn().__disown__())

    # add message for game challenges
    challenge_message: pyflamegpu.MessageBucket_Description = pdgame_model.newMessageBucket(
        "player_challenge_msg")
    challenge_message.newVariableID("challenger_id")
    challenge_message.newVariableID("responder_id")
    challenge_message.newVariableArrayUInt8(
        "challenger_strategies", AGENT_TRAIT_COUNT)
    challenge_message.newVariableUInt8("challenger_trait")
    challenge_message.newVariableFloat("challenger_energy")
    challenge_message.newVariableFloat("challenger_roll")
    challenge_message.newVariableUInt("challenger_x")
    challenge_message.newVariableUInt("challenger_y")
    challenge_message.newVariableUInt("challenger_bucket")
    
    challenge_message.newVariableID("challenger_game_memory_id")
    challenge_message.newVariableUInt8("challenger_game_memory_choice")
    challenge_message.setBounds(0, BUCKET_SIZE)

    resolve_message: pyflamegpu.MessageBucket_Description = pdgame_model.newMessageBucket(
        "play_resolve_msg")
    resolve_message.newVariableID("challenger_id")
    resolve_message.newVariableID("responder_id")
    resolve_message.newVariableFloat("challenger_energy")
    resolve_message.newVariableUInt8("challenger_strategy")
    resolve_message.newVariableUInt8("responder_response")
    resolve_message.setBounds(0, BUCKET_SIZE)

    pdgame_env: pyflamegpu.EnvironmentDescription = pdgame_model.Environment()
    add_env_vars(pdgame_env)
    add_pdgame_env_vars(pdgame_env)

    # create the submodel
    pdgame_submodel: pyflamegpu.SubModelDescription = model.newSubModel(
        "pdgame_model", pdgame_model)
    pdgame_subagent: pyflamegpu.AgentDescription = make_core_agent(
        pdgame_model)
    add_pdgame_vars(pdgame_subagent)

    agent_challenge_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(
        CUDA_AGENT_PLAY_CHALLENGE_FUNC_NAME, CUDA_AGENT_PLAY_CHALLENGE_FUNC)
    agent_challenge_fn.setMessageOutput("player_challenge_msg")
    agent_challenge_fn.setRTCFunctionCondition(
        CUDA_AGENT_PLAY_CHALLENGE_CONDITION)

    agent_response_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(
        CUDA_AGENT_PLAY_RESPONSE_FUNC_NAME, CUDA_AGENT_PLAY_RESPONSE_FUNC)
    agent_response_fn.setMessageInput("player_challenge_msg")
    agent_response_fn.setRTCFunctionCondition(
        CUDA_AGENT_PLAY_RESPONSE_CONDITION)
    agent_response_fn.setMessageOutput("play_resolve_msg")
    agent_response_fn.setAllowAgentDeath(True)

    agent_resolve_fn: pyflamegpu.AgentFunctionDescription = pdgame_subagent.newRTCFunction(
        CUDA_AGENT_PLAY_RESOLVE_FUNC_NAME, CUDA_AGENT_PLAY_RESOLVE_FUNC)
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
    movement_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription(
        "movement_model")
    movement_model.addExitConditionCallback(exit_move_fn().__disown__())

    move_request_msg: pyflamegpu.MessageBucket_Description = movement_model.newMessageBucket(
        "agent_move_request_msg")
    move_request_msg.newVariableID("requester_id")
    move_request_msg.newVariableFloat("requester_roll")
    move_request_msg.newVariableUInt("requested_x")
    move_request_msg.newVariableUInt("requested_y")

    move_request_msg.setBounds(0, BUCKET_SIZE)

    movement_env: pyflamegpu.EnvironmentDescription = movement_model.Environment()
    add_env_vars(movement_env)
    add_movement_env_vars(movement_env)

    movement_submodel: pyflamegpu.SubModelDescription = model.newSubModel(
        "movement_model", movement_model)
    movement_subagent: pyflamegpu.AgentDescription = make_core_agent(
        movement_model)
    add_movement_vars(movement_subagent)

    agent_move_request_fn: pyflamegpu.AgentFunctionDescription = movement_subagent.newRTCFunction(
        CUDA_AGENT_MOVE_REQUEST_FUNCTION_NAME, CUDA_AGENT_MOVE_REQUEST_FUNCTION)
    agent_move_request_fn.setMessageOutput("agent_move_request_msg")
    agent_move_request_fn.setRTCFunctionCondition(
        CUDA_AGENT_MOVE_REQUEST_CONDITION)
    # Agents can die if they should travel, but don't have enough energy to do so
    agent_move_request_fn.setAllowAgentDeath(True)

    agent_move_response_fn: pyflamegpu.AgentFunctionDescription = movement_subagent.newRTCFunction(
        CUDA_AGENT_MOVE_RESPONSE_FUNCTION_NAME, CUDA_AGENT_MOVE_RESPONSE_FUNCTION)
    agent_move_response_fn.setMessageInput("agent_move_request_msg")
    agent_move_response_fn.setRTCFunctionCondition(
        CUDA_AGENT_MOVE_RESPONSE_CONDITION)

    movement_submodel.bindAgent("prisoner", "prisoner", auto_map_vars=True)

    movement_submodel_layer1: pyflamegpu.LayerDescription = movement_model.newLayer()
    movement_submodel_layer1.addAgentFunction(agent_move_request_fn)

    movement_submodel_layer2: pyflamegpu.LayerDescription = movement_model.newLayer()
    movement_submodel_layer2.addAgentFunction(agent_move_response_fn)

    # update neighbours submodel
    neighbourhood_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription(
        "neighbourhood_model")
    neighbourhood_model.addExitConditionCallback(
        exit_neighbourhood_fn().__disown__())

    neighbourhood_broadcast_msg: pyflamegpu.MessageBucket_Description = neighbourhood_model.newMessageBucket(
        "neighbourhood_broadcast_msg")
    neighbourhood_broadcast_msg.newVariableID("id")
    neighbourhood_broadcast_msg.setBounds(0, BUCKET_SIZE)

    neighbourhood_env: pyflamegpu.EnvironmentDescription = neighbourhood_model.Environment()
    add_env_vars(neighbourhood_env)
    add_neighbourhood_env_vars(neighbourhood_env)
    neighbourhood_submodel: pyflamegpu.SubModelDescription = model.newSubModel(
        "neighbourhood_model", neighbourhood_model)
    neighbourhood_subagent: pyflamegpu.AgentDescription = make_core_agent(
        neighbourhood_model)

    agent_neighbourhood_broadcast_fn: pyflamegpu.AgentFunctionDescription = neighbourhood_subagent.newRTCFunction(
        CUDA_AGENT_NEIGHBOURHOOD_BROADCAST_FUNCTION_NAME, CUDA_AGENT_NEIGHBOURHOOD_BROADCAST_FUNCTION)
    agent_neighbourhood_broadcast_fn.setMessageOutput(
        "neighbourhood_broadcast_msg")

    agent_neighbourhood_update_fn: pyflamegpu.AgentFunctionDescription = neighbourhood_subagent.newRTCFunction(
        CUDA_AGENT_NEIGHBOURHOOD_UPDATE_FUNCTION_NAME, CUDA_AGENT_NEIGHBOURHOOD_UPDATE_FUNCTION)
    agent_neighbourhood_update_fn.setMessageInput(
        "neighbourhood_broadcast_msg")
    # only need to update agents who could reproduce, hence the condition
    agent_neighbourhood_update_fn.setRTCFunctionCondition(
        CUDA_AGENT_NEIGHBOURHOOD_UPDATE_CONDITION)

    neighbourhood_submodel.bindAgent(
        "prisoner", "prisoner", auto_map_vars=True)

    neighbourhood_submodel_layer1: pyflamegpu.LayerDescription = neighbourhood_model.newLayer()
    neighbourhood_submodel_layer1.addAgentFunction(
        agent_neighbourhood_broadcast_fn)

    neighbourhood_submodel_layer2: pyflamegpu.LayerDescription = neighbourhood_model.newLayer()
    neighbourhood_submodel_layer2.addAgentFunction(
        agent_neighbourhood_update_fn)

    # god submodel, asexual reproduction, and environmental slaughter
    god_model: pyflamegpu.ModelDescription = pyflamegpu.ModelDescription(
        "god_model")
    # only attempt reproduction if there are agents with the right status
    # AND the current count of agents is less than the maximum
    god_model.addInitFunctionCallback(init_god_fn().__disown__())
    god_model.addExitConditionCallback(exit_god_fn().__disown__())

    god_go_forth_msg: pyflamegpu.MessageBucket_Description = god_model.newMessageBucket(
        "god_go_forth_msg")
    god_go_forth_msg.newVariableID("id")
    god_go_forth_msg.newVariableUInt("requested_x")
    god_go_forth_msg.newVariableUInt("requested_y")
    god_go_forth_msg.newVariableFloat("die_roll")
    god_go_forth_msg.setBounds(0, BUCKET_SIZE)

    god_env: pyflamegpu.EnvironmentDescription = god_model.Environment()
    add_env_vars(god_env)
    add_god_env_vars(god_env)
    god_submodel: pyflamegpu.SubModelDescription = model.newSubModel(
        "god_model", god_model)
    god_subagent: pyflamegpu.AgentDescription = make_core_agent(god_model)
    add_god_vars(god_subagent)

    agent_god_go_forth_fn: pyflamegpu.AgentFunctionDescription = god_subagent.newRTCFunction(
        CUDA_AGENT_GOD_GO_FORTH_FUNCTION_NAME, CUDA_AGENT_GOD_GO_FORTH_FUNCTION)

    agent_god_go_forth_fn.setMessageOutput("god_go_forth_msg")
    agent_god_go_forth_fn.setRTCFunctionCondition(
        CUDA_AGENT_GOD_GO_FORTH_CONDITION)

    agent_god_multiply_fn: pyflamegpu.AgentFunctionDescription = god_subagent.newRTCFunction(
        CUDA_AGENT_GOD_MULTIPLY_FUNCTION_NAME, CUDA_AGENT_GOD_MULTIPLY_FUNCTION)
    agent_god_multiply_fn.setMessageInput("god_go_forth_msg")
    agent_god_multiply_fn.setRTCFunctionCondition(
        CUDA_AGENT_GOD_MULTIPLY_CONDITION)
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

    # Layer #6: Delete agents over hard limit, deduct environmental cost
    main_layer7: pyflamegpu.LayerDescription = model.newLayer()
    main_layer7.addAgentFunction(agent_environmental_punishment_fn)

    if not MULTI_RUN:
        print("Configuring simulation...")
        simulation = configure_simulation_single(model, sys.argv)
        print("Configuring logging...")
        step_log_cfg = configure_logging(model)
        simulation.setStepLog(step_log_cfg)
        if USE_VISUALISATION:
            print("Configuring visualisation...")
            visualisation = configure_visualisation(simulation)
            visualisation.activate()
        print("Running simulation...")
        simulation.simulate()
        if USE_VISUALISATION:
            visualisation.join() #type: ignore
    else:
        print("Configuring CUDAEnsemble...")
        ensemble = configure_ensemble(model, sys.argv)
        print("Configuring logging...")
        step_log_cfg = configure_logging(model)
        ensemble.setStepLog(step_log_cfg)
        print("Configuring run plan...")
        runs = configure_runplan(model)
        print("Running simulation...")
        ensemble.simulate(runs)

if __name__ == "__main__":
    main()
