FLAMEGPU_AGENT_FUNCTION(interact, flamegpu::MessageArray2D, flamegpu::MessageNone) {
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
    auto playspace = FLAMEGPU->environment.getMacroProperty<unsigned int, 16384, 16384>("playspace");
    // iterate over all cells in the neighbourhood
    // this also wraps across env boundaries.
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y, interaction_radius)) {
        flamegpu::id_t local_competitor = message.getVariable<flamegpu::id_t>("id");
        unsigned int opponent_grid_index = message.getVariable<unsigned int>("grid_index");
        // play with the competitor
        if (playspace[opponent_grid_index][my_grid_index] == 0) {
            // we haven't played before
            playspace[my_grid_index][opponent_grid_index]++;
        } else {
            // we have played before, and this is the second of two possible interactions
            // do nothing except reset grid
            playspace[opponent_grid_index][my_grid_index]--;
        }

    }
    if (my_energy <= 0.0) {
        return flamegpu::DEAD;
    }
    if (my_energy > reproduction_threshold) {
        // spawn child in a free adjacent cell
        FLAMEGPU->setVariable<float>("energy", my_energy - reproduction_cost);
        return flamegpu::ALIVE;
    }
    return flamegpu::ALIVE;
}