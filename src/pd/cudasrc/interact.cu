FLAMEGPU_AGENT_FUNCTION(interact, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int>("x_a");
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int>("y_a");
    const unsigned int interaction_radius = FLAMEGPU->environment.getProperty<unsigned int>("max_play_distance");
    
    const float reproduction_threshold = FLAMEGPU->environment.getProperty<float>("reproduce_min_energy");
    const float reproduction_cost = FLAMEGPU->environment.getProperty<float>("reproduce_cost");

    float my_energy = FLAMEGPU->getVariable<float>("energy");
    
    // iterate over all cells in the neighbourhood
    // this also wraps across env boundaries.
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y, interaction_radius)) {
        flamegpu::id_t local_competitor = message.getVariable<flamegpu::id_t>("id");
        // play with the competitor

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