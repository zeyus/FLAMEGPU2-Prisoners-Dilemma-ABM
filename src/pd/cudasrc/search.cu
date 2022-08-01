FLAMEGPU_AGENT_FUNCTION(search, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    const unsigned int my_grid_index = FLAMEGPU->getVariable<unsigned int>("grid_index");
    FLAMEGPU->message_out.setVariable<unsigned int>("grid_index", my_grid_index);
    FLAMEGPU->message_out.setVariable<float>("energy", FLAMEGPU->getVariable<float>("energy"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int>("x"), FLAMEGPU->getVariable<unsigned int>("y"));
    auto playspace = FLAMEGPU->environment.getMacroProperty<unsigned int, 16384, 16384>("playspace");
    // set playspace at my position to my current energy
    playspace[my_grid_index][my_grid_index] = FLAMEGPU->getVariable<float>("energy");
    return flamegpu::ALIVE;
}