FLAMEGPU_AGENT_FUNCTION(search, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<unsigned int>("grid_index", FLAMEGPU->getVariable<unsigned int>("grid_index"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int>("x_a"), FLAMEGPU->getVariable<unsigned int>("y_a"));
    return flamegpu::ALIVE;
}