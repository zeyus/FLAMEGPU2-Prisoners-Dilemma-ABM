FLAMEGPU_AGENT_FUNCTION(output, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<char>("lets_play", FLAMEGPU->getVariable<unsigned int>("lets_play"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int,>("x"), FLAMEGPU->getVariable<unsigned int>("y"));
    return flamegpu::ALIVE;
}