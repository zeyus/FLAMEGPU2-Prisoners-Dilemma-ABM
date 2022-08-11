# FLAMEGPU2 Based Prisoner's Dilemma ABM

## What is it?

A 2D ABM simulating executed on the GPU interactions of "games" between agents, specifically the prisoner's dilemma game, in which participants can either cooperate or defect, which has a payoff matrix.

Using [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2), agent counts can go into the millions without a problem, especially on a decent GPU.

## Details

One of the earlier descriptions of the game interactions:

<img width="326" alt="image" src="https://user-images.githubusercontent.com/75656/184105191-1f7af765-add8-4161-9998-062c39f65c18.png">

_Figure from Axelrod, R., & Hamilton, W. D. (1981). The Evolution of Cooperation. Science, 211(4489), 1390–1396. [https://doi.org/10.1126/science.7466396](https://doi.org/10.1126/science.7466396)_


In this setup the default payoff matrix for interactions is

|   | They Cooperate | They Defect |
|:---------:|:------------:|:---------:|
| I Cooperate | 3.0 | -1.0 |
| I Defect | 5.0 | 0.0 |

Agents play all of their neighbours and if they cannot, they move, and pay a cost of travel.

After play or movement each agent has the opportunity to reproduce if they have sufficient energy and an available space around them, if they reproduce, they pay the energy cost of reproduction.

Each round there is a cost of living imposed on all agents except agents that were just born in the reproduction phase.

Any agents that drop below zero energy die.

## Other features

- Agent strategy distributions can be configured (more than 4 strategies broken at the moment, but strategy probability can be set to 0.0)
- Agents have a random trait assigned to them (default 1 of 4 possible traits) (more than 4 traits broken at the moment)
- Strategy mutation can be configured at a specific mutation rate which applies during reproduction
- Agents can employ a global strategy (i.e. always cooperate, with any agent) or a strategy for agents with the same trait (kin) or others, OR (reporting broken for this but it works) a strategy per unique trait)
- Environmental noise can be configured for a chance of miscommunication (i.e. if i choose cooperate, it becomes defect)
- Can run in a CUDAEnsemble for a whole suite of simulation runs
- logging is configured for both single and multi runs, currently it collects the agent counts by their strategies, but it should also not bother doing any counts (for performance) if logging is disabled, which it still does


<img width="1238" alt="Prisoner's Dilemma ABM model flow" src="https://user-images.githubusercontent.com/75656/184108979-10fbb3d9-32f0-4610-9941-a67593097527.png">


![Screenshots from ABM simulation](https://user-images.githubusercontent.com/75656/184108676-8f6821eb-f792-484c-b4a8-ba02de789a1f.png)

ABM concepts for tags/traits adapted from:
- Hammond, R. A., & Axelrod, R. (2006). Evolution of contingent altruism when cooperation is expensive. Theoretical Population Biology, 69(3), 333–338. https://doi.org/10.1016/j.tpb.2005.12.002

Environmental pressure/cost of living concepts adapted from:
- Smaldino, P., Schank, J., & Mcelreath, R. (2013). Increased Costs of Cooperation Help Cooperators in the Long Run. The American Naturalist. https://doi.org/10.1086/669615
