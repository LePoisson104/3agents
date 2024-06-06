# 4368-project

Terminal state: all three drop off cells contain 5 blocks each
Initial state: Black agent is in cell (1,3), red agent is in cell (3,3) and blue agent is in cell (3,5) and each pickup cell contains 5 blocks

pickup cells: (1,5), (2,4), (5,2)
dropoff cells: (1,1), (3,1), (4,5)

PD-World:

- states: north, south, east, west. leaving the grid is not allow
- pickup is only appicable if the agnt is in an pickup cell that contain at least one block and if the agent does not already carry a block.
- dropoff is only applicable if the agent is in a dropoff cell that contains less than 4 blocks and the agent carries a block
- moving of agents: red, blue, and black alternate applying operators with the red agent acting first, the blue second, and the black agent acting third. Agents are not allowed to occupy the same cell at the same time

Rewards in the PD-World:

- picking up a block from a pickup state: +13
- dropping off a block in a dropoff state: +13
- applying north, south east, west: -1

Experiment setup:

- episodes: 4000
- if a terminal state is reached the sytem is resset to the initial state, but Q-Tables are not reinitialized, and the threee agents continnue to opearte in the PD-World until the operator application limit has been reached

PRandom: If pickup and dropoff is applicable,
choose this operator; otherwise, choose an operator
randomly.

PExploit: If pickup and dropoff is applicable, choose this
operator; otherwise, apply the applicable operator with the
highest q-value (break ties by rolling a dice for operators  
 with the same utility) with probability 0.80 and choose a
different applicable operator randomly with probability
0.20.

PGreedy: If pickup and dropoff is applicable, choose this
operator; otherwise, apply the applicable operator with the
highest q-value (break ties by rolling a dice for operators
with the same utility).

There are two approaches to choose from to implement 3-agent reinforcement learning:

a. Each agent uses his own reinforcement learning strategy and Q-Table. However, the position the other agent occupies is visible to each agent, and can be part of the employed reinforcement learning state space.

b. A single reinforcement learning strategy and Q-Table is used which moves the three agents, selecting an operator for each agent and then executing the selected three operators in the order red-blue-black.

- pick one one of the two choices (a or b?)

Suggested Implementation steps:

- Write a function aplop: (i, j, i’, j’, x, x’, a, b, c, d, e, f) --> 2^{n,s,e,w,p,d} that returns the set of applicable operators in (i, j, i’, j’, x, x’, a, b, c, d, e, f)

- Write a function apply: (i, j, i’, j’, x, x’, a, b, c, d, e, f) x {n,s,e,w,p,d} --> (i’,j’,x’,a’,b’,c’,d’,e’,f’)

- Implement the q-table data structure
- Implement the SARSA/Q-Learning q-table update
- Implement the 3 policies
- Write functions that enable an agent to act according to a policy
  for n steps which also computes the performance variables
- Develop visualization functions for Q-Tables
- Develop a visualization functions for the evolution of the PD-World
- Develop functions to run experiments 1-4
- Develop visualization functions for attractive paths
- Develop functions to analyze agent coordination.
