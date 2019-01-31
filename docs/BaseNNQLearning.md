# Base NN QLearning

In this base solution we consider a NN that learns the Q(s,a) function with single step forward.

The NN architecture has 4 layers

- Input layer based on state observation
- Hidden layer with $ h $ nodes and tanh activation
- Hidden layer with $ h $ nodes and tanh activation
- Output linear layer with a node for each action predicting the Q function

For each step the agent feeds the network with the current observation of state and record the values of Q(s, *).

Then it chooses the action with highest Q value
$$$
a = max_{a_i}(Q(s(t),a_i))
$$$
with a $ \varepsilon-greedy $ policy and records the reward and the new observation from environment.

To apply the RL then it computes the error as 
$$$
\delta = R + \gamma max_i(Q(s(t+1), a_i) - max_i(Q(s(t), a_i)
$$$
and the expected NN output as 
$$$
Q(s(t), a_i) = Q(s(t), a_i) + \delta, if a_i is the action taken
Q(s(t), a_i) = Q(s(t), a_i), otherwise
$$$
