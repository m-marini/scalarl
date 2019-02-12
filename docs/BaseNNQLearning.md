# Base NN QLearning

In this base solution we consider a NN that learns the Q(s, a) function with one step forward.

The NN architecture has 4 layers

- Input layer based on state observation
- Hidden layer with h nodes and tanh activation
- Hidden layer with h nodes and tanh activation
- Output linear layer with a node for each action predicting the Q function

For each step the agent feeds the network with the current observation of state and record the output values as Q(s, a).

Then it chooses the action with highest Q value

```math
a = max(Q(s(t), a))
```

with a epsilon-greedy policy and records the reward and the new observation from environment.

To apply the RL then it computes the error as 

```math
delta = R + gamma * max(Q(s(t+1), a)) - max(Q(s(t), a))
```
and apply backpropagation for the expected NN output as 

```math
Q(s(t), a) = Q(s(t), a) + delta, if a is the taken action
- Q(s(t), a) = Q(s(t), a), otherwise
```
