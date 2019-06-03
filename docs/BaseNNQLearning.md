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
a^* = max(Q(s(t), a))
```

with a epsilon-greedy policy and records the reward and the new observation from environment.

The state value is
```math
V(s(t)) = \max(Q(s(t), a))
```

To apply the RL then it computes the error as 

```math
\delta = R + \gamma V(s(t+1)) - V(s(t))
```

and apply backpropagation for the expected NN output as 

```math
Q'(s(t), a) = Q(s(t), a) + \delta, a  = a^*
\\
Q'(s(t), a) = Q(s(t), a), a \ne a^*
```

# Advantage learning

In the advantage learning the network architacture is the same as QLearning but the output layer is the estimation of advantage function $A(s,a)$

The state value is
```math
V(s(t)) = \max(A(s(t), a))
```

The error is
```math
\delta = V(s(t)) + \frac{R + \gamma * V(s(t+1)) - V(s(t))}{\kappa} - V(s(t))
\\
\delta = \frac{R + \gamma * V(s(t+1)) - V(s(t))}{\kappa}
```

and apply backpropagation for the expected NN output as 

```math
A'(s(t), a) = A(s(t), a) + \delta, a  = a^*
\\
A'(s(t), a) = A(s(t), a), a \ne a^*
```

AS you can see the QLearning is a special case of Advantage learning when $\kappa = 1$
