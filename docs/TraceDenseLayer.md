# Trace Dense Layer

The layer implement a full connected neural network layer base on eligible trace to update the layer parameters.

## Forward propagation

The forward propagation of the network is the same of classical network

```math
y_i = \sum_j x_j w_{ji} + b_i, j = 1 ... n

```

## Gradients

The gradients for weights and bias are

```math
\frac{\partial y_i}{\partial w_{ji}} = x_j
\\
\frac{\partial y_i}{\partial b_i} = 1
```

## Trace update

During the learning phase the network parameters are updates basing on eligibility trace.

The eligibility trace is a vector that traces the previous updated parameters allowing the network to learn from the past events.

Let be

- $w_{i,j}$ the weights of neural network for i-th input and j-th output
- $b_j$ the bias of j-th output
- $E_{w_{ji}}$ the weights eligibility trace after updating for i-th input and j-th output
- $E_{b_j}$ the bias eligibility trace after updating and j-th output
- $\delta_j$ the errors for the j-th output
- $\alpha$ the learing rate hyperparamter
- $\gamma$ the reward discount rate
- $\lambda$ the TD hyperparameter

The update parameter equations are

```math
E'_{w_{ij}} = E_{w_{ij}} \gamma \lambda + \frac{\partial y_j}{\partial w_{ij}}
\\
E'_{b_j} = E_{b_j} \gamma \lambda + \frac{\partial y_j}{\partial b_j}
\\
w'_{ij} = w_{ij} + E'_{w_{ij}} \delta_j \alpha
\\
b'_i = b_i + E'_{b_i} \delta_i \alpha
```

that concretely becomes

```math
E'_{w_{ij}} = E_{w_{ij}} \gamma \lambda + x_i
\\
E'_{b_j} = E_{b_j} \gamma \lambda + 1
\\
w'_{ij} = w_{ij} + E'_{w_{ij}} \delta_j \alpha
\\
b'_i = b_i + E'_{b_i} \delta_i \alpha
```

If we consider the replace method then the update parameters equation are
```math
E'_{w_{ij}} = hardtanh(E_{w_{ij}} \gamma \lambda + x_i)
\\
E'_{b_j} = hardtanh(E_{b_j} \gamma \lambda + 1)
\\
hardtanh(x) = -1, x \lt -1
\\
hardtanh(x) = x, -1 \le x \lt 1
\\
hardtanh(x) = 1, x > 1
\\
w'_{ij} = w_{ij} + E'_{w_{ij}} \delta_j \alpha
\\
b'_i = b_i + E'_{b_i} \delta_i \alpha
```

## Backward propagation

The backward propagation of errors in the input are

```math
\delta'_i = \sum_j \frac{\partial y_j}{\partial x_i} \delta_j,  j = 1 \dots m
\\
\delta'_i = \sum_j w_{ij} \delta_j, j = 1 \dots m
```

## Error mask

In Reinforcement Learning the Q-Learning algorithm must update the parameters for the action $a_k$ taken by the current policy so the update mechanism should be changed by the equations

```math
E'_{w_{ij}} = E_{w_{ij}} \gamma \lambda + \frac{\partial y_j}{\partial w_{ij}}, j = k
\\
E'_{w_{ij}} = E_{w_{ij}} \gamma \lambda, j \ne k
\\
E'_{b_j} = E_{b_j} \gamma \lambda + \frac{\partial y_j}{\partial b_j}, j = k
\\
E'_{b_j} = E_{b_j} \gamma \lambda, j \ne k
```

When a deep learning network composed by multiple layers is updated the update should not be applied for all the parameters but on for the releted to the action a(k).

To implement such requirement we use a mask for the errors that are backpropagated toghther the error depending on the layer type.

```math
E'_{w_{ij}} = E_{w_{ij}} \gamma \lambda + \frac{\partial y_j}{\partial w_{ij}} M_j
\\
E'_{b_j} = E_{b_j} \gamma \lambda + \frac{\partial y_j}{\partial b_j} * M_j
```

For the Dense layer each input is connected and therefore influenced by each output
so the backpropagated error mask is a ones vectors.

```math
M_i = 1
```

## MSE Loss function

The MSE Loss function is define as

```math
L = \frac{1}{2} \sum_i \left( Y_i - y_i \right)^2, i = 1 \dots m
\\
\frac{\partial L}{\partial y_i}= Y_i - y_i
```

where $Y_i$ is the expected output
