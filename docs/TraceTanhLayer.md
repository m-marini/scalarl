# Trace Tanh Layer

The layer implement a tanh activation function layer and does not have any parameters.

## Forward propagation

The forward propagation of the network is the same of classical network

```math
y_i = \tanh(x_i)
```

## Backward propagation

The backward propagation of errors in the input are

```math
\delta'_i = \frac{\partial y_j}{\partial x_i} \delta_i
\\
\delta'_i = (1 - y_i) (1 + y_i) = 1-y_i^2
```

## Error mask

For the activation layer the error mask is backpropagated without changes

```math
M'_i = M_i
```
