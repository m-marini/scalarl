# Trace Tanh Layer

The layer implement a tanh activation function layer and does not have any parameters.

## Forward propagation

The forward propagation of the network is the same of classical network

```latex
y(i) =  tanh(x(i))
```

## Backward propagation

The backward propagation of errors in the input are

```latex
delta'(i) = diff(y(i), x(i)) * delta(i))

delta'(i) = (1 - y(i)) * (1 + y(i))
```

## Error mask

For the activation layer the error mask is backpropagated without changes

```latex
M'(i) = M(i)
```
