# Trace Dense Layer

The layer implement a full connected nural network layer base on eligible trace
to update the layer parameters.

## Forward propagation

The forward propagation of the network is the same of classical network

y(j) = Sum (x(i) * w(i, j)) + b(j) among i = 1 ... n

## Gradients

The gradients for weights and bias are

grad(Y, w(i, j)) = x(i)

grad(Y, b(j)) = 1

## Trace update

During the learning phase the network parameters are apdates basing on
eligibility trace.

The eligibility trace is a vector that traces the previous updated parameters
allowing the network to learn from the past events.

Let be

- w(i, j) the weights of neural network for i-th input and j-th output
- b(j) the bias of j-th output
- Ew(i, j) the weights eligibility trace after updating for i-th input and j-th output
- Eb(j) the bias eligibility trace after updating and j-th output
- delta(j) the errors for the j-th output
- alpha the learing rate hyperparamter
- gamma the reward discount rate
- lambda the TD hyparameter

The update parameter equations are

Ew'(i, j) <- Ew(i, j) * gamma * lambda + grad(Y, w(i, j))

Eb'(j) <- Eb(j) * gamma * lambda + grad(Y, b(j))

w'(i, j) <- w(i, j) + Ew'(i, j) * delta(j)_ * alpha

b'(j) <- b(j) + Eb'(j) * e_j * alpha

the concretely become

Ew'(i, j) <- Ew(i, j) * gamma * lambda + X(i)

Eb'(i, j) <- Eb(j) * gamma * lambda  + 1

w'(i, j) <- w(i, j) + alpha * Ew'(i, j) * delta(j)

b'(j) <- b(j) + alpha * Eb'(j) * delta(j)

## Backward propagation

The backward propagation of errors in the input are

delta'(i) = Sum( diff(y(j), x(i)) * delta(j)) among j = 1 ... m

delta'(i) = Sum( w(i, j) * delta(j)) among j = 1 ... m

## Error mask

In Reinforcement learning the Q-Learning algorithm must update the parameters
for the action a(k) taken by the current policy so the update mechanism shuold
be changed by the equations

Ew'(i, j) <- Ew(i, j) * gamma * lambda + grad(Y, w(i, j)), for j = k
Ew'(i, j) <- Ew(i, j) * gamma * lambda, for j != k

Eb'(j) <- Eb(j) * gamma * lambda + grad(Y, b(j)), for j = k
Eb'(j) <- Eb(j) * gamma * lambda, for j != k

When a deep learning network composed by multiple layers is updated
the update should not be applied for all the parameters but on for the releted
to the action a(k).

To implement such requirement we use a mask for the errors that are backpropagated toghther
the error depending on the layer type.

Ew'(i, j) <- Ew(i, j) * gamma * lambda + grad(Y, w(i, j)) * M(j)
Eb'(j) <- Eb(j) * gamma * lambda + grad(Y, b(j)) * M(j)

For the Dense layer each input is connected and therefore influenced by each output
so the backpropagated error mask is a ones vectors.

M'(i) = ones(n)
