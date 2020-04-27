# Actor Critic Agent

## Actor Update

The output of actor function is $ \pi_i $ the probabilities of choose action $ a_i $ at status $ s_t $.

The function is the softmax of the actions preferences $ h_i(s_t) $

```math
    \pi(a, s) = \frac{e^{h_a(s)}}{\sum_k e^{h_k(s)}}
```

The TD error is defined as

```Math
    \delta_t = r_t - r_\pi + v_\pi(s_{t+1}) - v_\pi(s_t)
```

simplifying the notation with

```math
    \pi(x, s) = \pi_a
    \\
    h_a(s) = h_a
```

The update of policy gradient is

```math
    \nabla \ln \pi_a = \frac{1}{\pi_a} \frac{\partial}{\partial h_a} \pi_a =
     \frac{1}{\pi_a ( \sum_k e^{h_k} )^2}
    \left[
        e^{h_a} \nabla h_a - e^{h_a} \nabla \sum_k e^{h_k}
    \right] =
    \\
    =  \frac{1} { \sum_k e^{h_k}}
    \left[
        \nabla h_a - \sum_k \nabla e^{h_k}
    \right] = 
    =  \frac{1} { \sum_k e^{h_k}}
    \left[
        \nabla h_a - \sum_k e^{h_k} \nabla h_k
    \right] =
    \\
    = \frac{1}{\sum_k e^{h_k}} \sum_i
    \left[
        A_i(a) - e^{h_i} 
    \right] \nabla h_i
    = \sum_i
    \left[ \frac{ A_i(a)}{\sum_k e^{h_k}} - \pi_i
    \right] \nabla h_i
```

where

```math
    i = a \Rightarrow A_i(a) = 1
    \\
    i \ne a \Rightarrow A_i(a) = 0
```

so

```math
    \delta^h_i = \delta \nabla \ln \pi_a = 
    \delta\sum_i \left[
        \frac{A_i(a)}{\sum_k e^{h_k}} - \pi_i
    \right]
```

the update target of $ h_i $ is

```math
    h'_i = h_i + \alpha \delta^h_i
```

## Continuous actions

```math
    \pi(x, s) = \frac{1}{\sigma(s) \sqrt{2 \pi}} e^{-\frac{(x-\mu(s))^2}{\sigma(s)^2}}
    \\
    \sigma(s) = e^{h_\sigma(s)}
```

simplifying the notation with

```math
    \pi(x, s) = p
    \\
    \mu(s) = \mu
    \\
    \sigma(s) = \sigma
    \\
    h_\sigma(s) = h_\sigma
```

we get

```math
  \nabla \ln p = \left(
        \frac{\partial}{\partial \mu} + \frac{\partial}{\partial h_\sigma}
        \right)
    \ln p
```

let's divide the problem

```math
    \\
    \frac{\partial}{\partial \mu} \ln p = \frac{1}{p} \frac{\partial p}{\partial \mu} =
    \frac{1}{p \sigma \sqrt{2 \pi} } e^{\frac{-(x-\mu)^2}{\sigma^2}} \frac{\partial}{\partial \mu} \left[ -\frac{(x-\mu)^2}{\sigma^2} \right]=
    = - \frac{1}{\sigma^2}[2 (x-\mu) (-1))] = \frac{2}{\sigma^2}(x-\mu)
```

and

```math
    \frac{\partial}{\partial h_\sigma} \ln p = \frac{1}{p} \frac{\partial p}{\partial \sigma} \frac{\partial \sigma}{\partial h_\sigma}
```

and

```math
    \frac{\partial p}{\partial \sigma} = \frac{1}{\sigma^2 \sqrt{2 \pi}}
    \left\{
        \sigma \frac{\partial}{\partial \sigma}
        \left[
            e^{-\frac{(x-\mu)^2}{\sigma^2}}
         \right]
         - e^{-\frac{(x-\mu)^2}{\sigma^2}}
    \right\} =
    \\
    = \frac{p}{\sigma}
    \left\{
        \sigma \frac{\partial}{\partial \sigma}
        \left[
            -(x - \mu)^2\sigma^{-2}
        \right] - 1
    \right\} = \frac{p}{\sigma}
    \left[
        -\sigma (x - \mu)^2 (-2 \sigma^{-3}) - 1)
    \right] = 
    \\
    = \frac{p}{\sigma}
    \left[
        2(x-\mu)^2 \sigma^{-2} - 1
    \right]
```

additionaly

```math
    \frac{\partial \sigma}{\partial h_\sigma} = \sigma
```

putting all toghether

```math
    \frac{\partial}{\partial h_\sigma} \ln p = \frac{1}{p}
    \frac{p}{\sigma}
    \left[
        2(x-\mu)^2 \sigma^{-2} - 1
    \right] \sigma = 2(x-\mu)^2 \sigma^{-2} - 1
```

finally

```math
    \delta_\mu = \eta \frac{2}{\sigma^2}(x-\mu) \delta
    \\
    \delta_\sigma = \eta
    \left[
        2\frac{(x-\mu)^2}{\sigma^2} - 1
    \right] \delta
```