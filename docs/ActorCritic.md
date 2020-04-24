# Actor Critic Agent

## Actor Update

The output of actor function is $ \pi_i $ the probabilities of choose action $ a_i $ at status $ s_t $.

The function is the softmax of the actions preferences $ h_i(s_t) $

```math
    \pi_i(s) = \frac{e^{h_i(s)}}{\sum_k e^{h_k(s)}}
```

The TD error is defined as

```Math
    \delta_t = R_t - \bar R_\pi + v_\pi(s_{t+1}) - v_\pi(s_t)
```

The update of policy gradient is

```math
    \vec \theta' = \vec \theta + \Delta \vec \theta
    \\
    \Delta \vec \theta=\alpha \delta_t \nabla \ln \pi_a(s_t)
    \\
    \nabla \ln \pi_a(s_t) = \frac{\nabla \pi_a(s_t)}{\pi_a(s_t)} =
    \frac{
        e^{h_a(s_t)} \nabla h_a(s_t) - e^{h_a(st)} \nabla \sum_k e^{h_k(s_t)}
    } {(\sum_k e^{h_k(s_t)})^2}
    \frac{\sum_k e^{h_k(s)}}{e^{h_a(s)}} =
    \\
    =  \frac{
        \nabla h_a(s_t) - \sum_k \nabla e^{h_k(s_t)}
    } {\sum_k e^{h_k(s_t)}}
    = \frac{\nabla h_a(s_t) - \sum_k e^{h_k(s_t)} \nabla h_k(s_t)}{\sum_k e^{h_k(s_t)}} =
    \\
    = \sum_i \frac{ A_i(a) - e^{h_i(s_t)}}{\sum_k e^{h_k(s_t)}} \nabla h_i(s_t)
    = \sum_i
    \left[ \frac{ A_i(a)}{\sum_k e^{h_k(s_t)}} - \pi_i(s_t)
    \right] \nabla h_i(s_t)
```

where

```math
    i = a \Rightarrow A_i(a) = 1
    \\
    i \ne a \Rightarrow A_i(a) = 0
```

so

```math
    \Delta \vec \theta = \alpha \sum_i \delta^h_i \nabla h_i(s_t) \\
    \delta^h_i = \left(
        \frac{ A_i(a)}{\sum_k e^{h_k(s_t)}}  - \pi_i(s_t)
    \right) \delta_t
```

the update target of $ h_i(s) $ is

```math
    h'_i(s_t) = h_i(s_t) + \delta^h_i
```
