# KPIs

[TOC]

The agent have a lot of iper parameters to tune and optimaze the learning rate.

We defines key performance indicator to monitor the learing method to tune iper parameters.

## Actor Critic Agent

The actor critic agent use to different network to aproximate the environemnt model - critic - and the policy - actor -.

### Critic KPI

The critic computes the updated value of current state by appling the the bootstrap.

```math
v^*(s_t) = v(s_{t+1}) + r_t - r_\pi 
```

 We can monitor how the ratio of MSE after and before the learning activity varies.

```math
    J(s_t) = [ v^*(s_t) - v(s_t) ]^2
    \\
    J'(s_t) = [v^*(s_t) - v'(s_t)]^2
    \\
    \Kappa(s_t) = \frac{J'(s_t)}{J(s_t)}
```

A ratio $ \Kappa (s_t) \ge 1 $ means a step-size parameter $ \alpha $ too high.

A ratio $ \Kappa (s_t) \ll 1 $  means a step-size parameter too low with very poor capacity of learning.

Because the $ J(s_t) $ should approach to $ 0 $ in optimal conditions, we should take into consideration only the steps that have a $ J(s_t) > \varepsilon $.

In a learning session we can evaluate the maximum value of kpi and adjust the step-size parameter accordingly.
As empirical method we can adjust the step-size parameter is multiply by a factor of

```math
     C = \frac{1}{\sum_t^T \Kappa(s_t)} = \frac{1}{\sum_t^T \frac{J'(s_t)}{J(s_t)}}
```


### Policy Actor KPIs

The actor computes the updated preferernces of current state by adding a step-size parameter to gradient and TD error

```math
    pr^*(s_t, a) = pr(s_t, a) + \alpha \delta_t \ln \nabla \pi(s_t, a)
```

To avoid comuptation overflow the preferences are constratints to a limited range $ (-7, +7) $.
The changes of preferences should be limited too to a fraction of the range $ (-1, +1) $ so meaninful kpis are the distance of changes of preferences:

```math
    J(s_t) = \sum_a
    \left[
        pr^*(s_t, a) - pr(s_t, a)
    \right] ^2
    = \sum_a
    \alpha \delta_t \ln^2 \nabla \pi(s_t, a)
```

A kpi $ J(s_t) \ge 1 $ - out of defined range $ (-1, +1) $ - means an $ \alpha $ parameter value too high.

Then it adjusts the network to fit the updated preferences.
The same kpi for the critic is used for each action of actor:

```math
    J(s_t) = \sum_a (pr^*(s_t, a) - pr(s_t, a))^2
    \\
    J'(s_t) = \sum_a (pr^*(s_t, a) - pr'(s_t, a))^2
    \\
    \Kappa(s_t) = \frac{J'(s_t)}{J(s_t)}
    \\
    C = \frac{1}{\sum_t^T \Kappa(s_t)} = \frac{1}{\sum_t^T  \frac{J'(s_t)}{J(s_t)}}
```

## KPIs File

The kpis format is

| Length | Offset | Field      |
|-------:|-------:|------------|
|      1 |      0 | Critic J   |
|      1 |      1 | Critik J'  |
|      1 |      2 | X Actor J  |
|      1 |      3 | X Actor J' |
|      1 |      4 | Y Actor J  |
|      1 |      5 | Y Actor J' |
|      1 |      6 | Z Actor J  |
|      1 |      7 | Z Actor J' |