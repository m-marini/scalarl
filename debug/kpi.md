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

In learning session we can evaluate the value of kpi and adjust the step-size parameter accordingly.
As empirical method we can compute the 90th percentile and adjust the step-size parameter by multiply by a factor of

```math
     C_{90} = \frac{1}{\Kappa_{90}}
```

In such a way the 90% of sample should be corrected.

### Policy Actor KPIs

The actor computes the updated preferernces of current state by adding a step-size parameter to gradient and TD error

```math
    pr^*(s_t, a) = pr(s_t, a) + \alpha \delta_t \ln \nabla \pi(s_t, a)
```

> *The correct equation should consider the softmax effect on the preferences*


To avoid comuptation overflow the preferences are constratints to a limited range e.g. $ (-7, +7) $.
The changes of preferences should also be limited to a fraction of the range $ (-\xi, +\xi) $, so meaninful kpi is the squared distance of changes of preferences:

```math
    J(s_t) = \sum_a
    \left[
        pr^*(s_t, a) - pr(s_t, a)
    \right] ^2
    = \sum_a
    \alpha^2 \delta_t^2 \ln^2 \nabla \pi(s_t, a)
    = \alpha^2 \delta_t^2 \sum_a \ln^2 \nabla \pi(s_t, a)
```

A kpi $ J(s_t) \ge \xi^2 $ - out of defined range $ (-\xi, +\xi) $ - means an $ \alpha $ parameter value too high.

Let's suppose to correct the $ \alpha $ parameter by a $ C $ factor so that the corrected $ J_C(s_t) $ is equal to $ \xi^2 $, we have

```math
    J_C(s_t) = (C \alpha)^2 \delta_t^2 \sum_a \ln^2 \nabla \pi(s_t, a)
    = C^2 J_(s_t)
    \\
    \xi^2 = C^2 J(s_t)
    \\
    C = \frac{\xi}{\sqrt{J(s_t)}}
```

So to correct the agent we may apply the correction to the 90 percetage of J samples by computing $ J_{90} $, the 90 percenitle of $ J $,  and computes the correction factor

```math
    C_{90} = \frac{\xi}{\sqrt{J_{90} } }
```

Then actor adjusts the network to fit the updated preferences.
The same kpi for the critic is used for each action of actor:

```math
    J(s_t) = \sum_a (pr^*(s_t, a) - pr(s_t, a))^2
    \\
    J'(s_t) = \sum_a (pr^*(s_t, a) - pr'(s_t, a))^2
    \\
    \Kappa(s_t) = \frac{J'(s_t)}{J(s_t)}
    \\
    C_{90} = \frac{1}{\Kappa_{90}}
```

## KPIs File

The kpis format is

| Length | Offset | Field      |
|-------:|-------:|------------|
|      1 |      0 | Epoch      |
|      1 |      1 | Step       |
|      1 |      0 | Critic J   |
|      1 |      1 | Critik J'  |
|      1 |      2 | X Actor J  |
|      1 |      3 | X Actor J' |
|      1 |      4 | Y Actor J  |
|      1 |      5 | Y Actor J' |
|      1 |      6 | Z Actor J  |
|      1 |      7 | Z Actor J' |
