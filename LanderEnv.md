# Lander environment

[TOC]

## Abstract

The environment consists of a shuttle that must land in a circular landing site of

```math
r = 10 \; m
```

of radius laying in the plan along the x, y axis.

The suttle is attract by a gravity acceleration of

```math
g = -1.6 \; \frac{m}{s^2}
```

## Initial position

At initial step the shuttle is located randomly nearby the landing site at an height of

```math
h = 100 \; m
```

and at a distance of

```math
d \le 500 \; m
```

## Goal

The shuttle must land with a vertical speed 

```math
v_z \ge -4 \; \frac{m}{s}
```

and a horizontal speed

```math
v_h \le 0.5\; \frac{m}{s}
```

## Actions

The agent controls the activation the jet above the shuttle createing a vertcal acceleration

```math
0 \le j_z \le 3.2  \; \frac{m}{s^2}
```

and the sideward jets creating a horizontal acceleration along x and y axis of

```math
-1 \le j_x \le 1 \; \frac{m}{s^2}
\\
\,
\\
-1 \le j_y \le 1 \; \frac{m}{s^2}
```

At each step of the subject decides which action perfom between a set of available actions depending on the landing site position.

At each step the acceleration of shuttle is computed as

```math
a_x = j_x
\\
a_y = j_y
\\
a_z = j_z + g
```

The speed is computed as

```math
v_x(t+1) = v_x(t) + a_x \Delta t
\\
v_y(t+1) = v_y(t) + a_z \Delta t
\\
v_z(t+1) = v_z(t) + a_z \Delta t
```

and the position is computed as

```math
x(t+1) = x(t) + v_x(t+1) \Delta t
\\
y(t+1) = y(t) + v_y(t+1) \Delta t
\\
z(t+1) = z(t) + z_y(t+1) \Delta t
```

To have the control of horizontal and vertical speed the $\Delta t $ between each step should be

```math
\max(j_x) \Delta t \le 0.5
\\
\,
\\
\Delta t \le \frac{0.5}{\max(j_x)} = 0.5 \; s
\\
\,
\\
\max(j_z) - g \Delta t \le 4
\\
\,
\\
\Delta t \le \frac{4}{\max(j_z) - g} = \frac{4}{3.2 - 1.6}  = 2.5 \; s
```

so we apply

```math
\Delta t = 0.25
```

## Rewards

When the shuttle reaches the landing site at within expected speeds it receives a reword of ?

When the shuttle reaches the land at wrong site or at higher speeds it receive a reword of ?

When the shuttle exits the boundary area

```math
|x| > 600 \; m
\\
|y| > 600 \; m
\\
z > 150 \; m
```

it receives a reward of ?

In the all the other cases it receive a negative rewards of the sum of jet useage

```math
R = -j_x - j_y - j_z
```

### Reward framework

A reward system framework is necessary to better address the short term reward problem.

The reward should consider the distance from the platform and the speed of shuttle.
In particolar the reward should penalize

- the horizontal distance from the platform
- the height
- speed directions opposite the platform direction
- horizontal speeds exceeding the optimal horizontal speed 
- vertical speeds exceeding the optimal vertical speed

The speed direction reward is proportional to the scalar product of speed versor and platform versor:
```math
    \rho = -\frac{\vec v \cdot \vec r}{|\vec v| |\vec r|}
```

The optimal horizontal speed should be in range between 0 and the  horizontal landing speed.
The optimal vertical speed should be in range between but 0 and the vertical landing speed.
The speed surplus may be compute as
```math
    \Delta v = \max\left(
        \left|v - v_0 \right| - dv, 0
        \right) \\
    v_0 = \frac{v_{min} + v_{max}}{2} \\
    dv = \frac{v_{max} - v_{min}}{2} \\
```

For each state $ s $ we define the reward as
```math
    R = f_s(\vec r, \vec v) \\
    R = g_s(\rho, r_h, z, \Delta v_h, \Delta v_z) \\
    r_h = \sqrt{x^2+y^2} \\
    v_h = \sqrt{v_x^2 + v_y^2} \\
    g_s(\rho, r_h, z, v_h, v_z) = a_s + a_{\rho s} \rho + a_{rs} r_h + a_{zs} z + a_{hs} \Delta v_h  + a_{vs} \Delta v_z
```

The reward is composed by:
- a fixed base value,
- a value depending on direction difference,
- a value depending on horizontal distance from platform,
- a value depending on height,
- a value depending on vertical speed exceeding the optimal vertical speed
- a value depending on absolute difference between vertical speed and optimal speed.

The coefficents $ a_{rs}, a_{zs}, a_{hs}, a_{vs} $ are non positive, $ a_{\rho s} $ is non negative and $ a_s $ may be positive if the state is desired or negative if the status is undesired.

## Input signals

The base input signals are in the range -1, 1
- the direction of platform (0 toward x-axis, 0.5 toward y-axis) tiles length of 11.25 DEG for 32 tiles
- the direction of horizontal speed (0 toward x-axis, 0.5 toward y-axis) tiles length of 11.25 DEG for 32 tiles
- the platform distance (-1 = 0 m, 1 = 64 m), tiles length of 2m for 32 tiles
- the height from the ground (-1 = 0 m, 1 = 16 m) tiles length of 0.5 m for 32 tiles
- the horizontal speed (-1 = 0 m/s, 1 = 8 m/s) tiles length of 0.25 m/s for 32 tiles
- the vertical speed (-1 = -16 m/s, 1 = 16 m/s) tiles length of 1 m/s for 32 tiles

| Offset | Signal             |
|-------:|--------------------|
|      0 | Platform direction |
|      1 | Speed direction    |
|      2 | Platform distance  |
|      3 | Height             |
|      4 | H Speed            |
|      5 | V Speed            |

## Available action signals

The available actions consist of the 5 possible levels of jet powers for each of 3 directions (x, y, z).

## Output signals

The output signals consists of 15 output units representing the estimation of advantage action value. The higer value of the output indicates the action to be taken.

## Test cases

| Cases |   z | radius | fuel |
|-------|----:|-------:|-----:|
|     1 |   1 |      5 |   10 |
|     2 |   3 |     15 |   30 |
|     3 |  10 |     50 |  100 |
|     4 |  30 |    150 |  300 |
|     5 | 100 |    500 |  500 |

## Planner model

The state space of the planner is the 3D lander position and 3D lander speed tiled to have a sufficent precision to controll the lander.

| Dimension |  Full Range |  Precision | Clipped Range | Tiles | Signal range |
|-----------|------------:|-----------:|--------------:|------:|-------------:|
| x         |   -/+ 500 m |     250 mm |       -/+ 4 m |    32 |     -/+ 8e-3 |
| y         |   -/+ 500 m |     250 mm |       -/+ 4 m |    32 |     -/+ 8e-3 |
| z         |    0, 150 m |    62.5 mm |        0, 4 m |    32 |  0, 26.67e-3 |
| $v_x $    |  -/+ 24 m/s |    25 mm/s |   -/+ 0.4 m/s |    32 | -/+ 16.67e-3 |
| $v_y $    |  -/+ 24 m/s |    25 mm/s |   -/+ 0.4 m/s |    32 | -/+ 16.67e-3 |
| $v_z $    |  -/+ 12 m/s | 31.25 mm/s |   -/+ 0.5 m/s |    32 | -/+ 41.67e-3 |
