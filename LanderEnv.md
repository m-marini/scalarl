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

## Input signals

The input signals are

- the position x, y of landing site respect the shuttle normalized -500, 500 to -1, 1
- the position z of landing site respect the shuttle normalized 0, 100 to -1, 1
- the speeds vx, vy, vz of shuttle normalized ? to -1, 1
- the signals for correct horizontal position respect the landing site -1 when x, y before $r =10 \; m$, 1 when x, y after $r=10 \; m$, 0 otherwise.
- the signals for inrange speeds for landing, -1 when $ v_x, v_y, < 0.5 \; \frac{m}{s} $,  -1 when $ v_x, v_y, > 0.5 \; \frac{m}{s} $, 0 otherwise.
- the signals for inrange vertical speeds for landing, -1 when $ v_z < -4 \; \frac{m}{s} $,  1 when $ v_z > 0 \; \frac{m}{s} $, 0 otherwise.

| Offset |  Signal |
|-------:|--------:|
|      0 |       x |
|      1 |       y |
|      2 |       z |
|      3 |   $v_x$ |
|      4 |   $v_y$ |
|      5 |   $v_z$ |
|      6 |  land x |
|      7 |  land y |
|      8 | vland x |
|      9 | vland y |
|     10 | vland z |

## Available action signals

The available actions consist of the 5 possible levels of jet powers for each of 3 directions (x, y, z).

## Output signals

The output signals consists of 15 output units representing the estimation of advantage action value. The higer value of the output indicates the action to be taken.

# Test cases

| Cases |   z | radius | fuel |
|-------|----:|-------:|-----:|
|     1 |   1 |      5 |   10 |
|     2 |   3 |     15 |   30 |
|     3 |  10 |     50 |  100 |
|     4 |  30 |    150 |  300 |
|     5 | 100 |    500 |  500 |
