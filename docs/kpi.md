# KPIs

[TOC]

## KPIs File Policy Actor Agent

The kpis format is

| Length | Offset | Field            |
|-------:|-------:|------------------|
|      1 |      0 | Critic sqrt(J)   |
|      1 |      1 | Critic sqrt(J')  |
|      1 |      2 | Average reward   |
|      1 |      3 | X Alpha          |
|      1 |      4 | X Actor sqrt(J)  |
|      1 |      5 | X Actor sqrt(J') |
|      1 |      6 | Y Alpha          |
|      1 |      7 | Y Actor sqrt(J)  |
|      1 |      8 | Y Actor sqrt(J') |
|      1 |      9 | Y Alpha          |
|      1 |      0 | Z Actor sqrt(J)  |
|      1 |     11 | Z Actor sqrt(J') |

## KPIs File Continuous Actor Agent

The kpis format is

| Length | Offset | Field               |
|-------:|-------:|---------------------|
|      1 |      0 | Critic sqrt(J)      |
|      1 |      1 | Critic sqrt(J')     |
|      1 |      2 | Average reward      |
|      1 |      3 | X Alpha mu          |
|      1 |      4 | X Actor sqrt(J mu)  |
|      1 |      5 | X Actor sqrt(J' mu) |
|      1 |      6 | X Alpha h           |
|      1 |      7 | X Actor sqrt(J h)   |
|      1 |      8 | X Actor sqrt(J' h)  |
|      1 |      9 | Y Alpha mu          |
|      1 |     10 | Y Actor sqrt(J mu)  |
|      1 |     11 | Y Actor sqrt(J' mu) |
|      1 |     12 | Y Alpha h           |
|      1 |     13 | Y Actor sqrt(J h)   |
|      1 |     14 | Y Actor sqrt(J' h)  |
|      1 |     15 | Z Alpha mu          |
|      1 |     16 | Z Actor sqrt(J mu)  |
|      1 |     17 | Z Actor sqrt(J' mu) |
|      1 |     18 | Z Alpha h           |
|      1 |     19 | Z Actor sqrt(J h)   |
|      1 |     20 | Z Actor sqrt(J' h)  |
