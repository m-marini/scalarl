# KPIs

[TOC]

## KPIs File Policy Actor Agent

The kpis format is

| Length | Offset | Field            |
|-------:|-------:|------------------|
|      1 |      0 | Critic sqrt(J)   |
|      1 |      1 | Critik sqrt(J')  |
|      1 |      2 | X Actor sqrt(J)  |
|      1 |      3 | X Actor sqrt(J') |
|      1 |      4 | Y Actor sqrt(J)  |
|      1 |      5 | Y Actor sqrt(J') |
|      1 |      6 | Z Actor sqrt(J)  |
|      1 |      7 | Z Actor sqrt(J') |
|      1 |      8 | Model size       |
|      1 |      9 | Queue size       |

## KPIs File Continuous Actor Agent

The kpis format is

| Length | Offset | Field               |
|-------:|-------:|---------------------|
|      1 |      0 | Critic sqrt(J)      |
|      1 |      1 | Critik sqrt(J')     |
|      1 |      2 | X Actor sqrt(J mu)  |
|      1 |      3 | X Actor sqrt(J' mu) |
|      1 |      4 | X Actor sqrt(J h)   |
|      1 |      5 | X Actor sqrt(J' h)  |
|      1 |      6 | Y Actor sqrt(J mu)  |
|      1 |      7 | Y Actor sqrt(J' mu) |
|      1 |      8 | Y Actor sqrt(J h)   |
|      1 |      9 | Y Actor sqrt(J' h)  |
|      1 |     10 | Z Actor sqrt(J mu)  |
|      1 |     11 | Z Actor sqrt(J' mu) |
|      1 |     12 | Z Actor sqrt(J h)   |
|      1 |     13 | Z Actor sqrt(J' h)  |
|      1 |     14 | Model size          |
|      1 |     15 | Queue size          |
