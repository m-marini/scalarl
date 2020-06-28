# KPIs

[TOC]

## KPIs File Policy Actor Agent

The kpis format is

| Length | Offset | Field             |
|-------:|-------:|-------------------|
|      1 |      1 | v0                |
|      1 |      2 | v0*               |
|      1 |      3 | v0'               |
|      1 |      4 | Average reward    |
|      1 |      5 | $ \alpha_X $      |
|      5 |      6 | X Actor $ h_i $   |
|      5 |     11 | X Actor $ h_i^* $ |
|      5 |     16 | X Actor $ h_i' $  |
|      1 |     21 | $ \alpha_Y $      |
|      5 |     22 | Y Actor $ h_i $   |
|      5 |     27 | Y Actor $ h_i^* $ |
|      5 |     32 | Y Actor $ h_i' $  |
|      1 |     37 | $ \alpha_Z $      |
|      5 |     38 | Z Actor $ h_i $   |
|      5 |     43 | Z Actor $ h_i^* $ |
|      5 |     48 | Z Actor $ h_i' $  |

## KPIs File Continuous Actor Agent

The kpis format is

| Length | Offset | Field                   |
|-------:|-------:|-------------------------|
|      1 |      1 | v0                      |
|      1 |      2 | v0*                     |
|      1 |      3 | v0'                     |
|      1 |      4 | Average reward          |
|      1 |      5 | X $ \alpha_\mu $        |
|      1 |      6 | X Actor $ \mu $         |
|      1 |      7 | X Actor $ \mu^* $       |
|      1 |      8 | X Actor $ \mu' $        |
|      1 |      9 | X $ \alpha_{h_\sigma} $ |
|      1 |     10 | X Actor $ h_\sigma $    |
|      1 |     11 | X Actor $ h_\sigma^* $  |
|      1 |     12 | X Actor $ h_\sigma' $   |
|      1 |     13 | Y $ \alpha_\mu $        |
|      1 |     14 | Y Actor $ \mu $         |
|      1 |     15 | Y Actor $ \mu^* $       |
|      1 |     16 | Y Actor $ \mu' $        |
|      1 |     17 | Y $ \alpha_{h_\sigma} $ |
|      1 |     18 | Y Actor $ h_\sigma $    |
|      1 |     19 | Y Actor $ h_\sigma^* $  |
|      1 |     20 | Y Actor $ h_\sigma' $   |
|      1 |     21 | Z $ \alpha_\mu $        |
|      1 |     22 | Z Actor $ \mu $         |
|      1 |     23 | Z Actor $ \mu^* $       |
|      1 |     24 | Z Actor $ \mu' $        |
|      1 |     25 | Z $ \alpha_{h_\sigma} $ |
|      1 |     26 | Z Actor $ h_\sigma $    |
|      1 |     27 | Z Actor $ h_\sigma^* $  |
|      1 |     28 | Z Actor $ h_\sigma' $   |
