# KPIs

[TOC]

## KPIs File Policy Actor Agent

The kpis format is

| Length | Offset | Field                     |
|-------:|-------:|---------------------------|
|      1 |      1 | v0                        |
|      1 |      2 | v0*                       |
|      1 |      3 | v0'                       |
|      1 |      4 | Average reward            |
|      1 |      5 | J                         |
|      1 |      6 | J'                        |
|      1 |      7 | $ \alpha_{direction} $    |
|      8 |      8 | Direction Actor $ h_i $   |
|      8 |     16 | Direction Actor $ h_i^* $ |
|      8 |     24 | Direction Actor $ h_i' $  |
|      1 |     32 | $ \alpha_{H Jet} $        |
|      3 |     33 | H Jet Actor $ h_i $       |
|      3 |     36 | H Jet Actor $ h_i^* $     |
|      3 |     39 | H Jet Actor $ h_i' $      |
|      1 |     42 | $ \alpha_Z $              |
|      5 |     43 | Z Actor $ h_i $           |
|      5 |     48 | Z Actor $ h_i^* $         |
|      5 |     53 | Z Actor $ h_i' $          |

## KPIs File Continuous Actor Agent

The kpis format is

| Length | Offset | Field                            |
|-------:|-------:|----------------------------------|
|      1 |      1 | v0                               |
|      1 |      2 | v0*                              |
|      1 |      3 | v0'                              |
|      1 |      4 | Average reward                   |
|      1 |      5 | J                                |
|      1 |      6 | J'                               |
|      1 |      7 | Direction $ \alpha_\mu $         |
|      1 |      8 | Direction Actor $ \mu $          |
|      1 |      9 | Direction Actor $ \mu^* $        |
|      1 |     10 | Direction Actor $ \mu' $         |
|      1 |     11 | Direction $ \alpha_{h_\sigma}  $ |
|      1 |     12 | Direction Actor $ h_\sigma $     |
|      1 |     13 | Direction Actor $ h_\sigma^* $   |
|      1 |     14 | Direction Actor $ h_\sigma' $    |
|      1 |     15 | H Speed $ \alpha_\mu $           |
|      1 |     16 | H Speed Actor $ \mu $            |
|      1 |     17 | H Speed Actor $ \mu^* $          |
|      1 |     18 | H Speed Actor $ \mu' $           |
|      1 |     19 | H Speed $ \alpha_{h_\sigma} $    |
|      1 |     20 | H Speed Actor $ h_\sigma $       |
|      1 |     21 | H Speed Actor $ h_\sigma^* $     |
|      1 |     22 | H Speed Actor $ h_\sigma' $      |
|      1 |     23 | Z Speed $ \alpha_\mu $           |
|      1 |     24 | Z Speed Actor $ \mu $            |
|      1 |     25 | Z Speed Actor $ \mu^* $          |
|      1 |     26 | Z Speed Actor $ \mu' $           |
|      1 |     27 | Z Speed $ \alpha_{h_\sigma} $    |
|      1 |     28 | Z Speed Actor $ h_\sigma $       |
|      1 |     29 | Z Speed Actor $ h_\sigma^* $     |
|      1 |     30 | Z Speed Actor $ h_\sigma' $      |
