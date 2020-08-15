# File formats

[TOC]

## Dump file format

| Length | Offset | Field  |
|-------:|-------:|--------|
|      1 |      1 | Epoch  |
|      1 |      2 | Step   |
|      1 |      3 | Reward |
|      1 |      4 | Score  |

## Trace file format

| Length | Offset | Field                  |
|-------:|-------:|------------------------|
|      1 |      1 | Epoch                  |
|      1 |      2 | Step id                |
|      1 |      3 | Signal id              |
|      1 |      4 | Status Code            |
|      3 |      5 | Position               |
|      3 |      8 | Speed                  |
|      1 |     11 | Fuel                   |
|      1 |     12 | Time                   |
|      3 |     13 | Actions                |
|      1 |     16 | Reward                 |
|      1 |     17 | Score                 |
|        |        | **Policy actors**      |
|      8 |     18 | Direction h            |
|      8 |     26 | Direction probabilitis |
|      3 |     34 | H Speed h              |
|      3 |     37 | H Speed probabilitis   |
|      5 |     40 | V Speed h              |
|      5 |     45 | V Speed probabilitis   |
|        |        | **Gaussian actors**    |
|      1 |     18 | Direction $\mu$        |
|      1 |     19 | Direction $h_\sigma$   |
|      1 |     20 | Direction $\sigma$     |
|      1 |     21 | H Speed $\mu$          |
|      1 |     22 | H Speed $h_\sigma$     |
|      1 |     23 | H Speed $\sigma$       |
|      1 |     24 | V Speed $\mu$          |
|      1 |     25 | V Speed $h_\sigma$     |
|      1 |     26 | V Speed $\sigma$       |


## Status Codes

| Value | Description           |
|------:|-----------------------|
|     0 | Flying                |
|     1 | Landed                |
|     2 | LandedOutOfPlatform   |
|     3 | VCrashedOnPlatform    |
|     4 | VCrashedOutOfPlatform |
|     5 | HCrashedOnPlatform    |
|     6 | HCrashedOutOfPlatform |
|     7 | OutOfRange            |
|     8 | OutOfFuel             |
