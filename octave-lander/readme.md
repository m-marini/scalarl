# Debug utilities

[TOC]

## Ravel

The ravel function of nd4j transform a tensor in a flat colum vector such that

```octave
X = [[[1 2 3]; [4 5 6]]; [[10 20 30] [40 50 60]]];
X.ravel = [ 1 2 3 4 5 6 10 20 30 40 50 60];
```

the octave inverse function is

```octave
    permute(reshape(X, 3, 2, 2), [3 : -1 : 1])
```

## Features

These features are defined to maneuver the shuttle jets:


| Field          |   Size | Offset | Description                                |
|:---------------|-------:|-------:|:-------------------------------------------|
| POS            |      3 |      1 | Normalized (sh,sh,sz) position (x,y,z)     |
| SPEEED         |      3 |      4 | Normalized (svh,svh,svz) speed (x,y,z) m/s |
| LOW_NW_POS     |      1 |      7 | x < 0, y < 0, rh >= Rl, z < z1             |
| LOW_NE_POS     |      1 |      8 | x < 0, y > 0, rh >= Rl, z < z1             |
| LOW_SW_POS     |      1 |      9 | x > 0, y < 0, rh >= Rl, z < z1             |
| LOW_SW_POS     |      1 |     10 | x > 0, y > 0, rh >= Rl, z < z1             |
| HIGH_SW_POS    |      1 |     11 | x < 0, y < 0, rh >= Rl, z >= z1            |
| HIGH_NW_POS    |      1 |     12 | x < 0, y > 0, rh >= Rl, z >= z1            |
| HIGH_SE_POS    |      1 |     13 | x > 0, y < 0, rh >= Rl, z >= z1            |
| HIGH_NE_POS    |      1 |     14 | x > 0, y > 0, rh >= Rl, z >= z1            |
| LOW_CENTER     |      1 |     15 | rh < Rl, z < z1                            |
| HIGH_CENTER    |      1 |     16 | rh < Rl, z >= z1                           |
| LOW_NW_SPEED   |      1 |     17 | vx < 0, vy < 0, vh < v0                    |
| LOW_NE_SPEED   |      1 |     18 | vx < 0, vy > 0, vh < v0                    |
| LOW_SW_SPEED   |      1 |     19 | vx > 0, vy < 0, vh < v0                    |
| LOW_SE_SPEED   |      1 |     20 | vx > 0, vy > 0, vh < v0                    |
| HIGH_NW_SPEED  |      1 |     21 | vx < 0, vy < 0, vh >= v0                   |
| HIGH_NE_SPEED  |      1 |     22 | vx < 0, vy > 0, vh >= v0                   |
| HIGH_SW_SPEED  |      1 |     23 | vx > 0, vy < 0, vh >= v0                   |
| HIGH_SE_SPEED  |      1 |     24 | vx > 0, vy > 0, vh >= v0                   |
| POS_Z_SPEED    |      1 |     25 | vz >= 0                                    |
| LOW_Z_SPEED    |      1 |     26 | v1 < vz <= 0                               |
| MID_Z_SPEED    |      1 |     27 | v2 < vz <= v1                              |
| HIGH_Z_SPEED   |      1 |     28 | vz <= v2                                   |
| **Total size** | **28** |

- sh = 1 / 500 (1/m)
- sz = 1 / 150 (1/m)
- svh = 1/ 24 (s/m)
- svz = 1 / 12 (s/m)
- Rl = 10 (m)
- z1 = 10 (m)
- v0 = 0.25 (m/s)
- v1 = 2 (m/s)
- v2 = 4 (m/s)

## Trace file format

The trace file format is composed by:

| Field          | Offset | Size   |
|:---------------|-------:|-------:|
| EPISODE        |      1 |      1 |
| STEP           |      2 |      1 |
| ACTION         |      3 |     15 |
| REWARD         |     18 |      1 |
| ENDUP          |     19 |      1 |
| POS_0          |     20 |      3 |
| SPEED_0        |     23 |      3 |
| POS_1          |     26 |      3 |
| SPEED_1        |     29 |      3 |
| AVG_REWARD     |     32 |      3 |
| Q_0            |     35 |     15 |
| Q_1            |     50 |     15 |
| Q_01           |     65 |     15 |
| **Total size** |        | **79** |

## Dump file format

Each record of the dump file represents an episode and the format is composed by:

| Field          | Offset |                  Size |
|:---------------|-------:|----------------------:|
| STEPCOUNT      |      1 |                     1 |
| RETURNS        |      2 |                     1 |
| ERRORS         |      3 |                     1 |
| **Total size** |        |                 **3** |

## Samples file format

Each reacord of samples file contains the data for batch learning.

| Field            |   Size | Offset |
|:-----------------|-------:|-------:|
| SIGNALS          |     28 |      1 |
| ACTION           |     15 |     29 |
| REWARD           |      1 |     44 |
| ENDUP            |      1 |     45 |
| **Total size**   | **45** |        |

## Trace analysis

The trace analysis allows to understand the learning performance of the agent step by step.
The trace file contains information of each step for a given session.
To load the trace run the octave:

```octave
  X = csvread("trace file");
```

then you extracts some useful information:

### Plot trajectory

```octave
  plotTrajectoriesFromTrace(X)
```

### Plot 