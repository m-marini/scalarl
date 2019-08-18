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

## Trace file format

The trace file format is composed by:

| Field              | Offset | Size   |
|:-------------------|-------:|-------:|
| EPISODE            |      1 |      1 |
| STEP               |      2 |      1 |
| ACTION             |      3 |     15 |
| REWARD             |     18 |      1 |
| ENDUP              |     19 |      1 |
| PREV_POS           |     20 |      3 |
| PREV_SPEED         |     23 |      3 |
| RESULT_POS         |     26 |      3 |
| RESULT_SPEED       |     29 |      3 |
| PREV_Q             |     32 |     15 |
| RESULT_Q           |     47 |     15 |
| PREV_Q1            |     62 |     15 |
| **Total size**     |        | **76** |

## Dump file format

Each record of the dump file rapresents an episode and the format is composed by:

| Field          | Offset |                  Size |
|:---------------|-------:|----------------------:|
| STEPCOUNT      |      1 |                     1 |
| RETURNS        |      2 |                     1 |
| ERRORS         |      3 |                     1 |
| **Total size** |        |                 **3** |

## Trace analysis

The trace analysis allows to understand the learning performance of the agent step by step.
The trace file contains information of each step for a given session.
To load the trace run the octave:

```octave
  X = csvread("trace file");
```

then you extracts some useful information:
