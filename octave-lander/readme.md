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

Each record of the dump file represents an episode and the format is composed by:

| Field          | Offset |                  Size |
|:---------------|-------:|----------------------:|
| STEPCOUNT      |      1 |                     1 |
| RETURNS        |      2 |                     1 |
| ERRORS         |      3 |                     1 |
| **Total size** |        |                 **3** |

## Samples file format

Each reacord of samples file contains the data for batch learning.

| Field            | Offset |    Size |
|:-----------------|-------:|--------:|
| POS_SIG          |      1 |       3 |
| SPEED_SIG        |      4 |       3 |
| SQR_POS_SIG      |      7 |       3 |
| SQR_SPEED_SIG    |     10 |       3 |
| H_POS_DIR_SIG    |     13 |       2 |
| H_SPEED_DIR_SIG  |     15 |       3 |
| NO_LAND_POS__SIG |     18 |       1 |
| NO_LAND_VH__SIG  |     19 |       1 |
| NO_LAND_VZ__SIG  |     20 |       2 |
| Q                |     22 |      15 |
| ACTION           |     37 |      15 |
| REWARD           |     52 |       1 |
| ENDUP            |     53 |       1 |
| **Total size**   |        |  **53** |


## Trace analysis

The trace analysis allows to understand the learning performance of the agent step by step.
The trace file contains information of each step for a given session.
To load the trace run the octave:

```octave
  X = csvread("trace file");
```

then you extracts some useful information:
