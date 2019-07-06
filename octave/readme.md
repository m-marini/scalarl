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

| Field              | Offset | Size |
|:-------------------|-------:|-----:|
| EPISODE            |      1 |    1 |
| STEP               |      2 |    1 |
| ACTION             |      3 |    1 |
| REWARD             |      4 |    1 |
| ENDUP              |      5 |    1 |
| PREV_POS           |      6 |    2 |
| RESULT_POS         |      8 |    2 |
| PREV_Q             |     10 |    8 |
| RESULT_Q           |     18 |    8 |
| PREV_Q1            |     26 |    8 |
| AVAILABLE_ACTIONS  |     34 |    8 |
| AVAILABLE_ACTIONS1 |     42 |    8 |

## Dump file format

Each record of the dump file rapresents an episode and the format is composed by:

| Field     | Offset |                  Size |
|:----------|-------:|----------------------:|
| STEPCOUNT |      1 |                     1 |
| RETURNS   |      2 |                     1 |
| ERRORS    |      3 |                     1 |
| Q         |      4 |     8 x 10 x 10 = 800 |
| MASK      |    804 |     8 x 10 x 10 = 800 |

`Q` consists of 8 action values for 10 columns for 10 rows of possible subject locations

```octave
Q = permute(reshape(..., 8, 10, 10), [3 : -1 : 1])
```

`MASK` consists of 8 action available flags for 10 columns for 10 rows of possible subject locations

```octave
MASK = permute(reshape(..., 8, 10, 10), [3 : -1 : 1])
```

## Statistics file format

Each record of statistic file rapresentes the statistical data of sampled episode sequence

| Field              | Offset | Size |
|:-------------------|-------:|-----:|
| Average            |      1 |    1 |
| Standard deviation |      2 |    1 |
| Minimum            |      3 |    1 |
| 5% percentile      |      4 |    1 |
| 25% percentile     |      5 |    1 |
| Median             |      6 |    1 |
| 75% percentile     |      7 |    1 |
| 95% percentile     |      8 |    1 |
| Maximum            |      9 |    1 |

## Trace analysis

The trace analysis allows to understand the learning performance of the agent step by step.
The trace file contains information of each step for a given session.
To load the trace run the octave:

```octave
  X = csvread("trace file");
```

then you extracts some useful information:

### vFromTrace

It extracs the estimated state values for each step

```octave
  V = vFromTrace(X);
```

### afterVFromTrace

It estracts the estimated state value for the resulting state of each step

```octave
  V = afterVFromTrace(X);
```

### fitVFromTrace

It extracts the estiamtes state value of fitted agent of each step

```octave
  V = fitVFromTrace(X);
```

### greedyActionFromTrace

It extracts the greedy action from the trace file

```octavce
  GA = greedyActionForma(X);
```

### policyStats

It extracts the statistics of session related to the states of maze

```octave
  [N, M, AF, GF] = policyStats(X);
```

`N` is the frequence the subject lay in the cells
`M` is the number of actions for each cells
`AF` is the maximum frequence the subject selected an action
`GF` is the maximum frequence the greedy policy selected an action

### plotCoverage

It plots the frequence the subject layed in a cell

```octave
  plotCoverage("../trace.csv");
```

### plotMu

It plots the ratio of maximum frequence per action of agent policy related to the random policy.

```octave
  plotMu("../trace.csv");
```

### plotGMu

It plots the ratio of maximum frequence per action of greedy policy related to the random policy.

```octave
  plotGMu("../trace.csv");
```

### plotVFromTrace

Plot the V values from trace file for a given cell position

Syntax

```octave
H = plotVTraceFromFile(filename, pos)
```

Example:

```octave
H = plotVTraceFromFile("../trace.cvs", [9,9]])
```

## plotQFromTrace

Plot the policy values from trace file for a given cell position

Syntax

```octave
H = plotQTraceFromFile(filename, pos)
```

Example

```octave
H = plotQTraceFromFile("../trace.cvs", [9,9]])
```

## Dump analysis

The trace analysis allows to understand the learning performance of the agent for each episode.
The trace file contains information of each episode for a given session.
To load the dump run the octave:

```octave
  X = csvread("../maze-dump.csv");
```

### vFromEpisode

It extracts the final estimation of state values for an episode

```octave
  V = vFromEpisode(X(i, :));
```

### plotReturnsFromDump

It plots the returns of session:

```octave
  plotReturnsFromDump(X);
```

### plotStepsFromDump

It plots the steps count of session:

```octave
  plotStepsFromDump(X);
```

### plotErrorsFromDump

It plots the errors of session:

```octave
  plotErrorsFromDump(X);
```

### stats

Computes the statistics on data dump folder.
Creates the steps, returns and errors files containing statistics

### plotFile

Plot the data on a file in y linear scale

### logyPlotFile

Plot the data on a file in y logaritmic scale

## Hyperparameters statistics

To create the statistics of hyper parameters run the simulation that creates the statistics folders

```bash
./runhypers.sh
```

Then run the aggregation of samples running in Octave

```octave
stats("../dump-base")
stats("../dump-alpha-x")
...
stats("../dump-adams")
```

Than create charts by running 
```octave
plotAdam
plotLambda
plotAlpha
....
```