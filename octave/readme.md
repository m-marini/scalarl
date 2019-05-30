# Debug utilities

## Trace file format

The trace file format is composed by:

| Field      | Offset | Size |
|:-----------|-------:|-----:|
| EPISODE    |      1 |    1 |
| STEP       |      2 |    1 |
| ACTION     |      3 |    1 |
| REWARD     |      4 |    1 |
| ENDUP      |      5 |    1 |
| PREV_POS   |      6 |    2 |
| RESULT_POS |      8 |    2 |
| PREV_Q     |     10 |    8 |
| RESULT_Q   |     18 |    8 |
| PREV_Q1    |     26 |    8 |

## Dump file format

Each record of the dump file rapresents an episode and the format is composed by:

| Field     | Offset |                  Size |
|:----------|-------:|----------------------:|
| STEPCOUNT |      1 |                     1 |
| RETURNS   |      2 |                     1 |
| ERRORS    |      3 |                     1 |
| Q(s,a)    |      4 |     8 x 10 x 10 = 800 |

Q(s,a) consist of 8 action values for 10 columns for 10 rows of possible subject locations

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

## stats

Computes the statistics on data dump folder.
Creates the steps, returns and errors files containing statistics

## plotFile

Plot the data on a file in y linear scale

## logyPlotFile

Plot the data on a file in y logaritmic scale

## readTrace

Returns the data of state transition capture by the trace file:

- the subject position in the maze environment before step
- the subject position in the maze environment after step
- the action applied
- the reward obtained
- the action values estimated before step
- the action values estimated at result position before step
- the action values estimated at result position after step

## qlearn 

Returns the analysis of state transition data:

- the errors after reinforcement learning
- the estimation errors
- the state value estimated at the initial position
- the state value estimated at the final position
- the expected action value estimated at initial position after the transition
- the subject position in the maze environment
- the action value estimated at that position
- the action applied
- the reward obtained
- the resulting subject position
- the action value estimated at the resulting position
- the action value estimated at inital position after the reinforcement learning

## readErrors

Returns the errors from dump of episodes

## readReturns

Returns the returns (discounted sum of rewards) from dump of episodes

## readStepCount

Returns the returns (discounted sum of rewards) from dump of episodes

## readDumpAgent

Returns the agent status for a given episode from dump of episodes:

- `Q` the action values at each position and direction (10, 10, 8)
- `POS` best action at position (10, 10)
