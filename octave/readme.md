# Debug utilities

## Trace file format

The trace file format is composed by:

| Field  | Offset | Size |
|:-------|-------:|-----:|
| S0     |      1 |  200 |
| Q0     |    201 |    8 |
| ACTION |    209 |    1 |
| REWARD |    210 |    1 |
| S1     |    211 |  410 |
| Q1     |    411 |    8 |
| NQ0    |    419 |    8 |

## Dump file format

Each record of the dump file rappresents an episode and the format is composed by:

| Field     | Offset |                  Size |
|:----------|-------:|----------------------:|
| STEPCOUNT |      1 |                     1 |
| RETURNS   |      2 |                     1 |
| ERRORS    |      3 |                     1 |
| STATES    |      4 | 10 x 10 x 200 = 20000 |
| Q         |  20004 |     10 x 10 x 8 = 800 |

the states consist of 10 x 10 possible subject location of
(10 x 10 values with subject location flag and
10 x 10 values with obstacle flags)

the actions consist of 10 x 10 possible subject location of
8 action values

## readTrace

Returns the data of state transition capture by the trace file:

- the subject position in the maze environment
- the action value estimated at that position
- the action applied
- the reward obtained
- the resulting subject position
- the action value estimated at the resulting position
- the action value estimated at inital position after the reinforcement learning

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

Returns the agent status at a step from dump of episodes:

- the position in the maze
- the action values at position
- the action values at position
- the best action at position
- the best action map
