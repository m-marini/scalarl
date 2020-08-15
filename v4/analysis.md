# Analysis Functions

[TOC]

## Flight Analysis

Draws charts of behaviour indicators from trace data.

1. The rewards of flying status during the session.
2. The direction error of flying status during the session.
3. The horizontal speed of flying status during the session.
4. The vertical speed of flying status during the session.

```octave
analyzeFlight(
  X,              # The trace data
  NPOINTS = 300,  # The number of points in the charts
  LENGTH = 300    # The number of averaging samples in the charts
  )
```

## Episodes Analysis

Draws charts of behaviour indicators from trace data.

1. The rewards of bad landing states during the session.
2. The status code distribution of bad landing states.
3. The distance from platform of bad landing states during the session.

```octave
analyzeEpisodes(
  X,              # The trace data
  NPOINTS = 300,  # The number of points in the charts
  LENGTH = 300    # The number of averaging samples in the charts
  )
```

## Learning Analysis for Policy Actors Agent

Draw charts of learing performance for Policy Actors Agent from kpi file.

1. Steps lerning classes chart
2. K distribution chart
3. Historical average rewards and trend
4. Historical squared errors and trend
5. J distribution chart for each actor
5. Alpha percentile and average for each actor

The step learning classes allow to estimate the ANN learning parameter.
The red slice indicates the percentage of steps that worse ANN, it should be less than 10%. A large slice indicates a learning parameter too high.
The yellow slice indicates the percentage of steps that did not improve significantly the ANN. A large slice indicates a learning parameter too small.
The green slice indicates the percentage of steps that improved the ANN.

The historical squared errors chart should indicate a descreasing trend. An increasing trend may be caused by a wrong ANN learning parameter (too high or too low).

The historical reward chart should indicate a increasing trend. A decreasing trend may be caused be worng ANN learning parameter or wrong actor alpha parameters not able to change significantly the behavior of agent.

The alpha average should indicate the optimal alpha parameter for the actors.

```octave
analyzeDiscreteAgent(
  X,                      # the kpi data
  EPSH = 0.24,            # the optimal range of h to be considered
  K0 = 0.7,               # the K threshold for C1 class
  EPS = 100e-6,           # the minimun J value to be considered
  PRC = [50 : 10 : 90]',  # the percentiles in the chart
  BINS = 20,              # the number of bins in histogram
  NPOINTS = 300,          # the number of point for historical charts
  LENGTH = 300            # the number of averaging samples for historical chrts
  )
```

## Learning Analysis for Gaussian Actors Agent

Draw charts of learing performance for Gaussian Actors Agent from kpi file.

```octave
analyzeGaussianAgent(
    DATA,
    EPSMU = [10e-3 62.5e-3 250e-3],
    EPSHS = [347e-3 347e-3 347e-3],
    K0 = 0.7,
    EPS = 10e-6,
    PRC = [50 : 10 : 90]',
    BINS = 20)
```
