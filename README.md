# ScalaRL

[TOC]

Study of Reinforcement Learning algorithm in a simple environment.

## Running

To run the generator execute

```bash
./runnit.sh
```

It repeats forever running the generator createing `data/maze-dump-$(date).csv` file for each generation.

## Configuration

The configuration is defined in `maze.yaml` file and consists of a maze map with `'*'` initial position of agent, `'o'` target position and `'X'` forbidden position (wall).

## Analysis

The analysis of data can be done using octave script in `octave` folders.
See `octave/readme.md`

## Hyper parameters measuring

Analyze the learning state values step by step.
Run the lerning session by

```yaml
---
env:
  map:
  - "|   O      |"
  - "|          |"
  - "|          |"
  - "|          |"
  - "|XXXXXX    |"
  - "|          |"
  - "|          |"
  - "|    XXXXXX|"
  - "|          |"
  - "|         *|"
agent:
  numInputs: 208
  numActions: 8
  numHiddens: []
  seed: 1234
  learningRate: 1e-3
  epsilon: 0.01
  gamma: 0.999
  lambda: 0.7
  kappa: 1
  maxAbsGradients: 1
  maxAbsParameters: 1e3
  type: TDAAgent
  model: maze.zip
session:
  numEpisodes: 10
  maxEpisodeLength: 10
  sync: 0
  mode: stats
  trace: trace.csv
  dump: maze-dump.csv
```

Analize the results plotting V and Q for traget cell positions with Octave functions `plotVFromTrace` and `plotQFromTrace`.

The experiments did not show any capability of learning. When the subject is far away the target, the agent behave with random exploring actions. If this behavior does not lead to the achievement of the objective, the rewards are negative and the Q values ​​tend to the asymptopic value of $\frac{R}{1-\gamma}$.
At each step the Q value associated with the action taken is decreased making a new strategy that rewards another action be adopted.
making a new strategy that rewards another action be adopted.
The finial behaviour is an equal distribution of probability of choosing the available actions that can be measured by the frequency of actions.

 If the probability is equal distributed between actions the expected  maximum frequency of all action is

 ```math
 F = \max(f_i) = \left| \frac{n+m-1}{m} \right|
 ```

the relative difference between the maximun frequency and the expected value is

```math
\mu = \frac{\max(f_i) - F}{F} = \frac{\max(f_i)}{F} - 1
```

Let us consider valid the uniform distribution if $\mu \le \mu_0$ then

```math
\max(f_i) \le (1+\mu_0) F
```

Use Octave script 

```octave
plotCoverage(trace-file)
```

and

```octave
plotMu(trace-file)
```

to plot the maze stats from trace file.

The cause of unability to learng was the learning rate

```math
  \alpha = 1 \times 10^{-3}
```

to low, increasing it to
```math
\alpha = 100 \times 10^{-3}
```
allowed the network to learn the maze solution wihtin about 40 episodes.