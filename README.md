# ScalaRL

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

## Current Goal 19

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
