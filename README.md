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

## Current goal #18

Run different sessions on maze environment with different agent (QAgent, TDQAgent) and analize the results.
It is expected that learning rate of TDQAgent should be heighr of QAgent.

Analyze different network configurations with no hidden layer, one hidden layer or two hidden layer.

Run a QAgent without hidden layer (linear regretion) with 300 episodes limited to max 300 steps.
