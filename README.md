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
