# Hyper parameters analysis

The performances of neural network are going to be measured by running 100 training sessions over different configurations varying hyper parameters.

The values of hyper parameter are

| Hyper parameter   |       |       |        |      |       |
|-------------------|------:|------:|-------:|-----:|------:|
| $\alpha$          |  0.01 |  0.03 |  * 0.1 |  0.3 |     1 |
| $\varepsilon$     | 0.001 | 0.003 | * 0.01 | 0.03 | 0.100 |
| $\lambda$         |     0 |   0.5 |  * 0.7 |  0.8 |   0.9 |
| $\kappa$          |   * 1 |     2 |      4 |    8 |    16 |
| no. hiddens nodes |   * 0 |     3 |     10 |   30 |   100 |

(*) are the base values

The average of returns and the average of errors per episode are computed over the different sessions.
