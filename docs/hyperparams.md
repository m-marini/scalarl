# Hyper parameters analysis

## Non normalized inputs

The performances of neural network are going to be measured by running 100 training sessions over different configurations varying hyper parameters.
The neural network is feed with binary non normilzed data (0, 1 values).

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

## Normalized inputs with batch and ADAM

The performances of neural network are going to be measured by running 100 training sessions over different configurations varying hyper parameters.
The neural network is feed with binary normilzed data (-1, 1 values).
The neural network is a linear regression network.

The values of hyper parameter are

| Hyper parameter   |                     |                     |                        |                      |            |
|-------------------|--------------------:|--------------------:|-----------------------:|---------------------:|-----------:|
| $\alpha$          | $10 \times 10^{-6}$ | $30 \times 10^{-6}$ | * $100 \times 10^{-6}$ | $300 \times 10^{-6}$ |  $10^{-3}$ |
| $\varepsilon$     |               0.001 |               0.003 |                 * 0.01 |                 0.03 |        0.1 |
| $\lambda$         |                   0 |                 0.5 |                  * 0.7 |                  0.8 |        0.9 |
| $\kappa$          |                 * 1 |                   2 |                      4 |                    8 |         16 |
| max history       |                   1 |                  10 |                  * 100 |                  300 |       1000 |
| batch iterations  |                 * 1 |                   3 |                     10 |                   30 |        100 |

(*) base values

The average of returns and the average of errors per episode are computed over the different sessions.
