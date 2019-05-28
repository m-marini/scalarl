# State value

Lets $\gamma = 0.999$

Value $v=1$

```text
9|  1O1     |
8|  111     |
7|   *      |
6|          |
5|XXXXXX    |
4|          |
3|          |
2|    XXXXXX|
1|          |
0|          |
  0123456789
```

Value $v=\gamma-1=0.999-1 = -0.001$

```text
| 2 O 2    |
| 2   2    |
| 22222    |
|          |
|XXXXXX    |
|          |
|          |
|    XXXXXX|
|          |
|          |
```

Value $v=\gamma^2-\gamma - 1 \approx -1.001$

```text
|3  O  3   |
|3     3   |
|3     3   |
|3333333   |
|XXXXXX    |
|          |
|          |
|    XXXXXX|
|          |
|          |
```

Value $v=\gamma^3- \gamma^2-\gamma - 1 \approx -2$

```text
|   O   4  |
|       4  |
|       4  |
|       4  |
|XXXXXX44  |
|          |
|          |
|    XXXXXX|
|          |
|          |
```

Value $v = \dots \approx -2.998$

```text
|   O    5 |
|        5 |
|        5 |
|        5 |
|XXXXXX  5 |
|     5555 |
|          |
|    XXXXXX|
|          |
|          |
```

Value $v= \dots \approx -3.995$

```text
|   O     6|
|         6|
|         6|
|         6|
|XXXXXX   6|
|    6    6|
|    666666|
|    XXXXXX|
|          |
|          |
```

Value $v= \dots \approx -4.991$

```text
|   O      |
|          |
|          |
|          |
|XXXXXX    |
|   7      |
|   7      |
|   7XXXXXX|
|          |
|          |
```

Value $v= \dots \approx -5.986$

```text
|   O      |
|          |
|          |
|          |
|XXXXXX    |
|  8       |
|  8       |
|  8 XXXXXX|
|  888     |
|          |
```

Value $v= \dots \approx -6.980$

```text
|   O      |
|          |
|          |
|          |
|XXXXXX    |
| 9        |
| 9        |
| 9  XXXXXX|
| 9   9    |
| 99999    |
```

Value $v= \dots \approx -7.973$

```text
|   O      |
|          |
|          |
|          |
|XXXXXX    |
|0         |
|0         |
|0   XXXXXX|
|0     0   |
|0     0   |
```

Value $v= \dots \approx -8.965$

```text
|   O      |
|          |
|          |
|          |
|XXXXXX    |
|          |
|          |
|    XXXXXX|
|       1  |
|       1  |
```

Value $v= \dots \approx -9.956$

```text
|   O      |
|          |
|          |
|          |
|XXXXXX    |
|          |
|          |
|    XXXXXX|
|        2 |
|        2 |
```

Value $v= \dots \approx -10.946$

```text
|   O      |
|          |
|          |
|          |
|XXXXXX    |
|          |
|          |
|    XXXXXX|
|         3|
|         3|
```

Let us consider the cell $(2,8)$ and compute the action value.

$
q(s, ne) = 1 \\
q(s, n) = q(s,e) = \gamma v(s(2,9)) -1 \approx 0.999 \cdot  0.999-1 = -1.999 \times 10^-3 \\
q(s, se) = q(s, s) = q(s, sw) = q(s, w) = q(s, nw) =
\gamma v(s(2,7)) -1 \approx -\gamma 10^{-3} - 1 = -1.001
$
