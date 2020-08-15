# Test log

[TOC]

## 2020-08-13 - #1 lander-11.yaml

Configuration `debug\lander-11.yaml`

```yaml
---
version: "5"
env:
  dt: 0.25
  g: 1.6
  fuel: 300
  initialLocationRange:
    - [-100, 100]
    - [-100, 100]
    - [100, 100]
  spaceRanges:
    - [-500, 500]
    - [-500, 500]
    - [0, 150]
  landingRadius: 10
  landingSpeedLimits: [2, -4]
  optimalSpeedRanges:
   - [0, 1]
   - [-2, -2]
  jetAccRange:
    - [-1, 1]
    - [-1, 1]
    - [0, 3.2]
  rewards:
    landed:
      base: 100
    hCrashedOnPlatform:
      base: 100
      hSpeed: -5
    vCrashedOnPlatform:
      base: 100
      vSpeed: -0.125
    landedOutOfPlatform:
      base: -30
      distance: -0.3
    hCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    vCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    outOfRange:
      base: -100
      distance: -0.3
      height: -0.3
    outOfFuel:
      base: -100
      distance: -0.3
      height: -0.3
    flying:
      base: -11
      direction: 10
      hSpeed: -3
      vSpeed: -1
  actionDimensions: 3
  encoder: LanderContinuous
  signalRanges:
    - [-3.14, 3.14]
    - [-3.14, 3.14]
    - [0, 64]
    - [0, 16]
    - [0, 8]
    - [-16, 16]
agent:
  type: ActorCritic
  avgReward: -14
  rewardDecay: 0.999
  valueDecay: 0.99
  rewardRange: [-100, 100]
  network:
    seed: 1234
    activation: SOFTPLUS
    numHiddens:
    - 100
    - 30
    - 30
    shortcuts:
      - [ 1, 4 ]
    updater: Sgd
    learningRate: 1000e-3
    maxAbsGradients: 100
    maxAbsParameters: 10e3
    dropOut: 0.8
  actors:
    - type: PolicyActor
      noValues: 8
      range: 
        - [0, 6.09]
      alpha: 100e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 3
      range:
        - [0, 3]
      alpha: 100e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 5
      range:
        - [-6, 6]
      alpha: 100e-3
      prefRange:
        - [-2.4, 2.4]
session:
  numSteps: 30000
  seed: 1234
  kpisOnPlanning: true
```

**Results**

**ANN**
ANN
Linear Increasing rewards trend from -13.6 to -11.8
Linear Decreasing MSE (RMSE) trend from 105.9 (10.3) to 52.8 (7.3)
20% red class
65% yellow class
14% green class
Optimal actor alpha: 8.8e-02, 8.2e-02, 7.5e-02

**Flying status**
Linear Increasing rewards trend from -12.8 to -11.7.
Linear Decreasing direction error trend from 125 DEG to 81 DEG, drop after 10K steps to about 90 DEG and waves between 40 DEG and 140 DEG.
Linear Increasing horizontal speed trend fwrom 0.4 m/s to 1.1 m/s, rise after 10K steps to 1 m/s and quite constant.
Linear Decreasing vertical speed trend from 0.5 m/s to -3.0 m/s great waves after 10K steps from -4 m/s to 0.2 m/s.

**Bad status**
Linear Increasing rewards trend from -151.8 to -25.6 rise at 38 after episodes to -50 and then some spikes to -150 and a spike to 100.
Linear Increasing platform distance trend from 68 m to 80 m.
66 cases of landed out of platform.
1 cases of vertical crashed on platform.
23 cases of vertical crash out of platform.
1 cases of horizontal crash out of platform.
8 cases of out of range.
47 cases of out of fuel.

***Diagnosis***
No actor alpha correction needed.
Retry with more steps (100 k) to see the long run behavior.
Retry with higher learning rate (3) to see if improvements can be obtained.
The trend of distance from platform in termination states indicates a lack of improvement to learn to approach the platform, improvement may be obtained in correlation to the ability to fix the direction during the flight.

## 2020-08-13 - #2 lander-11.yaml

Configuration `debug\lander-11.yaml`

```yaml
---
version: "5"
env:
  dt: 0.25
  g: 1.6
  fuel: 300
  initialLocationRange:
    - [-100, 100]
    - [-100, 100]
    - [100, 100]
  spaceRanges:
    - [-500, 500]
    - [-500, 500]
    - [0, 150]
  landingRadius: 10
  landingSpeedLimits: [2, -4]
  optimalSpeedRanges:
   - [0, 1]
   - [-2, -2]
  jetAccRange:
    - [-1, 1]
    - [-1, 1]
    - [0, 3.2]
  rewards:
    landed:
      base: 100
    hCrashedOnPlatform:
      base: 100
      hSpeed: -5
    vCrashedOnPlatform:
      base: 100
      vSpeed: -0.125
    landedOutOfPlatform:
      base: -30
      distance: -0.3
    hCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    vCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    outOfRange:
      base: -100
      distance: -0.3
      height: -0.3
    outOfFuel:
      base: -100
      distance: -0.3
      height: -0.3
    flying:
      base: -11
      direction: 10
      hSpeed: -3
      vSpeed: -1
  actionDimensions: 3
  encoder: LanderContinuous
  signalRanges:
    - [-3.14, 3.14]
    - [-3.14, 3.14]
    - [0, 64]
    - [0, 16]
    - [0, 8]
    - [-16, 16]
agent:
  type: ActorCritic
  avgReward: -14
  rewardDecay: 0.999
  valueDecay: 0.99
  rewardRange: [-100, 100]
  network:
    seed: 1234
    activation: SOFTPLUS
    numHiddens:
    - 100
    - 30
    - 30
    shortcuts:
      - [ 1, 4 ]
    updater: Sgd
    learningRate: 3000e-3
    maxAbsGradients: 100
    maxAbsParameters: 10e3
    dropOut: 0.8
  actors:
    - type: PolicyActor
      noValues: 8
      range: 
        - [0, 6.09]
      alpha: 100e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 3
      range:
        - [0, 3]
      alpha: 100e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 5
      range:
        - [-6, 6]
      alpha: 100e-3
      prefRange:
        - [-2.4, 2.4]
session:
  numSteps: 30000
  seed: 1234
  kpisOnPlanning: true
```
**Results**

**ANN**
Linear Decreasing rewards trend from -16.8 to -17.4
Linear Decreasing MSE (RMSE) trend from 284.4 (16.9) to 282.6 (16.8)
0% red class
100% yellow class
0% green class
Optimal actor alpha: 4.6e-02, 1.6e-01, 3.9e-02

**Flying status**
Linear Increasing rewards trend from -15.3 to -15.3.
Linear Increasing direction error trend from 178 DEG to 179 DEG.
Linear Decreasing horizontal speed trend from 0.0 m/s to 0.0 m/s.
Linear Decreasing vertical speed trend from 2.5 m/s to 2.5 m/s.

**Bad status**
Linear Increasing rewards trend from -168.4 to -168.1.
Linear Decreasing platform distance trend from 78 m to 77 m.
367 cases of out of range.

***Diagnosis***
Behavior seems to be completly random.
The RMSE is quite high maybe due to learning parameter too high .

## 2020-08-14 - #1 lander-12.yaml

Configuration `debug\lander-12.yaml`

```yaml

---
version: "5"
env:
  dt: 0.25
  g: 1.6
  fuel: 300
  initialLocationRange:
    - [-100, 100]
    - [-100, 100]
    - [100, 100]
  spaceRanges:
    - [-500, 500]
    - [-500, 500]
    - [0, 150]
  landingRadius: 10
  landingSpeedLimits: [2, -4]
  optimalSpeedRanges:
   - [0, 1]
   - [-2, -2]
  jetAccRange:
    - [-1, 1]
    - [-1, 1]
    - [0, 3.2]
  rewards:
    landed:
      base: 100
    hCrashedOnPlatform:
      base: 100
      hSpeed: -5
    vCrashedOnPlatform:
      base: 100
      vSpeed: -0.125
    landedOutOfPlatform:
      base: -30
      distance: -0.3
    hCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    vCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    outOfRange:
      base: -100
      distance: -0.3
      height: -0.3
    outOfFuel:
      base: -100
      distance: -0.3
      height: -0.3
    flying:
      base: -11
      direction: 10
      hSpeed: -3
      vSpeed: -1
  actionDimensions: 3
  encoder: LanderContinuous
  signalRanges:
    - [-3.14, 3.14]
    - [-3.14, 3.14]
    - [0, 64]
    - [0, 16]
    - [0, 8]
    - [-16, 16]
agent:
  type: ActorCritic
  avgReward: -14
  rewardDecay: 0.999
  valueDecay: 0.99
  rewardRange: [-100, 100]
  network:
    seed: 1234
    activation: SOFTPLUS
    numHiddens:
    - 100
    - 30
    - 30
    shortcuts:
      - [ 1, 4 ]
    updater: Sgd
    learningRate: 300e-3
    maxAbsGradients: 100
    maxAbsParameters: 10e3
    dropOut: 0.8
  actors:
    - type: PolicyActor
      noValues: 8
      range: 
        - [0, 6.09]
      alpha: 40e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 3
      range:
        - [0, 3]
      alpha: 20e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 5
      range:
        - [-6, 6]
      alpha: 40e-3
      prefRange:
        - [-2.4, 2.4]
  planner:
    planningSteps: 3
    threshold: 1e-3
    minModelSize: 300
    maxModelSize: 1000
    stateKey:
      type: NormalTiles
      noTiles: [32, 32, 32, 32, 32, 32]
    actionsKey:
      type: Tiles
      noTiles: [8, 3, 5]
      ranges:
        - [0, 6.09]
        - [0, 3]
        - [-6, 6]
session:
  numSteps: 30000
  seed: 1234
  kpisOnPlanning: true
```

***Results***

**ANN**
Linear Increasing rewards trend from -16.9 to -16.6
Linear Increasing MSE (RMSE) trend from 23360.1 (152.8) to 25718.9 (160.4)
8% red class
45% yellow class
47% green class
Optimal actor alpha: 3.9e-03, 2.2e-03, 3.0e-03


**Flying status**
Linear Increasing rewards trend from -15.7 to -14.7, with rise after 2000 steps to about quite constant -15.
Linear Increasing direction error trend from 116 DEG to 132 DEG with a rise after 2600 step to about quite constant 130 DEG.
Linear Decreasing horizontal speed trend from 0.5 m/s to 0.2 m/s with drop at 2600 steps to about quite constant -0.27 m/s.
Linear Increasing vertical speed trend from 2.0 m/s to 2.3 m/s with rise at 500 steps to about quite constant 2.2 m/s.

**Bad status**
Linear Decreasing rewards trend from -167.3 to -167.7.
Linear Decreasing platform distance trend from 76 m to 75 m.
321 cases of out of range.
3 cases of out of fuel.

**Diagnosis**

Good class partitioning, fix the actor alphas, rewards trend quite constant and no improvement, worse error trend.

## 2020-08-14 - #2 lander-11.yaml

Configuration `debug\lander-11.yaml`

```yaml
---
version: "5"
env:
  dt: 0.25
  g: 1.6
  fuel: 300
  initialLocationRange:
    - [-100, 100]
    - [-100, 100]
    - [100, 100]
  spaceRanges:
    - [-500, 500]
    - [-500, 500]
    - [0, 150]
  landingRadius: 10
  landingSpeedLimits: [2, -4]
  optimalSpeedRanges:
   - [0, 1]
   - [-2, -2]
  jetAccRange:
    - [-1, 1]
    - [-1, 1]
    - [0, 3.2]
  rewards:
    landed:
      base: 100
    hCrashedOnPlatform:
      base: 100
      hSpeed: -5
    vCrashedOnPlatform:
      base: 100
      vSpeed: -0.125
    landedOutOfPlatform:
      base: -30
      distance: -0.3
    hCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    vCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    outOfRange:
      base: -100
      distance: -0.3
      height: -0.3
    outOfFuel:
      base: -100
      distance: -0.3
      height: -0.3
    flying:
      base: -11
      direction: 10
      hSpeed: -3
      vSpeed: -1
  actionDimensions: 3
  encoder: LanderContinuous
  signalRanges:
    - [-3.14, 3.14]
    - [-3.14, 3.14]
    - [0, 64]
    - [0, 16]
    - [0, 8]
    - [-16, 16]
agent:
  type: ActorCritic
  avgReward: -14
  rewardDecay: 0.999
  valueDecay: 0.99
  rewardRange: [-100, 100]
  network:
    seed: 1234
    activation: SOFTPLUS
    numHiddens:
    - 100
    - 30
    - 30
    shortcuts:
      - [ 1, 4 ]
    updater: Sgd
    learningRate: 10
    maxAbsGradients: 100
    maxAbsParameters: 10e3
    dropOut: 0.8
  actors:
    - type: PolicyActor
      noValues: 8
      range: 
        - [0, 6.09]
      alpha: 100e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 3
      range:
        - [0, 3]
      alpha: 100e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 5
      range:
        - [-6, 6]
      alpha: 100e-3
      prefRange:
        - [-2.4, 2.4]
session:
  numSteps: 30000
  seed: 1234
  kpisOnPlanning: true
```

**ANN**
Linear Increasing rewards trend from -15.7 to -15.4
Linear Decreasing MSE (RMSE) trend from 194.8 (14.0) to 168.3 (13.0)
0% red class
100% yellow class
0% green class
Optimal actor alpha: 5.8e-02, 4.3e-02, 5.5e-02

**Flying status**
Linear Increasing rewards trend from -14.9 to -14.4.
Linear Decreasing direction error trend from 120 DEG to 112 DEG.
Linear Increasing horizontal speed trend from 0.3 m/s to 0.4 m/s.
Linear Decreasing vertical speed trend from 1.4 m/s to 1.2 m/s.

**Bad status**
Linear Increasing rewards trend from -168.9 to -166.7.
Linear Decreasing platform distance trend from 79 m to 72 m.
189 cases of out of range.
1 cases of out of fuel.


## 2020-08-14 - #3 lander-12.yaml

Configuration `debug\lander-11.yaml`

```yaml
---
version: "5"
env:
  dt: 0.25
  g: 1.6
  fuel: 300
  initialLocationRange:
    - [-100, 100]
    - [-100, 100]
    - [100, 100]
  spaceRanges:
    - [-500, 500]
    - [-500, 500]
    - [0, 150]
  landingRadius: 10
  landingSpeedLimits: [2, -4]
  optimalSpeedRanges:
   - [0, 1]
   - [-2, -2]
  jetAccRange:
    - [-1, 1]
    - [-1, 1]
    - [0, 3.2]
  rewards:
    landed:
      base: 100
    hCrashedOnPlatform:
      base: 100
      hSpeed: -5
    vCrashedOnPlatform:
      base: 100
      vSpeed: -0.125
    landedOutOfPlatform:
      base: -30
      distance: -0.3
    hCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    vCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    outOfRange:
      base: -100
      distance: -0.3
      height: -0.3
    outOfFuel:
      base: -100
      distance: -0.3
      height: -0.3
    flying:
      base: -11
      direction: 10
      hSpeed: -3
      vSpeed: -1
  actionDimensions: 3
  encoder: LanderContinuous
  signalRanges:
    - [-3.14, 3.14]
    - [-3.14, 3.14]
    - [0, 64]
    - [0, 16]
    - [0, 8]
    - [-16, 16]
agent:
  type: ActorCritic
  avgReward: -14
  rewardDecay: 0.999
  valueDecay: 0.99
  rewardRange: [-100, 100]
  network:
    seed: 1234
    activation: SOFTPLUS
    numHiddens:
    - 100
    - 30
    - 30
    shortcuts:
      - [ 1, 4 ]
    updater: Sgd
    learningRate: 300e-3
    maxAbsGradients: 100
    maxAbsParameters: 10e3
    dropOut: 0.8
  actors:
    - type: PolicyActor
      noValues: 8
      range: 
        - [0, 6.09]
      alpha: 4e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 3
      range:
        - [0, 3]
      alpha: 2e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 5
      range:
        - [-6, 6]
      alpha: 4e-3
      prefRange:
        - [-2.4, 2.4]
  planner:
    planningSteps: 3
    threshold: 1e-3
    minModelSize: 300
    maxModelSize: 1000
    stateKey:
      type: NormalTiles
      noTiles: [32, 32, 32, 32, 32, 32]
    actionsKey:
      type: Tiles
      noTiles: [8, 3, 5]
      ranges:
        - [0, 6.09]
        - [0, 3]
        - [-6, 6]
session:
  numSteps: 30000
  seed: 1234
  kpisOnPlanning: true
```

**Result**

**ANN**
Linear Decreasing rewards trend from -14.8 to -15.4
Linear Increasing MSE (RMSE) trend from 19264.3 (138.8) to 20799.6 (144.2)
14% red class
1% yellow class
85% green class
Optimal actor alpha: 4.3e-03, 2.2e-03, 3.2e-03

**Flying status**
Linear Decreasing rewards trend from -14.7 to -14.8 quite constant at -13.5 for 4500 steps and then high variable wawes from -21 to -7.
Linear Decreasing direction error trend from 135 DEG to 69 DEG, quite constant at 170 DEG for 4500 steps then high waves from 30 DEG to 130 DEG.
Linear Increasing horizontal speed trend from 0.7 m/s to 1.5 m/s, quite constant at 0.03 m/s for 4500 step then constant at 1.2 m/s.
Linear Decreasing vertical speed trend from -2.1 m/s to -4.6 m/s variable for 4200 steps then constant at -4 m/s.

**Bad status**
Linear Increasing rewards trend from -78.5 to -41.9 variable for 70 episodes from -150 to -50 then constant at -50 with a sipke to 100 after 200 episodes.
Linear Decreasing platform distance trend from 85 m to 80 m.
90 cases of landed out of platform.
1 cases of vertical crashed on platform.
161 cases of vertical crash out of platform.
9 cases of out of range.
6 cases of out of fuel.

**Diagnosis**
Both reward trend and RMSE trends are bad.
Right actor alphas.
Quit good clolred classes distribution.
Unstable improvements of direction during flying after 4500 step and not significant improvements of distance from platform in bad status.
It learnt to not run out of fuel or out of ranges.
Quit high number of vertical crash out of platform.
A single case of vertical crash on platform probably the 100 reward after 200 episodes.
Try with higher leraning paramter (1).

## 2020-08-15 - #1 lander-12.yaml

Configuration `debug\lander-12.yaml`
```yaml
---
version: "5"
env:
  dt: 0.25
  g: 1.6
  fuel: 300
  initialLocationRange:
    - [-100, 100]
    - [-100, 100]
    - [100, 100]
  spaceRanges:
    - [-500, 500]
    - [-500, 500]
    - [0, 150]
  landingRadius: 10
  landingSpeedLimits: [2, -4]
  optimalSpeedRanges:
   - [0, 1]
   - [-2, -2]
  jetAccRange:
    - [-1, 1]
    - [-1, 1]
    - [0, 3.2]
  rewards:
    landed:
      base: 100
    hCrashedOnPlatform:
      base: 100
      hSpeed: -5
    vCrashedOnPlatform:
      base: 100
      vSpeed: -0.125
    landedOutOfPlatform:
      base: -30
      distance: -0.3
    hCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    vCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    outOfRange:
      base: -100
      distance: -0.3
      height: -0.3
    outOfFuel:
      base: -100
      distance: -0.3
      height: -0.3
    flying:
      base: -11
      direction: 10
      hSpeed: -3
      vSpeed: -1
  actionDimensions: 3
  encoder: LanderContinuous
  signalRanges:
    - [-3.14, 3.14]
    - [-3.14, 3.14]
    - [0, 64]
    - [0, 16]
    - [0, 8]
    - [-16, 16]
agent:
  type: ActorCritic
  avgReward: -14
  rewardDecay: 0.999
  valueDecay: 0.99
  rewardRange: [-100, 100]
  network:
    seed: 1234
    activation: SOFTPLUS
    numHiddens:
    - 100
    - 30
    - 30
    shortcuts:
      - [ 1, 4 ]
    updater: Sgd
    learningRate: 1000e-3
    maxAbsGradients: 100
    maxAbsParameters: 10e3
    dropOut: 0.8
  actors:
    - type: PolicyActor
      noValues: 8
      range: 
        - [0, 6.09]
      alpha: 4e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 3
      range:
        - [0, 3]
      alpha: 2e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 5
      range:
        - [-6, 6]
      alpha: 4e-3
      prefRange:
        - [-2.4, 2.4]
  planner:
    planningSteps: 3
    threshold: 1e-3
    minModelSize: 300
    maxModelSize: 1000
    stateKey:
      type: NormalTiles
      noTiles: [32, 32, 32, 32, 32, 32]
    actionsKey:
      type: Tiles
      noTiles: [8, 3, 5]
      ranges:
        - [0, 6.09]
        - [0, 3]
        - [-6, 6]
session:
  numSteps: 30000
  seed: 1234
  kpisOnPlanning: true
```
**Result**

**ANN**
Linear Decreasing rewards trend from -14.8 to -22.0
Linear Decreasing MSE (RMSE) trend from 21150.0 (145.4) to 20671.9 (143.8)
11% red class
43% yellow class
47% green class
Optimal actor alpha: 4.2e-03, 2.6e-03, 3.6e-03

**Flying status**
Linear Decreasing rewards trend from -14.2 to -22.0.
Linear Increasing direction error trend from 125 DEG to 127 DEG.
Linear Increasing horizontal speed trend from -0.0 m/s to 2.7 m/s rise from about 0.3 m/s to 2 m/S after 12K steps.
Logarithmic Decreasing vertical speed trend from 1.8 m/s to 0.3 m/s, drop from 1.3 m/s to -1.2 m/s after 12K steps.

**Bad status**
Linear Increasing rewards trend from -185.3 to -108.8, change from quit constant at -150 to variable -150, -50 after 78 episodes.
Linear Increasing platform distance trend from 58 m to 173 m.
10 cases of landed out of platform.
14 cases of horizontal crash out of platform.
65 cases of out of range.
44 cases of out of fuel.

**Diagnosis**
Bad results. Try with lower learning rate (0.1).


## 2020-08-15 - #2 lander-12.yaml

Configuration `debug\lander-12.yaml`

```yaml
---
version: "5"
env:
  dt: 0.25
  g: 1.6
  fuel: 300
  initialLocationRange:
    - [-100, 100]
    - [-100, 100]
    - [100, 100]
  spaceRanges:
    - [-500, 500]
    - [-500, 500]
    - [0, 150]
  landingRadius: 10
  landingSpeedLimits: [2, -4]
  optimalSpeedRanges:
   - [0, 1]
   - [-2, -2]
  jetAccRange:
    - [-1, 1]
    - [-1, 1]
    - [0, 3.2]
  rewards:
    landed:
      base: 100
    hCrashedOnPlatform:
      base: 100
      hSpeed: -5
    vCrashedOnPlatform:
      base: 100
      vSpeed: -0.125
    landedOutOfPlatform:
      base: -30
      distance: -0.3
    hCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    vCrashedOutOfPlatform:
      base: -30
      distance: -0.3
    outOfRange:
      base: -100
      distance: -0.3
      height: -0.3
    outOfFuel:
      base: -100
      distance: -0.3
      height: -0.3
    flying:
      base: -11
      direction: 10
      hSpeed: -3
      vSpeed: -1
  actionDimensions: 3
  encoder: LanderContinuous
  signalRanges:
    - [-3.14, 3.14]
    - [-3.14, 3.14]
    - [0, 64]
    - [0, 16]
    - [0, 8]
    - [-16, 16]
agent:
  type: ActorCritic
  avgReward: -14
  rewardDecay: 0.999
  valueDecay: 0.99
  rewardRange: [-100, 100]
  network:
    seed: 1234
    activation: SOFTPLUS
    numHiddens:
    - 100
    - 30
    - 30
    shortcuts:
      - [ 1, 4 ]
    updater: Sgd
    learningRate: 100e-3
    maxAbsGradients: 100
    maxAbsParameters: 10e3
    dropOut: 0.8
  actors:
    - type: PolicyActor
      noValues: 8
      range: 
        - [0, 6.09]
      alpha: 4e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 3
      range:
        - [0, 3]
      alpha: 2e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 5
      range:
        - [-6, 6]
      alpha: 4e-3
      prefRange:
        - [-2.4, 2.4]
  planner:
    planningSteps: 3
    threshold: 1e-3
    minModelSize: 300
    maxModelSize: 1000
    stateKey:
      type: NormalTiles
      noTiles: [32, 32, 32, 32, 32, 32]
    actionsKey:
      type: Tiles
      noTiles: [8, 3, 5]
      ranges:
        - [0, 6.09]
        - [0, 3]
        - [-6, 6]
session:
  numSteps: 30000
  seed: 1234
  kpisOnPlanning: true
```

**Result**

**ANN**
Linear Increasing rewards trend from -12.7 to -12.4
Linear Increasing MSE (RMSE) trend from 20656.9 (143.7) to 21095.0 (145.2)
4% red class
0% yellow class
96% green class
Optimal actor alpha: 4.2e-03, 2.1e-03, 3.1e-03

**Flying status**
Linear Decreasing rewards trend from -12.2 to -12.3.
Linear Decreasing direction error trend from 171 DEG to 171 DEG.
Linear Decreasing horizontal speed trend from 0.0 m/s to 0.0 m/s.
Linear Decreasing vertical speed trend from -1.5 m/s to -1.9 m/s.

**Episodes**
Linear Increasing rewards trend from -82.4 to -60.8.
Linear Increasing platform distance trend from 68 m to 76 m.
58 cases of landed out of platform.
39 cases of vertical crash out of platform.
35 cases of out of fuel.
1 cases of landed.
1 cases of out of range.