[TOC]

File: `lander21.yaml`

## 2020-08-19 - #1

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
  encoder: LanderTiles
  hash: 300
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
  kpisOnPlanning: true
```

**ANN Exam**
Linear Increasing rewards trend from -14.0 to -13.4
Linear Decreasing MSE (RMSE) trend from 117.7 (10.8) to 87.0 (9.3)
0% red class
100% yellow class
0% green class
Optimal actor alpha: 7.7e-02, 5.8e-02, 7.3e-02

**Flying Status Exam**
Linear Increasing rewards trend from -13.4 to -12.8.
Linear Decreasing direction error trend from 131 DEG to 126 DEG.
Linear Increasing horizontal speed trend from 0.2 m/s to 0.2 m/s.
Linear Decreasing vertical speed trend from 0.2 m/s to -0.5 m/s.


**Episodes Exam**
Linear Increasing rewards trend from -158.0 to -128.8.
Linear Decreasing platform distance trend from 83 m to 73 m.
65 cases of out of fuel between 544 and 29746 steps.
42 cases of out of range between 119 and 29261 steps.
13 cases of landed out of platform between 3738 and 29445 steps.

**Diagnosis**
No sign of learning.

**Therapy**
Reduce learning parameter to 300mU.

## 2020-08-22 - #1

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
  encoder: LanderTiles
  hash: 300
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
      alpha: 100e-3
      prefRange:
 ♠       - [-2.4, 2.4]
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
  kpisOnPlanning: true
```

**ANN Exam**
Linear Increasing rewards trend from -11.2 to -6.4
Linear Decreasing MSE (RMSE) trend from 81.2 (9.0) to 69.1 (8.3)
12% red class
57% yellow class
32% green class
Optimal actor alpha: 8.9e-02, 8.8e-02, 8.1e-02

**Flying Status Exam**
Linear Increasing rewards trend from -10.2 to -6.5.
Linear Increasing direction error trend from 97 DEG to 103 DEG.
Linear Decreasing horizontal speed trend from 0.9 m/s to 0.9 m/s.
Linear Decreasing vertical speed trend from -1.6 m/s to -2.1 m/s.

**Episodes Exam**
Linear Increasing rewards trend from -78.6 to -23.0.
Linear Decreasing platform distance trend from 67 m to 43 m.
120 cases of landed out of platform between 627 and 29967 steps.
15 cases of out of fuel between 928 and 28386 steps.
9 cases of landed between 12017 and 29108 steps.
5 cases of out of range between 292 and 8817 steps.
3 cases of horizontal crash out of platform between 5102 and 9013 steps.
1 cases of vertical crash out of platform between 1123 and 1123 steps.

**Diagnosis**
The agent learnt to land at right speed and some cases of land on platform. The distance of platform at end episode is reducing.

**Therapy**
Increase the number of steps to 100K to see if the landed case increase.

## 2020-08-19 - #1

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
  encoder: LanderTiles
  hash: 300
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
  numSteps: 100000
  kpisOnPlanning: true
```

**ANN Exam**
Linear Increasing rewards trend from -9.5 to -6.2
Linear Decreasing MSE (RMSE) trend from 105.7 (10.3) to 66.7 (8.2)
13% red class
62% yellow class
25% green class
Optimal actor alpha: 8.2e-02, 7.8e-02, 7.4e-02

**Flying Status Exam**
Linear Increasing rewards trend from -8.9 to -6.2.
Linear Decreasing direction error trend from 91 DEG to 87 DEG.
Linear Increasing horizontal speed trend from 0.9 m/s to 1.0 m/s.
Linear Decreasing vertical speed trend from -1.4 m/s to -2.2 m/s.

**Episodes Exam**
Linear Increasing rewards trend from -77.9 to -36.4.
Linear Decreasing platform distance trend from 55 m to 48 m.
350 cases of landed out of platform between 593 and 99658 steps.
83 cases of out of fuel between 300 and 98963 steps.
33 cases of vertical crash out of platform between 7374 and 99963 steps.
23 cases of landed between 8445 and 95374 steps.
21 cases of out of range between 1141 and 90968 steps.
9 cases of horizontal crash out of platform between 22284 and 93138 steps.
1 cases of vertical crashed on platform between 77685 and 77685 steps.
1 cases of horizontal crash on platform between 92550 and 92550 steps.

**Diagnosis**
The agent is learning

**Therapy**
No changes
