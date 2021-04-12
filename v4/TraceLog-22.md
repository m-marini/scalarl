[TOC]

File: `lander22.yaml`

## 2020-08-23 - #1

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
  planner:
    planningSteps: 3
    threshold: 1e-3
    minModelSize: 300
    maxModelSize: 1000
    stateKey:
      type: Binary
    actionsKey:
      type: Tiles
      noTiles: [8, 3, 5]
      ranges:
        - [0, 6.09]
        - [0, 3]
        - [-6, 6]
session:
  numSteps: 30000
  kpisOnPlanning: true
```

**ANN Exam**
Linear Increasing rewards trend from -11.7 to -10.0
Linear Increasing MSE (RMSE) trend from 16373.1 (128.0) to 18798.3 (137.1)
4% red class
36% yellow class
60% green class
Optimal actor alpha: 4.6e-03, 2.5e-03, 3.5e-03

**Flying Status Exam**
Linear Increasing rewards trend from -10.9 to -9.9.
Linear Decreasing direction error trend from 118 DEG to 102 DEG.
Linear Increasing horizontal speed trend from 0.6 m/s to 0.8 m/s.
Linear Decreasing vertical speed trend from -0.7 m/s to -2.1 m/s.

**Episodes Exam**
Linear Increasing rewards trend from -103.3 to -42.2.
Linear Increasing platform distance trend from 60 m to 62 m.
83 cases of landed out of platform between 259 and 29959 steps.
29 cases of out of fuel between 1679 and 19878 steps.
12 cases of out of range between 129 and 29578 steps.
8 cases of horizontal crash out of platform between 15454 and 25197 steps.
6 cases of vertical crash out of platform between 1292 and 9987 steps.
3 cases of landed between 4838 and 24374 steps.
1 cases of vertical crashed on platform between 10169 and 10169 steps.

**Diagnosis**
Wrong actor alpha. High RMSE.

**Therapy**
Correct the actor alpha to 4 mU, 2 mU, 3 mU.
Reduce the learning parameter to 100 mU.

## 2020-08-23 - #2

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
      alpha: 3e-3
      prefRange:
        - [-2.4, 2.4]
  planner:
    planningSteps: 3
    threshold: 1e-3
    minModelSize: 300
    maxModelSize: 1000
    stateKey:
      type: Binary
    actionsKey:
      type: Tiles
      noTiles: [8, 3, 5]
      ranges:
        - [0, 6.09]
        - [0, 3]
        - [-6, 6]
session:
  numSteps: 30000
  kpisOnPlanning: true
```

**ANN Exam**
Logarithmic Increasing rewards trend from -15.2 to -13.1
Linear Increasing MSE (RMSE) trend from 18671.4 (136.6) to 19521.6 (139.7)
15% red class
1% yellow class
84% green class
Optimal actor alpha: 4.4e-03, 2.2e-03, 3.2e-03

**Flying Status Exam**
Linear Increasing rewards trend from -15.3 to -13.5.
Linear Decreasing direction error trend from 98 DEG to 88 DEG, high variablility.
Linear Increasing horizontal speed trend from 1.1 m/s to 1.3 m/s, quite constant at 1.0 m/s after 20K steps.
Linear Decreasing vertical speed trend from -1.1 m/s to -2.7 m/s, quite constant at -2 m/s after 20K steps.

**Episodes Exam**
Linear Increasing rewards trend from -101.7 to -36.8.
Linear Increasing platform distance trend from 83 m to 89 m.
63 cases of landed out of platform between 2294 and 29948 steps.
60 cases of vertical crash out of platform between 2592 and 29725 steps.
16 cases of horizontal crash out of platform between 5258 and 25130 steps.
14 cases of out of range between 53 and 1720 steps.
9 cases of out of fuel between 2021 and 28798 steps.
1 cases of vertical crashed on platform between 22370 and 22370 steps.

**Diagnosis**
Not available

**Therapy**
Increase steps to 100K.

## 2020-08-25 - #1

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
      alpha: 3e-3
      prefRange:
        - [-2.4, 2.4]
  planner:
    planningSteps: 3
    threshold: 1e-3
    minModelSize: 300
    maxModelSize: 1000
    stateKey:
      type: Binary
    actionsKey:
      type: Tiles
      noTiles: [8, 3, 5]
      ranges:
        - [0, 6.09]
        - [0, 3]
        - [-6, 6]
session:
  numSteps: 100000
  kpisOnPlanning: true
```

**ANN Exam**
Linear Increasing rewards trend from -14.5 to -11.7 consant at -12.5 after 100K iterations (25K steps).
Linear Increasing MSE (RMSE) trend from 20889.6 (144.5) to 24078.5 (155.2) constant at 23000 (150) after 100K iteration (25K steps)
5% red class
0% yellow class
95% green class
Optimal actor alpha: 4.1e-03, 2.1e-03, 3.0e-03

**Flying Status Exam**
Linear Increasing rewards trend from -14.0 to -11.6 constant at -12 after 25K steps.
Linear Increasing direction error trend from 111 DEG to 195 DEG constant at 170 DEG after 25k steps.
Linear Decreasing horizontal speed trend from 0.7 m/s to -0.2 m/s constant at 0 m/s after 25K steps.
Linear Decreasing vertical speed trend from -0.4 m/s to -2.1 m/s drop to around -1.5 m/s after 25K steps.

**Episodes Exam**
Linear Increasing rewards trend from -123.6 to -52.6.
Linear Decreasing platform distance trend from 81 m to 78 m.
130 cases of out of fuel between 536 and 99496 steps.
129 cases of landed out of platform between 3035 and 99662 steps.
127 cases of vertical crash out of platform between 3661 and 99927 steps.
43 cases of out of range between 110 and 24194 steps.
4 cases of landed between 48925 and 97096 steps.
2 cases of vertical crashed on platform between 43844 and 45838 steps.
1 cases of horizontal crash out of platform between 9685 and 9685 steps.

**Disgnosis**
Not available
