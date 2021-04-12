[TOC]

File: `lander31.yaml`

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
  numSteps: 30000
  kpisOnPlanning: true
```

**ANN Exam**
Linear Increasing rewards trend from -11.4 to -8.0
Linear Increasing MSE (RMSE) trend from 108.9 (10.4) to 122.2 (11.1)
16% red class
51% yellow class
32% green class
Optimal actor alpha: 7.4e-02, 7.4e-02, 6.2e-02

**Flying Status Exam**
Linear Increasing rewards trend from -10.2 to -8.1.
Linear Decreasing direction error trend from 105 DEG to 80 DEG.
Linear Decreasing horizontal speed trend from 1.0 m/s to 0.8 m/s.
Linear Decreasing vertical speed trend from -0.5 m/s to -1.8 m/s.

**Episodes Exam**
Linear Increasing rewards trend from -116.9 to -51.8.
Linear Decreasing platform distance trend from 63 m to 49 m.
56 cases of landed out of platform between 616 and 29906 steps.
41 cases of out of fuel between 485 and 26902 steps.
26 cases of out of range between 58 and 29153 steps.
19 cases of vertical crash out of platform between 2852 and 29764 steps.
5 cases of landed between 13588 and 25737 steps.
2 cases of vertical crashed on platform between 6114 and 12087 steps.
2 cases of horizontal crash out of platform between 18802 and 28389 steps.
