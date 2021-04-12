[TOC]

## 2020-08-17 - #1 lander11.yaml

```yaml
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
  kpisOnPlanning: true
```

**ANN Exam**
Linear Increasing rewards trend from -18.3 to -16.6
Linear Decreasing MSE (RMSE) trend from 81.5 (9.0) to 69.3 (8.3)
0% red class
100% yellow class
0% green class
Optimal actor alpha: 9.0e-02, 2.5e-01, 8.5e-02

**Flying status Exam**
Linear Increasing rewards trend from -18.5 to -16.0.
Linear Decreasing direction error trend from 113 DEG to 94 DEG.
Linear Decreasing horizontal speed trend from 1.8 m/s to 1.8 m/s.
Linear Decreasing vertical speed trend from -2.2 m/s to -4.8 m/s, constant at -2.5 after 2700 steps, constant at -4 after 7000 steps.

**Episodes Exam**
Linear Increasing rewards trend from -66.9 to -48.1, after 2000 steps rise to constant -50.
Linear Decreasing platform distance trend from 95 m to 76 m, very variable.
155 cases of vertical crash out of platform between 2274 and 29985 steps.
53 cases of landed out of platform between 1480 and 28611 steps.
48 cases of horizontal crash out of platform between 1278 and 29349 steps.
4 cases of out of range between 284 and 1048 steps.
3 cases of out of fuel between 585 and 2575 steps.
1 cases of landed between 2947 and 2947 steps.
1 cases of vertical crashed on platform between 14502 and 14502 steps.

**Diagnosis**
Not available

**Therapy**
Correct alpha actors to 90 mU, 200 mU, 80 mU.
Increase steps to 100K.

## 2020-08-17 - #2 lander11.yaml

```yaml
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
      alpha: 90e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 3
      range:
        - [0, 3]
      alpha: 250e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 5
      range:
        - [-6, 6]
      alpha: 80e-3
      prefRange:
        - [-2.4, 2.4]
session:
  numSteps: 100000
  kpisOnPlanning: true
```

**ANN Exam**
Linear Decreasing rewards trend from -21.1 to -22.1
Linear Increasing MSE (RMSE) trend from 130.5 (11.4) to 159.5 (12.6)
0% red class
100% yellow class
0% green class
Optimal actor alpha: 6.5e-02, 1.5e-01, 6.2e-02

**Flying status**
Linear Decreasing rewards trend from -20.8 to -21.3.
Linear Decreasing direction error trend from 124 DEG to 124 DEG.
Linear Increasing horizontal speed trend from 2.0 m/s to 2.0 m/s.
Linear Increasing vertical speed trend from -0.3 m/s to 0.5 m/s.

**Episodes Exam**
Linear Decreasing rewards trend from -143.0 to -176.5.
Linear Increasing platform distance trend from 124 m to 128 m.
244 cases of out of range between 5408 and 99919 steps.
170 cases of out of fuel between 5206 and 99584 steps.
38 cases of horizontal crash out of platform between 234 and 93932 steps.
27 cases of landed out of platform between 533 and 95403 steps.

**Diagnosis**
Not available.

**Therapy**
Reduce leraning rate to 1.

## 2020-08-17 - #3 lander11.yaml

```yaml
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
      alpha: 90e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 3
      range:
        - [0, 3]
      alpha: 250e-3
      prefRange:
        - [-2.4, 2.4]
    - type: PolicyActor
      noValues: 5
      range:
        - [-6, 6]
      alpha: 80e-3
      prefRange:
        - [-2.4, 2.4]
session:
  numSteps: 100000
  kpisOnPlanning: true
```

**ANN Exam**
Linear Increasing rewards trend from -14.8 to -10.4, rise from -15 to -11 after 27K steps.
Linear Decreasing MSE (RMSE) trend from 102.5 (10.1) to 47.9 (6.9)
7% red class
58% yellow class
35% green class
Optimal actor alpha: 9.2e-02, 1.1e-01, 7.9e-02

**Flying Status Exam**
Linear Increasing rewards trend from -14.3 to -10.2, less variable at -11 after 27K steps.
Linear Increasing direction error trend from 82 DEG to 125 DEG.
Linear Decreasing horizontal speed trend from 1.0 m/s to 0.6 m/s.
Linear Decreasing vertical speed trend from -0.7 m/s to -1.5 m/s.

**Episodes Exam**
Linear Increasing rewards trend from -118.9 to -74.6.
Linear Decreasing platform distance trend from 85 m to 53 m.
201 cases of out of fuel between 300 and 99729 steps.
131 cases of vertical crash out of platform between 17713 and 99918 steps.
76 cases of landed out of platform between 17116 and 97298 steps.
19 cases of out of range between 25037 and 60323 steps.
6 cases of horizontal crash out of platform between 24648 and 90816 steps.
2 cases of landed between 45223 and 89760 steps.
1 cases of horizontal crash on platform between 99127 and 99127 steps.

**Diagnosis**
Not Available

**Therapy**
Not Available
