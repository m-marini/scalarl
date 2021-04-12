[TOC]

File: `lander12.yaml`

## 2020-08-17 - #4

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
  kpisOnPlanning: true
```

**ANN Exam**
Linear Decreasing rewards trend from -14.6 to -17.7, drop from -12 to -19 after 23K steps and then rise to descendant -13.
Linear Increasing MSE (RMSE) trend from 8888.0 (94.3) to 10300.9 (101.5)
23% red class
1% yellow class
75% green class
Optimal actor alpha: 6.6e-03, 3.5e-03, 5.0e-03

**Flying Status Exam**
Linear Decreasing rewards trend from -14.4 to -17.6, quite constant at -12.5 before 4.9K steps and the variable.
Linear Decreasing direction error trend from 135 DEG to 82 DEG, quite constant at 170 DEG before 4.9K step and the variable.
Linear Increasing horizontal speed trend from 0.9 m/s to 2.3 m/s, rise from 0.03 m/s to constant 2 m/s.
Linear Decreasing vertical speed trend from -3.0 m/s to -3.8 m/s variable before 4.8K steps then constant at -3.7 then rise to -2.7 after 27K.

**Episodes Exam**
Linear Increasing rewards trend from -56.4 to -52.3.
Linear Increasing platform distance trend from 77 m to 86 m varibale.
125 cases of vertical crash out of platform between 596 and 26834 steps unstable.
62 cases of landed out of platform between 497 and 27491 steps unstable.
59 cases of horizontal crash out of platform between 5082 and 29930 steps unstable.
3 cases of out of fuel between 300 and 4413 steps.
1 cases of vertical crashed on platform between 26515 and 26515 steps.
1 cases of horizontal crash on platform between 11229 and 11229 steps.

**Diagnosis**
The constant horizontal and vertical speed to the exterme range maybe caused by a saturation of the outputs due to high learning parameter.

**Therapy**
Reduce the learning parameter to 30 mU.

## 2020-08-18 - #1 lander12.yaml

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
    activation: SOFTPLUS
    numHiddens:
    - 100
    - 30
    - 30
    shortcuts:
      - [ 1, 4 ]
    updater: Sgd
    learningRate: 30e-3
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
  kpisOnPlanning: true
```

**ANN Exam**
Linear Increasing rewards trend from -13.3 to -13.1
Linear Increasing MSE (RMSE) trend from 16156.0 (127.1) to 17503.0 (132.3)
3% red class
2% yellow class
95% green class
Optimal actor alpha: 4.7e-03, 2.4e-03, 3.5e-03

**Flying Status Exam**
Linear Decreasing rewards trend from -12.8 to -12.8 constant at -12.8 after 900 steps.
Linear Increasing direction error trend from 165 DEG to 175 DEG constant at 170 DEG after 900 steps
Linear Decreasing horizontal speed trend from 0.1 m/s to 0.0 m/s constant at 0.3 m/s after 800 steps..
Linear Decreasing vertical speed trend from -3.4 m/s to -4.0 m/s constant at 3.8 after 900 steps.

**Episodes Exam**
Linear Increasing rewards trend from -58.7 to -46.2 quite constant at -51 after 900 steps.
Linear Decreasing platform distance trend from 76 m to 74 m.
170 cases of vertical crash out of platform between 403 and 29774 steps.
100 cases of landed out of platform between 836 and 29993 steps.
3 cases of out of range between 508 and 680 steps.
2 cases of landed between 9008 and 29442 steps.
1 cases of vertical crashed on platform between 22508 and 22508 steps.
1 cases of out of fuel between 300 and 300 steps.

**Diagnosis**
Not availble

**Therapy**
Increase steps to 100K.

## 2020-08-18 - #2 lander12.yaml

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
    activation: SOFTPLUS
    numHiddens:
    - 100
    - 30
    - 30
    shortcuts:
      - [ 1, 4 ]
    updater: Sgd
    learningRate: 30e-3
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
  numSteps: 100000
  kpisOnPlanning: true
```

***ANN Exam***
Linear Decreasing rewards trend from -13.6 to -14.3
Linear Increasing MSE (RMSE) trend from 20418.6 (142.9) to 21375.1 (146.2)
4% red class
4% yellow class
92% green class
Optimal actor alpha: 4.2e-03, 2.2e-03, 3.1e-03

**Flying Status Exam**
Linear Decreasing rewards trend from -13.0 to -14.1, constant at -13 after 7K steps and then variable after 80K steps.
Linear Decreasing direction error trend from 174 DEG to 127 DEG, constant at 170DEG after 7K steps and then constant around 90 DEG +-40 DEG  after 80K steps.
Linear Increasing horizontal speed trend from -0.1 m/s to 0.8 m/s, constant at 0 m/s after 7K steps and then constant at 1.3 m/s  after 80K steps..
Linear Decreasing vertical speed trend from 0.3 m/s to -3.2 m/s, variabale around -1 m/s +-2 m/s and tehn constant at 3.6 m/s after 80K steps.

**Episodes Exam**
Linear Increasing rewards trend from -132.2 to -46.6.
Linear Increasing platform distance trend from 73 m to 84 m.
216 cases of vertical crash out of platform between 6778 and 99613 steps with high frequency at the end.
185 cases of landed out of platform between 861 and 99918 steps with high frequency at the end.
143 cases of out of range between 109 and 79065 steps.
74 cases of out of fuel between 641 and 79008 steps.
3 cases of landed between 7732 and 89845 steps.
2 cases of vertical crashed on platform between 82529 and 91832 steps.
