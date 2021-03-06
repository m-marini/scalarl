# Changelog

[TOC]

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Issue #30: Batch agent
- Issue #31: Continuos environment
- Issue #36: Batch optimization
- Issue #38: DynaQ+ Agent
- Issue #39: Model tolerance
- Issue #43: Tile coarse code
- Issue #44: Actor Critic and sofmax policy
- Issue #48: Simplify without multiple actions
- Issue #54: Continuous Action
- Issue #56: Multidimension Actions
- Issue #58: Priority Sweeping Agent
- Issue #59: Planner kpi
- Issue #62: Move Multilayer Network to Computational Graph
- Issue #63: Actor Critic with single network
- Issue #64: Resnet
- Issue #68: KPI for Gaussian policy
- Issue #69: Lander with crash
- Issue #76: Cleaning up priority sweeping model by priority
- Issue #78: int array key for key generator is not unique
- Issue #80: Optimize model key
- Issue #82: more detailed crash states
- Issue #84: Multiprocess benchmark
- Issue #86: VCrash priority
- Issue #91: Dump output on kpi
- Issue #93: Report session duration
- Issue #95: ANN Output Normalization and Relu
- Issue #100: Automatic jet controls
- Issue #112: Separate the observation from encoder (v6)
- Issue #119: dynamic actor alpha
- Issue #120: Saving model parameters

### Changed

- Issue #89: KPI review
- Issue #97: Redesign rewards
- Issue #106: New reward model
- Issue #111: Enqueue only by threshold on priority planner
- Issue #125: More test on environment
- Issue #102: Final states rewards depending on platform distance and speed
- Issue #128: Speed to platform reward
- Issue #130: Fix direction rewards when lower distance

### Fixed

- Issue #73: Mounting car error
- Issue #117: Change version configuration
- Issue #122: Upload network error

## [0.1.0] - 2019-01-28

### Added

- Issue #18:
    Run different sessions on maze environment with different agent (QAgent, TDQAgent) and analize the results.
    It is expected that learning rate of TDQAgent should be heighr of QAgent.
    Analyze different network configurations with no hidden layer, one hidden layer or two hidden layer.
    Run a QAgent without hidden layer (linear regretion) with 300 episodes limited to max 300 steps.
    Results in `analysis/Analysis.pdf`
- Issue #23: Adams algorithm implementation
- Issue #27: Verify Adam alghorithm
