# Changelog

[TOC]

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Issue #36: Batch optimization

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
