
PlannerLearn
```dot
digraph PlannerLearn {
    in -> signals [label=feedback]
    in -> actions [label=feedback]
    signals -> stateKey
    actions -> actionKey
    stateKey -> createKey
    actionKey -> createKey
    in -> agent [label=feedback]
    agent -> score
    in -> score [label=feedback]
    createKey -> enqueue
    score -> enqueue
    in -> enqueue [label=feedback]
    in -> enqueue [label=planner]
    enqueue -> out [label=planner]
}
```

Planner plan

```dot
digraph plannerLoop {
  in -> max [label="planner"]
  in -> directLearn [label="agent"]
  max -> directLearn [label="feedback"]
  directLearn -> addEntry [label="score"]
  in -> model [label="planner"]
  model -> addEntry [label="model"]
  addEntry -> addAndSweep [label="model"]
  in -> addAndSweep [label="planner"]
  addAndSweep -> out [label="planner"]
  directLearn -> out [label="agent"]
}
```

Agent directLearn

```dot
digraph plannerLoop {
    processForTrain1 [label="processForTrain"]
    in -> processForTrain [label=feedback]
    in -> signals [label=feedback]
    processForTrain->labels [label="training dictionary"]

    network -> fit
    signals -> fit
    labels -> fit
    fit -> processForTrain1 [label=none]
    in -> processForTrain1 [label=feedback]
    this -> out [label=agent]
    processForTrain1 -> score [label="training dictionary"]
    processForTrain1 -> newAverage [label="training dictionary"]
    newAverage -> out
    score -> out
}
```

Agent processForTrain

```dot
digraph plannerLoop {
    foutputs0 [label="network.outputs(...)"]
    foutputs1 [label="network.outputs(...)"]
    fv0 [label="agent.v(...)"]
    fv1 [label="agent.v(...)"]
    ra [label="residual advantage(...)"]
    normalizeAndClip [label="normalizeAndClip(...)"]
    computeLabels [label="actors.computeLabels(...)"]
    concat [label="concat(...)"]
    toMap [label="toMap(...)"]
    pow [label="pow(...)"]

    feedback->signals0->foutputs0->outputs0->fv0->v0

    feedback->signals1->foutputs1->outputs1->fv1->v1
    
    feedback->actions

    v0 -> ra
    v1 -> ra

    ra -> newV0 ->normalizeAndClip -> criticalLabels
    ra -> delta
    ra -> newAvg
    
    outputs1 -> computeLabels
    actions-> computeLabels
    delta ->  computeLabels

    computeLabels -> actorLabels -> toMap-> actorsDictionary

    criticalLabels -> concat
    actorLabels -> concat-> labels

    labels -> out [label="labels"]
    v0 -> out [label="v0"]
    v1 -> out [label="v1"]
    newV0-> out [label="v0*"]
    outputs0 -> out [label="outputs0"]
    outputs1 -> out [label="outputs1"]
    delta -> out [label="delta"]
    delta -> pow
    pow ->out [label="score"]
    newAvg-> out [label="newAverage"]
    avg-> out [label="avg"]
    actorsDictionary-> out
}
```
