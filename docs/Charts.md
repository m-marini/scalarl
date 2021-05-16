## PolicyActor.chooseAction

```scala
def chooseAction(outputs: Array[INDArray], random: Random): INDArray
```

```dot
digraph chooseAction {

    netOutputs->fpreferences
    prefRange->fpreferences
    fpreferences->h
    fpreferences [label="denormalize(...)"]

    h->softmax->pi
    softmax [label="softmax(...)"]
    pi [label="&pi;"]

    pi->randomInt
    randomizer->randomInt
    randomInt->actionIndex
    randomInt [label="randomInt(...)"]

    actionIndex->decode
    noValues->decode
    range->decode
    decode->action
    decode [label="decode(...)"]
}
```
---

## PolicyActor.computeLabels

```scala
def computeLabels(outputs: Array[INDArray],
                  actions: INDArray,
                  delta: INDArray): Map[String, Any]
```

```dot
digraph computeLabels {

    noValues->encode
    range->encode
    actions->encode
    index->encode
    encode->featureVector
    encode [label="encode(...)"]

    featureVector->fdeltaH
    delta->fdeltaH
    alpha->fdeltaH
    fdeltaH->deltaH
    fdeltaH [label="featureVector * &delta; * &alpha;"]
    delta [label="&delta;"]
    alpha [label="&alpha;"]
    deltaH [label="&Delta;h"]
    
    prefRange->fpreferences
    netOutputs->fpreferences
    fpreferences->h
    fpreferences [label="denormalize(...)"]

    h->softmax->pi
    softmax [label="softmax(...)"]
    pi [label="&pi;"]

    deltaH->fhStar
    h->fhStar
    fhStar->hStar
    fhStar [label="h + &Delta;h"]
    hStar [label="h*"]

    prefRange->invPreference
    hStar->invPreference->labels
    invPreference [label="normalize(...)"]

    pi->out
    hStar->out
    labels->out
    deltaH->out
    h->out


}
```
---

## GaussianActor.chooseAction

```scala
def chooseAction(outputs: Array[INDArray], random: Random): INDArray
```

```dot
digraph chooseAction {
    
    netOutputs->muHSigma
    muRange->muHSigma
    sigmaRange->muHSigma
    muHSigma [label="muHSigma(...)"]
    muHSigma->mu
    muHSigma->sigma
    mu [label="&mu;"]
    sigma [label="&sigma;"]

    mu->randomGaussian
    sigma->randomGaussian
    randomizer->randomGaussian
    randomGaussian->action
}
```
---

## GaussianActor.computeLabels

```scala
def computeLabels(outputs: Array[INDArray],
                             actions: INDArray,
                             delta: INDArray): Map[String, Any]

```

```dot
digraph chooseAction {

    netOutputs->muHSigma
    muRange->muHSigma
    sigmaRange->muHSigma
    muHSigma->mu
    muHSigma->sigma
    muHSigma->h
    muHSigma [label="muHSigma(...)"]
    mu [label="&mu;"]
    sigma [label="&sigma;"]
    muRange [label="&mu;Range"]
    sigmaRange [label="&sigma;Range"]
    
    action->fdeltaMu
    mu->fdeltaMu
    sigma->fdeltaMu
    delta->fdeltaMu
    alphaMu->fdeltaMu
    fdeltaMu -> deltaMu
    fdeltaMu [label="2 * (action - &mu;) / &sigma;^2 * &delta; * &alpha;&mu;"]
    deltaMu [label="&Delta;&mu;"]
    alphaMu [label="&alpha;&mu;"]
    delta [label="&delta;"]

    action->fdeltaH
    mu->fdeltaH
    sigma->fdeltaH
    delta->fdeltaH
    alphaH->fdeltaH
    fdeltaH -> deltaH
    fdeltaH [label="(2 (action - &mu;)^2 / &sigma;^2) - 1) * &delta; * &alpha;h"]
    alphaH [label="&alpha;h"]
    deltaH [label="&Delta;h"]

    mu->fmuStar    
    deltaMu->fmuStar    
    fmuStar->muStar
    fmuStar [label="&mu; + &Delta;&mu;"]
    muStar [label="&mu;*"]

    h->fhStar
    deltaH->fhStar
    fhStar->hStar
    fhStar [label="h + &Delta;h"]
    hStar [label="h*"]

    muStar->concat
    hStar->concat
    concat->muH
    concat [label="[&mu;*, h*]"]
    muH [label="&mu;h"]

    muH->flabels
    muRange->flabels
    sigmaRange->flabels
    flabels->labels
    flabels [label="normalize(...)"]

    labels->out
    muStar->out
    hStar->out
    deltaMu->out
    deltaH->out
    mu->out
    h->out
}
```
---

## PriorityPlanner.learn

```scala
def learn(feedback: Feedback, agent: Agent): Planner
```

```dot
digraph PlannerLearn {

    feedback -> fscore
    agent -> fscore->score
    fscore [label="agent.score(...)"]

    feedback -> signals0
    feedback -> actions

    signals0 -> fstatusKey->statusKey
    fstatusKey [label="conf.statusKey(...)"]

    actions -> factionKey->actionKey
    factionKey [label="conf.actionKey(...)"]

    statusKey -> createKey
    actionKey -> createKey->key
    createKey [label="createKey(...)"]

    key -> enqueue

    score -> enqueue
    feedback -> enqueue
    planner -> enqueue
    enqueue -> newPlanner
    enqueue [label="planner.enqueue(...)"]
}
```
---

## PriorityPlanner.planLoop

```scala
def planLoop(ctx: (Agent, PriorityPlanner[KS, KA]), n: Int): (Agent, PriorityPlanner[KS, KA])
```

```dot
digraph plannerLoop {
  ctx->agent
  ctx->planner

  planner->max
  max->sample
  max [label="max(...)"]

  sample->feedback

  feedback->directLearn
  agent->directLearn
  directLearn->newAgent
  directLearn-> score
  directLearn [label="agent.directLearn(...)"]

  sample->key
  
  
  planner -> model -> addEntry
  key->addEntry
  feedback->addEntry
  score->addEntry
  addEntry-> newModel
  addEntry [label="model.addEntry(...)"]


  feedback->signals0->addAndSweep
  newModel->addAndSweep
  planner -> addAndSweep
  newAgent->addAndSweep
  addAndSweep -> newPlanner
  addAndSweep [label="planner.sweepBackward(...)"]

  newPlanner->out
  newAgent->out

}
```
---

## PriorityPlanner.sweepBackward

```scala
def sweepBackward(signals: INDArray, agent: Agent): PriorityPlanner[KS, KA]
```

```dot
digraph plannerLoop {
    signals->fpredecessors
    fpredecessors->predecessors
    fpredecessors [label="predecessors(...)"]

    thisPlanner->foreach
    predecessors->foreach
    foreach->planner
    foreach->predecessor
    foreach [label="thisPlanner.foldLeft(...)"]

    predecessor->key
    predecessor->feedback


    feedback->fscore
    agent->fscore
    fscore->score
    fscore [label="agent.score(...)"]

    planner->model

    model->add
    feedback->add
    key->add
    score->add
    add->newModel
    add [label="model.add(...)"]
    
    newModel->withModel
    planner->withModel
    withModel->newPlanner
    withModel [label="planner.withModel(...)"]

    newPlanner->foreach
}
```
---

## ActorCriticAgent.chooseAction

```scala
def chooseAction(observation: Observation, random: Random): INDArray
```

```dot
digraph chooseAction {
    observation->signals->fencoder
    fencoder [label="conf.encode(...)"]

    fencoder->output
    network->output
    output->outputs
    output [label="network.output(...)"]

    outputs->chooseAction0
    chooseActionn [label="actors(n).chooseAction(...)"]

    outputs->chooseActions
    chooseActions [label="actors(...).chooseAction(...)"]

    outputs->chooseActionn
    chooseAction0 [label="actors(0).chooseAction(...)"]

    chooseAction0->concat
    chooseActions->concat
    chooseActionn->concat
    -> actions
    concat [label="concat(...)"]

}
```

---

## ActorCriticAgent.directLearn

def Agent.directLearn(feedback: Feedback, random: Random): (Agent, INDArray, INDArray)

```dot
digraph {

    feedback->signals0->fencode
    signals0 [label="s0.signals"]

    fencode->netSignals
    fencode [label="encode(...)"]

    feedback->processForTraining
    processForTraining->map
    processForTraining [label="processForTraining(...)"]

    map->labels
    map->score
    map->newAvg

    netSignals->ffit
    network->ffit
    labels->ffit
    ffit->void
    ffit [label="fit(...)"]

    void->processForTraining1
    feedback->processForTraining1
    processForTraining1->mapAfter
    processForTraining1 [label="processForTraining(...)"]

    map->event
    mapAfter->event
    feedback->event
    event->emit
    emit [label="emit(...)"]

    newAvg->out
    score->out
    emit->out
}
```
---

## ActorCriticAgent.processForTrain

 def Agent.processForTrain(feedback: Feedback): Map[String, Any]

```dot
digraph plannerLoop {
    feedback->signals0
    feedback->signals1
    feedback->reward
    feedback->actions

    signals0->fs0norm
    signalRanges->fs0norm
    fs0norm [label="normalize(...)"]

    network->foutputs0
    fs0norm->foutputs0
    foutputs0->outputs0
    foutputs0 [label="network.outputs(...)"]

    outputs0->fv0->v0
    fv0 [label="v(...)"]

    signals1->fs1norm
    signalRanges->fs1norm
    fs1norm [label="normalize(...)"]

    network->foutputs1
    fs1norm->foutputs1
    foutputs1->outputs1
    foutputs1 [label="network.outputs(...)"]
    
    outputs1->fv1->v1
    fv1 [label="v(...)"]

    v0->fnewv0
    v1->fnewv0
    reward->fnewv0
    valueDecay->fnewv0
    avg->fnewv0
    fnewv0->newV0
    fnewv0 [label="(v1 + reward - avg) * valueDecay + (1 - valueDecay) * avg"]

    newV0->fdelta
    v0->fdelta
    fdelta->delta
    fdelta [label="newV0 - v0"]

    avg->fnewAvg
    reward->fnewAvg
    rewardDecay->fnewAvg
    fnewAvg->newAvg
    fnewAvg [label="rewardDecay * avg + (1 - rewardDecay) * reward"]

    newV0 ->normalizeAndClip
    rewardRange->normalizeAndClip 
    normalizeAndClip -> criticLabels
    normalizeAndClip [label="normalize(...)"]
    
    outputs1 -> computeLabels
    actions-> computeLabels
    delta ->  computeLabels
    computeLabels -> actorLabels
    computeLabels [label="actors.computeLabels(...)"]

    actorLabels -> toMap-> actorsDictionary
    toMap [label="toMap(...)"]

    criticLabels -> concat
    actorLabels -> concat-> labels
    concat [label="[criticLabels, actorLabels]"]

    delta -> pow -> delta2
    pow [label="delta^2"]

    labels -> out
    v0 -> out
    v1 -> out
    newV0-> out
    outputs0 -> out
    outputs1 -> out
    delta -> out
    delta2 ->out
    newAvg-> out
    avg-> out
    actorsDictionary-> out
}
```
