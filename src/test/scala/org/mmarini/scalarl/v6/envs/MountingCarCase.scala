// Copyright (c) 2019 Marco Marini, marco.marini@mmarini.org
//
// Licensed under the MIT License (MIT);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/MIT
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

package org.mmarini.scalarl.v6.envs

import com.typesafe.scalalogging.LazyLogging
import monix.eval.Task
import monix.execution.Scheduler
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.v6.agents._
import org.mmarini.scalarl.v6.envs.MountingCarEnv._
import org.mmarini.scalarl.v6.reactive.Implicits._
import org.mmarini.scalarl.v6.{Feedback, Session}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.scalatest.Matchers

import java.io.File

object MountingCarCase extends App with Matchers with LazyLogging {
  private val Seed = 12345L
  private val NoSteps = 1000
  private val Alpha = 30000e-6
  private val EtaMu = 3000e-3
  private val EtaHs = 300e-3
  private val PlanningStep = 0
  private val MinModelSize = 100
  private val MaxModelSize = 300
  private val ModelThreshold = 1e-9
  private val ValueDecay = 0.99
  private val RewardDecay = 0.97
  private val StateModelSize = 11
  private val ActionModelSize = 11
  private val Trace = false
  private val KpiFile = new File("mc-kpi.csv")

  private val MuRange = create(Array(MinActionValue, MaxActionValue)).muli(2).transposei()
  private val SigmaRange = create(Array(0.1, 1)).transposei()

  private val StateRanges = create(Array(
    Array(XLeft, VMin),
    Array(XRight, VMax)
  ))

  private val ActionRange = create(Array(-1, 2.0)).transpose()

  private val (stateEncode, netInputDimensions) = Encoder.tiles(
    ranges = StateRanges,
    sizes = ones(2).muli(3),
    hash = None)

  private implicit val sch: Scheduler = Scheduler.global

  create()
  private val random: Random = getRandomFactory.getNewRandomInstance(Seed)

  private val MidX: INDArray = ones(1).muli((MountingCarEnv.XLeft + MountingCarEnv.XRight) / 2)
  private val RewardRange: INDArray = create(Array(-2.0, 2.0)).transposei()

  private val actor = GaussianActor(dimension = 0,
    alphaMu = EtaMu,
    alphaSigma = EtaHs,
    muRange = MuRange,
    sigmaRange = SigmaRange
  )

  private val agentConf = ActorCriticAgentConf(
    rewardDecay = ones(1).muli(RewardDecay),
    valueDecay = ones(1).muli(ValueDecay),
    rewardRange = RewardRange,
    actors = Seq(actor),
    stateEncode = stateEncode,
    netInputDimensions = netInputDimensions
  )

  private val network: ComputationGraph = {
    val criticOutLayer = new OutputLayer.Builder().
      nIn(netInputDimensions).
      nOut(1).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val actorOutLayer = new OutputLayer.Builder().
      nIn(netInputDimensions).
      nOut(2).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Sgd(Alpha)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      miniBatch(false).
      constrainAllParameters(new MinMaxNormConstraint(-10e3, 10e3, 1)).
      gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).
      gradientNormalizationThreshold(10.0).
      graphBuilder().
      addInputs("inputs").
      addLayer("critic", criticOutLayer, "inputs").
      addLayer("actor", actorOutLayer, "inputs").
      setOutputs("critic", "actor").
      build()

    val net = new ComputationGraph(conf)
    net.init()
    net
  }

  private val planner = if (PlanningStep > 0) {
    Some(PriorityPlanner(
      stateKeyGen = INDArrayKeyGenerator.discrete(StateRanges, ones(2).muli(StateModelSize)),
      actionsKeyGen = INDArrayKeyGenerator.discrete(ActionRange, ones(1).muli(ActionModelSize)),
      planningSteps = PlanningStep,
      minModelSize = MinModelSize,
      maxModelSize = MaxModelSize,
      threshold = ModelThreshold,
      model = Map[(ModelKey, ModelKey), (Feedback, Double)]()
    ))
  } else {
    None
  }

  private val agent = ActorCriticAgent(
    conf = agentConf,
    network = network,
    avg = zeros(1),
    planner = planner)

  val s0 = MountingCarEnv.initial(random)

  val session = new Session(numSteps = NoSteps,
    epoch = 0,
    env = s0,
    agent = agent)

  KpiFile.delete()
  if (Trace) {
    session.monitorInfo().logInfo().subscribe()
    agentConf.agentObserver.doOnNext(_ => Task.eval {
    }).kpis().writeCsv(KpiFile).subscribe()
  } else {
    agentConf.agentObserver.kpis().writeCsv(KpiFile).subscribe()
  }

  logger.info("Mounting car case")

  val (_, agent1: ActorCriticAgent) = session.run(random)

  val s1 = MountingCarEnv(
    MidX,
    ones(1).muli(-0.07),
    zeros(1))
  val outs1 = agent1.network.output(stateEncode(s1.observation.signals))
  val (mu1, _, _) = actor.muHSigma(outs1)
  mu1.getDouble(0L) should be < 0.0

  val s2 = MountingCarEnv(
    MidX,
    ones(1).muli(0.07),
    zeros(1))
  val outs2 = agent1.network.output(stateEncode(s2.observation.signals))
  val (mu2, _, _) = actor.muHSigma(outs2)
  mu2.getDouble(0L) should be > 0.0

  logger.info("Test completed")
}