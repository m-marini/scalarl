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

package org.mmarini.scalarl.v4.envs

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import monix.eval.Task
import monix.execution.Scheduler
import monix.reactive.subjects.PublishSubject
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.v4.Utils._
import org.mmarini.scalarl.v4.agents._
import org.mmarini.scalarl.v4.reactive.Implicits._
import org.mmarini.scalarl.v4.{Feedback, Session}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.ops.transforms.Transforms._
import org.scalatest.{FunSpec, Matchers}

class MountingCarTest extends FunSpec with Matchers with LazyLogging {
  private val Seed = 12345L
  private val NoSteps = 1000
  private val Alpha = 30000e-6
  private val EtaMu = 3000e-3
  private val EtaHs = 300e-3
  private val PlanningStep = 5
  private val MinModelSize = 100
  private val MaxModelSize = 300
  private val ModelThreshold = 1e-9
  private val Trace = false
  private val KpiFile = new File("mc-kpi.csv")

  private val NoInputs: Long = Tiles(1, 1).noFeatures
  private implicit val s: Scheduler = Scheduler.global

  create()

  private val MidX: INDArray = ones(1).muli((MountingCarEnv.XLeft + MountingCarEnv.XRight) / 2)
  private val RewardRange: INDArray = create(Array(-2.0, 2.0)).transpose()
  private val Range: INDArray = create(Array(Array(-10.0, 10.0), Array(-3.0, 3.0)))
  private val denorm = denormalize(Range)
  private val norm = normalize(Range)

  private val random: Random = getRandomFactory.getNewRandomInstance(Seed)
  private val events: PublishSubject[AgentEvent] = PublishSubject[AgentEvent]()

  def agent: ActorCriticAgent = ActorCriticAgent(
    network = network,
    avg = zeros(1),
    rewardDecay = ones(1).muli(0.97),
    valueDecay = ones(1).muli(0.99),
    denormalize = denormalize(RewardRange),
    normalizer = denormalize(RewardRange),
    actors = Array(GaussianActor(dimension = 0,
      eta = create(Array(EtaMu, EtaHs)),
      denormalize = denorm,
      normalize = norm)),
    planner = planner,
    agentObserver = events)

  def planner: Option[PriorityPlanner[ModelKey, ModelKey]] = if (PlanningStep > 0) {
    Some(PriorityPlanner(
      stateKeyGen = INDArrayKeyGenerator.binary,
      actionsKeyGen = INDArrayKeyGenerator.tiles(
        min = ones(1).negi(),
        max = ones(1),
        noTiles = ones(1).muli(100)),
      planningSteps = PlanningStep,
      minModelSize = MinModelSize,
      maxModelSize = MaxModelSize,
      threshold = ModelThreshold,
      model = Map()
    ))
  } else {
    None
  }

  def network: ComputationGraph = {
    val criticOutLayer = new OutputLayer.Builder().
      nIn(NoInputs).
      nOut(1).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val actorOutLayer = new OutputLayer.Builder().
      nIn(NoInputs).
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
      gradientNormalizationThreshold(10).
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

  def trace(event: AgentEvent) {
    val Feedback(s0, a, r, s1) = event.feedback
    val ag0 = event.agent0.asInstanceOf[ActorCriticAgent]
    val ac0 = ag0.actors.head.asInstanceOf[GaussianActor]
    val o0 = ag0.network.output(s0.signals)
    val (mu0, h0, sigma0) = ac0.muHSigma(o0)
    val ag1 = event.agent1.asInstanceOf[ActorCriticAgent]
    val ac1 = ag1.actors.head.asInstanceOf[GaussianActor]
    val o1 = ag1.network.output(s0.signals)
    val (mu1, h1, sigma1) = ac1.muHSigma(o1)
    val s = ag0.score(event.feedback)

    val v0 = ag0.v(o0)
    val o01 = ag0.network.output(s1.signals)
    val v1 = ag0.v(o01)

    val (delta, _, _) = ag0.computeDelta(v0, v1, r)

    val actorLabels = ac0.computeLabels(o0, a, delta, random)
    val muStar = actorLabels.getColumn(0)
    val hStar = actorLabels.getColumn(1)
    val sigmaStart = exp(hStar)

    logger.debug("s0={}, a={}, r={}, score={}, delta={}", find(s0.signals), a, r, s, delta)
    logger.debug("   mu ={}, h= {}, sigma= {}", mu0, h0, sigma0)
    logger.debug("   mu*={}, h*={}, sigma*={}", muStar, hStar, sigmaStart)
    logger.debug("   mu'={}, h'={}, sigma'={}", mu1, h1, sigma1)

  }

  describe("TestEnv") {
    val s0 = MountingCarEnv.initial(random)

    val session = new Session(numSteps = NoSteps,
      epoch = 0,
      env = s0,
      agent = agent)

    KpiFile.delete()
    if (Trace) {
      session.monitorInfo().logInfo().subscribe()
      events.doOnNext(event => Task.eval {
        trace(event)
      }).kpis().writeCsv(KpiFile).subscribe()
    } else {
      events.kpis().writeCsv(KpiFile).subscribe()
    }

    val (_, agent1: ActorCriticAgent) = session.run(random)

    val actor = agent1.actors.head.asInstanceOf[GaussianActor]

    it("should accelerate left when going to left") {
      val s = MountingCarEnv(
        MidX,
        ones(1).muli(-0.07),
        zeros(1))
      val outs = agent1.network.output(s.observation.signals)
      val (mu, _, _) = actor.muHSigma(outs)
      //mu.getDouble(0L) should be < 0.0
    }

    it("should accelerate right when going to right") {
      val s = MountingCarEnv(
        MidX,
        ones(1).muli(0.07),
        zeros(1))
      val outs = agent1.network.output(s.observation.signals)
      val (mu, _, _) = actor.muHSigma(outs)
      //mu.getDouble(0L) should be > 0.0
    }
  }
}