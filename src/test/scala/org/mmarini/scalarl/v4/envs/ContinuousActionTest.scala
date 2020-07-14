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

import com.typesafe.scalalogging.LazyLogging
import monix.eval.Task
import monix.execution.Scheduler.global
import monix.reactive.subjects.PublishSubject
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.v4.Session
import org.mmarini.scalarl.v4.Utils._
import org.mmarini.scalarl.v4.agents.{ActorCriticAgent, AgentEvent, GaussianActor}
import org.mmarini.scalarl.v4.reactive.Implicits._
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.ops.transforms.Transforms._
import org.scalatest.{FunSpec, Matchers}

class ContinuousActionTest extends FunSpec with Matchers with LazyLogging {
  private val Seed = 12345L
  private val NoSteps = 1000
  private val NoInputs: Long = Tiles(2).noFeatures
  private val EtaMu = 0.03
  private val EtaH = 0.03
  private val RewardDecay = 0.97
  private val ValueDecay = 0.99
  private val MuRange = 10
  private val HRange = Math.log(10 / 3)
  private val LearningRate = 300e-3

  create()

  private val random: Random =
    getRandomFactory.getNewRandomInstance(Seed)
  private val events: PublishSubject[AgentEvent] = PublishSubject[AgentEvent]()
  private val Range: INDArray = create(Array(Array(-MuRange, -HRange), Array(MuRange, HRange)))
  private val RewardRange: INDArray = create(Array(-100.0, 0.0)).transpose()
  private val Eta = create(Array(EtaMu, EtaH))
  private val denorm = denormalize(Range)
  private val norm = normalize(Range)

  def agent: ActorCriticAgent = ActorCriticAgent(
    network = network,
    avg = zeros(1),
    rewardDecay = ones(1).muli(RewardDecay),
    valueDecay = ones(1).muli(ValueDecay),
    denormalize = denormalize(RewardRange),
    normalizer = normalize(RewardRange),
    actors = Array(GaussianActor(dimension = 0,
      eta = Eta,
      denormalize = denorm,
      normalize = norm
    )),
    planner = None,
    agentObserver = events)

  def network: ComputationGraph = {
    val criticOutLayer = new OutputLayer.Builder().
      nIn(NoInputs).
      nOut(1).
      lossFunction(LossFunction.MSE).
      activation(Activation.TANH).
      build()
    val actorOutLayer = new OutputLayer.Builder().
      nIn(NoInputs).
      nOut(2).
      lossFunction(LossFunction.MSE).
      activation(Activation.TANH).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Sgd(LearningRate)).
      //updater(new Adam(1000e-3 / (4), 0.9, 0.999, 0.1)).
      //updater(new Adam(1000e-3 / (4), 0.9, 0.999, 0.1)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      miniBatch(false).
      //      constrainAllParameters(new MinMaxNormConstraint(-10e3, 10e3, 1)).
      //      gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).
      //      gradientNormalizationThreshold(100).
      graphBuilder().
      addInputs("inputs").
      addLayer("critic", criticOutLayer, "inputs").
      addLayer("actor", actorOutLayer, "inputs").
      setOutputs("critic", "actor")
      .build()

    val net = new ComputationGraph(conf)
    net.init()
    net
  }

  describe("ContinuousActionEnv") {
    val s0 = ContinuousActionEnv(ones(1).mul(2), zeros(1))

    val session = new Session(numSteps = NoSteps,
      epoch = 0,
      env = s0,
      agent = agent)

    //events.logKpi().subscribe()(global)

    val (_, agent1: ActorCriticAgent) = session.run(random)
    val actor = agent1.actors.head.asInstanceOf[GaussianActor]
    val outputs = agent1.network.output(s0.observation.signals)
    val (mu, _, sigma) = actor.muHSigma(outputs)
    logger.info("mu=    {}", mu)
    logger.info("sigma= {}", sigma)
    it("should result mu = 2") {
      mu.getDouble(0L) shouldBe 2.0 +- 1
    }
    it("should result sigma less then 1.5") {
      sigma.getDouble(0L) should be < 1.5
    }
  }
}