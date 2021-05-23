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
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.v6.Session
import org.mmarini.scalarl.v6.Utils._
import org.mmarini.scalarl.v6.agents._
import org.mmarini.scalarl.v6.envs.ContinuousActionEnv._
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.ops.transforms.Transforms._
import org.scalatest.Matchers

object ContinuousActionCase extends Matchers with LazyLogging with App {
  val s0 = ContinuousActionEnv(ones(1).muli(2), zeros(1))
  val session = new Session(numSteps = NoSteps,
    epoch = 0,
    env = s0,
    agent = agent)
  val (_, agent1: ActorCriticAgent) = session.run(random)
  val outputs = agent1.network.output(s0.observation.signals)
  val (mu, _, sigma) = actor.muHSigma(outputs)
  private val Trace = false
  private val Seed = 12345L
  private val NoSteps = 1000
  private val NoInputs: Long = 1

  create()

  private implicit val sch: Scheduler = Scheduler.global
  private val EtaMu = 0.03
  private val EtaH = 0.03
  private val RewardDecay = 0.97
  private val ValueDecay = 0.99
  private val LearningRate = 1000e-3
  private val random: Random =
    getRandomFactory.getNewRandomInstance(Seed)
  private val MuRange = create(Array(MinActionValue, MaxActionValue)).transposei()
  private val SigmaRange = create(Array(0.05, 1)).transposei()
  private val RewardRange: INDArray = pow(create(Array(MaxStateValue - MinStateValue, 0.0)), 2).
    transposei().
    negi()
  private val SignalsRange = create(Array(MinStateValue, MaxStateValue)).
    muli(2).
    transposei()

  logger.info(
    s"""
       | Given a process with continuous 1D state
       |   and the rewards maximum when the action is equal to the state
       |   and an agent implementing actor critic method
       | When acting in the environment
       | Then the actor should improve its policy approaching the optimal policy
       |   and reaching a good policy within $NoSteps interaction steps
       |""".stripMargin)
  private val stateEncode = clipAndNormalize(ranges = SignalsRange)
  private val actor = GaussianActor.createActor(
    dimension = 0,
    alphaMu = EtaMu,
    alphaSigma = EtaH,
    epsilon = 0.1,
    alphaDecay = 1.0,
    muRange = MuRange,
    sigmaRange = SigmaRange)

  if (Trace) {
    agentConf.agentObserver.doOnNext(event => Task {
      logger.debug(
        s"""
           |  s=${event.feedback.s0.signals}
           |  a=${event.feedback.actions}
           |  r=${event.feedback.reward}
           |  v0=${event.map("v0")}
           |  v1=${event.map("v1")}
           |  v0*=${event.map("v0*")}
           |  mu=${event.map("mu(0)")}
           |  mu*=${event.map("mu*(0)")}
           |  sigma=${event.map("sigma(0)")}
           |  h0=${event.map("h(0)")}
           |  h*0=${event.map("h*(0)")}
           |  avg=${event.map("avg")}
           |  avg*=${event.map("newAverage")}""".stripMargin)
    }).subscribe()
  }
  private val agentConf = ActorCriticAgentConf(
    rewardDecay = ones(1).muli(RewardDecay),
    valueDecay = ones(1).muli(ValueDecay),
    rewardRange = RewardRange,
    actors = Seq(actor),
    stateEncode = stateEncode,
    netInputDimensions = 1)
  private val network = {
    val criticOutLayer = new OutputLayer.Builder().
      nIn(NoInputs).
      nOut(1).
      lossFunction(LossFunction.MSE).
      activation(Activation.TANH).
      build()
    val actorOutLayer = new OutputLayer.Builder().
      nIn(NoInputs).
      nOut(actor.noOutputs).
      lossFunction(LossFunction.MSE).
      activation(Activation.TANH).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Sgd(LearningRate)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      miniBatch(false).
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
  private val agent = ActorCriticAgent(
    conf = agentConf,
    network = network,
    avg = zeros(1),
    planner = None,
    alpha = Seq(create(Array(EtaMu, EtaH))))

  logger.info("mu=    {}", mu)
  logger.info("sigma= {}", sigma)

  logger.info("mu for s0=2 should be close to 2 +- 0.5")
  mu.getDouble(0L) shouldBe 2.0 +- 0.5

  logger.info("sigma for s0=2 should be less then to 0.2")
  sigma.getDouble(0L) should be < 0.2

  logger.info("Test completed")
}
