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
import org.mmarini.scalarl.v6.agents._
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.ops.transforms.Transforms._
import org.scalatest.Matchers

object TestEnv2Case extends Matchers with LazyLogging with App {
  private val Trace = false
  private val Seed = 12345L
  private val NoSteps = 1000
  private val ValueDecay = 0.99
  private val RewardDecay = 0.98
  private val Alpha = 1000e-3
  private implicit val sch: Scheduler = Scheduler.global

  create()
  private val LearningRate = 1
  private val RewardRange = create(Array(0.0, 1.0)).transposei()
  private val NoActionValues = 2
  private val StateRanges = create(Array(0.0, 2.0)).transposei()
  private val StateValues = ones(1).muli(3)
  private val ActionRange = create(Array(0.0, 1.0)).transposei()
  private val PrefRange = create(Array(Math.log(0.1), Math.log(10))).transposei()
  private val random: Random = getRandomFactory.getNewRandomInstance(Seed)
  private val conf = TestEnvConfBuilder().numState(3).
    p(0, 0, 1.0, 0, 0).
    p(0, 1, 1.0, 1, 0).
    p(1, 0, 1.0, 1, 0).
    p(1, 1, 1.0, 2, 1).
    p(2, 0, 1.0, 0, 0).
    p(2, 1, 1.0, 0, 0).build

  private val network: ComputationGraph = {
    val criticOutLayer = new OutputLayer.Builder().
      nIn(3).
      nOut(1).
      lossFunction(LossFunction.MSE).
      activation(Activation.TANH).
      build()

    val actorOutLayer = new OutputLayer.Builder().
      nIn(3).
      nOut(2).
      lossFunction(LossFunction.MSE).
      activation(Activation.TANH).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Sgd(LearningRate)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
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

  private val actor = PolicyActor(
    dimension = 0,
    noValues = NoActionValues,
    actionRange = ActionRange,
    prefRange = PrefRange,
    alpha = ones(1).muli(Alpha)
  )

  private val agentConfig = {
    val (signalEncode, netInputDimensions) = Encoder.features(StateRanges, StateValues)

    ActorCriticAgentConf(
      rewardDecay = ones(1).mul(RewardDecay),
      valueDecay = ones(1).mul(ValueDecay),
      rewardRange = RewardRange,
      actors = Seq(actor),
      stateEncode = signalEncode,
      netInputDimensions = netInputDimensions)
  }

  private val agent = ActorCriticAgent(
    conf = agentConfig,
    network = network,
    avg = zeros(1),
    planner = None
  )

  logger.info(
    s"""TestEnv
       | Given a continuous MDP process with 3 state, deterministic transitions driven by 2 possible actions
       |   and an agent implementing actor critic method
       | When acting in the environment
       | Then the actor should improve its policy approaching the optimal policy
       |   and reaching a good policy within $NoSteps interaction steps""".stripMargin)
  private val s0 = TestEnv(zeros(1), 0, conf)
  private val s1 = TestEnv(zeros(1), 1, conf)
  private val session = new Session(numSteps = NoSteps,
    epoch = 0,
    env = s0,
    agent = agent)

  if (Trace) {
    agent.conf.agentObserver.
      filter(event => {
        (event.feedback.s0.signals.getInt(0), event.feedback.actions.getInt(0)) == (0, 1)
      }).
      doOnNext(event => Task.eval {
        val agent = event.agent.asInstanceOf[ActorCriticAgent]
        logger.debug(
          s"""
             |  v0=${event.map("v0")}
             |  v1=${event.map("v1")}
             |  v0*=${event.map("v0*")}
             |  avg=${event.map("avg")}
             |  avg*=${event.map("newAverage")}
             |  h0=${event.map("h(0)")}
             |  h0*=${event.map("h*(0)")}""".stripMargin)
        for (i <- 0 until 3) {
          val st = TestEnv(zeros(1), i, conf).observation.signals
          val in = agent.conf.stateEncode(st)
          val out = agent.network.output(in)
          val v = agent.v(out)
          val pref = agent.conf.actors(0).asInstanceOf[PolicyActor].preferences(out)
          logger.debug(s" s$i: $v $pref")
        }
      }).subscribe()
  }
  private val (_, agent1: ActorCriticAgent) = session.run(random)

  private val o0 = agent1.network.output(agent1.conf.stateEncode(s0.observation.signals))
  private val ac0 = agent1.conf.actors.head.asInstanceOf[PolicyActor]
  private val h0 = ac0.preferences(o0)
  private val o1 = agent1.network.output(agent1.conf.stateEncode(s1.observation.signals))
  private val h1 = ac0.preferences(o1)

  logger.info(
    """
      | Preference for action 1 at state s0
      | should be greater than preference for action 0 for state s0""".stripMargin)
  logger.debug("h(s0) = {}, pi(s0) = {}", h0, softmax(h0))
  h0.getDouble(1L) shouldBe >(h0.getDouble(0L))

  logger.info(
    """
      | Preference for action 1 at state s1
      | should be greater than preference for action 0 for state s1""".stripMargin)
  logger.debug("h(s1) = {}, pi(s1) = {}", h1, softmax(h1))
  h1.getDouble(1L) shouldBe >(h1.getDouble(0L))

  logger.info("Test completed")
}
