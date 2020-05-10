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

package org.mmarini.scalarl.v3.envs

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.v3.agents.{ActorCriticAgent, PolicyActor}
import org.mmarini.scalarl.v3.{Agent, Session}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.ops.transforms.Transforms._
import org.scalatest.{FunSpec, Matchers}

class TestEnv2Test extends FunSpec with Matchers with LazyLogging {
  val Seed = 12345L
  val NoSteps = 500
  val Hiddens = 10

  create()
  val random: Random = getRandomFactory.getNewRandomInstance(Seed)

  def agent: Agent = ActorCriticAgent(
    critic = critic,
    rewardDecay = ones(1).mul(0.97),
    valueDecay = ones(1).mul(0.99),
    avg = zeros(1),
    actors = Array(PolicyActor(
      dimension = 0,
      actor = actor,
      alpha = ones(1).mul(3))),
    planner = None)

  def critic: ComputationGraph = {
    val outLayer = new OutputLayer.Builder().
      nIn(3).
      nOut(1).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Sgd(1.0 / 4)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      miniBatch(false).
      list(outLayer)
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.toComputationGraph
  }

  def actor: ComputationGraph = {
    val outLayer = new OutputLayer.Builder().
      nIn(3).
      nOut(2).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Sgd(1.0 / (4 * 2))).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      constrainAllParameters(new MinMaxNormConstraint(-10e3, 10e3, 1)).
      list(outLayer)
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.toComputationGraph
  }

  describe(
    s"""TestEnv
       | Given a continuous MDP process with 3 state, deterministic transitions driven by 2 possible actions
       |   and an agent implementing actor critic method
       | When acting in the environment
       | Then the actor should improve its policy approching the optimal policy
       |   and reaching a good policy within $NoSteps interaction steps""".stripMargin) {

    val conf = TestEnvConfBuilder().numState(3).
      p(0, 0, 0, 0, 1.0).
      p(1, 0, 0, 1, 1.0).
      p(1, 0, 1, 0, 1.0).
      p(2, 1, 1, 1, 1.0).
      p(0, 0, 2, 0, 1.0).
      p(0, 0, 2, 1, 1.0).build

    val s0 = TestEnv(zeros(1), 0, conf)
    val s1 = TestEnv(zeros(1), 1, conf)

    val session = new Session(numSteps = NoSteps,
      epoch = 0,
      env = s0,
      agent = agent)

    it("should compare first with last episode") {
      val (_, agent1) = session.run(random)

      val q0 = agent1.asInstanceOf[ActorCriticAgent].
        actors.head.asInstanceOf[PolicyActor].
        actor.output(s0.observation.signals)(0)
      logger.debug("q(s0) = {}, pi(s0) = {}", q0, softmax(q0))
      q0.getDouble(1L) shouldBe >(q0.getDouble(0L))

      val q1 = agent1.asInstanceOf[ActorCriticAgent].
        actors.head.asInstanceOf[PolicyActor].
        actor.output(s1.observation.signals)(0)
      logger.debug("q(s1) = {}, pi(s1) = {}", q1, softmax(q1))
      q1.getDouble(1L) shouldBe >(q1.getDouble(0L))
    }
  }
}