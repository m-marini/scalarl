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
import monix.reactive.subjects.PublishSubject
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.v4.agents._
import org.mmarini.scalarl.v4.{Agent, Session}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.ops.transforms.Transforms.softmax
import org.scalatest.{FunSpec, Matchers}

class TestEnvWithPlanningTest extends FunSpec with Matchers with LazyLogging {
  private val Seed = 123L
  private val NoSteps = 100
  private val PlanningSteps = 5
  private val MinModelSize = 6
  private val MaxModelSize = 10

  create()

  private val random: Random = getRandomFactory.getNewRandomInstance(Seed)
  private val events: PublishSubject[AgentEvent] = PublishSubject[AgentEvent]()
  private val sKey = INDArrayKeyGenerator.binary
  private val aKey = INDArrayKeyGenerator.binary

  def agent: Agent = ActorCriticAgent(
    network = network,
    rewardDecay = ones(1).mul(0.97),
    valueDecay = ones(1).mul(0.99),
    avg = zeros(1),
    actors = Array(PolicyActor(
      dimension = 0,
      noOutputs = 2,
      alpha = ones(1).mul(3))),
    planner = Some(planner),
    agentObserver = events)

  def planner: PriorityPlanner[Array[Int], Array[Int]] = PriorityPlanner(
    stateKeyGen = sKey,
    actionsKeyGen = aKey,
    planningSteps = PlanningSteps,
    minModelSize = MinModelSize,
    maxModelSize = MaxModelSize,
    threshold = 0.1,
    model = Map()
  )

  def network: ComputationGraph = {
    val criticOutLayer = new OutputLayer.Builder().
      nIn(3).
      nOut(1).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val actorOutLayer = new OutputLayer.Builder().
      nIn(3).
      nOut(2).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Sgd(1.0 / 4)).
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
      setOutputs("critic", "actor").
      build()

    val net = new ComputationGraph(conf)
    net.init()
    net
  }

  describe(
    s"""TestEnv
       | Given a continuous MDP process with 3 state, deterministic transitions driven by 2 possible actions
       |   and an agent implementing actor critic method with planning
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
    //  val s2 = TestEnv(0, 2, conf)

    val session = new Session(numSteps = NoSteps,
      epoch = 0,
      env = s0,
      agent = agent)

    val (_, agent1: ActorCriticAgent) = session.run(random)

    val actor = agent1.actors.head.asInstanceOf[PolicyActor]

    it("should result an actor with q(s0, a1) > q(s0, a1) ") {
      val o0 = agent1.network.output(s0.observation.signals)
      val pr = actor.preferences(o0)
      logger.debug("q(s0) = {}, pi(s0) = {}", pr, softmax(pr))
      pr.getDouble(1L) shouldBe >(pr.getDouble(0L))
    }

    it("should result an actor with q(s1, a1) > q(s1, a1) ") {
      val o1 = agent1.network.output(s0.observation.signals)
      val pr = actor.preferences(o1)
      logger.debug("q(s1) = {}, pi(s1) = {}", pr, softmax(pr))
      pr.getDouble(1L) shouldBe >(pr.getDouble(0L))
    }

    val planner = agent1.planner.get.asInstanceOf[PriorityPlanner[Seq[Int], Seq[Int]]]
    it("should have model with 6 entries") {
      planner.model should have size 6
    }
  }
}