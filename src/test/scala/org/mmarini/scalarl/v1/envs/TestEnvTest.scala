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

package org.mmarini.scalarl.v1.envs

import com.typesafe.scalalogging.LazyLogging
import monix.eval.Task
import monix.execution.Scheduler
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.v1.agents.ExpSarsaAgent
import org.mmarini.scalarl.v1.{Agent, Session}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.scalatest.{FunSpec, Matchers}

class TestEnvTest extends FunSpec with Matchers with LazyLogging {
  val Seed = 12345L
  val NoSteps = 1000
  val MaxEpisodeLength = 100
  val Hiddens = 10

  Nd4j.create()

  def agent: Agent =
    ExpSarsaAgent(net = network,
      model = Seq(),
      noActions = 2,
      avgReward = 0.25,
      beta = 0.5,
      maxModelSize = 0,
      minModelSize = 1000,
      epsilon = 0.01,
      kappa = 1,
      kappaPlus = 0,
      planningStepsCounter = 10,
      tolerance = Some(Nd4j.create(Array(0.5, 0.5))))

  def network: MultiLayerNetwork = {
    val hidden = new DenseLayer.Builder().
      nIn(3).
      nOut(Hiddens).
      activation(Activation.TANH).
      dropOut(0.8).
      build()

    val outLayer = new OutputLayer.Builder().
      nIn(Hiddens).
      nOut(2).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      biasInit(1).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Adam(10.0 / (4 * Hiddens + (Hiddens + 1) * 2), 0.9, 0.999, 0.1)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      constrainAllParameters(new MinMaxNormConstraint(-10e3, 10e3, 1)).
      gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).
      gradientNormalizationThreshold(1).
      list(hidden, outLayer)
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net
  }

  def random: Random =
    Nd4j.getRandomFactory.getNewRandomInstance(Seed)

  describe("TestEnv") {

    val conf = TestEnvConfBuilder().numState(3).
      p(0, 0, 0, 0, 2.0 / 3).
      p(1, 0, 0, 0, 1.0 / 3).
      p(0, 0, 0, 1, 1.0 / 3).
      p(1, 0, 0, 1, 2.0 / 3).
      p(1, 0, 1, 0, 2.0 / 3).
      p(2, 1, 1, 0, 1.0 / 3).
      p(1, 0, 1, 1, 1.0 / 3).
      p(2, 1, 1, 1, 2.0 / 3).
      p(0, 0, 2, 0, 1.0).
      p(0, 0, 2, 1, 1.0).build

    val s0 = TestEnv(0, 0, conf)
    val s1 = TestEnv(0, 1, conf)
    //  val s2 = TestEnv(0, 2, conf)

    val session = Session(noSteps = NoSteps,
      env0 = s0,
      agent0 = agent,
      maxEpisodeLength = MaxEpisodeLength)

    val firstEpisode = session.episodes.take(1)
    val lastEpisode = session.episodes.last

    it("should compare first with last episode") {
      firstEpisode.combineLatest(lastEpisode).doOnNext {
        case (first, last) => Task.eval {
          logger.info("stepCount = {}/{}, score={}/{}",
            first.stepCount, last.stepCount, first.totalScore, last.totalScore)
          //          first.stepCount shouldBe >=(last.stepCount)
          //          first.totalScore shouldBe >=(last.totalScore)
        }
      }.subscribe()(Scheduler.global)

      val (_, agent1) = session.run(random)

      //t1.asInstanceOf[ExpSarsaAgent].avgReward shouldBe 0.25 +- 0.25

      val q0 = agent1.asInstanceOf[ExpSarsaAgent].q(s0.observation)
      q0.getDouble(0L) shouldBe <(q0.getDouble(1L))
      //      q0.getDouble(0L) shouldBe 0.125 +- 0.05
      //      q0.getDouble(1L) shouldBe 0.25 +- 0.05

      val q1 = agent1.asInstanceOf[ExpSarsaAgent].q(s1.observation)
      q1.getDouble(0L) shouldBe <(q1.getDouble(1L))
      //      q1.getDouble(0L) shouldBe 0.5 +- 0.05
      //      q1.getDouble(1L) shouldBe 0.625 +- 0.05
    }
  }
}