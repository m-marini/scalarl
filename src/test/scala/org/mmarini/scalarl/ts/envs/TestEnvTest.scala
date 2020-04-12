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

package org.mmarini.scalarl.ts.envs

import com.typesafe.scalalogging.LazyLogging
import monix.eval.Task
import monix.execution.Scheduler
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.ts.agents.ExpSarsaAgent
import org.mmarini.scalarl.ts.{Agent, DiscreteActionChannels, Session}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.scalatest.{FunSpec, Matchers}

class TestEnvTest extends FunSpec with Matchers with LazyLogging {
  val Seed = 1234L
  val NoSteps = 100
  val MaxEpisodeLength = 100


  Nd4j.create()

  def agent: Agent =
    ExpSarsaAgent(net = network,
      model = Seq(),
      config = DiscreteActionChannels(Array(2)),
      avgReward = Nd4j.ones(1).muli(-1),
      beta = 0.1,
      maxModelSize = 0,
      epsilon = 0.01,
      kappa = 1,
      kappaPlus = 0,
      planningStepsCounter = 0,
      tolerance = Some(Nd4j.create(Array(0.5, 0.5))))

  def network: MultiLayerNetwork = {
    val hidden = new DenseLayer.Builder().
      nIn(3).
      nOut(3).
      activation(Activation.TANH).
      dropOut(0.8).
      build()

    val outLayer = new OutputLayer.Builder().
      nIn(3).
      nOut(2).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Adam(0.3 / 20, 0.9, 0.999, 0.1)).
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
      add(0, 0, 0, 7.0, 0.0).
      add(0, 0, 1, 1.0, 0.0).
      add(0, 1, 0, 1.0, 0.0).
      add(0, 1, 1, 7.0, 0.0).
      add(1, 0, 1, 1.0, 0.0).
      add(1, 0, 2, 7.0, 1.0).
      add(1, 1, 1, 7.0, 0.0).
      add(1, 1, 2, 1.0, 1.0).
      add(2, 0, 0, 1.0, 0.0).
      add(2, 1, 0, 1.0, 0.0).build

    val s0 = TestEnv(0, 0, conf)
    val s1 = TestEnv(0, 1, conf)
    //  val s2 = TestEnv(0, 2, conf)

    val session = Session(noSteps = NoSteps,
      env0 = s0,
      agent0 = agent,
      maxEpisodeLength = MaxEpisodeLength)

    val firstEpisode = session.episodes.take(1)
    val lastEpisode = session.episodes.last

    firstEpisode.combineLatest(lastEpisode).doOnNext {
      case (first, last) => Task.eval {
        logger.info("stepCount = {}/{}, score={}/{}",
          first.stepCount, last.stepCount, first.totalScore, last.totalScore)
        first.stepCount shouldBe >=(last.stepCount)
        first.totalScore shouldBe >=(last.totalScore)
      }
    }.subscribe()(Scheduler.global)

    val (_, agent1) = session.run(random)

    it("should have policy at state s0 with higher value on action 0 ") {
      val p0 = agent1.asInstanceOf[ExpSarsaAgent].q(s0.observation)
      p0.getDouble(0L) shouldBe >(p0.getDouble(1L))
    }
    it("should have policy at state s1 with higher value on action 1 ") {
      val p0 = agent1.asInstanceOf[ExpSarsaAgent].q(s1.observation)
      p0.getDouble(1L) shouldBe >(p0.getDouble(0L))
    }
  }
}