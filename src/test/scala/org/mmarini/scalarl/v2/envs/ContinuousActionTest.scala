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

package org.mmarini.scalarl.v2.envs

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.v2.agents.ACContinuousActionAgent
import org.mmarini.scalarl.v2.{AgentContinuousAction, SessionContinuousAction}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.scalatest.{FunSpec, Matchers}

class ContinuousActionTest extends FunSpec with Matchers with LazyLogging {
  val Seed = 12345L
  val NoSteps = 1000
  val Hiddens = 10
  val NoInputs: Long = Tiles(2).noFeatures

  Nd4j.create()

  def agent: AgentContinuousAction = ACContinuousActionAgent(
    actor = actor,
    critic = critic,
    avg = 0.0,
    alpha = 0.03,
    rewardDecay = 0.97,
    valueDecay = 0.99)

  def critic: MultiLayerNetwork = {
    val outLayer = new OutputLayer.Builder().
      nIn(NoInputs).
      nOut(1).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      updater(new Sgd(3e-3)).
      //updater(new Adam(1000e-3 / (4), 0.9, 0.999, 0.1)).
      //updater(new Adam(1000e-3 / (4), 0.9, 0.999, 0.1)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      miniBatch(false).
      //      constrainAllParameters(new MinMaxNormConstraint(-10e3, 10e3, 1)).
      //      gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).
      //      gradientNormalizationThreshold(100).
      list(outLayer)
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net
  }

  def actor: MultiLayerNetwork = {
    val outLayer = new OutputLayer.Builder().
      nIn(NoInputs).
      nOut(2).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(Seed).
      weightInit(WeightInit.XAVIER).
      //      updater(new Adam(10.0 / (4 * Hiddens + (Hiddens + 1) * 2), 0.9, 0.999, 0.1)).
      //updater(new Adam(1.0 / (4 * 2), 0.9, 0.999, 0.1)).
      updater(new Sgd(1e-3)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      constrainAllParameters(new MinMaxNormConstraint(-10e3, 10e3, 1)).
      //    gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).
      //      gradientNormalizationThreshold(1).
      list(outLayer)
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net
  }

  val random: Random =
    Nd4j.getRandomFactory.getNewRandomInstance(Seed)

  describe("ContinuousActionEnv") {


    val s0 = ContinuousActionEnv(Nd4j.ones(1).mul(2), Nd4j.zeros(1))

    val session = new SessionContinuousAction(numSteps = NoSteps,
      epoch = 0,
      env = s0,
      agent = agent)

    val (_, agent1) = session.run(random)
    val (mu, _, sigma) = agent1.asInstanceOf[ACContinuousActionAgent].muHSigma(s0.observation)
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