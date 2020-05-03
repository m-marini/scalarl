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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.mmarini.scalarl.v3.Session
import org.mmarini.scalarl.v3.agents._
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.scalatest.{FunSpec, Matchers}

class MountingCarTest extends FunSpec with Matchers with LazyLogging {
  val Seed = 12345L
  val NoSteps = 1000
  val Hiddens = 10
  val NoInputs: Long = Tiles(1, 1).noFeatures

  create()

  def agent: ActorCriticAgent = ActorCriticAgent(
    critic = critic,
    avg = zeros(1),
    rewardDecay = ones(1).muli(0.97),
    valueDecay = ones(1).muli(0.99),
    actors = Array(GaussianActor(dimension = 0,
      actor = actor,
      alpha = ones(1).muli(0.1))))

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
      updater(new Sgd(1.0e-3)).
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
    getRandomFactory.getNewRandomInstance(Seed)

  describe("TestEnv") {


    val s0 = MountingCarEnv.initial(random)

    val session = new Session(numSteps = NoSteps,
      epoch = 0,
      env = s0,
      agent = agent)

    val (_, agent1) = session.run(random)

    it("should accelerate left when going to left") {
      val s = MountingCarEnv(zeros(1), ones(1).muli(-0.07), zeros(1))
      val (mu, _, _) = agent1.asInstanceOf[ActorCriticAgent].
        actors.head.asInstanceOf[GaussianActor].
        muHSigma(s.observation)
      mu.getDouble(0L) should be < 0.0
    }

    it("should accelerate right when going to right") {
      val s = MountingCarEnv(zeros(1), ones(1).muli(0.07), zeros(1))
      val (mu, _, _) = agent1.asInstanceOf[ActorCriticAgent].
        actors.head.asInstanceOf[GaussianActor].
        muHSigma(s.observation)
      mu.getDouble(0L) should be > 0.0
    }
  }
}