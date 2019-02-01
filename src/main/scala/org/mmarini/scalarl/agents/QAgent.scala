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

package org.mmarini.scalarl.agents

import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.Observation
import org.mmarini.scalarl.Action
import org.mmarini.scalarl.Feedback
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.learning.config.Sgd
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.lossfunctions.impl.LossMSLE
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.factory.Nd4j

case class QAgent(net: MultiLayerNetwork, random: Random, epsilon: Double, gamma: Double) extends Agent {

  private def maxIdxWithMask(a: INDArray, mask: INDArray): Int = {
    var idx = -1

    for {
      i <- 0 until a.size(0).toInt
      if (mask.getInt(i) > 0)
      if (idx < 0 || a.getDouble(i.toLong) > a.getDouble(idx.toLong))
    } {
      idx = i
    }
    idx
  }
  private def maxWithMask(a: INDArray, mask: INDArray): Double = a.getDouble(maxIdxWithMask(a, mask).toLong)

  private def q(observation: Observation): INDArray = {
    val out = net.feedForward(observation.observation.ravel())
    out.get(out.size() - 1)
  }

  override def chooseAction(observation: Observation): (Agent, Action) = {
    val actions = observation.actions
    val action = if (random.nextDouble() < epsilon) {
      val validActions = (0 until actions.size(0).toInt).filter(i => actions.getInt(i) > 0)
      validActions(random.nextInt(validActions.length))
    } else {
      // Greedy policy
      maxIdxWithMask(q(observation), actions)
    }
    (this, action)
  }

  def fit(feedback: Feedback): Agent = feedback match {
    case (obs0, action, reward, obs1, endUp, _) =>
      val q0 = q(obs0)
      val q1 = q(obs1)
      val v0 = maxWithMask(q0, obs0.actions)
      val v1 = maxWithMask(q1, obs1.actions)
      val err = reward + gamma * v1 - v0
      val delta = Nd4j.zeros(q0.shape(): _*)
      delta.putScalar(action, err)
      val expected = q0.add(delta)
      net.fit(obs0.observation.ravel(), expected)
      this
  }

}

case class QAgentBuilder(
  numInputs:    Int,
  numActions:   Int,
  _numHiddens1: Option[Int]    = None,
  _numHiddens2: Option[Int]    = None,
  _epsilon:     Option[Double] = None,
  _gamma:       Option[Double] = None) {

  val DefaultEpsilon = 0.01
  val DefaultGamma = 0.99
  val DefaultHidden = 10

  def numHiddens1(numHidden: Int): QAgentBuilder = copy(_numHiddens1 = Some(numHidden))

  def numHiddens2(numHidden: Int): QAgentBuilder = copy(_numHiddens2 = Some(numHidden))

  def epsilon(epsilon: Double): QAgentBuilder = copy(_epsilon = Some(epsilon))

  def gamma(gamma: Double): QAgentBuilder = copy(_gamma = Some(gamma))

  def build(): QAgent = {
    val net = buildNet()
    net.init()
    QAgent(
      net = net,
      random = new DefaultRandom(),
      epsilon = _epsilon.getOrElse(DefaultEpsilon),
      gamma = _gamma.getOrElse(DefaultGamma))
  }

  private def buildNet(): MultiLayerNetwork = {
    val layer1 = new DenseLayer.Builder().
      nIn(numInputs).
      nOut(_numHiddens1.getOrElse(DefaultHidden)).
      weightInit(WeightInit.XAVIER).
      activation(Activation.TANH).
      build()
    val layer2 = new DenseLayer.Builder().
      nIn(_numHiddens1.getOrElse(DefaultHidden)).
      nOut(_numHiddens2.getOrElse(DefaultHidden)).
      weightInit(WeightInit.XAVIER).
      activation(Activation.TANH).
      build()
    val layer3 = new OutputLayer.Builder().
      nIn(_numHiddens2.getOrElse(DefaultHidden)).
      nOut(numActions).
      lossFunction(LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      weightInit(WeightInit.XAVIER).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      list(layer1, layer2, layer3).
      build()

    new MultiLayerNetwork(conf);
  }
}

