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

import java.io.File

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.Action
import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.Feedback
import org.mmarini.scalarl.FileUtils.withFile
import org.mmarini.scalarl.FileUtils.writeINDArray
import org.mmarini.scalarl.Observation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import com.typesafe.scalalogging.LazyLogging

/**
 * The agent acting in the environment by QLearning T(0) algorithm.
 *
 *  Generates actions to change the status of environment basing on observation of the environment
 *  and the internal strategy policy.
 *
 *  Updates its strategy policy to optimize the return value (discount sum of rewards)
 *  and the observation of resulting environment
 */
case class QAgent(
  net:     MultiLayerNetwork,
  random:  Random,
  epsilon: Double,
  gamma:   Double) extends Agent {

  /**
   * Returns the index containing the max value of a by masking mask
   *
   * @param a the vector of values
   * @param mask the vector mask with valid value
   */
  private def maxIdxWithMask(a: INDArray, mask: INDArray): Int = {
    var idx = -1

    for {
      i <- 0 until a.size(1).toInt
      if (mask.getInt(i) > 0)
      if (idx < 0 || a.getDouble(0L, i.toLong) > a.getDouble(0L, idx.toLong))
    } {
      idx = i
    }
    idx
  }

  /**
   * Returns the the max value of a by masking mask
   *
   * @param a the vector of values
   * @param mask the vector mask with valid value
   */
  private def maxWithMask(a: INDArray, mask: INDArray): Double = a.getDouble(maxIdxWithMask(a, mask).toLong)

  /**
   * Returns the q function with action value for an observation
   *
   * @param observationt the observation
   */
  def q(observation: Observation): INDArray = {
    val out = net.feedForward(observation.observation)
    val q = out.get(out.size() - 1)
    q
  }

  /**
   * Chooses the action for an observation
   *
   * It return the new agent and the chosen action.
   * This actor apply a epsilon greedy policy choosing a random actin with probability epsilon
   * and the action with highest action value with probability 1 - epsilon.
   *
   *  @param observation the observation of environment status
   */
  override def chooseAction(observation: Observation): (Agent, Action) = {
    val actions = observation.actions
    val action = if (random.nextDouble() < epsilon) {
      val validActions = (0 until actions.size(0).toInt).filter(i => actions.getInt(i) > 0)
      val action = validActions(random.nextInt(validActions.length))
      action
    } else {
      // Greedy policy
      val q0 = q(observation)
      val action = maxIdxWithMask(q0, actions)
      action
    }
    require(actions.getInt(action) > 0)
    (this, action)
  }

  /**
   * Return an agent that fits the expected result and the error.
   * Updates the q function to fit the expected result.
   *
   *  @param feedback the [[Feedback]] from environment after a state transition
   */
  override def fit(feedback: Feedback): (Agent, Double) = feedback match {
    case Feedback(obs0, action, reward, obs1, endUp) =>
      val q0 = q(obs0)
      val q1 = q(obs1)
      val v0 = maxWithMask(q0, obs0.actions)
      val v1 = if (endUp) {
        0.0
      } else {
        maxWithMask(q1, obs1.actions)
      }
      val err = reward + gamma * v1 - v0
      val delta = Nd4j.zeros(q0.shape(): _*)
      delta.putScalar(action, err)
      val expected = q0.add(delta)
      val newNet = net.clone()
      newNet.fit(obs0.observation, expected)
      (copy(net = newNet), err)
  }

  def writeModel(file: String): QAgent = {
    ModelSerializer.writeModel(net, file, true)
    this
  }

  override def reset: Agent = this
}

/**
 * The builder of a QAgent that build a QAgent
 *
 *  @constructor Creates a [[QAgentBuilder]]
 *  @param numInputs the number of input nodes
 *  @param numActions the total number of actions or output nodes
 *  @param _numHiddens array of the number of hidden nodes per layer
 *  @param _learningRate the learning rate
 *  @param _epsilon the epsilon value of epsilon-greedy policy
 *  @param _gamma the discount factor for total return
 *  @param _seed the seed of random generators
 *  @param _maxAbsParams max absloute value of parameters
 *  @param _maxAbsGradient max absolute value of gradients
 *  @param _file the filename of model to load
 */
case class QAgentBuilder(
  numInputs:       Int,
  numActions:      Int,
  _numHiddens:     Array[Int]     = Array(),
  _learningRate:   Double         = Adam.DEFAULT_ADAM_LEARNING_RATE,
  _epsilon:        Double         = 1e-2,
  _gamma:          Double         = 0.99,
  _seed:           Long           = 0L,
  _maxAbsParams:   Double         = 1e3,
  _maxAbsGradient: Double         = 1e2,
  _file:           Option[String] = None) extends LazyLogging {

  val DefaultEpsilon = 0.01
  val DefaultGamma = 0.99

  /** Returns the builder with a number of hidden nodes */
  def numHiddens(numHiddens: Array[Int]): QAgentBuilder = copy(_numHiddens = numHiddens)

  /** Returns the builder with a number of hidden nodes in the third layer */
  def addLayer(numHidden: Int): QAgentBuilder = numHiddens(_numHiddens :+ numHidden)

  /** Returns the builder with epsilon value */
  def epsilon(epsilon: Double): QAgentBuilder = copy(_epsilon = epsilon)

  /** Returns the builder with a discount factor of total return */
  def gamma(gamma: Double): QAgentBuilder = copy(_gamma = gamma)

  /** Returns the builder with a learning rate */
  def learningRate(learningRate: Double): QAgentBuilder = copy(_learningRate = learningRate)

  /** Returns the builder with a seed random generator */
  def seed(seed: Long): QAgentBuilder = copy(_seed = seed)

  /** Returns the builder with a maximum absolute gradient value */
  def maxAbsGradient(value: Double): QAgentBuilder = copy(_maxAbsGradient = value)

  /** Returns the builder with a maximum absolute gradient value */
  def maxAbsParams(value: Double): QAgentBuilder = copy(_maxAbsParams = value)

  /** Returns the builder with filename model to load */
  def file(file: String): QAgentBuilder = copy(_file = Some(file))

  /** Builds and returns the [[QAgent]] */
  def build(): QAgent = {
    val file = _file.map(f => new File(f)).filter(_.canRead())
    val net = file.map(loadNet).getOrElse(buildNet())
    net.init()
    QAgent(
      net = net,
      random = if (_seed != 0) new DefaultRandom(_seed) else new DefaultRandom(),
      epsilon = _epsilon,
      gamma = _gamma)
  }

  private def loadNet(file: File): MultiLayerNetwork = {
    logger.info(s"Loading ${file.toString} ...")
    ModelSerializer.restoreMultiLayerNetwork(file, true)
  }

  private def buildHiddenLayers(): Array[DenseLayer] = if (_numHiddens.isEmpty) {
    Array()
  } else {
    val inpAct = numInputs +: _numHiddens.init
    val layers = for {
      (n, m) <- inpAct.zip(_numHiddens)
    } yield new DenseLayer.Builder().
      nIn(n).
      nOut(m).
      weightInit(WeightInit.XAVIER).
      activation(Activation.TANH).
      build()
    layers
  }

  /** Returns the built network for the QAgent */
  private def buildNet(): MultiLayerNetwork = {
    val layers = buildHiddenLayers()
    val outActivations = _numHiddens.lastOption.getOrElse(numInputs)
    val outLayer = new OutputLayer.Builder().
      nIn(outActivations).
      nOut(numActions).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(_seed).
      weightInit(WeightInit.XAVIER).
      updater(new Adam(_learningRate)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      constrainAllParameters(new MinMaxNormConstraint(-_maxAbsParams, _maxAbsParams, 1)).
      gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).
      gradientNormalizationThreshold(_maxAbsGradient).
      list((layers :+ outLayer): _*).
      build()

    new MultiLayerNetwork(conf);
  }
}

