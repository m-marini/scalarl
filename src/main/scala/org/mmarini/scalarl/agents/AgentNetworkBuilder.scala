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

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
 * The builder of neural network for agent
 *
 * @constructor Creates a [[AgentNetworkBuilder]]
 * @param _numInputs      the number of input nodes
 *                        The number of input nodes must be the number of observations signals
 *                        that is (number of observations + number of actions)
 * @param _numOutputs     the number of output nodes
 * @param _numHiddens     the number of hidden nodes in the second layer
 * @param _seed           the seed of random generators
 * @param _learningRate   the learning rate
 * @param _beta1          beta1 parameter of ADAM algorithm
 * @param _beta2          beta2 parameter of ADAM algorithm
 * @param _epsilonAdam    epsilon parameter of ADAM algorithm
 * @param _maxAbsParams   max absolute value of parameters
 * @param _maxAbsGradient max absolute value of gradients
 * @param _file           the filename of model to load
 */
case class AgentNetworkBuilder(
                                _numInputs: Int,
                                _numOutputs: Int,
                                _numHiddens: Array[Int],
                                _seed: Long,
                                _learningRate: Double,
                                _beta1: Double,
                                _beta2: Double,
                                _epsilonAdam: Double,
                                _maxAbsParams: Double,
                                _maxAbsGradient: Double,
                                _file: Option[String]) extends LazyLogging {

  /**
   * Returns the builder with a number of output nodes
   * The number of input nodes must be the number of observations signals
   * that is number of observations + number of actions
   */
  def numInputs(numInputs: Int): AgentNetworkBuilder = copy(_numInputs = numInputs)

  /**
   * Returns the builder with a number of input nodes
   * The number of outputs nodes must be the number of actions
   */
  def numOutputs(numOutputs: Int): AgentNetworkBuilder = copy(_numOutputs = numOutputs)

  /** Returns the builder with a number of hidden nodes in the third layer */
  def numHiddens(numHiddens: Int*): AgentNetworkBuilder = copy(_numHiddens = numHiddens.toArray)

  /** Returns the builder with a learning rate */
  def learningRate(learningRate: Double): AgentNetworkBuilder = copy(_learningRate = learningRate)

  /** Returns the builder with a seed random generator */
  def seed(seed: Long): AgentNetworkBuilder = copy(_seed = seed)

  /** Returns the builder with a maximum absolute gradient value */
  def maxAbsGradient(value: Double): AgentNetworkBuilder = copy(_maxAbsGradient = value)

  /** Returns the builder with a maximum absolute gradient value */
  def maxAbsParams(value: Double): AgentNetworkBuilder = copy(_maxAbsParams = value)

  /** Returns the builder with filename model to load */
  def file(file: String): AgentNetworkBuilder = copy(_file = Some(file))

  /** Returns the builder for beta1 adam parameter */
  def beta1(beta1: Double): AgentNetworkBuilder = copy(_beta1 = beta1)

  /** Returns the builder for beta2 adam parameter */
  def beta2(beta2: Double): AgentNetworkBuilder = copy(_beta2 = beta2)

  /** Returns the builder for epsilonAdam adam parameter */
  def epsilonAdam(epsilonAdam: Double): AgentNetworkBuilder = copy(_epsilonAdam = epsilonAdam)

  def build(): MultiLayerNetwork =
    _file.map(f => loadNet(new File(f))).getOrElse(createNetwork())

  private def loadNet(file: File): MultiLayerNetwork = {
    logger.info(s"Loading ${file.toString} ...")
    ModelSerializer.restoreMultiLayerNetwork(file, true)
  }

  /** Returns the built network for the QAgent */
  private def createNetwork(): MultiLayerNetwork = {
    // Computes the number of nodes of initial layers
    val initialNodes = _numInputs +: _numHiddens
    // Creates the hidden layers
    val hiddenLayers = for {
      (ins, outs) <- initialNodes.init.zip(initialNodes.tail)
    } yield new DenseLayer.Builder().
      nIn(ins).
      nOut(outs).
      activation(Activation.TANH).
      build()

    val outLayer = new OutputLayer.Builder().
      nIn(initialNodes.last).
      nOut(_numOutputs).
      lossFunction(LossFunction.MSE).
      activation(Activation.IDENTITY).
      build()

    val conf = new NeuralNetConfiguration.Builder().
      seed(_seed).
      weightInit(WeightInit.XAVIER).
      updater(new Adam(_learningRate, _beta1, _beta2, _epsilonAdam)).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      constrainAllParameters(new MinMaxNormConstraint(-_maxAbsParams, _maxAbsParams, 1)).
      gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).
      gradientNormalizationThreshold(_maxAbsGradient).
      list(hiddenLayers :+ outLayer: _*).
      build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net
  }
}

/** Utility */
object AgentNetworkBuilder {
  val DefaultNumInput = 0
  val DefaultNumOutput = 0
  val DefaultNumHidden: Array[Int] = Array[Int]()
  val DefaultSeed = 0L
  val DefaultLearningRate = 0.1
  val DefaultBeta1 = 0.9
  val DefaultBeta2 = 0.999
  val DefaultEpsilonAdam = 0.001
  val DefaultMaxAbsParams = 0.0
  val DefaultMaxAbsGradient = 0.0

  /**
   * Returns [AgentNetworkBuilder for a configuration
   *
   * @param agentCursor json configuration
   * @param numInputs   number of inputs
   * @param numOutputs  number of outputs
   */
  def apply(agentCursor: ACursor,
            numInputs: Int,
            numOutputs: Int): AgentNetworkBuilder =
  // Returns the builder from file
    agentCursor.get[String]("modelFile").toOption.
      map(f => AgentNetworkBuilder().file(f)).getOrElse(
      // Returns the new builder from configuration
      AgentNetworkBuilder().
        numInputs(numInputs).
        numOutputs(numOutputs).
        seed(agentCursor.get[Long]("seed").getOrElse(DefaultSeed)).
        numHiddens(agentCursor.get[List[Int]]("numHiddens").right.get.toArray: _ *).
        learningRate(agentCursor.get[Double]("learningRate").right.get).
        beta1(agentCursor.get[Double]("beta1").right.get).
        beta2(agentCursor.get[Double]("beta2").right.get).
        epsilonAdam(agentCursor.get[Double]("epsilonAdam").right.get).
        maxAbsGradient(agentCursor.get[Double]("maxAbsGradients").right.get).
        maxAbsParams(agentCursor.get[Double]("maxAbsParameters").right.get)
    )

  /** Returns default [AgentNetworkBuilder] */
  def apply(): AgentNetworkBuilder = AgentNetworkBuilder(
    _numInputs = DefaultNumInput,
    _numHiddens = DefaultNumHidden,
    _numOutputs = DefaultNumOutput,
    _seed = DefaultSeed,
    _learningRate = DefaultLearningRate,
    _beta1 = DefaultBeta1,
    _beta2 = DefaultBeta2,
    _epsilonAdam = DefaultEpsilonAdam,
    _maxAbsParams = DefaultMaxAbsParams,
    _maxAbsGradient = DefaultMaxAbsGradient,
    _file = None)
}
