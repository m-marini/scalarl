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
import org.mmarini.scalarl.Agent
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import com.typesafe.scalalogging.LazyLogging

object AgentType extends Enumeration {
  val QAgent, TDQAgent = Value
}

/**
 * The builder of a QAgent that build a QAgent
 *
 *  @constructor Creates a [[QAgentBuilder]]
 *  @param _numInputs the number of input nodes
 *          The number of input nodes must be the number of observations signals
 *          that is (number of observations + number of actions)
 *  @param _numActions the total number of actions or output nodes
 *  @param _numHiddens1 the number of hidden nodes in the second layer
 *  @param _numHiddens1 the number of hidden nodes in the third layer
 *  @param _learningRate the learning rate
 *  @param _epsilon the epsilon value of epsilon-greedy policy
 *  @param _gamma the discount factor for total return
 *  @param _seed the seed of random generators
 *  @param _maxAbsParams max absolute value of parameters
 *  @param _maxAbsGradient max absolute value of gradients
 *  @param _file the filename of model to load
 */
case class AgentBuilder(
  _numInputs:      Int             = 0,
  _numActions:     Int             = 0,
  _numHiddens:     Array[Int]      = Array(),
  _learningRate:   Double          = 0,
  _epsilon:        Double          = 0.01,
  _gamma:          Double          = 0.99,
  _lambda:         Double          = 0,
  _seed:           Long            = 0,
  _maxAbsParams:   Double          = 0,
  _maxAbsGradient: Double          = 0,
  _agentType:      AgentType.Value = AgentType.QAgent,
  _file:           Option[String]  = None) extends LazyLogging {

  /**
   * Returns the builder with a number of input nodes
   * The number of input nodes must be the number of observations signals
   * that is number of observations + number of actions
   */
  def numInputs(numInput: Int): AgentBuilder = copy(_numInputs = numInput)

  /** Returns the builder with a number of actions */
  def numActions(numActions: Int): AgentBuilder = copy(_numActions = numActions)

  /** Returns the builder with a number of hidden nodes in the third layer */
  def numHiddens(numHiddens: Int*): AgentBuilder = copy(_numHiddens = numHiddens.toArray)

  /** Returns the builder with epsilon value */
  def epsilon(epsilon: Double): AgentBuilder = copy(_epsilon = epsilon)

  /** Returns the builder with a discount factor of total return */
  def gamma(gamma: Double): AgentBuilder = copy(_gamma = gamma)

  /** Returns the builder with a learning rate */
  def learningRate(learningRate: Double): AgentBuilder = copy(_learningRate = learningRate)

  /** Returns the builder with a seed random generator */
  def seed(seed: Long): AgentBuilder = copy(_seed = seed)

  /** Returns the builder with a maximum absolute gradient value */
  def maxAbsGradient(value: Double): AgentBuilder = copy(_maxAbsGradient = value)

  /** Returns the builder with a maximum absolute gradient value */
  def maxAbsParams(value: Double): AgentBuilder = copy(_maxAbsParams = value)

  /** Returns the builder with filename model to load */
  def file(file: String): AgentBuilder = copy(_file = Some(file))

  def agentType(agentType: AgentType.Value): AgentBuilder = copy(_agentType = agentType)

  /** Builds and returns the [[QAgent]] */
  def build(): Agent = {
    val random = if (_seed != 0) new DefaultRandom(_seed) else new DefaultRandom()

    _agentType match {
      case AgentType.QAgent =>
        val file = _file.map(f => new File(f)).filter(_.canRead())
        val net = file.map(loadNet).getOrElse(buildNet())
        net.init()
        TD0QAgent(
          net = net,
          random = random,
          epsilon = _epsilon,
          gamma = _gamma)
      case AgentType.TDQAgent =>
        val file = _file.map(f => new File(f)).filter(_.canRead())
        val net = file.map(loadTraceNet).getOrElse(buildTraceNet())
        TDQAgent(
          net = net,
          random = random,
          epsilon = _epsilon,
          gamma = _gamma,
          lambda = _lambda)
    }
  }

  private def loadTraceNet(file: File): TraceNetwork = {
    logger.info(s"Loading ${file.toString} ...")
    TraceModelSerializer.restoreTraceNetwork(file)
  }

  private def buildTraceNet(): TraceNetwork = {
    // Computes the number of nodes of initial layers
    val initialNodes = _numInputs +: _numHiddens
    // Creates the hidden layers
    val hiddenLayers = for {
      (ins, outs) <- initialNodes.init.zip(initialNodes.tail)
      layer <- Seq(
        TraceDenseLayer(
          noInputs = ins,
          noOutputs = outs,
          gamma = _gamma,
          lambda = _lambda,
          learningRate = _learningRate),
        TraceTanhLayer())
    } yield layer

    // Creates the output layer
    val outLayer = TraceDenseLayer(
      noInputs = initialNodes.last,
      noOutputs = _numActions,
      gamma = _gamma,
      lambda = _lambda,
      learningRate = _learningRate)
    new TraceNetwork(layers = hiddenLayers :+ outLayer)
  }

  private def loadNet(file: File): MultiLayerNetwork = {
    logger.info(s"Loading ${file.toString} ...")
    ModelSerializer.restoreMultiLayerNetwork(file, true)
  }

  /** Returns the built network for the QAgent */
  private def buildNet(): MultiLayerNetwork = {
    // Computes the number of nodes of initial layers
    val initialNodes = _numInputs +: _numHiddens
    // Creates the hidden layers
    val hiddenLayers = for {
      (ins, outs) <- initialNodes.init.zip(initialNodes.tail)
    } yield new DenseLayer.Builder().
      nIn(ins).
      nOut(outs).
      weightInit(WeightInit.XAVIER).
      activation(Activation.TANH).
      build()

    val outLayer = new OutputLayer.Builder().
      nIn(initialNodes.last).
      nOut(_numActions).
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
      list((hiddenLayers :+ outLayer): _*).
      build()

    new MultiLayerNetwork(conf);
  }
}
