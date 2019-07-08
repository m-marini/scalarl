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
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.mmarini.scalarl.nn.NetworkBuilder
import org.mmarini.scalarl.nn.SGDOptimizer
import org.mmarini.scalarl.nn.AccumulateTraceMode
import org.mmarini.scalarl.nn.DenseLayerBuilder
import org.mmarini.scalarl.nn.ActivationLayerBuilder
import org.mmarini.scalarl.nn.TanhActivationFunction
import org.mmarini.scalarl.nn.LayerBuilder
import org.mmarini.scalarl.nn.AdamOptimizer
import org.mmarini.scalarl.nn.NoneTraceMode
import org.mmarini.scalarl.nn.Normalizer

object AgentType extends Enumeration {
  val QAgent, TDAAgent = Value
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
  _agentType:      String         = "QAgent",
  _numInputs:      Int            = 0,
  _numActions:     Int            = 0,
  _numHiddens:     Array[Int]     = Array(),
  _epsilon:        Double         = 0.01,
  _gamma:          Double         = 0.99,
  _seed:           Long           = 0,
  _kappa:          Double         = 1,
  _optimizer:      String         = "SGD",
  _learningRate:   Double         = 0.1,
  _beta1:          Double         = 0.9,
  _beta2:          Double         = 0.999,
  _epsilonAdam:    Double         = 0.001,
  _trace:          String         = "ACCUMULATE",
  _lambda:         Double         = 0,
  _maxAbsParams:   Double         = 0,
  _maxAbsGradient: Double         = 0,
  _file:           Option[String] = None
//  _traceUpdater:   TraceUpdater    = AccumulateTraceUpdater
) extends LazyLogging {

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

  /** Returns the builder for a given lambda hyper parameter */
  def lambda(lambda: Double): AgentBuilder = copy(_lambda = lambda)

  /** Returns the builder for a given kappa hyper parameter */
  def kappa(kappa: Double): AgentBuilder = copy(_kappa = kappa)

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

  /** Returns the builder for agent type */
  def agentType(agentType: String): AgentBuilder = copy(_agentType = agentType)

  /** Returns the builder for trace mode */
  def trace(trace: String): AgentBuilder = copy(_trace = trace)

  /** Returns the builder for the optimizer */
  def optimizer(optimizer: String): AgentBuilder = copy(_optimizer = optimizer)

  /** Returns the builder for beta1 adam parameter */
  def beta1(beta1: Double): AgentBuilder = copy(_beta1 = beta1)

  /** Returns the builder for beta2 adam parameter */
  def beta2(beta2: Double): AgentBuilder = copy(_beta2 = beta2)

  /** Returns the builder for epsilonAdam adam parameter */
  def epsilonAdam(epsilonAdam: Double): AgentBuilder = copy(_epsilonAdam = epsilonAdam)

  /** Builds and returns the [[QAgent]] */
  def build(): Agent = {
    val random = if (_seed != 0) {
      Nd4j.getRandomFactory().getNewRandomInstance(_seed)
    } else {
      Nd4j.getRandom()
    }
    _agentType match {
      case "QAgent" =>
        val file = _file.map(f => new File(f)).filter(_.canRead())
        val net = file.map(loadNet).getOrElse(buildNet(random))
        net.init()
        TD0QAgent(
          net = net,
          random = random,
          epsilon = _epsilon,
          gamma = _gamma)
      case "TDAAgent" =>
        val file = _file.map(f => new File(f)).filter(_.canRead())
        //        val builder = file.map(loadTraceNet).getOrElse(buildTraceNet(random))
        val builder = buildTraceNet()
        val proc = builder.buildProcessor
        val data = builder.buildData(random)
        TDAAgent(
          netProc = proc,
          netData = data,
          random = random,
          epsilon = _epsilon,
          gamma = _gamma,
          lambda = _lambda,
          kappa = _kappa)
      case s => throw new IllegalArgumentException(s"""agent type "${s}" invalid""")
    }
  }

  //  private def loadTraceNet(file: File): TraceNetwork = {
  //    logger.info(s"Loading ${file.toString} ...")
  //    TraceModelSerializer.restoreTraceNetwork(file)
  //  }
  //
  private def buildTraceNet(): NetworkBuilder = {

    val hiddens = for {
      (nodes, layer) <- _numHiddens.zipWithIndex
      dense = DenseLayerBuilder(s"${layer}.dense", nodes)
      act = ActivationLayerBuilder(s"${layer}.activation", TanhActivationFunction)
      builder <- Array[LayerBuilder](dense, act)
    } yield builder

    val withOutputs = hiddens :+
      DenseLayerBuilder("outputs.dense", _numActions)

    val optimizer = _optimizer match {
      case "SGD"  => SGDOptimizer(alpha = _learningRate)
      case "ADAM" => AdamOptimizer(alpha = _learningRate, beta1 = _beta1, beta2 = _beta2, epsilon = _epsilonAdam)
      case s      => throw new IllegalArgumentException(s"""optimizer "${s}" invalid""")
    }

    val trace = _trace match {
      case "ACCUMULATE" => AccumulateTraceMode(gamma = _gamma, lambda = _lambda)
      case "NONE"       => NoneTraceMode
      case s            => throw new IllegalArgumentException(s"""trace "${s}" invalid""")
    }

    NetworkBuilder().
      setNoInputs(_numInputs).
      setOptimizer(optimizer).
      setTraceMode(trace).
      setNormalizer(Normalizer.minMax(_numInputs, 0, 1)).
      addLayers(withOutputs: _*)
  }

  private def loadNet(file: File): MultiLayerNetwork = {
    logger.info(s"Loading ${file.toString} ...")
    ModelSerializer.restoreMultiLayerNetwork(file, true)
  }

  /** Returns the built network for the QAgent */
  private def buildNet(random: Random): MultiLayerNetwork = {
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
