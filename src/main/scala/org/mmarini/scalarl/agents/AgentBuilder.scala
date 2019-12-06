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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.ActionChannelConfig
import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.nn.AccumulateTraceMode
import org.mmarini.scalarl.nn.ActivationLayerBuilder
import org.mmarini.scalarl.nn.AdamOptimizer
import org.mmarini.scalarl.nn.DenseLayerBuilder
import org.mmarini.scalarl.nn.LayerBuilder
import org.mmarini.scalarl.nn.NetworkBuilder
import org.mmarini.scalarl.nn.NoneTraceMode
import org.mmarini.scalarl.nn.Normalizer
import org.mmarini.scalarl.nn.SGDOptimizer
import org.mmarini.scalarl.nn.TanhActivationFunction
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.mmarini.scalarl.ActionChannelConfig
import io.circe.ACursor

/**
 * The builder of a QAgent that build a QAgent
 *
 *  @constructor Creates a [[QAgentBuilder]]
 *  @param _numInputs the number of input nodes
 *          The number of input nodes must be the number of observations signals
 *          that is (number of observations + number of actions)
 *  @param _numActions the total number of actions or output nodes
 *  @param _epsilon the epsilon value of epsilon-greedy policy
 *  @param _gamma the discount factor for total return
 *  @param _seed the seed of random generators
 *  @param _minHistory batch history minimal length to learn
 *  @param _maxHistory batch history length
 *  @param _stepInterval number of step interval between learning
 *  @param _numBootsrapIteration number of iteration to bootsrap the model
 *  @param _numBatchIteration number of iteration batch
 *  @param _config channel action configuration
 */
case class AgentBuilder(
  _agentType:             String,
  _numInputs:             Int,
  _config:                ActionChannelConfig,
  _epsilon:               Double,
  _gamma:                 Double,
  _seed:                  Long,
  _kappa:                 Double,
  _lambda:                Double,
  _minHistory:            Int,
  _maxHistory:            Int,
  _stepInterval:          Int,
  _numBootstrapIteration: Int,
  _numBatchIteration:     Int,
  _networkBuilder:        Option[AgentNetworkBuilder]) extends LazyLogging {

  /**
   * Returns the builder with a number of input nodes
   * The number of input nodes must be the number of observations signals
   * that is number of observations + number of actions
   */
  def numInputs(numInput: Int): AgentBuilder = copy(_numInputs = numInput)

  /** Returns the builder with for a given action configuration */
  def config(config: ActionChannelConfig): AgentBuilder = copy(_config = config)

  /** Returns the builder with a number of hidden nodes in the third layer */
  def numBatchIteration(value: Int): AgentBuilder = copy(_numBatchIteration = value)

  /** Returns the builder with a number of hidden nodes in the third layer */
  def numBootstrapIteration(value: Int): AgentBuilder = copy(_numBootstrapIteration = value)

  /** Returns the builder with epsilon value */
  def epsilon(epsilon: Double): AgentBuilder = copy(_epsilon = epsilon)

  /** Returns the builder with a discount factor of total return */
  def gamma(gamma: Double): AgentBuilder = copy(_gamma = gamma)

  /** Returns the builder for a given lambda hyper parameter */
  def lambda(lambda: Double): AgentBuilder = copy(_lambda = lambda)

  /** Returns the builder for a given kappa hyper parameter */
  def kappa(kappa: Double): AgentBuilder = copy(_kappa = kappa)

  /** Returns the builder with a seed random generator */
  def seed(seed: Long): AgentBuilder = copy(_seed = seed)

  /** Returns the builder with a maximum history  */
  def minHistory(value: Int): AgentBuilder = copy(_minHistory = value)

  /** Returns the builder with a maximum history  */
  def stepInterval(value: Int): AgentBuilder = copy(_stepInterval = value)

  /** Returns the builder with a maximum history  */
  def maxHistory(value: Int): AgentBuilder = copy(_maxHistory = value)

  /** Returns the builder for agent type */
  def agentType(agentType: String): AgentBuilder = copy(_agentType = agentType)

  /** Returns the builder for agent type */
  def networkBuilder(networkBuilder: AgentNetworkBuilder): AgentBuilder = copy(_networkBuilder = Some(networkBuilder))

  /** Builds and returns the [[QAgent]] */
  def build(): Agent = {
    val random = if (_seed != 0) {
      Nd4j.getRandomFactory().getNewRandomInstance(_seed)
    } else {
      Nd4j.getRandom()
    }
    _agentType match {
      case "TDBatchAgent" =>
        val net = _networkBuilder.get.build()
        TDBatchAgent(
          config = _config,
          net = net,
          history = new AgentHistory(_maxHistory, Seq()),
          random = random,
          epsilon = _epsilon,
          gamma = _gamma,
          kappa = _kappa,
          minData = _minHistory,
          stepData = _stepInterval,
          stepCounter = 0,
          noBootstrapIter = _numBootstrapIteration,
          noEpochIter = _numBatchIteration)
      case s => throw new IllegalArgumentException(s"""agent type "${s}" invalid""")
    }
  }
}

object AgentBuilder {
  val DefaultAgentType = "TDBatchAgent"
  val DefaultInputs = 0
  val DefaultConfig = Array[Int]();
  val DefaultEpsilon = 0.01
  val DefaultGamma = 0.99
  val DefaultSeed = 0
  val DefaultKappa = 1
  val DefaultLambda = 0
  val DefaultMinHistory = 1
  val DefaultMaxHistory = 1
  val DefaultStepInterval = 1
  val DefaultNumBootstrapIteration = 1
  val DefaultNumBatchIteration = 1

  /** Returns default agent builder */
  def apply(): AgentBuilder = AgentBuilder(
    _agentType = DefaultAgentType,
    _numInputs = DefaultInputs,
    _config = DefaultConfig,
    _epsilon = DefaultEpsilon,
    _gamma = DefaultGamma,
    _seed = DefaultSeed,
    _kappa = DefaultKappa,
    _lambda = DefaultLambda,
    _minHistory = DefaultMinHistory,
    _maxHistory = DefaultMaxHistory,
    _stepInterval = DefaultStepInterval,
    _numBootstrapIteration = DefaultNumBootstrapIteration,
    _numBatchIteration = DefaultNumBatchIteration,
    _networkBuilder = None)

  /** Returns agent builder from configuration */
  def apply(agentCursor: ACursor): AgentBuilder = {
    val agentType = agentCursor.get[String]("type").getOrElse("TDBatchAgent")
    val ni = agentCursor.get[Int]("numInputs").right.get
    val seed = agentCursor.get[Long]("seed").toOption
    val epsilon = agentCursor.get[Double]("epsilon").toOption
    val gamma = agentCursor.get[Double]("gamma").toOption
    val kappa = agentCursor.get[Double]("kappa").toOption
    val lambda = agentCursor.get[Double]("lambda").toOption
    val minHistory = agentCursor.get[Int]("minHistory").toOption
    val maxHistory = agentCursor.get[Int]("maxHistory").toOption
    val stepInterval = agentCursor.get[Int]("stepInterval").toOption
    val numBootstrapIteration = agentCursor.get[Int]("numBootstrapIteration").toOption
    val numBatchIteration = agentCursor.get[Int]("numBatchIteration").toOption

    val builder1 = AgentBuilder().numInputs(ni)
    val builder2 = seed.map(builder1.seed).getOrElse(builder1)
    val builder4 = epsilon.map(builder2.epsilon).getOrElse(builder2)
    val builder5 = gamma.map(builder4.gamma).getOrElse(builder4)
    val builder6 = kappa.map(builder5.kappa).getOrElse(builder5)

    val builder8 = lambda.map(builder6.lambda).getOrElse(builder6)

    val builder16 = maxHistory.map(builder8.maxHistory).getOrElse(builder8)
    val builder17 = numBatchIteration.map(builder16.numBatchIteration).getOrElse(builder16)
    val builder18 = minHistory.map(builder17.minHistory).getOrElse(builder17)
    val builder19 = stepInterval.map(builder18.stepInterval).getOrElse(builder18)
    val builder20 = numBootstrapIteration.map(builder19.numBootstrapIteration).getOrElse(builder19)

    builder20
  }
}
