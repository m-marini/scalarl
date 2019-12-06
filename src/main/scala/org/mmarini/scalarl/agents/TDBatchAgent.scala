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

import org.mmarini.scalarl.Action
import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.Feedback
import org.mmarini.scalarl.Observation
import org.mmarini.scalarl.nn.NetworkData
import org.mmarini.scalarl.nn.NetworkProcessor
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.mmarini.scalarl.nn.NetDataMaterializer
import java.io.File
import org.mmarini.scalarl.ChannelAction
import org.mmarini.scalarl.ActionChannelConfig
import org.mmarini.scalarl.ActionChannelConfig
import org.mmarini.scalarl.StatusValues
import org.mmarini.scalarl.Policy
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator
import org.nd4j.linalg.primitives.Pair
import scala.collection.JavaConversions._
import org.mmarini.scalarl.ChannelAction
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.util.ModelSerializer

/**
 * The agent acting in the environment by QLearning T(0) algorithm.
 *
 *  Generates actions to change the status of environment basing on observation of the environment
 *  and the internal strategy policy.
 *
 *  Updates its strategy policy to optimize the return value (discount sum of rewards)
 *  and the observation of resulting environment
 */
case class TDBatchAgent(
  net:             MultiLayerNetwork,
  history:         AgentHistory,
  random:          Random,
  config:          ActionChannelConfig,
  epsilon:         Double,
  gamma:           Double,
  kappa:           Double,
  minData:         Int,
  stepData:        Int,
  stepCounter:     Int,
  noBootstrapIter: Int,
  noEpochIter:     Int) extends TDAgent with LazyLogging {

  override def policy(observation: Observation): Policy = if (observation.endUp) {
    TDAgentUtils.endStatePolicy(config)
  } else {
    net.output(observation.signals)
  }

  override def greedyAction(observation: Observation): ChannelAction =
    TDAgentUtils.actionAndStatusValuesFromPolicy(
      policy = policy(observation),
      valueMask = observation.actions,
      conf = config)._1

  /**
   * Chooses the action for an observation
   *
   * It return the new agent and the chosen action.
   * This actor apply a epsilon greedy policy choosing a random actin with probability epsilon
   * and the action with highest action value with probability 1 - epsilon.
   *
   *  @param observation the observation of environment status
   */
  override def chooseAction(observation: Observation): (Agent, ChannelAction) = {
    val valueMask = observation.actions
    val action = if (random.nextDouble() < epsilon) {
      TDAgentUtils.randomAction(valueMask, config)(random)
    } else {
      // Greedy policy
      greedyAction(observation)
    }
    (this, action)
  }

  override def fit(feedback: Feedback): Agent = {
    // Update history
    val agentWithNewHistory = copy(history = history :+ feedback)

    if (agentWithNewHistory.history.length >= minData && stepCounter + 1 >= stepData) {
      // Iterates creating dataset and training the network
      val newAgent = (1 to noBootstrapIter).
        foldLeft(agentWithNewHistory)((agent, _) => {
          // prepares data for learning session
          val iter = agent.createDataSetIterator
          val newNet = agent.net.clone()
          newNet.fit(iter, noEpochIter)
          agent.copy(net = newNet)
        })
      val score = newAgent.net.score()
      newAgent.copy(stepCounter = 0)
    } else {
      agentWithNewHistory.copy(stepCounter = stepCounter + 1)
    }
  }

  override def score(feedback: Feedback): Double = feedback match {
    case Feedback(obs0, action, reward, obs1) =>
      val policy0 = policy(obs0)
      val valueMask0 = obs0.actions
      val policy1 = policy(obs1)
      val valueMask1 = obs1.actions
      val label = TDAgentUtils.bootstrapPolicy(policy0, valueMask0, policy1, valueMask1, action, reward, gamma, kappa, config)
      val d2 = label.squaredDistance(policy0)
      val nch = valueMask0.sumNumber().doubleValue()
      d2 / nch
  }

  /**
   * Returns inputs, labels, mask, noClearTrace data to fit the network from history
   */
  def createDataSetIterator: DataSetIterator = {
    // Computes the labels and mask dataset
    require(history.length >= 1)
    val data = history.data.map {
      case Feedback(obs0, action, reward, obs1) =>
        val policy0 = policy(obs0)
        val valueMask0 = obs0.actions
        val policy1 = policy(obs1)
        val valueMask1 = obs1.actions
        val label = TDAgentUtils.bootstrapPolicy(policy0, valueMask0, policy1, valueMask1, action, reward, gamma, kappa, config)
        Pair.of(obs0.signals, label)
    }
    val batchSize = data.length
    val iter = new INDArrayDataSetIterator(data, batchSize)
    iter
  }

  override def writeModel(file: String): TDBatchAgent = {
    ModelSerializer.writeModel(net, new File(file), false)
    this
  }

  override def reset: TDBatchAgent = this
}
