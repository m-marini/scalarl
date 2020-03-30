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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl._
import org.nd4j.linalg.api.rng.Random

/**
 * The agent acting in the environment by QLearning T(0) algorithm.
 *
 * Generates actions to change the status of environment basing on observation of the environment
 * and the internal strategy policy.
 *
 * Updates its strategy policy to optimize the return value (discount sum of rewards)
 * and the observation of resulting environment
 *
 * @param net                  the neural network
 * @param model                the environment model
 * @param random               the random generator
 * @param config               the action channel configuration
 * @param epsilon              the e-greedy parameter
 * @param gamma                the gamma reward discount
 * @param kappa                the Residual Advantage Learning parameter
 * @param kappaPlus            the reward bonus parameter for planning in model learning phase
 * @param planningStepsCounter the number of planing steps in model learning
 */
case class DynaQPlusAgent(net: MultiLayerNetwork,
                          model: AgentModel,
                          random: Random,
                          config: ActionChannelConfig,
                          epsilon: Double,
                          gamma: Double,
                          kappa: Double,
                          kappaPlus: Double,
                          planningStepsCounter: Int) extends TDAgent with LazyLogging {

  /**
   * Chooses the action for an observation
   *
   * It return the new agent and the chosen action.
   * This actor apply a epsilon greedy policy choosing a random actin with probability epsilon
   * and the action with highest action value with probability 1 - epsilon.
   *
   * @param observation the observation of environment status
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

  override def greedyAction(observation: Observation): ChannelAction =
    TDAgentUtils.actionAndStatusValuesFromPolicy(
      policy = policy(observation),
      valueMask = observation.actions,
      conf = config)._1

  override def fit(feedback: Feedback): Agent = {
    // Direct RL
    val (inputs, labels) = createData(feedback)
    val newNet = net.clone()
    newNet.fit(inputs, labels)
    // Model learning
    val newModel = model :+ feedback
    // Planning
    val learntNet = learnModel(newNet, newModel)
    val newAgent = copy(net = learntNet, model = newModel)
    newAgent
  }

  /**
   * Returns the trained network by planning with the model
   *
   * The implementation changes the input network as side effect of learinig process
   *
   * @param net   the network
   * @param model the environment model
   */
  private def learnModel(net: MultiLayerNetwork, model: AgentModel): MultiLayerNetwork = {
    for {_ <- 1 to planningStepsCounter} {
      val feedback = model.nextFeedback(random)
      val (inputs, labels) = createData(feedback.get)
      net.fit(inputs, labels)
    }
    net
  }

  /**
   * Returns the pair of input and labels to feed the neural network
   *
   * @param feedback the feedback
   */
  def createData(feedback: Feedback): (StatusValues, Policy) = feedback match {
    case Feedback(obs0, action, reward, obs1) =>
      val policy0 = policy(obs0)
      val valueMask0 = obs0.actions
      val policy1 = policy(obs1)
      val valueMask1 = obs1.actions
      val label: Policy = TDAgentUtils.bootstrapPolicy(policy0, valueMask0, policy1, valueMask1, action, reward, gamma, kappa, config)
      (obs0.signals, label)
  }

  override def policy(observation: Observation): Policy = if (observation.endUp) {
    TDAgentUtils.endStatePolicy(config)
  } else {
    net.output(observation.signals)
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

  override def writeModel(file: String): DynaQPlusAgent = {
    ModelSerializer.writeModel(net, new File(file), false)
    this
  }

  override def reset: DynaQPlusAgent = this
}
