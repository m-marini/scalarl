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

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.Action
import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.Feedback
import org.mmarini.scalarl.Observation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

/**
 * The agent acting in the environment by QLearning T(0) algorithm.
 *
 *  Generates actions to change the status of environment basing on observation of the environment
 *  and the internal strategy policy.
 *
 *  Updates its strategy policy to optimize the return value (discount sum of rewards)
 *  and the observation of resulting environment
 */
case class TD0QAgent(
  net:     MultiLayerNetwork,
  random:  Random,
  epsilon: Double,
  gamma:   Double) extends QAgent {

  /**
   * Returns the q function with action value for an observation
   *
   * @param observationt the observation
   */
  override def q(observation: Observation): INDArray = {
    val out = net.feedForward(observation.signals)
    val q = out.get(out.size() - 1)
    q
  }

  /**
   * Chooses the action for an observation
   *
   * It return the new agent and the chosen action.
   * This actor apply a epsilon greedy policy choosing a random action with probability epsilon
   * and the action with highest action value with probability 1 - epsilon.
   *
   *  @param observation the observation of environment status
   */
  override def chooseAction(observation: Observation): (Agent, Action) = {
    val actions = observation.actions
    val action = if (random.nextDouble() < epsilon) {
      val validActions = (0 until actions.size(1).toInt).filter(i => actions.getInt(i) > 0)
      val action = validActions(random.nextInt(validActions.length))
      action
    } else {
      greedyAction(observation)
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
      val v0 = v(obs0)
      val v1 = if (endUp) 0.0 else v(obs1)
      val expected = reward + gamma * v1
      val err = expected - v0
      val q0 = q(obs0)
      q0.putScalar(action, expected)

      val newNet = net.clone()
      newNet.fit(obs0.signals, q0)
      (copy(net = newNet), err)
  }

  override def writeModel(file: String): Agent = {
    ModelSerializer.writeModel(net, file, true)
    this
  }

  override def reset: Agent = this
}
