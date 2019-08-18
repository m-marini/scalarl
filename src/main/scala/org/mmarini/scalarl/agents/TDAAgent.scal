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

/**
 * The agent acting in the environment by QLearning T(0) algorithm.
 *
 *  Generates actions to change the status of environment basing on observation of the environment
 *  and the internal strategy policy.
 *
 *  Updates its strategy policy to optimize the return value (discount sum of rewards)
 *  and the observation of resulting environment
 */
case class TDAAgent(
  netProc: NetworkProcessor,
  netData: NetworkData,
  random:  Random,
  epsilon: Double,
  gamma:   Double,
  lambda:  Double,
  kappa:   Double) extends TDAgent {

  /** Returns the estimated state value for an observation */
  def v(observation: Observation): Double =
    TDAgentUtils.maxWithMask(policy(observation), observation.actions)

  /**
   * Returns the q function with action value for an observation
   *
   * @param observationt the observation
   */
  override def policy(observation: Observation): INDArray =
    netProc.forward(netData, observation.signals)

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
      val validActions = (0 until actions.size(1).toInt).filter(i => actions.getInt(i) > 0)
      require(!validActions.isEmpty, s"actions=${actions}, validActions=${validActions}")
      val action = validActions(random.nextInt(validActions.length))
      action
    } else {
      // Greedy policy
      greedyAction(observation)
    }
    require(actions.getInt(action) > 0)
    (this, action)
  }

  /**
   * Updates the q function to fit the expected result.
   *
   *  @param feedback the [[Feedback]] from environment after a state transition
   */
  override def fit(feedback: Feedback): (Agent, Double) = feedback match {
    case Feedback(obs0, action, reward, obs1, endUp) =>
      val v0 = v(obs0)
      val v1 = if (endUp) 0 else v(obs1)

      val a0 = policy(obs0)
      a0.putScalar(action, v0 + (reward + gamma * v1 - v0) / kappa)

      val mask = Nd4j.zeros(a0.shape(): _*)
      mask.putScalar(action, 1)

      val aStar = greedyAction(obs0)
      val noClearTrace = if (action == aStar) Nd4j.ones(1) else Nd4j.zeros(1)
      val newNetData = netProc.fit(netData, obs0.signals, a0, mask, noClearTrace)
      val loss = newNetData("loss").getDouble(0L)

//      if (!new File("data.yaml").exists()) {
//        NetDataMaterializer.write("data.yaml", newNetData)
//        println("Dumped")
//      }
      
      (copy(netData = newNetData), loss)
  }

  override def writeModel(file: String): TDAAgent = {
    //    TraceModelSerializer.writeModel(net, file)
    //    this
    ???
  }

  override def reset: TDAAgent = this
}
