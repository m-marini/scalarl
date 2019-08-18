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
case class TDABatchAgent(
  netProc:     NetworkProcessor,
  netData:     NetworkData,
  history:     AgentHistory,
  random:      Random,
  epsilon:     Double,
  gamma:       Double,
  lambda:      Double,
  kappa:       Double,
  noBatchIter: Int) extends TDAgent {

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
  override def fit(feedback: Feedback): (Agent, Double) = {
    val newHistory = history :+ feedback

    val (inputs, labels, mask, noClearTrace) = prepareData(newHistory)
    val n = inputs.shape()(0)
    val newNetData = (0 until noBatchIter).foldLeft(netData)((data, _) =>
      (0L until n).foldLeft(netData)((data, i) =>
        netProc.fit(data, inputs.getRow(i), labels.getRow(i), mask.getRow(i), noClearTrace.getRow(i))))
    val loss = newNetData("loss").getDouble(0L)
    (
      copy(
        netData = newNetData,
        history = newHistory),
      loss)
  }

  /**
   * Returns inputs, labels, mask, noClearTrace data to fit the network from history
   */
  def prepareData(history: AgentHistory): (INDArray, INDArray, INDArray, INDArray) = {
    val historyData = history.data
    val inputsAry = historyData.map(_.s0.signals).toArray
    val inputs = Nd4j.vstack(inputsAry: _*)

    val noClearTraceSeq = if (historyData.length > 1) {
      0.0 +: historyData.tail.map {
        case Feedback(obs0, action, _, _, _) =>
          val aStar = greedyAction(obs0)
          val noClearTrace = if (action == aStar) 1.0 else 0.0
          noClearTrace
      }
    } else {
      historyData.map {
        case Feedback(obs0, action, _, _, _) =>
          val aStar = greedyAction(obs0)
          val noClearTrace = if (action == aStar) 1.0 else 0.0
          noClearTrace
      }
    }
    val noClearTrace = Nd4j.create(noClearTraceSeq.toArray).transpose()

    val (labelsSeq, maskSeq) = (historyData.map {
      case Feedback(obs0, action, reward, obs1, endUp) =>
        val v0 = v(obs0)
        val v1 = if (endUp) 0 else v(obs1)

        val a0 = policy(obs0)
        a0.putScalar(action, v0 + (reward + gamma * v1 - v0) / kappa)

        val mask = Nd4j.zeros(a0.shape(): _*)
        mask.putScalar(action, 1)
        (a0, mask)
    }).unzip

    val labels = Nd4j.vstack(labelsSeq.toArray: _*)
    val mask = Nd4j.vstack(maskSeq.toArray: _*)
    (inputs, labels, mask, noClearTrace)
  }

  override def writeModel(file: String): TDABatchAgent = {
    //    TraceModelSerializer.writeModel(net, file)
    //    this
    ???
  }

  override def reset: TDABatchAgent = this
}
