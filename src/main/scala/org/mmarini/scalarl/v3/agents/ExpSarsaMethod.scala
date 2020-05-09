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

package org.mmarini.scalarl.v3.agents

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v3.Utils._
import org.mmarini.scalarl.v3._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * The agent acting in the environment by QLearning T(0) algorithm.
 *
 * Generates actions to change the status of environment basing on observation of the environment
 * and the internal strategy policy.
 *
 * Updates its strategy policy to optimize the return value (discount sum of rewards)
 * and the observation of resulting environment
 *
 * @param dimension the dimension index
 * @param net       the neural network
 * @param noActions number of actions
 * @param avgReward the average reward
 * @param epsilon   epsilon greedy parameter
 * @param beta      average reward step parameter
 */
case class ExpSarsaMethod(dimension: Int,
                          net: MultiLayerNetwork,
                          noActions: Int,
                          avgReward: INDArray,
                          epsilon: INDArray,
                          beta: INDArray) extends LazyLogging {
  /**
   * Returns the new agent and the chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  def chooseAction(observation: Observation, random: Random): INDArray = {
    val q0 = q(observation)
    val pr = egreedy(q0, epsilon)
    val action = ones(1).muli(randomInt(pr)(random))
    action
  }

  /**
   * Returns the policy for an observation
   *
   * @param observation the observation
   */
  def q(observation: Observation): INDArray = net.output(observation.signals)

  /**
   * Returns the fit agent by optimizing its strategy policy and the score
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  def fit(feedback: Feedback, random: Random): (ExpSarsaMethod, INDArray) = {
    val newNet = net.clone()
    val (newAvg, score) = train(newNet, feedback, avgReward)
    //val postPlan = newNet.output(feedback.s0.signals)
    val newAgent = copy(net = newNet, avgReward = newAvg)
    (newAgent, score)
  }

  /**
   * Returns the new average reward and the score by training from feedback
   *
   * @param net      the neural network
   * @param feedback the feedback
   * @param avg      the average reward
   */
  private def train(net: MultiLayerNetwork, feedback: Feedback, avg: INDArray): (INDArray, INDArray) = {
    val (inputs, labels, newAvg, score) = createData(feedback, avg)
    // Train network
    net.fit(inputs, labels)
    (newAvg, score)
  }

  /**
   * Returns the score a feedback
   *
   * @param feedback the feedback from the last step
   */
  def score(feedback: Feedback): INDArray = createData(feedback, avgReward)._4

  /**
   * Returns signals, labels, new average, score
   *
   * @param feedback the feedback
   * @param avg      the average rewards
   * @return the input signals, the output label, the new advantage reward, the score
   */
  private def createData(feedback: Feedback, avg: INDArray): (INDArray, INDArray, INDArray, INDArray) = feedback match {
    case Feedback(obs0, actions, reward, obs1) =>
      val action = actions.getInt(dimension)
      // Computes state values
      val q0 = q(obs0)
      val q1 = q(obs1)
      val (labels, newAvg, score) = ExpSarsaMethod.createData(q0, q1, action, reward, beta, epsilon, avg)
      (obs0.signals, labels, newAvg, score)
  }

  /**
   * Writes the agent status to file
   *
   * @param path the path
   * @return the agents
   */
  def writeModel(path: File): ExpSarsaMethod = {
    ModelSerializer.writeModel(net, new File(path, s"network-$dimension.zip"), false)
    this
  }
}

/** The factory of [[ExpSarsaMethod]] */
object ExpSarsaMethod extends LazyLogging {
  /**
   * Returns output labels, new average, score
   *
   * @param q0      the initial action value
   * @param q1      the final action value
   * @param action  the action
   * @param reward  the reward
   * @param beta    the beta parameter
   * @param epsilon the epsilon parameter
   * @param avg     the average reward
   */
  def createData(q0: INDArray,
                 q1: INDArray,
                 action: Int,
                 reward: INDArray,
                 beta: INDArray,
                 epsilon: INDArray,
                 avg: INDArray): (INDArray, INDArray, INDArray) = {
    // Computes state values
    val v0 = Utils.v(q0, action)
    val v1 = vExp(q1, egreedy(q1, epsilon))

    // Compute new v0' = v1 - Rm + R and delta = v1 - Rm + R - v0
    val newv0 = v1.add(reward).subi(avg)
    val delta = newv0.sub(v0)

    // Update average rewards
    val newAvg = reward.mul(beta).addi(beta.sub(1).negi().muli(avg))
    //val newAvg = delta.mul(beta).add(avg)

    // Computes labels
    val labels = q0.dup()
    labels.put(action, newv0)
    val score = pow(delta, 2)
    (labels, newAvg, score)
  }
}