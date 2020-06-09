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

package org.mmarini.scalarl.v4

import java.io.File

import org.deeplearning4j.nn.api.NeuralNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random

/**
 * The agent acting in the environment
 *
 * Generates actions to change the status of environment basing on observation of the environment
 * and the internal strategy policy.
 *
 * Updates its strategy policy to optimize the return value (discount sum of rewards)
 * and the observation of resulting environment
 */
trait Agent {

  /**
   * Returns chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  def chooseAction(observation: Observation, random: Random): INDArray

  /**
   * Returns the fit agent and the score
   * Optimizes the policy based on the feedback and model/planning
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  def fit(feedback: Feedback, random: Random): (Agent, INDArray)

  /**
   * Returns the fit network, the averageReward and, the score
   * Optimizes the policy based on the feedback for a single feedback
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  def directLearn(feedback: Feedback, random: Random): (Agent, INDArray, INDArray)


  /**
   * Returns the score for a feedback
   *
   * @param feedback the feedback from the last step
   */
  def score(feedback: Feedback): INDArray

  /**
   * Writes the agent status to a path
   *
   * @param path the path to store the files of model
   * @return the agents
   */
  def writeModel(path: File): Agent
}
