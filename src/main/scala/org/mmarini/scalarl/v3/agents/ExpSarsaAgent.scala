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
import org.mmarini.scalarl.v3.{Agent, Feedback, Observation}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._

/**
 * The agent generates action based on the signals from the environment and learns the correct behaviour trying to
 * maximize the average rewords
 *
 * @param actionAgents the agents
 */
case class ExpSarsaAgent(actionAgents: Array[ExpSarsaMethod]) extends Agent with LazyLogging {
  /**
   * Returns chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): INDArray = {
    val actions = actionAgents.map(_.chooseAction(observation, random))
    val result = hstack(actions: _*)
    result
  }


  /**
   * Returns the fit agent and the score
   * Optimizes the policy based on the feedback
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  override def fit(feedback: Feedback, random: Random): (ExpSarsaAgent, INDArray) = {
    val (agents, scores) = actionAgents.map(_.fit(feedback, random)).unzip
    val score = hstack(scores: _ *).sum()
    (copy(actionAgents = agents), score)
  }

  /**
   * Returns the score for a feedback
   *
   * @param feedback the feedback from the last step
   */
  override def score(feedback: Feedback): INDArray = {
    val score = hstack(actionAgents.map(_.score(feedback)): _*).sum()
    score
  }

  /**
   * Writes the agent status to a path
   *
   * @param path the path to store the files of model
   * @return the agents
   */
  override def writeModel(path: File): ExpSarsaAgent = {
    actionAgents.foreach(_.writeModel(path))
    this
  }
}
