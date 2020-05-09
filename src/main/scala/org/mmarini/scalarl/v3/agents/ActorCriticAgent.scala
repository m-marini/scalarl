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

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v3.{Agent, Feedback, Observation}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms.pow

/**
 * The agent generates action based on the signals from the environment and learns the correct behaviour trying to
 * maximize the average rewords
 *
 * @param actors      the actors
 * @param critic      the critic network
 * @param avg         the average reward
 * @param valueDecay  the value decay parameter
 * @param rewardDecay the reward decay parameter
 * @param planner     the model to run planning
 */
case class ActorCriticAgent(actors: Array[Actor],
                            critic: MultiLayerNetwork,
                            avg: INDArray,
                            valueDecay: INDArray,
                            rewardDecay: INDArray,
                            planner: Option[Planner]) extends Agent {

  /**
   * Returns chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): INDArray = {
    val actions = actors.map(_.chooseAction(observation, random))
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
  override def fit(feedback: Feedback, random: Random): (Agent, INDArray) = {
    val result = directLearn(feedback, random)
    planner.map(mod => {
      // Model learn and planning fase
      val (agent1, score) = result
      val (agent2, model1) = mod.learn(feedback, agent1).plan(agent1, random)
      val agent3 = agent2.asInstanceOf[ActorCriticAgent].copy(planner = Some(model1))
      (agent3, score)
    }).getOrElse(result)
  }

  /**
   * Returns the fit agent and the score
   * Optimizes the policy based on a single feedback
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  override def directLearn(feedback: Feedback, random: Random): (Agent, INDArray) = {
    val Feedback(s0, _, reward, s1) = feedback
    val v0 = v(s0)
    val v1 = v(s1)
    val (delta, newv0, newAvg) = computeDelta(v0, v1, reward)

    // Critic update
    val criticLabel = newv0
    val newCritic = critic.clone()
    newCritic.fit(s0.signals, criticLabel)
    val score = pow(delta, 2)
    val newActors = actors.map(_.fit(feedback, delta, random))
    val newAgent = copy(actors = newActors, critic = newCritic, avg = newAvg)
    (newAgent, score)
  }

  /**
   *
   * @param v0     initail state value
   * @param v1     final state value
   * @param reward reward
   */
  def computeDelta(v0: INDArray, v1: INDArray, reward: INDArray): (INDArray, INDArray, INDArray) =
    ActorCriticAgent.computeDelta(v0, v1, reward, avg, valueDecay, rewardDecay)

  /**
   * Returns the value of a state
   *
   */
  def v(obs: Observation): INDArray = ActorCriticAgent.v(critic, obs)

  /**
   * Returns the score for a feedback
   *
   * @param feedback the feedback from the last step
   */
  override def score(feedback: Feedback): INDArray = {
    val Feedback(s0, _, reward, s1) = feedback
    val v0 = v(s0)
    val v1 = v(s1)
    val (delta, _, _) = ActorCriticAgent.computeDelta(v0, v1, reward, avg, valueDecay, rewardDecay)
    val score = pow(delta, 2)
    score
  }

  /**
   * Writes the agent status to a path
   *
   * @param path the path to store the files of model
   * @return the agents
   */
  override def writeModel(path: File): Agent = {
    ModelSerializer.writeModel(critic, new File(path, s"critic.zip"), false)
    actors.foreach(_.writeModel(path))
    this
  }
}

/**
 *
 */
object ActorCriticAgent {

  /**
   * Returns delta, v0', avg'
   *
   * @param v0          the initial state value
   * @param v1          the final state value
   * @param reward      the reward
   * @param avg         the average reward
   * @param valueDecay  the value decay
   * @param rewardDecay the reward decay
   */
  def computeDelta(v0: INDArray,
                   v1: INDArray,
                   reward: INDArray,
                   avg: INDArray,
                   valueDecay: INDArray,
                   rewardDecay: INDArray): (INDArray, INDArray, INDArray) = {
    //    val newv0 = (v1 + reward - avg) * valueDecay + (1 - valueDecay) * avg
    val target = v1.add(reward).subi(avg)
    val delta = target.sub(v0)
    val newv0 = target.mul(valueDecay).subi(valueDecay.sub(1).muli(avg))
    //val newAvg = rewardDecay * avg + (1 - rewardDecay) * reward
    val newAvg = rewardDecay.mul(avg).subi(rewardDecay.sub(1).muli(reward))
    (delta, newv0, newAvg)
  }

  /**
   * Returns the value of a state
   *
   * @param critic the critic
   * @param obs    the observation
   */
  def v(critic: MultiLayerNetwork, obs: Observation): INDArray =
    critic.output(obs.signals)
}