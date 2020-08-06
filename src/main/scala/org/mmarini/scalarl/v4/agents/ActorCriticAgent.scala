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

package org.mmarini.scalarl.v4.agents

import java.io.File

import monix.reactive.Observer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v4.Utils.clip
import org.mmarini.scalarl.v4.{Agent, Feedback, Observation}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * The agent generates action based on the signals from the environment and learns the correct behaviour trying to
 * maximize the average rewords
 *
 * @param actors        the actors
 * @param network       the  network
 * @param avg           the average reward
 * @param valueDecay    the value decay parameter
 * @param rewardDecay   the reward decay parameter
 * @param denormalize   the output denormalizer function
 * @param normalizer    the output normalizer function
 * @param planner       the model to run planning
 * @param agentObserver the agent event observer
 */
case class ActorCriticAgent(actors: Seq[Actor],
                            network: ComputationGraph,
                            avg: INDArray,
                            valueDecay: INDArray,
                            rewardDecay: INDArray,
                            denormalize: INDArray => INDArray,
                            normalizer: INDArray => INDArray,
                            planner: Option[Planner],
                            private val agentObserver: Observer[AgentEvent]) extends Agent {

  /**
   * Returns chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): INDArray = {
    val outputs = network.output(observation.signals)
    val actions = actors.map(_.chooseAction(outputs, random))
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
    val (agent: ActorCriticAgent, ar, score) = directLearn(feedback, random)
    val agent1 = agent.copy(avg = ar)
    val res = planner.map(mod => {
      // Model learn and planning fase
      val (agent2, model1) = mod.learn(feedback, agent1).plan(agent1, random)
      val agent3 = agent2.asInstanceOf[ActorCriticAgent].copy(planner = Some(model1))
      (agent3, score)
    }).getOrElse((agent1, score))
    res
  }

  /**
   * Returns the fit agent, the average reward and and the score
   * Optimizes the policy based on a single feedback
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  override def directLearn(feedback: Feedback, random: Random): (Agent, INDArray, INDArray) = {
    val map = processForTrain(feedback)
    //val newNet = network.clone()
    val labels = map("labels").asInstanceOf[Array[INDArray]]

    //    newNet.fit(Array(feedback.s0.signals), labels)
    network.fit(Array(feedback.s0.signals), labels)
    //    val newAgent = copy(network = newNet)
    val map1 = processForTrain(feedback)
    val event = AgentEvent(feedback, this, map, map1)
    agentObserver.onNext(event)
    val score = map("score").asInstanceOf[INDArray]
    val newAvg = map("newAverage").asInstanceOf[INDArray]
    (this, newAvg, score)
  }

  /**
   *
   * @param feedback
   * @return
   */
  def processForTrain(feedback: Feedback): Map[String, Any] = {
    val Feedback(s0, actions, reward, s1) = feedback
    val outputs0 = network.output(s0.signals)
    val v0 = v(outputs0)
    val outputs1 = network.output(s1.signals)
    val v1 = v(outputs1)

    //    val newv0 = (v1 + reward - avg) * valueDecay + (1 - valueDecay) * avg
    val target = v1.add(reward).subi(avg)
    val newv0 = target.mul(valueDecay).subi(valueDecay.sub(1).muli(avg))
    //val delta = target.sub(v0)
    val delta = newv0.sub(v0)
    //val newAvg = rewardDecay * avg + (1 - rewardDecay) * reward
    val newAvg = rewardDecay.mul(avg).subi(rewardDecay.sub(1).muli(reward))

    // Critic update
    val criticLabel = clip(normalizer(newv0), -1, 1, copy = false)

    val actorMap = actors.flatMap(_.computeLabels(outputs0, actions, delta)).toMap
    val actorLabels = actors.map(actor => actorMap(s"labels(${actor.dimension})").asInstanceOf[INDArray]).toArray

    val labels = criticLabel +: actorLabels

    val result = actorMap +
      ("outputs0" -> outputs0) +
      ("outputs1" -> outputs1) +
      ("avg" -> avg) +
      ("v0" -> v0) +
      ("v1" -> v1) +
      ("newAverage" -> newAvg) +
      ("v0*" -> newv0) +
      ("delta" -> delta) +
      ("score" -> pow(delta, 2)) +
      ("outputs0" -> outputs0) +
      ("outputs1" -> outputs1) +
      ("labels" -> labels)
    result
  }


  /**
   * Returns the score for a feedback
   *
   * @param feedback the feedback from the last step
   */
  override def score(feedback: Feedback): INDArray = {
    val result = processForTrain(feedback)("score").asInstanceOf[INDArray]
    result
  }

  /**
   * Writes the agent status to a path
   *
   * @param path the path to store the files of model
   * @return the agents
   */
  override def writeModel(path: File): Agent = {
    ModelSerializer.writeModel(network, new File(path, s"network.zip"), false)
    this
  }

  /**
   * Returns the estimation of state value
   *
   * @param outputs the network outputs
   */
  def v(outputs: Array[INDArray]): INDArray = denormalize(outputs(0))
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
    val newv0 = target.mul(valueDecay).subi(valueDecay.sub(1).muli(avg))
    //val delta = target.sub(v0)
    val delta = newv0.sub(v0)
    //val newAvg = rewardDecay * avg + (1 - rewardDecay) * reward
    val newAvg = rewardDecay.mul(avg).subi(rewardDecay.sub(1).muli(reward))
    (delta, newv0, newAvg)
  }
}
