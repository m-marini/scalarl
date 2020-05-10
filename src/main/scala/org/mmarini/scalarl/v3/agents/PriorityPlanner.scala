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

import io.circe.ACursor
import org.mmarini.scalarl.v3.{Agent, Feedback}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random

import scala.annotation.tailrec

/**
 * The priority planner learns the behavior of environment and simulate it for planning the agent policy
 *
 * @param stateKeyGen   the state key generator
 * @param actionsKeyGen the actions key generator
 * @param planningSteps the number of step to plan
 * @param model         the model
 * @param queue         the sweeping queue
 * @tparam KS the status key type
 * @tparam KA the actions key type
 */
case class PriorityPlanner[KS, KA](stateKeyGen: INDArray => KS,
                                   actionsKeyGen: INDArray => KA,
                                   planningSteps: Int,
                                   model: Model[(KS, KA), Feedback],
                                   queue: PriorityQueue[(KS, KA)]) extends Planner {

  /**
   * Returns the mode updated by a new feedback
   *
   * @param feedback the feedback
   */
  override def learn(feedback: Feedback, agent: Agent): Planner = {
    val skey = stateKeyGen(feedback.s0.signals)
    val akey = actionsKeyGen(feedback.actions)
    val key = (skey, akey)
    // Adds the feedback to the model
    val m1 = model + (key, feedback)
    // Compute the score for the feedback
    val score = agent.score(feedback).getDouble(0L)

    val newQueue1 = if (m1.data.size < model.data.size) {
      queue.keep(m1.data.keySet)
    } else {
      queue
    }
    // Adds the feedback to the updating queue
    val newQueue = newQueue1 + (key, score)
    val result = copy(queue = newQueue, model = m1)
    result
  }

  /**
   * Returns the agent fit by planning and updated model
   *
   * @param agent  the agent to fit
   * @param random the random generator
   */
  override def plan(agent: Agent, random: Random): (Agent, Planner) = {
    @tailrec
    def planLoop(ctx: (Agent, PriorityPlanner[KS, KA]), n: Int): (Agent, PriorityPlanner[KS, KA]) = if (n <= 0) {
      ctx
    } else {
      val (agent, planner) = ctx
      // get the feedback with higher score
      val (targetKey, newQueue) = planner.queue.dequeue()
      val planner1 = copy(queue = newQueue)
      val ctx3 = targetKey match {
        case None => ctx
        case Some(key) =>
          val ctx2 = model.get(key).map(feedback => {
            val (agent1, score) = agent.directLearn(feedback, random)
            val newQueue = planner1.queue + (key, score.getDouble(0L))
            val planner2 = planner1.copy(queue = newQueue).
              sweepBackward(feedback.s0.signals, agent1)
            (agent1, planner2)
          }).getOrElse(ctx)
          ctx2
      }
      planLoop(ctx3, n - 1)
    }

    planLoop((agent, this), planningSteps)
  }

  /**
   * Returns the updated planner by sweeping backward.
   * Sweep backward enqueuing status action bringing to target
   *
   * @param signals the target state signals
   * @param agent   the agent used to compute the score
   */
  def sweepBackward(signals: INDArray, agent: Agent): PriorityPlanner[KS, KA] = {
    // For each feedback bringing yo the target status
    val planner = predecessors(signals).foldLeft(this) {
      case (planner1, (key, feedback)) =>
        // enqueues the feedback with the score
        val score = agent.score(feedback).getDouble(0L)
        val newQueue = planner1.queue + (key, score)
        planner1.copy(queue = newQueue)
    }
    planner
  }

  /**
   * Returns the set of predecessor state, action key
   *
   * @param status the target status
   */
  def predecessors(status: INDArray): Map[(KS, KA), Feedback] = {
    val key = stateKeyGen(status)
    val result = model.filterValues(feedback => stateKeyGen(feedback.s1.signals).equals(key))
    result
  }
}

/** The object factory form [[PriorityPlanner]] */
object PriorityPlanner {
  /**
   * Returns the planner from json configuration
   *
   * @param conf the configuration
   */
  def fromJson(conf: ACursor): PriorityPlanner[INDArray, INDArray] = {
    val planningSteps = conf.get[Int]("planningSteps").toTry.get
    val model = Model.fromJson(conf.downField("model"))
    val queue = PriorityQueue.fromJson(conf)
    val stateKeyGen = INDArrayKeyGenerator.fromJson(conf.downField("stateKey"))
    val actionsKeyGen = INDArrayKeyGenerator.fromJson(conf.downField("actionsKey"))
    PriorityPlanner(
      stateKeyGen = stateKeyGen,
      actionsKeyGen = actionsKeyGen,
      planningSteps = planningSteps,
      model = model,
      queue = queue)
  }
}