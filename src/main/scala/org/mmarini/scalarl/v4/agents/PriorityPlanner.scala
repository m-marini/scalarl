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

import io.circe.ACursor
import org.mmarini.scalarl.v4.{Agent, Feedback}
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
 * @tparam KS the status key type
 * @tparam KA the actions key type
 */
case class PriorityPlanner[KS, KA](stateKeyGen: INDArray => KS,
                                   actionsKeyGen: INDArray => KA,
                                   planningSteps: Int,
                                   threshold: Double,
                                   minModelSize: Int,
                                   maxModelSize: Int,
                                   model: Map[(KS, KA), (Feedback, Double)]) extends Planner {

  /**
   * Returns the mode updated by a new feedback
   *
   * @param feedback the feedback
   */
  override def learn(feedback: Feedback, agent: Agent): Planner = {
    val skey = stateKeyGen(feedback.s0.signals)
    val akey = actionsKeyGen(feedback.actions)
    val key = (skey, akey)
    // Compute the score for the feedback
    val score = agent.score(feedback).getDouble(0L)

    // Adds the feedback to the model
    val m1 = enqueue(key -> (feedback, score))

    // Adds the feedback to the updating queue
    val result = copy(model = m1)
    result
  }

  /**
   * Returns the model with a new entry
   *
   * @param entry the entry
   */
  def enqueue(entry: ((KS, KA), (Feedback, Double))): Map[(KS, KA), (Feedback, Double)] = {
    val model1 = model + entry
    val result = if (model1.size >= maxModelSize) {
      // shrink data removing the lower ones
      val ordering = Ordering.by((t: ((KS, KA), (Feedback, Double))) => t._2._2).reverse
      val sorted = model1.toSeq.sorted(ordering)
      val shrunk = sorted.take(minModelSize)
      shrunk.toMap
    } else {
      model1
    }
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
      planner.max() match {
        case None => ctx
        case Some((_, (_, score))) if score < threshold => ctx
        case Some((targetKey, (feedback, _))) =>
          val (agent1, score) = agent.directLearn(feedback, random)
          val newModel = planner.model + (targetKey -> (feedback, score.getDouble(0L)))
          val planner1 = planner.copy(model = newModel).
            sweepBackward(feedback.s0.signals, agent1)
          planLoop((agent1, planner1), n - 1)
      }
    }

    planLoop((agent, this), planningSteps)
  }

  /** Returns the entry with maximum priority */
  def max(): Option[((KS, KA), (Feedback, Double))] = if (model.isEmpty) {
    None
  } else {
    Some(model.maxBy {
      case (_, (_, score)) => score
    })
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
      case (planner1, (key, (feedback, _))) =>
        // enqueues the feedback with the score
        val score = agent.score(feedback).getDouble(0L)
        val newModel = planner1.model + (key -> (feedback, score))
        planner1.copy(model = newModel)
    }
    planner
  }

  /**
   * Returns the set of predecessor state, action key
   *
   * @param status the target status
   */
  def predecessors(status: INDArray): Map[(KS, KA), (Feedback, Double)] = {
    val key = stateKeyGen(status)
    val result = model.filter {
      case (_, (feedback, _)) => stateKeyGen(feedback.s1.signals) == key
    }
    result
  }
}

/** The object factory form [[PriorityPlanner]] */
object PriorityPlanner {

  /**
   * Returns the planner from json configuration
   *
   * @param conf      the configuration
   * @param noInputs  the number of inputs
   * @param noOutputs the number of outputs
   */
  def fromJson(conf: ACursor)(noInputs: Int, noOutputs: Int): PriorityPlanner[Array[Int], Array[Int]] = {
    val planningSteps = conf.get[Int]("planningSteps").toTry.get
    val minModelSize = conf.get[Int]("minModelSize").toTry.get
    val maxModelSize = conf.get[Int]("maxModelSize").toTry.get
    val threshold = conf.get[Double]("threshold").toTry.get
    val stateKeyGen = INDArrayKeyGenerator.fromJson(conf.downField("stateKey"))(noInputs)
    val actionsKeyGen = INDArrayKeyGenerator.fromJson(conf.downField("actionsKey"))(noOutputs)
    PriorityPlanner(
      stateKeyGen = stateKeyGen,
      actionsKeyGen = actionsKeyGen,
      planningSteps = planningSteps,
      minModelSize = minModelSize,
      maxModelSize = maxModelSize,
      threshold = threshold,
      model = Map())
  }
}