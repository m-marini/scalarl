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

package org.mmarini.scalarl.v2

import com.typesafe.scalalogging.LazyLogging
import monix.reactive.Observable
import monix.reactive.subjects.PublishSubject
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

import scala.annotation.tailrec

/**
 * The session executes the interactions between the environment and the agent
 *
 * @constructor Creates a [[Session]]
 * @param env      the environment
 * @param agent    the agent
 * @param epoch    the number of epochs
 * @param numSteps the number of steps for the session
 */
class SessionContinuousAction(env: => EnvContinuousAction,
                              agent: => AgentContinuousAction,
                              epoch: Int,
                              numSteps: Int) extends LazyLogging {

  private val stepsPub = PublishSubject[StepContinuousAction]()

  /** Returns the observable of steps */
  def steps: Observable[StepContinuousAction] = stepsPub

  /**
   * Runs the interactions for a number of episodes
   *
   * Each episode is composed by:
   *  - reset of environment
   *  - render of the environment
   *  - a iteration of
   *    - choose of action by the agent
   *    - step the environment with the chosen action
   *    - render the environment
   *    - fit the agent
   *    - until detection of end of episode
   *
   * @param random the random generator
   * @return the environment and agent after the interaction session
   */
  def run(random: Random): (EnvContinuousAction, AgentContinuousAction) = {
    val env0 = env
    val obs0 = env0.observation
    val context0 = SessionContextContinuousAction(
      env = env0,
      agent = agent,
      obs = obs0)

    val context = runSession(random, context0)

    stepsPub.onComplete()
    (context.env, context.agent)
  }

  /**
   *
   * @param random  the random generator
   * @param context the session context
   * @return
   */
  @tailrec
  private def runSession(random: Random, context: SessionContextContinuousAction): SessionContextContinuousAction = {
    if (context.step >= numSteps) {
      context
    } else {
      val nextCtx = runStep(random, context)
      runSession(random, nextCtx)
    }
  }

  /**
   * Returns the context for a single step interation.
   *
   * @param random  the random generator
   * @param context the sessin context
   */
  private def runStep(random: Random, context: SessionContextContinuousAction): SessionContextContinuousAction = {
    // unfold context data
    val SessionContextContinuousAction(step, env0, agent0, obs0, totalScore, returnValue) = context

    // Agent chooses the action
    val action = agent0.chooseAction(obs0, random)

    // Updates environment
    val (env1, reward) = env0.change(action, random)
    val obs1 = env1.observation
    val feedback = FeedbackContinuousAction(obs0, action, reward, obs1)
    val (agent1, score) = agent0.fit(feedback, random)

    val returnValue1 = returnValue.add(reward)
    val totalScore1 = score.mul(score).add(totalScore)

    val ctx0 = context.copy(env = env1,
      agent = agent1,
      obs = obs1,
      step = step + 1,
      totalScore = totalScore1,
      returnValue = returnValue1)

    // Generate step event
    val stepInfo = StepContinuousAction(
      epoch = epoch,
      step = step,
      feedback = feedback,
      env0 = env0,
      agent0 = agent0,
      env1 = env1,
      agent1 = agent1,
      score,
      session = this,
      context = ctx0)

    stepsPub.onNext(stepInfo)


    ctx0
  }
}

/**
 * The session context
 *
 * @param step        the step counter
 * @param env         the environment
 * @param agent       the agent
 * @param obs         the observable
 * @param totalScore  the total loss
 * @param returnValue the return value
 */
case class SessionContextContinuousAction(step: Int = 0,
                                          env: EnvContinuousAction,
                                          agent: AgentContinuousAction,
                                          obs: Observation,
                                          totalScore: INDArray = Nd4j.zeros(1),
                                          returnValue: INDArray = Nd4j.zeros(1))
