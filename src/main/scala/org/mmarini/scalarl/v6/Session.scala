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

package org.mmarini.scalarl.v6

import com.typesafe.scalalogging.LazyLogging
import monix.reactive.Observable
import monix.reactive.subjects.PublishSubject
import org.mmarini.scalarl.v6.agents.ActorCriticAgent
import org.nd4j.linalg.api.rng.Random

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
class Session(env: => Env,
              agent: => Agent,
              epoch: Int,
              numSteps: Int) extends LazyLogging {

  private val stepsPub = PublishSubject[Step]()

  /** Returns the observable of steps */
  def steps: Observable[Step] = stepsPub

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
  def run(random: Random): (Env, Agent) = {
    logger.info("Running session with agent with {} inputs and {} actors ...",
      agent.asInstanceOf[ActorCriticAgent].network.layerInputSize(0),
//      env.signalsSize,
      env.actionDimensions
    )

    val env0 = env
    val obs0 = env0.observation
    val context0 = SessionContext(
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
  private def runSession(random: Random, context: SessionContext): SessionContext = {
    if (context.step >= numSteps) {
      context
    } else {
      val nextCtx = runStep(random, context)
      runSession(random, nextCtx)
    }
  }

  /**
   * Returns the context for a single step interaction.
   *
   * @param random  the random generator
   * @param context the session context
   */
  private def runStep(random: Random, context: SessionContext): SessionContext = {
    // unfold context data
    val SessionContext(step, env0, agent0, obs0) = context

    // Agent chooses the action
    val action = agent0.chooseAction(obs0, random)

    // Updates environment
    val (env1, reward) = env0.change(action, random)
    val obs1 = env1.observation
    val feedback = Feedback(obs0, action, reward, obs1)
    val (agent1, score) = agent0.fit(feedback, random)

    val ctx0 = context.copy(env = env1,
      agent = agent1,
      obs = obs1,
      step = step + 1)

    // Generate step event
    val stepInfo = Step(
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
 * @param step  the step counter
 * @param env   the environment
 * @param agent the agent
 * @param obs   the observable
 */
case class SessionContext(step: Int = 0,
                          env: Env,
                          agent: Agent,
                          obs: Observation)
