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

package org.mmarini.scalarl.v1

import com.typesafe.scalalogging.LazyLogging
import monix.reactive.Observable
import monix.reactive.subjects.PublishSubject
import org.nd4j.linalg.api.rng.Random

/**
 * The session executes the interactions between the environment and the agent
 *
 * @constructor Creates a [[Session]]
 * @param env              the environment
 * @param agent            the agent
 * @param noSteps          the number of steps for the session
 * @param maxEpisodeLength the maximum number of steps per episode
 */
class Session(env: Env,
              agent: Agent,
              noSteps: Int,
              maxEpisodeLength: Long) extends LazyLogging {

  private val stepsPub = PublishSubject[Step]()
  private val episodesPub = PublishSubject[Episode]()

  /** Returns the observable of steps */
  def steps: Observable[Step] = stepsPub

  /** Returns the observable of episodes */
  def episodes: Observable[Episode] = episodesPub

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
    val env0 = env.reset(random)
    val obs0 = env0.observation
    val agent0 = agent.reset(random)
    val context0 = SessionContext(
      env = env0,
      agent = agent0,
      obs = obs0)

    val context = (1 to noSteps).foldLeft(context0)(runStep(random))

    stepsPub.onComplete()
    episodesPub.onComplete()

    (context.env, context.agent)
  }

  /**
   * Returns the context for a single step interation.
   *
   * @param random      the random generator
   * @param context     the sessin context
   * @param sessionStep the session count
   */
  private def runStep(random: Random)(context: SessionContext, sessionStep: Int): SessionContext = {
    // unfold context data
    val SessionContext(env0, agent0, obs0, step, episode, totalScore, returnValue) = context

    // Agent chooses the action
    val action = agent0.chooseAction(obs0, random)

    // Updates environment
    val (env1, reward) = env0.change(action, random)
    val obs1 = env1.observation
    val feedback = Feedback(obs0, action, reward, obs1)
    val (agent1, score) = agent0.fit(feedback, random)

    val returnValue1 = returnValue + reward
    val totalScore1 = totalScore + score * score
    val step1 = step + 1

    // Generate step event
    val stepInfo = Step(
      step = sessionStep,
      episode = episode,
      episodeStep = step1,
      feedback = feedback,
      env0 = env0,
      agent0 = agent0,
      env1 = env1,
      agent1 = agent1,
      session = this)
    stepsPub.onNext(stepInfo)

    if (obs1.endUp || step1 >= maxEpisodeLength) {
      // Episode completed

      // Emits episode event
      val episodeInfo = Episode(step = sessionStep,
        episode = episode,
        stepCount = step1,
        returnValue = returnValue1,
        totalScore = totalScore1,
        env = env1,
        agent = agent1,
        session = this)
      episodesPub.onNext(episodeInfo)

      // Reinitialize session for next episode
      val initialEnv = env1.reset(random)
      val initialObs = initialEnv.observation
      SessionContext(
        env = initialEnv,
        agent = agent1.reset(random),
        obs = initialObs,
        episode = episode + 1)
    } else {
      context.copy(
        env = env1,
        agent = agent1,
        obs = obs1,
        step = step1,
        totalScore = totalScore1,
        returnValue = returnValue1)
    }
  }
}

/**
 * A builder of [[Session]]
 */
object Session {

  /**
   * Returns a [[Session]]
   *
   * @param noSteps          the number of steps
   * @param env0             the initial environment
   * @param agent0           the initial agent
   * @param maxEpisodeLength the max number of steps per episode
   */
  def apply(noSteps: Int,
            env0: Env,
            agent0: Agent,
            maxEpisodeLength: Long): Session =
    new Session(
      noSteps = noSteps,
      env = env0,
      agent = agent0,
      maxEpisodeLength = maxEpisodeLength)
}

/**
 * The session context
 *
 * @param env         the environment
 * @param agent       the agent
 * @param obs         the observable
 * @param step        the step counter
 * @param episode     the episode counter
 * @param totalScore  the total loss
 * @param returnValue the return value
 */
case class SessionContext(env: Env,
                          agent: Agent,
                          obs: Observation,
                          step: Int = 0,
                          episode: Int = 0,
                          totalScore: Double = 0,
                          returnValue: Double = 0);
