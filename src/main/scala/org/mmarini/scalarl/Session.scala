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

package org.mmarini.scalarl

import rx.lang.scala.Observable
import rx.lang.scala.Subject

/**
 * The session executes the interactions between the environment and the agent
 *
 *  @constructor Creates a [[Session]]
 *  @param numEpisode the number of episodes of the current session
 *  @param env the environment
 *  @param agent the agent
 *  @param mode the rendering mode of environment
 *  @param close true if closing rendering window
 *  @param sync the time interval between each step in millis
 */
class Session(
  env:              Env,
  agent:            Agent,
  noSteps:          Int,
  maxEpisodeLength: Long) {

  private val episodeSubj: Subject[Episode] = Subject()
  private val stepSubj: Subject[Step] = Subject()

  def episodeObs: Observable[Episode] = episodeSubj
  def stepObs: Observable[Step] = stepSubj

  /**
   *
   */
  private def runStep(context: SessionContext, sessionStep: Int): SessionContext = {
    val SessionContext(env0, agent0, obs0, step, episode, totalLoss, returnValue, discount) = context

    // Agent chooses the action
    val (agent_1, action) = agent0.chooseAction(obs0)
    // Updates environment
    val (env1, obs1, reward) = env0.step(action)

    // Updates agent
    val feedback = Feedback(obs0, action, reward, obs1)
    val agent1 = agent_1.fit(feedback)
    val error = agent1.score(feedback)
    val stepInfo = Step(
      step = sessionStep,
      episode = context.episode,
      episodeStep = context.step,
      reward = reward,
      endUp = obs1.endUp,
      action = action,
      beforeEnv = env0,
      beforeAgent = agent0,
      afterEnv = env1,
      afterAgent = agent1,
      session = this)
    stepSubj.onNext(stepInfo)

    val step1 = step + 1
    val returnValue1 = returnValue + reward * discount
    val totalLoss1 = totalLoss + error * error
    val discount1 = discount * agent.gamma

    if (obs1.endUp || step1 >= maxEpisodeLength) {
      // Episode completed

      // Emits episode event
      val episodeInfo = Episode(step = sessionStep, episode = episode, stepCount = step1, returnValue = returnValue1,
        avgLoss = if (step1 > 1) totalLoss1 / step else totalLoss1, env = env1, agent = agent1, session = this)
      episodeSubj.onNext(episodeInfo)

      // Reinitialize session for next episode
      val (initialEnv, initialObs) = env1.reset()
      SessionContext(
        env = initialEnv,
        agent = agent1.reset,
        obs = initialObs,
        episode = episode + 1)
    } else {
      context.copy(
        env = env1,
        agent = agent1,
        obs = obs1,
        step = step1,
        totalLoss = totalLoss1,
        returnValue = returnValue1,
        discount = discount1)
    }
  }

  /**
   * Runs the interactions for the number of episodes
   *
   *  Each episode is compo
   *  sed by the
   *  - reset of environment
   *  - render of the environment
   *  - a iteration of
   *    - choose of action by the agent
   *    - step the environment with the chosen action
   *    - render the environment
   *    - fit the agent
   *    - until detection of end of episode
   */
  def run(): (Env, Agent) = {
    val (initialEnv, initialObs) = env.reset()
    val sessionContext = SessionContext(
      env = initialEnv,
      agent = agent.reset,
      obs = initialObs)

    val x = (1 to noSteps).foldLeft(sessionContext)(runStep)

    (x.env, x.agent)
  }
}

/**
 *
 */
object Session {

  /**
   * Returns a session
   */
  def apply(
    noSteps:          Int,
    env0:             Env,
    agent0:           Agent,
    maxEpisodeLength: Long): Session =
    new Session(
      noSteps = noSteps,
      env = env0,
      agent = agent0,
      maxEpisodeLength = maxEpisodeLength)
}

/**
 *
 */
case class SessionContext(
  env:         Env,
  agent:       Agent,
  obs:         Observation,
  step:        Int         = 0,
  episode:     Int         = 0,
  totalLoss:   Double      = 0,
  returnValue: Double      = 0,
  discount:    Double      = 1);
