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
  noEpisode:        Int,
  maxEpisodeLength: Long) {

  private val episodeSubj: Subject[Episode] = Subject()
  private val stepSubj: Subject[Step] = Subject()

  def episodeObs: Observable[Episode] = episodeSubj
  def stepObs: Observable[Step] = stepSubj

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
  def run(): Session = {
    var currentEnv = env
    var currentAgent = agent
    for { episode <- 0 until noEpisode } {
      // Initialize before starting episode
      val (initialEnv, initialObs) = currentEnv.reset()
      currentEnv = initialEnv
      currentAgent = currentAgent.reset
      var obs = initialObs
      var endUp = true
      var step = 0
      var totalLoss = 0.0
      var returnValue = 0.0
      do {
        val env0 = currentEnv
        val agent0 = currentAgent
        val obs0 = obs

        val (agent_1, action) = agent0.chooseAction(obs0)
        val (env1, obs1, reward, endUp1) = env0.step(action)
        val (agent1, error) = agent_1.fit(Feedback(obs0, action, reward, obs1, endUp1))

        val stepInfo = Step(
          episode = episode,
          step = step,
          reward = reward,
          endUp = endUp1,
          action = action,
          beforeEnv = env0,
          beforeAgent = agent0,
          afterEnv = env1,
          afterAgent = agent1,
          session = this)
        stepSubj.onNext(stepInfo)

        step += 1
        returnValue = returnValue * agent1.gamma + reward
        totalLoss += error * error
        currentEnv = env1
        currentAgent = agent1
        obs = obs1
        endUp = endUp1
      } while (!endUp && step < maxEpisodeLength)
      val episodeInfo = Episode(
        episode = episode,
        stepCount = step,
        returnValue = returnValue,
        avgLoss = totalLoss / step,
        env = currentEnv,
        agent = currentAgent,
        session = this)
      episodeSubj.onNext(episodeInfo)
    }
    this
  }
}

object Session {
  /**
   * Returns a session
   */
  def apply(
    noEpisode:        Int,
    env0:             Env,
    agent0:           Agent,
    maxEpisodeLength: Long): Session =
    new Session(
      noEpisode = noEpisode,
      env = env0,
      agent = agent0,
      maxEpisodeLength = maxEpisodeLength)
}
