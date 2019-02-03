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
  noEpisode: Int,
  env0:      => Env,
  agent0:    => Agent,
  mode:      String                  = "human",
  close:     Boolean                 = false,
  sync:      Long                    = 0,
  onEpisode: Option[Session => Void] = None) {

  private var _env: Env = None.orNull
  private var _agent: Agent = None.orNull

  def env: Env = _env

  def agent: Agent = _agent

  /**
   * Runs the interactions for the number of episodes
   *
   *  Each episode is composed by the
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
    _env = env0
    _agent = agent0

    for { episode <- 1 to noEpisode } {
      val (env1, obs1) = _env.reset()
      val env2 = env1.render(mode, close)
      var _endUp = true
      var _obs = obs1
      _env = env2
      do {
        val obs0 = _obs
        val (agent1, action) = _agent.chooseAction(obs0)
        val (env1, obs1, reward, endUp, info) = _env.step(action)
        _env = env1.render(mode, close)
        if (sync > 0) Thread.sleep(sync)
        _agent = agent1.fit((obs0, action, reward, obs1, endUp, info))
        _obs = obs1
        _endUp = endUp
      } while (!_endUp)
      onEpisode.foreach(_(this))
    }
    this
  }
}
