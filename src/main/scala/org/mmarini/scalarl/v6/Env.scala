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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random

/**
 * The environment simulates the environment changing the status by action chosen by an agent
 * and notifying the reward to the agent.
 * Checks for end of episode by identifying the final states.
 */
trait Env {

  /**
   * Returns the next status and the reward.
   * Computes the next status of environment executing an action.
   *
   * @param actions the executing actions
   * @param random  the random generator
   * @return a n-tuple with:
   *         - the environment in the next status,
   *         - the reward for the actions,
   */
  def change(actions: INDArray, random: Random): (Env, INDArray)

  /** Returns the action space dimension */
  def actionDimensions: Int

  /** Returns the number of signals */
  def signalsSize: Int

  /** Returns the [[Observation]] for the environment */
  def observation: Observation
}
