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

package org.mmarini.scalarl.agents

import org.mmarini.scalarl.Action
import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.Observation
import org.nd4j.linalg.api.ndarray.INDArray

/**
 *
 */
trait StateValueFunction {
  /** Returns the estimated state value for an observation */
  def v(observation: Observation): Double
}

/**
 *
 */
trait PolicyFunction {
  /** Returns the estimated action values for an observation */
  def policy(observation: Observation): INDArray

  /** Returns the estimated action value for an observation and an action */
  def actionPolicy(observation: Observation, action: Action): Double = {
    require(observation.actions.getDouble(action.toLong) > 0.0)
    policy(observation).getDouble(action.toLong)
  }

}

/**
 *
 */
trait GreedyActionFunction {
  /** Returns the estimated greedy action */
  def greedyAction(observation: Observation): Action
}

/**
 * The agent acting in the environment
 *
 *  Generates actions to change the status of environment basing on observation of the environment
 *  and the internal strategy policy.
 *
 *  Updates its strategy policy to optimize the return value (discount sum of rewards)
 *  and the observation of resulting environment
 */
trait TDAgent extends Agent with StateValueFunction with GreedyActionFunction with PolicyFunction {

  /** Returns the estimated greedy action */
  def greedyAction(observation: Observation): Action =
    TDAgentUtils.maxIdxWithMask(policy(observation), observation.actions)
}

object TDAgentUtils {
  /**
   * Returns the index containing the max value of a by masking mask
   *
   * @param a the vector of values
   * @param mask the vector mask with valid value
   */
  def maxIdxWithMask(a: INDArray, mask: INDArray): Int = {
    var idx = -1

    for {
      i <- 0 until a.size(1).toInt
      if (mask.getInt(i) > 0)
      if (idx < 0 || a.getDouble(0L, i.toLong) > a.getDouble(0L, idx.toLong))
    } {
      idx = i
    }
    idx
  }

  /**
   * Returns the the max value of a by masking mask
   *
   * @param a the vector of values
   * @param mask the vector mask with valid value
   */
  def maxWithMask(a: INDArray, mask: INDArray): Double = a.getDouble(maxIdxWithMask(a, mask).toLong)
}
