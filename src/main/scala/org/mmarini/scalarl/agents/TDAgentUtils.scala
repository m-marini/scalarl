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

import org.mmarini.scalarl._
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

object TDAgentUtils {

  /** Returns the end state policy for a channel configuration */
  def endStatePolicy(conf: ActionChannelConfig): Policy =
    Nd4j.zeros(numOutputs(conf))

  /** Returns the number of output for a given action channel configuration */
  def numOutputs(config: ActionChannelConfig): Int =
    config.sum

  /** Returns a random action for a valueMask, channel configuration */
  def randomAction(valueMask: ActionMask, conf: ActionChannelConfig)(random: Random): ChannelAction = {
    val action = Nd4j.zeros(valueMask.shape(): _*)
    for {
      (start, to) <- sliceIdxFromChannels(conf)
    } {
      val indexes = for {
        i <- start to to
        if (valueMask.getInt(i) != 0)
      } yield i
      val idx = random.nextInt(indexes.length)
      val value = indexes(idx)
      action.putScalar(value, 1)
    }
    action
  }

  /** Returns the start,end indexes for each channel */
  def sliceIdxFromChannels(conf: ActionChannelConfig): Array[(Int, Int)] = {
    val (indexes, _) = conf.foldLeft((Array[(Int, Int)](), 0)) {
      case ((indexes, start), length) =>
        val next = start + length
        val newIndexes = indexes :+ (start, next - 1)
        (newIndexes, next)
    }
    indexes
  }

  /**
   * Returns the bootstrap policy for a transaction feedback
   *
   * Given `policy0` the policy for a state s0 with the relative action mask
   * and `policy1` the policy for the resulting state after the application of
   * `action` action with `reword` received from environment,
   * then the fit policy is the `policy0` with the action value changed for the any `action`
   * channel selected and computed applying the formula
   * ```
   * A' = v0 + (R * gamma * v1 -v0) / kappa
   * ```
   * `v` is the highest value for each the action channel for the given policy
   */
  def bootstrapPolicy(policy0: Policy, valueMask0: ActionMask,
                      policy1: Policy, valueMask1: ActionMask,
                      action: ChannelAction,
                      reward: Reward,
                      gamma: Double,
                      kappa: Double,
                      conf: ActionChannelConfig): Policy = {
    val (_, v0) = actionAndStatusValuesFromPolicy(policy0, valueMask0, conf)
    val (_, v1) = actionAndStatusValuesFromPolicy(policy1, valueMask1, conf)
    // a' = v0 + (R + gamma * v1 - v0) / kappa
    val p1 = v0.add(v1.mul(gamma).subi(v0).addi(reward).divi(kappa)).muli(action)
    val fit = policy0.mul(action.sub(1).neg()).add(p1)
    fit
  }

  /**
   * Returns the action for a policy, a value mask and a channel configuration
   */
  def actionAndStatusValuesFromPolicy(
                                       policy: Policy,
                                       valueMask: ActionMask,
                                       conf: ActionChannelConfig): (ChannelAction, StatusValues) = {

    val action = Nd4j.zeros(policy.shape(): _*)
    val values = action.dup()
    for {
      (start, to) <- sliceIdxFromChannels(conf)
    } {
      var max = Double.NegativeInfinity
      var idx = -1L
      for {
        i <- start to to
        if (valueMask.getInt(i) != 0)
        v = policy.getDouble(i.toLong)
        if (v > max)
      } {
        max = v
        idx = i
      }
      if (idx >= 0) {
        action.putScalar(idx, 1)
      }
      for {
        i <- start to to
      } {
        values.putScalar(i, max)
      }
    }
    (action, values)
  }
}
