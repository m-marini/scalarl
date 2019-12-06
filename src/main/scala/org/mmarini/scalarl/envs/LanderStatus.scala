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

package org.mmarini.scalarl.envs

import scala.math.abs

import org.mmarini.scalarl.ActionChannelConfig
import org.mmarini.scalarl.ChannelAction
import org.mmarini.scalarl.Env
import org.mmarini.scalarl.INDArrayObservation
import org.mmarini.scalarl.Observation
import org.mmarini.scalarl.Reward
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

/**
 * The LanderStatus with position and speed
 *
 * @constructor create a LanderStatus with position and speed
 * @param pos the position
 * @param speed the speed
 */
case class LanderStatus(
  pos:    INDArray,
  speed:  INDArray,
  fuel:   Int,
  conf:   LanderConf,
  random: Random) extends Env {

  override def actionConfig: ActionChannelConfig = LanderConf.ActionChannels

  override def reset(): (Env, Observation) = {
    val newEnv = copy(
      pos = conf.initialPos(random),
      speed = Nd4j.zeros(3),
      fuel = conf.fuel)
    (newEnv, newEnv.observation)
  }

  def isOutOfFuel: Boolean = fuel <= 0

  override def step(action: ChannelAction): (Env, Observation, Reward) = {
    val newEnv = drive(action)
    val reward = if (newEnv.isOutOfRange) {
      conf.outOfRangeReward
    } else if (newEnv.isCrashed) {
      conf.crashReward
    } else if (newEnv.isLanded) {
      conf.landedReward
    } else if (newEnv.isOutOfFuel) {
      conf.outOfFuelReward
    } else {
      conf.rewardFromPosition(newEnv.pos)
    }
    (newEnv, newEnv.observation, reward)
  }

  /**
   * Return the observation of the current land status
   *
   * The signals are composed with
   *  - (0) 2 signals for the horizontal position in the range -1 : 1 (-600 : 600)
   *  - (2) 1 signal for the vertical position in the range 0 : 1 (0 : 100)
   *  - (3) 2 signal for horizontal speed in the range -1 : 1 (-24 : 24) sqrt(600 * 1)
   *  - (5) 1 signal for vertical speed in the range -1 : 1 (-12 : 12) sqrt(100 * 1.6)
   *  - (6) 3 signals for squared position 0 : 1
   *  - (9) 3 signals for squared speed 0 : 1
   *  - (12) 2 signals for horizontal position direction 0, 1
   *  - (14) 3 signals for speed direction 0, 1
   *  - (17) 1 signal for horizontal no landing position 0, 1
   *  - (18) 3 signals for no land speed 0, 1 (vh high, vz low, vz high)
   */
  lazy val observation: Observation = {

    INDArrayObservation(
      signals = conf.signals(pos, speed),
      actions = LanderConf.ValidActions,
      endUp = isLanded || isCrashed || isOutOfRange || isOutOfFuel)
  }

  /** Returns true if the shuttle is out of range */
  def isOutOfRange: Boolean = conf.isOutOfRange(pos)

  /** Returns true if the shuttle has landed */
  def isLanded: Boolean = conf.isLanded(pos, speed)

  /** Returns true is the shuttle crush */
  def isCrashed: Boolean = conf.isCrashed(pos, speed)

  /** Returns the new status by driving with action */
  def drive(action: ChannelAction): LanderStatus = {
    val (p, v) = conf.drive(action, pos, speed)
    copy(pos = p, speed = v, fuel = fuel - 1)
  }
}

/** Factory for [[LanderStatus]] instances */
object LanderStatus {

  def apply(conf: LanderConf, random: Random): LanderStatus = LanderStatus(
    pos = Nd4j.zeros(3),
    speed = Nd4j.zeros(3),
    conf = conf,
    fuel = conf.fuel,
    random = random)

  def apply(pos: INDArray, speed: INDArray): LanderStatus = {
    val conf = LanderConf()
    LanderStatus(
      pos = pos,
      speed = speed,
      conf = conf,
      fuel = conf.fuel,
      random = Nd4j.getRandom())
  }
}
