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

package org.mmarini.scalarl.v4.envs

import com.typesafe.scalalogging.LazyLogging
import org.mmarini.scalarl.v4._
import org.mmarini.scalarl.v4.envs.StatusCode._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._

/**
 * The LanderStatus with position and speed
 *
 * @constructor create a LanderStatus with position and speed
 * @param encoder      the status coder
 * @param pos          the position
 * @param time         the time instant
 * @param speed        the speed
 * @param fuel         the fuel stock
 * @param conf         the configuration
 * @param actionConfig the actions configuration
 */
case class LanderStatus(encoder: LanderEncoder,
                        pos: INDArray,
                        speed: INDArray,
                        time: INDArray,
                        fuel: INDArray,
                        conf: LanderConf,
                        actionConfig: Seq[ActionConfig]) extends Env with LazyLogging {
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
  override lazy val observation: Observation = INDArrayObservation(
    signals = encoder.signals(this),
    time = time)

  /** Returns true if the status is final */
  def isFinal: Boolean = status != Flying

  /** Returns the status code */
  def status: StatusCode.Value = conf.status(pos, speed, fuel)

  /**
   * Returns the next status and the reward.
   * Computes the next status of environment executing an action.
   *
   * @param actions the executing actions
   * @param random  the random generator
   * @return a n-uple with:
   *         - the environment in the next status,
   *         - the reward for the actions,
   */
  override def change(actions: INDArray, random: Random): (Env, INDArray) = status match {
    case Flying =>
      val newEnv = drive(actions)
      val reward = newEnv.status match {
        case OutOfRange =>
          conf.outOfRangeReward
        case VCrash =>
          conf.vCrashReward
        case HCrash =>
          conf.hCrashReward
        case Landed =>
          conf.landedReward
        case OutOfPlatform =>
          conf.outOfPlatformReward
        case OutOfFuel =>
          conf.outOfFuelReward
        case Flying =>
          conf.rewardFromMovement(pos, newEnv.pos).
            add(conf.flyingReward).
            add(conf.rewardFromDirection(pos, speed)).
            add(conf.rewardFromSpeed(speed))
      }
      (newEnv, reward)
    case _ =>
      (initial(random), zeros(1))
  }

  /**
   *
   * @param random the random generator
   */
  private def initial(random: Random): LanderStatus = {
    val newEnv = copy(
      pos = conf.initialPos(random),
      speed = zeros(3),
      fuel = conf.fuel,
      time = time.add(conf.dt))
    newEnv
  }

  /**
   * Returns the new status by driving with actions
   *
   * @param actions the actions
   */
  def drive(actions: INDArray): LanderStatus = {
    val (p, v) = conf.drive(actions, pos, speed)
    copy(pos = p, speed = v, fuel = fuel.sub(1), time = time.add(conf.dt))
  }

  /** Returns the number of signals */
  override def signalsSize: Int = encoder.noSignals
}

/** Factory for [[LanderStatus]] instances */
object LanderStatus {

  /**
   * Returns the initial environment
   *
   * @param conf         the configuration
   * @param encoder      the encoder
   * @param actionConfig the action configuration
   * @param random       the random generator
   */
  def apply(conf: LanderConf,
            encoder: LanderEncoder,
            actionConfig: Seq[ActionConfig],
            random: Random): LanderStatus = LanderStatus(
    conf = conf,
    encoder = encoder,
    pos = conf.initialPos(random),
    speed = zeros(3),
    time = zeros(1),
    fuel = conf.fuel,
    actionConfig = actionConfig)
}
