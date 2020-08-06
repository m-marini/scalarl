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
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * The LanderStatus with position and speed
 *
 * @constructor create a LanderStatus with position and speed
 * @param pos   the position of lander
 * @param time  the time instant
 * @param speed the speed of lander
 * @param fuel  the fuel stock
 * @param conf  the configuration
 */
case class LanderStatus(pos: INDArray,
                        speed: INDArray,
                        time: INDArray,
                        fuel: INDArray,
                        conf: LanderConf) extends Env with LazyLogging {
  /**
   * Return the observation of the current land status
   */
  override lazy val observation: Observation = INDArrayObservation(
    signals = conf.signals(this),
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
        case VCrashedOnPlatform =>
          conf.vCrashedOnPlatformReward
        case HCrashedOnPlatform =>
          conf.hCrashedOnPlatformReward
        case Landed =>
          conf.landedReward
        case LandedOutOfPlatform =>
          conf.landedOutOfPlatformReward
        case HCrashedOutOfPlatform =>
          conf.hCrashedOutOfPlatformReward
        case VCrashedOutOfPlatform =>
          conf.vCrashedOutOfPlatformReward
        case OutOfFuel =>
          conf.outOfFuelReward
        case Flying =>
          conf.flyingReward.
            add(conf.rewardFromDirection(pos, speed)).
            addi(conf.rewardFromVSpeed(speed)).
            addi(conf.rewardFromHSpeed(speed))
      }
      (newEnv, reward)
    case _ =>
      (initial(random), zeros(1))
  }

  /**
   * Returns the direction of target
   * The direction is in the range of -Pi, Pi
   * 0 toward x axis
   * Pi/2 toward y axis
   * Pi backward x axis
   * -Pi/2 backward y axis
   */
  def direction: INDArray = atan2(pos.getColumn(1).neg(), pos.getColumn(0).neg())

  /** Returns the distance from platform */
  def distance: INDArray = pos.getColumns(0, 1).norm2()

  /**
   * Returns the speed direction
   * The direction is in the range of -Pi, Pi
   * 0 toward x axis
   * Pi/2 toward y axis
   * Pi backward x axis
   * -Pi/2 backward y axis
   */
  def speedDirection: INDArray = atan2(speed.getColumn(1), speed.getColumn(0))

  /** Returns the horizontal speed */
  def hSpeed: INDArray = speed.getColumns(0, 1).norm2()

  /** Returns the height from ground */
  def height: INDArray = pos.getColumn(2)

  /** Returns the vertical speed */
  def vSpeed: INDArray = speed.getColumn(2)

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
  override def signalsSize: Int = conf.noSignals

  /** Returns the action configuration */
  override val actionDimensions: Int = LanderConf.NumActors
}

/** Factory for [[LanderStatus]] instances */
object LanderStatus {

  /**
   * Returns the initial environment
   *
   * @param conf   the configuration
   * @param random the random generator
   */
  def apply(conf: LanderConf,
            random: Random): LanderStatus = LanderStatus(
    conf = conf,
    pos = conf.initialPos(random),
    speed = zeros(3),
    time = zeros(1),
    fuel = conf.fuel)
}
