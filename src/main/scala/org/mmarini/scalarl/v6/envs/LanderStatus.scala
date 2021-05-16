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

package org.mmarini.scalarl.v6.envs

import com.typesafe.scalalogging.LazyLogging
import org.mmarini.scalarl.v6._
import org.mmarini.scalarl.v6.envs.StatusCode._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms
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

  /** Return the observation of the current land status */
  override lazy val observation: Observation =
    INDArrayObservation(
      signals = baseSignals,
      time = time)

  /** Returns true if the status is final */
  lazy val isFinal: Boolean = status != Flying
  /**
   * Returns the direction of target
   * The direction is in the range of -Pi, Pi
   * 0 toward x axis
   * Pi/2 toward y axis
   * Pi backward x axis
   * -Pi/2 backward y axis
   */
  lazy val direction: INDArray = atan2(pos.getColumn(1).neg(), pos.getColumn(0).neg())
  /** Returns the distance from platform */
  lazy val distance: INDArray = pos.getColumns(0, 1).norm2()
  /**
   * Returns the speed direction
   * The direction is in the range of -Pi, Pi
   * 0 toward x axis
   * Pi/2 toward y axis
   * Pi backward x axis
   * -Pi/2 backward y axis
   */
  lazy val speedDirection: INDArray = atan2(speed.getColumn(1), speed.getColumn(0))
  /** Returns the horizontal speed */
  lazy val hSpeed: INDArray = speed.getColumns(0, 1).norm2()
  /** Returns the height from ground */
  lazy val height: INDArray = pos.getColumn(2)
  /** Returns the vertical speed */
  lazy val vSpeed: INDArray = speed.getColumn(2)

  /**
   * Returns the input base signals
   * The signals are composed by:
   *  - platform direction
   *  - speed direction
   *  - platform distance
   *  - height
   *  - horizontal speed
   *  - vertical speed
   */
  lazy val baseSignals: INDArray = {
    val signals = hstack(direction, speedDirection, distance, height, hSpeed, vSpeed)
    signals
  }
  /** Returns the status of lander */
  lazy val status: StatusCode.Value = {
    if (pos.getDouble(2L) <= 0) {
      // has touched ground
      val vh = hSpeed.getDouble(0L)
      val vz = vSpeed.getDouble(0L)
      val dist = distance.getDouble(0L)
      val isLandPosition = dist <= conf.landingRadius.getDouble(0L)
      if (vz < conf.landingSpeed.getDouble(1L)) {
        if (isLandPosition) {
          VCrashedOnPlatform
        } else {
          VCrashedOutOfPlatform
        }
      } else if (vh > conf.landingSpeed.getDouble(0L)) {
        if (isLandPosition) {
          HCrashedOnPlatform
        } else {
          HCrashedOutOfPlatform
        }
      } else if (isLandPosition) {
        Landed
      } else {
        LandedOutOfPlatform
      }
    } else if (greaterThanOrEqual(pos, conf.spaceRange.getRow(1)).sumNumber().intValue() > 0) {
      OutOfRange
    } else if (lessThanOrEqual(pos, conf.spaceRange.getRow(0)).sumNumber().intValue() > 0) {
      OutOfRange
    } else if (fuel.getDouble(0L) <= 0) {
      OutOfFuel
    } else {
      Flying
    }
  }
  /**
   * Returns the vector for reward in the order:
   * <ul>
   * <li>1</li>
   * <li>rho (-1, 1)</li>
   * <li>hDistance (m)</li>
   * <li>deltaHSpeed (m/s)</li>
   * <li>deltaVSpeed (m/s)</li>
   * </li>
   * </ul>
   */
  lazy val rewardVector: INDArray = {
    val v = hstack(hSpeed, vSpeed)
    val d1 = abs(v.sub(conf.v0))
    val d2 = d1.sub(conf.dv)
    val deltaV = Transforms.max(d2, 0.0)
    val hDistance = pos.getColumns(0, 1).norm2()
    val result = hstack(ones(1), rho, hDistance, pos.getColumn(2), deltaV)
    result
  }
  /** Returns the direction coefficient */
  lazy val rho: INDArray = {
    val dir = pos.getColumns(0, 1)
    val vDir = speed.getColumns(0, 1)
    val prod = dir.norm2().muli(vDir.norm2())
    val result = if (prod.getDouble(0L) > EPS_THRESHOLD) {
      dir.mmul(vDir.transpose()).negi().divi(prod)
    } else {
      zeros(1)
    }
    result
  }
  /** Returns the reward */
  lazy val reward: INDArray = conf.rewardFunctionTable(status)(this)
  /** Returns the action configuration */
  override val actionDimensions: Int = LanderConf.NumActors

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
  override def change(actions: INDArray, random: Random): (Env, INDArray) = status match {
    case Flying =>
      val newEnv = drive(actions)
      (newEnv, newEnv.reward)
    case _ =>
      (initial(random), zeros(1))
  }

  /**
   * Returns a new initial status
   *
   * @param random the random generator
   */
  private def initial(random: Random): LanderStatus = {
    val newEnv = copy(
      pos = conf.initial(random),
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
    val a = acceleration(actions).add(conf.gVector)
    val dv = a.mul(conf.dt)
    val dp = speed.mul(conf.dt)
    val p = pos.add(dp)
    val v = speed.add(dv)
    copy(pos = p, speed = v, fuel = fuel.sub(1), time = time.add(conf.dt))
  }

  /**
   * Returns the jet accelerations
   *
   * @param actions the actions
   */
  private def acceleration(actions: INDArray): INDArray = {
    val dir = actions.getColumn(0)
    val speedHVersor = hstack(cos(dir), sin(dir))
    val targetHSpeed = speedHVersor.mul(actions.getColumn(1))
    val targetSpeed = hstack(targetHSpeed, actions.getColumn(2))
    val targetJetAcc = targetSpeed.sub(speed).div(conf.dt).sub(conf.gVector)
    val jetAcc = Utils.clip(conf.jetAccRange)(targetJetAcc)
    jetAcc
  }

  /** Returns the number of signals */
  override def signalsSize: Int = LanderConf.NumDims
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
    pos = conf.initial(random),
    speed = zeros(3),
    time = zeros(1),
    fuel = conf.fuel)
}
