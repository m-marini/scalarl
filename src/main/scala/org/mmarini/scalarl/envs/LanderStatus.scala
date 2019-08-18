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
  hRange: Double,
  height: Double,
  random: Random) extends Env {

  override def actionConfig: ActionChannelConfig = LanderStatus.ActionChannels

  override def reset(): (Env, Observation) = {
    val x = random.nextDouble() * 2 * hRange - hRange
    val y = random.nextDouble() * 2 * hRange - hRange
    val pos = Nd4j.create(Array[Double](x, y, height))
    val newEnv = copy(
      pos = pos,
      speed = Nd4j.zeros(3))
    (newEnv, newEnv.observation)
  }

  override def step(action: ChannelAction): (Env, Observation, Reward) = {
    val newEnv = drive(action)
    val reward = if (newEnv.isOutOfRange) {
      LanderStatus.OutOfRangeReward
    } else if (newEnv.isCrush) {
      LanderStatus.CrashReward
    } else if (newEnv.isLanded) {
      LanderStatus.LandedReward
    } else {
      val dist = -newEnv.pos.mul(newEnv.pos).sumNumber().doubleValue()
      dist / 10000
    }
    (newEnv, newEnv.observation, reward)
  }

  /** Returns true if the shuttle is out of range */
  def isOutOfRange: Boolean = {
    abs(pos.getDouble(0L)) > LanderStatus.MaxHorizontal ||
      abs(pos.getDouble(1L)) > LanderStatus.MaxHorizontal ||
      pos.getDouble(2L) > LanderStatus.MaxHeight
  }

  /**
   * Return the observatin of the current land status
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

    val x = pos.getDouble(0L)
    val y = pos.getDouble(1L)

    val vx = speed.getDouble(0L)
    val vy = speed.getDouble(1L)
    val vz = speed.getDouble(2L)

    val posSignals: INDArray = pos.mmul(LanderStatus.SignalsFromPos)
    val speedSignals: INDArray = speed.mmul(LanderStatus.SignalsFromSpeed)

    val squaredPosSignals = posSignals.mul(posSignals)
    val squaredSpeedSignals = speedSignals.mul(speedSignals)

    val hPosDirSignals = Nd4j.create(Array[Double](
      if (x >= 0) 1 else 0,
      if (y >= 0) 1 else 0))
    val hSpeedDirSignals = Nd4j.create(Array[Double](
      if (vx >= 0) 1 else 0,
      if (vy >= 0) 1 else 0,
      if (vz >= 0) 1 else 0))

    val radius2 = x * x + y * y
    val noLandPosSignals = Nd4j.create(Array[Double](
      if (radius2 > LanderStatus.SquareRadius) 1 else 0))

    val vh2 = vx * vx + vy * vy

    val noLandVHSignals = Nd4j.create(Array[Double](
      if (vh2 > LanderStatus.SquareLandHorizontalSpeed) 1 else 0))
    val noLandVZSignals = Nd4j.create(Array[Double](
      if (vz < -LanderStatus.LandVerticalSpeed) 1 else 0,
      if (vz >= 0) 1 else 0))

    val signal = Nd4j.hstack(
      posSignals,
      speedSignals,
      squaredPosSignals,
      squaredSpeedSignals,
      hPosDirSignals,
      hSpeedDirSignals,
      noLandPosSignals,
      noLandVHSignals,
      noLandVZSignals)

    INDArrayObservation(
      signals = signal,
      actions = Nd4j.ones(15),
      endUp = isLanded || isCrush || isOutOfRange)
  }

  /** Returns true is the shuttle has touched the ground */
  private def hasTouchedGround: Boolean =
    pos.getDouble(2L) < 0

  /** Returns true if the shuttle is in land location */
  private def isLandPosition: Boolean = {
    val x = pos.getDouble(0L)
    val y = pos.getDouble(1L)
    val squareR = x * x + y * y
    val landPosition = squareR <= LanderStatus.SquareRadius &&
      hasTouchedGround
    landPosition
  }

  /** Returns true if the shuttle has land speed */
  private def isLandSpeed: Boolean = {
    val vx = speed.getDouble(0L)
    val vy = speed.getDouble(1L)
    val vz = speed.getDouble(2L)
    val squareHV = vx * vx + vy * vy
    val landSpeed = vz >= -LanderStatus.LandVerticalSpeed &&
      squareHV <= LanderStatus.SquareLandHorizontalSpeed
    landSpeed
  }

  /** Returns true if the shuttle has landed */
  def isLanded: Boolean = {
    val landed = isLandPosition && isLandSpeed
    landed
  }

  /** Returns true is the shuttle crush */
  def isCrush: Boolean = {
    val x = pos.getDouble(0L)
    val y = pos.getDouble(1L)
    val z = pos.getDouble(2L)
    val squareR = x * x + y * y

    val vx = speed.getDouble(0L)
    val vy = speed.getDouble(1L)
    val vz = speed.getDouble(2L)
    val squareHV = vx * vx + vy * vy

    val crush = hasTouchedGround && (
      !isLandPosition ||
      vz < -LanderStatus.LandVerticalSpeed ||
      squareHV > LanderStatus.SquareLandHorizontalSpeed ||
      !isLandPosition)

    crush
  }

  /** Returns the new status by driving with action */
  def drive(action: ChannelAction): LanderStatus = {
    val a = action.mmul(LanderStatus.AccelerationFromAction).addi(LanderStatus.GravityVector)
    val v = speed.add(a.mul(LanderStatus.Deltat))
    val p = pos.add(v.mul(LanderStatus.Deltat))
    copy(pos = p, speed = v)
  }
}

/** Factory for [[Maze]] instances */
object LanderStatus {
  val MaxHorizontal = 600.0
  val MaxHeight = 100.0
  val Radius = 10.0
  val LandVerticalSpeed = 4.0
  val LandHorizontalSpeed = 0.5
  val Deltat = 0.25
  val Gravity = 1.6
  val MaxHorizontalJet = 1.0
  val MaxVerticalJet = 3.2
  val NumDiscreteJet = 5

  val LandedReward = 100
  val CrashReward = -100
  val OutOfRangeReward = -100

  val ActionChannels = Array(NumDiscreteJet, NumDiscreteJet, NumDiscreteJet)
  val SquareRadius = Radius * Radius
  val SquareLandHorizontalSpeed = LandHorizontalSpeed * LandHorizontalSpeed

  val GravityVector = Nd4j.create(Array[Double](0, 0, -Gravity))
  val HorizontalJet = (0 until NumDiscreteJet).map(i => (i - 2) * MaxHorizontalJet / 2)
  val VerticalJet = (0 until NumDiscreteJet).map(i => i * MaxVerticalJet / (NumDiscreteJet - 1))
  val Filler = (1 to NumDiscreteJet).map(_ => 0.0)

  val AccelerationFromAction = Nd4j.create(Array(
    (HorizontalJet ++ Filler ++ Filler).toArray,
    (Filler ++ HorizontalJet ++ Filler).toArray,
    (Filler ++ Filler ++ VerticalJet).toArray)).transpose

  val SignalsFromPos = Nd4j.create(Array(
    Array[Double](1.0 / 600, 0, 0),
    Array[Double](0, 1.0 / 600, 0),
    Array[Double](0, 0, 1.0 / 100))).transpose()
  val SignalsFromSpeed = Nd4j.create(Array(
    Array[Double](1.0 / 24, 0, 0),
    Array[Double](0, 1.0 / 24, 0),
    Array[Double](0, 0, 1.0 / 12))).transpose()

  def apply(hRange: Double, height: Double, random: Random): LanderStatus = LanderStatus(
    pos = Nd4j.zeros(3),
    speed = Nd4j.zeros(3),
    height = height,
    hRange = hRange,
    random = random)

  def apply(pos: INDArray, speed: INDArray): LanderStatus = LanderStatus(
    pos = pos,
    speed = speed,
    height = MaxHeight,
    hRange = MaxHorizontal,
    random = Nd4j.getRandom())
}
