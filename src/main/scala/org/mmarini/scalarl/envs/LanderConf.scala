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

import org.mmarini.scalarl.ChannelAction
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

/**
 * The LanderConf with lander parameters
 */
case class LanderConf(
  h0Range:             Double,
  z0:                  Double,
  hRange:              Double,
  zMax:                Double,
  landingRadius:       Double,
  landingVH:           Double,
  landingVZ:           Double,
  dt:                  Double,
  g:                   Double,
  maxAH:               Double,
  maxAZ:               Double,
  vhSignalScale:       Double,
  vzSignalScale:       Double,
  landedReward:        Double,
  crashReward:         Double,
  outOfRangeReward:    Double,
  outOfFuelReward:     Double,
  rewardDistanceScale: Double,
  fuel:                Int) {

  import LanderConf._

  private val squaredRadius = landingRadius * landingRadius
  private val squaredLandingVH = landingVH * landingVH

  private val gVect = Nd4j.create(Array[Double](0, 0, -g))
  private val horizontalJet = (0 until NumDiscreteJet).map(i => (i - 2) * maxAH / 2)
  private val verticalJet = (0 until NumDiscreteJet).map(i => i * maxAZ / (NumDiscreteJet - 1))

  private val accelerationFromAction = Nd4j.create(Array(
    (horizontalJet ++ Filler ++ Filler).toArray,
    (Filler ++ horizontalJet ++ Filler).toArray,
    (Filler ++ Filler ++ verticalJet).toArray)).transpose

  private val signalsFromPos = Nd4j.create(Array(
    Array[Double](1.0 / hRange, 0, 0),
    Array[Double](0, 1.0 / hRange, 0),
    Array[Double](0, 0, 1.0 / zMax))).transpose()

  private val signalsFromSpeed = Nd4j.create(Array(
    Array[Double](1 / vhSignalScale, 0, 0),
    Array[Double](0, 1 / vhSignalScale, 0),
    Array[Double](0, 0, 1 / vzSignalScale))).transpose()

  /** Returns a random initial position */
  def initialPos(random: Random): INDArray = {
    val x = random.nextDouble() * 2 * h0Range - h0Range
    val y = random.nextDouble() * 2 * h0Range - h0Range
    val pos = Nd4j.create(Array[Double](x, y, z0))
    pos
  }

  /** Returns the reward from distance */
  def rewardFromPosition(pos: INDArray): Double = {
    val dist = -pos.mul(pos).sumNumber().doubleValue()
    val reward = dist * rewardDistanceScale * rewardDistanceScale
    reward
  }

  /** Returns true if the shuttle is out of range */
  def isOutOfRange(pos: INDArray): Boolean = {
    abs(pos.getDouble(0L)) > hRange ||
      abs(pos.getDouble(1L)) > hRange ||
      pos.getDouble(2L) > zMax
  }

  /** Returns true is the shuttle has touched the ground */
  private def hasTouchedGround(pos: INDArray): Boolean =
    pos.getDouble(2L) < 0

  /** Returns true if the shuttle is in land location */
  private def isLandPosition(pos: INDArray): Boolean = {
    val x = pos.getDouble(0L)
    val y = pos.getDouble(1L)
    val squareR = x * x + y * y
    val landPosition = squareR <= squaredRadius &&
      hasTouchedGround(pos)
    landPosition
  }

  /** Returns true if the shuttle has land speed */
  private def isLandSpeed(speed: INDArray): Boolean = {
    val vx = speed.getDouble(0L)
    val vy = speed.getDouble(1L)
    val vz = speed.getDouble(2L)
    val squareHV = vx * vx + vy * vy
    val landSpeed = vz >= -landingVZ &&
      squareHV <= squaredLandingVH
    landSpeed
  }

  /** Returns true if the shuttle has landed */
  def isLanded(pos: INDArray, speed: INDArray): Boolean = {
    val landed = isLandPosition(pos) && isLandSpeed(speed)
    landed
  }

  /** Returns true is the shuttle crush */
  def isCrashed(pos: INDArray, speed: INDArray): Boolean = {
    val x = pos.getDouble(0L)
    val y = pos.getDouble(1L)
    val z = pos.getDouble(2L)
    val squareR = x * x + y * y

    val vx = speed.getDouble(0L)
    val vy = speed.getDouble(1L)
    val vz = speed.getDouble(2L)
    val squareHV = vx * vx + vy * vy

    val crashed = hasTouchedGround(pos) && (
      !isLandPosition(pos) ||
      vz < -landingVZ ||
      squareHV > squaredLandingVH ||
      !isLandPosition(pos))

    crashed
  }

  /** Returns the new status by driving with action */
  def drive(action: ChannelAction, pos: INDArray, speed: INDArray): (INDArray, INDArray) = {
    val a = action.mmul(accelerationFromAction).addi(gVect)
    val v = speed.add(a.mul(dt))
    val p = pos.add(v.mul(dt))
    (p, v)
  }

  /**
   * Returns the input signals as a vector of
   *
   */
  def signals(pos: INDArray, speed: INDArray): INDArray = {
    val x = pos.getDouble(0L)
    val y = pos.getDouble(1L)

    val vx = speed.getDouble(0L)
    val vy = speed.getDouble(1L)
    val vz = speed.getDouble(2L)

    val posSignals: INDArray = pos.mmul(signalsFromPos)
    val speedSignals: INDArray = speed.mmul(signalsFromSpeed)

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
      if (radius2 > squaredRadius) 1 else 0))

    val vh2 = vx * vx + vy * vy

    val noLandVHSignals = Nd4j.create(Array[Double](
      if (vh2 > squaredLandingVH) 1 else 0))
    val noLandVZSignals = Nd4j.create(Array[Double](
      if (vz < -landingVZ) 1 else 0,
      if (vz >= 0) 1 else 0))

    val signals = Nd4j.hstack(
      posSignals,
      speedSignals,
      squaredPosSignals,
      squaredSpeedSignals,
      hPosDirSignals,
      hSpeedDirSignals,
      noLandPosSignals,
      noLandVHSignals,
      noLandVZSignals)
    signals
  }
}

/** Factory for [[LanderConf]] instances */
object LanderConf {
  val NumDiscreteJet = 5
  val ActionChannels = Array(NumDiscreteJet, NumDiscreteJet, NumDiscreteJet)
  val Filler = (1 to NumDiscreteJet).map(_ => 0.0)
  val ValidActions = Nd4j.ones(NumDiscreteJet * 3)
  val DefaultZ = 100.0
  val DefaultZMax = 150.0
  val DefaultH0Range = 500.0
  val DefaultHRange = 600.0
  val DefaultLandingRadius = 10.0
  val DefaultLandingVH = 0.5
  val DefaultLandingVZ = 4.0
  val DefaultDt = 0.25
  val DefaultG = 1.6
  val DefaultMaxAH = 1.0
  val DefaultMaxAZ = DefaultG * 2
  val DefaultVHSignalScale = 24.0
  val DefaultVZSignalScale = 12.0
  val DefaultLandedReward = 100.0
  val DefaultCrashReward = -100.0
  val DefaultOutOfRangeReward = -100.0
  val DefaultOutOfFuelReward = -100.0
  val DefaultRewardDistanceScale = 1.0 / 100.0
  val DefaultFuel = 1000

  def apply(): LanderConf = LanderConf(
    h0Range = DefaultH0Range,
    z0 = DefaultZ,
    hRange = DefaultHRange,
    zMax = DefaultZMax,
    landingRadius = DefaultLandingRadius,
    landingVH = DefaultLandingVH,
    landingVZ = DefaultLandingVZ,
    dt = DefaultDt,
    g = DefaultG,
    maxAH = DefaultMaxAH,
    maxAZ = DefaultMaxAZ,
    vhSignalScale = DefaultVHSignalScale,
    vzSignalScale = DefaultVZSignalScale,
    landedReward = DefaultLandedReward,
    crashReward = DefaultCrashReward,
    outOfRangeReward = DefaultOutOfRangeReward,
    outOfFuelReward = DefaultOutOfFuelReward,
    rewardDistanceScale = DefaultRewardDistanceScale,
    fuel = DefaultFuel)
}
