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

package org.mmarini.scalarl.ts.envs

import io.circe.ACursor
import org.mmarini.scalarl.ts.{ChannelAction, DiscreteActionChannels}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

import scala.math._

/**
 * The LanderConf with lander parameters
 *
 * @param dt                  the simulation interval
 * @param h0Range             the horizontal range of initial area m
 * @param z0                  the the initial height m
 * @param zMax                m the maximum admitted height m
 * @param landingRadius       the radius of landing area m
 * @param landingVH           the maximum horizontal landing speed m/s
 * @param landingVZ           the maximum vertical landing speed m/s
 * @param g                   the acceleration m/(s**2)
 * @param hRange              the maximum admitted horizontal range
 * @param zRange              the signal z range
 * @param vhRange             the horizontal speed scale
 * @param vzRange             the vertical speed scale
 * @param maxAH               the maximum horizontal acceleration reactor
 * @param maxAZ               the maximum vertical acceleration reactor
 * @param landedReward        the reward when landed
 * @param crashReward         the reward when crashed
 * @param outOfRangeReward    the reward when out of range
 * @param outOfFuelReward     the rewaord when out of fuel
 * @param rewardDistanceScale the reward by distance
 * @param fuel                the initial available fuel
 */
case class LanderConf(dt: Double,
                      h0Range: Double,
                      z0: Double,
                      zMax: Double,
                      fuel: Int,
                      landingRadius: Double,
                      landingVH: Double,
                      landingVZ: Double,
                      g: Double,
                      hRange: Double,
                      zRange: Double,
                      vhRange: Double,
                      vzRange: Double,
                      maxAH: Double,
                      maxAZ: Double,
                      z1: Double,
                      landedReward: Double,
                      crashReward: Double,
                      outOfRangeReward: Double,
                      outOfFuelReward: Double,
                      rewardDistanceScale: Double) {

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

  private val signalsFromPos = {
    val result = Nd4j.zeros(3L, 3L)
    result.put(0, 0, 1 / hRange)
    result.put(1, 1, 1 / hRange)
    result.put(2, 2, 1 / zRange)
    result
  }

  private val signalsFromSpeed = {
    val result = Nd4j.zeros(3L, 3L)
    result.put(0, 0, 1 / vhRange)
    result.put(1, 1, 1 / vhRange)
    result.put(2, 2, 1 / vzRange)
    result
  }

  /**
   * Returns a random initial position
   *
   * @param random the random generator
   */
  def initialPos(random: Random): INDArray = {
    val x = random.nextDouble() * 2 * h0Range - h0Range
    val y = random.nextDouble() * 2 * h0Range - h0Range
    val pos = Nd4j.create(Array[Double](x, y, z0))
    pos
  }

  /**
   * Returns the reward from distance
   *
   * @param pos the shuttle position (x,y,z)
   */
  def rewardFromPosition(pos: INDArray): Double = {
    val dist = -pos.mul(pos).sumNumber().doubleValue()
    val reward = dist * rewardDistanceScale
    reward
  }

  /**
   * Returns true if the shuttle is out of range
   *
   * @param pos the shuttle position (x,y,z)
   */
  def isOutOfRange(pos: INDArray): Boolean = {
    abs(pos.getDouble(0L)) > hRange ||
      abs(pos.getDouble(1L)) > hRange ||
      pos.getDouble(2L) > zMax
  }

  /**
   * Returns true if the shuttle has landed
   *
   * @param pos   the shuttle position (x,y,z)
   * @param speed the speed the shuttle speed (x,y,z)
   */
  def isLanded(pos: INDArray, speed: INDArray): Boolean = {
    val landed = isLandPosition(pos) && isLandSpeed(speed)
    landed
  }

  /**
   * Returns true if the shuttle has land speed
   *
   * @param speed the shuttle speed (x,y,z)
   */
  private def isLandSpeed(speed: INDArray): Boolean = {
    val vx = speed.getDouble(0L)
    val vy = speed.getDouble(1L)
    val vz = speed.getDouble(2L)
    val squareHV = vx * vx + vy * vy
    val landSpeed = vz >= -landingVZ &&
      squareHV <= squaredLandingVH
    landSpeed
  }

  /**
   * Returns true if the shuttle is in land location
   *
   * @param pos the shuttle position (x,y,z)
   */
  private def isLandPosition(pos: INDArray): Boolean = {
    val x = pos.getDouble(0L)
    val y = pos.getDouble(1L)
    val squareR = x * x + y * y
    val landPosition = squareR <= squaredRadius &&
      hasTouchedGround(pos)
    landPosition
  }

  /**
   * Returns true is the shuttle has touched the ground
   *
   * @param pos the shuttle position (x,y,z)
   */
  private def hasTouchedGround(pos: INDArray): Boolean =
    pos.getDouble(2L) < 0

  /**
   * Returns true is the shuttle crush
   *
   * @param pos   the shuttle position (x,y,z)
   * @param speed the speed the shuttle speed (x,y,z)
   */
  def isCrashed(pos: INDArray, speed: INDArray): Boolean = {

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

  /**
   * Returns the new status by driving with action
   *
   * @param actions the action
   * @param pos     the shuttle position (x,y,z)
   * @param speed   the speed the shuttle speed (x,y,z)
   */
  def drive(actions: ChannelAction, pos: INDArray, speed: INDArray): (ChannelAction, ChannelAction) = {
    val a = actions.mmul(accelerationFromAction).addi(gVect)
    val dv = a.mul(dt)
    val dp = speed.mul(dt)
    val pos1 = pos.add(dp)
    val speed1 = speed.add(dv)
    (pos1, speed1)
  }

  /**
   * Returns the input signals as a vector
   *
   * @param pos   the shuttle position (x,y,z)
   * @param speed the speed the shuttle speed (x,y,z)
   */
  def signals(pos: INDArray, speed: INDArray): INDArray = {
    val x = pos.getDouble(0L)
    val y = pos.getDouble(1L)
    val z = pos.getDouble(2L)

    val vx = speed.getDouble(0L)
    val vy = speed.getDouble(1L)
    val vz = speed.getDouble(2L)

    val posSignals: INDArray = pos.mmul(signalsFromPos)
    val speedSignals: INDArray = speed.mmul(signalsFromSpeed)

    // Computes the position feature index
    val radius2 = x * x + y * y
    val xIdx = if (x > 0) 1 else 0
    val yIdx = if (y > 0) 1 else 0
    val zIdx = if (z >= z1) 1 else 0
    val posIdx = if (radius2 >= squaredRadius) {
      yIdx + 2 * xIdx + 4 * zIdx
    } else {
      8 + zIdx
    }
    val posFeatures = Nd4j.zeros(10)
    posFeatures.put(0, posIdx, 1)

    // Computes the horizontal speed feature index
    val vh = vx * vx + vy * vy
    val vxIdx = if (vx > 0) 1 else 0
    val vyIdx = if (vy > 0) 1 else 0
    val v02 = v0 * v0
    val vhhIdx = if (vh >= v02) 1 else 0
    val vhIdx = vyIdx + 2 * vxIdx + 4 * vhhIdx
    val vhFeatures = Nd4j.zeros(8)
    vhFeatures.put(0, vhIdx, 1)

    // Computes the vertical speed features
    val vzFeatures = Nd4j.zeros(4)
    if (vz >= 0) {
      vzFeatures.put(0, 0, 1)
    } else if (vz > v1) {
      vzFeatures.put(0, 1, 1)
    } else if (vz > v2) {
      vzFeatures.put(0, 2, 1)
    } else {
      vzFeatures.put(0, 3, 1)
    }

    val signals = Nd4j.hstack(
      posSignals,
      speedSignals,
      posFeatures,
      vhFeatures,
      vzFeatures)
    signals
  }

  /** Returns the feature safe horizontal speed */
  private def v0 = landingVH / 2

  /** Returns the feature safe vertical speed */
  private def v1 = -landingVZ / 2

  /** Returns the feature max vertical speed */
  private def v2 = -landingVZ
}

/** Factory for [[LanderConf]] instances */
object LanderConf {
  val NumDiscreteJet = 5
  val ActionChannels: DiscreteActionChannels = DiscreteActionChannels(Array(NumDiscreteJet, NumDiscreteJet, NumDiscreteJet))
  val Filler: Seq[Double] = (1 to NumDiscreteJet).map(_ => 0.0)
  val ValidActions: ChannelAction = Nd4j.ones(NumDiscreteJet * 3)

  /**
   *
   * @param conf the json configuration
   */
  def apply(conf: ACursor): LanderConf = {
    LanderConf(
      h0Range = conf.get[Double]("h0Range").right.get,
      z0 = conf.get[Double]("z0").right.get,
      z1 = conf.get[Double]("z1").right.get,
      hRange = conf.get[Double]("hRange").right.get,
      zRange = conf.get[Double]("zRange").right.get,
      zMax = conf.get[Double]("zMax").right.get,
      landingRadius = conf.get[Double]("landingRadius").right.get,
      landingVH = conf.get[Double]("landingVH").right.get,
      landingVZ = conf.get[Double]("landingVZ").right.get,
      dt = conf.get[Double]("dt").right.get,
      g = conf.get[Double]("g").right.get,
      maxAH = conf.get[Double]("maxAH").right.get,
      maxAZ = conf.get[Double]("maxAZ").right.get,
      vhRange = conf.get[Double]("vhRange").right.get,
      vzRange = conf.get[Double]("vzRange").right.get,
      landedReward = conf.get[Double]("landedReward").right.get,
      crashReward = conf.get[Double]("crashReward").right.get,
      outOfRangeReward = conf.get[Double]("outOfRangeReward").right.get,
      outOfFuelReward = conf.get[Double]("outOfFuelReward").right.get,
      rewardDistanceScale = conf.get[Double]("rewardDistanceScale").right.get,
      fuel = conf.get[Int]("fuel").right.get)
  }
}
