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

package org.mmarini.scalarl.v1.envs

import io.circe.ACursor
import org.mmarini.scalarl.v1._
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
 * @param maxAH               the maximum horizontal acceleration reactor
 * @param maxAZ               the maximum vertical acceleration reactor
 * @param landedReward        the reward when landed
 * @param crashReward         the reward when crashed
 * @param outOfRangeReward    the reward when out of range
 * @param outOfFuelReward     the rewaord when out of fuel
 * @param rewardDistanceScale the reward by distance
 * @param fuel                the initial available fuel
 */
class LanderConf(val dt: Double,
                 val h0Range: Double,
                 val z0: Double,
                 val zMax: Double,
                 val fuel: Int,
                 val landingRadius: Double,
                 val landingVH: Double,
                 val landingVZ: Double,
                 val g: Double,
                 val hRange: Double,
                 val maxAH: Double,
                 val maxAZ: Double,
                 val landedReward: Double,
                 val crashReward: Double,
                 val outOfRangeReward: Double,
                 val outOfFuelReward: Double,
                 val rewardDistanceScale: Double) {

  import LanderConf._

  private val squaredRadius = landingRadius * landingRadius
  private val squaredLandingVH = landingVH * landingVH
  private val jetScale = Nd4j.create(Array(maxAH * 2 / (NumDiscreteJet - 1),
    maxAH * 2 / (NumDiscreteJet - 1),
    maxAZ / (NumDiscreteJet - 1)))

  private val gVect = Nd4j.create(Array[Double](0, 0, -g))

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
   * Returns the reward from movement
   *
   * @param pos0 the shuttle position (x,y,z) before movement
   * @param pos1 the shuttle position (x,y,z) after movement
   */
  def rewardFromMovement(pos0: INDArray, pos1: INDArray): Double = {
    val dist0 = sqrt(pos0.mul(pos0).sumNumber().doubleValue())
    val dist1 = sqrt(pos1.mul(pos1).sumNumber().doubleValue())
    val reward = (dist0 - dist1) * rewardDistanceScale
    reward
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
   * Returns the new status by driving with action
   *
   * @param action the action
   * @param pos    the shuttle position (x,y,z)
   * @param speed  the speed the shuttle speed (x,y,z)
   */
  def drive(action: Action, pos: INDArray, speed: INDArray): (INDArray, INDArray) = {

    val a = acceleration(action).addi(gVect)
    val dv = a.mul(dt)
    val dp = speed.mul(dt)
    val pos1 = pos.add(dp)
    val speed1 = speed.add(dv)
    (pos1, speed1)
  }

  /**
   * Returns the acceleration
   *
   * @param action the action
   */
  private def acceleration(action: Action): INDArray = {
    val j0 = ActionConf.vector(action)
    val j1 = j0.add(JetOffset)
    val result = j1.muli(jetScale)
    result
  }
}

/** Factory for [[LanderConf]] instances */
object LanderConf {
  val NumDiscreteJet = 5
  val ActionConf: MultiDimensionAction = new MultiDimensionAction(NumDiscreteJet, NumDiscreteJet, NumDiscreteJet)
  val ValidActions: ActionMask = 0L until ActionConf.actions
  val JetOffset: QValues = Nd4j.create(Array(-(NumDiscreteJet - 1) / 2.0, -(NumDiscreteJet - 1) / 2.0, 0.0))

  /**
   *
   * @param conf the json configuration
   */
  def apply(conf: ACursor): LanderConf = new LanderConf(
    dt = conf.get[Double]("dt").right.get,
    h0Range = conf.get[Double]("h0Range").right.get,
    z0 = conf.get[Double]("z0").right.get,
    hRange = conf.get[Double]("hRange").right.get,
    zMax = conf.get[Double]("zMax").right.get,
    landingRadius = conf.get[Double]("landingRadius").right.get,
    landingVH = conf.get[Double]("landingVH").right.get,
    landingVZ = conf.get[Double]("landingVZ").right.get,
    g = conf.get[Double]("g").right.get,
    maxAH = conf.get[Double]("maxAH").right.get,
    maxAZ = conf.get[Double]("maxAZ").right.get,
    landedReward = conf.get[Double]("landedReward").right.get,
    crashReward = conf.get[Double]("crashReward").right.get,
    outOfRangeReward = conf.get[Double]("outOfRangeReward").right.get,
    outOfFuelReward = conf.get[Double]("outOfFuelReward").right.get,
    rewardDistanceScale = conf.get[Double]("rewardDistanceScale").right.get,
    fuel = conf.get[Int]("fuel").right.get)
}
