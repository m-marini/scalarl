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

import io.circe.ACursor
import org.mmarini.scalarl.v4.Utils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

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
 * @param g                   the acceleration m / (s**2)
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
class LanderConf(val dt: INDArray,
                 val h0Range: INDArray,
                 val z0: INDArray,
                 val zMax: INDArray,
                 val fuel: INDArray,
                 val landingRadius: INDArray,
                 val landingVH: INDArray,
                 val landingVZ: INDArray,
                 val g: INDArray,
                 val hRange: INDArray,
                 val maxAH: INDArray,
                 val maxAZ: INDArray,
                 val landedReward: INDArray,
                 val crashReward: INDArray,
                 val outOfRangeReward: INDArray,
                 val outOfFuelReward: INDArray,
                 val rewardDistanceScale: INDArray) {

  import LanderConf._
  import StatusCode._

  private val minRange = hstack(hRange.neg(), hRange.neg(), zeros(1))
  private val maxRange = hstack(hRange, hRange, zMax)

  private val jetScale = hstack(maxAH, maxAH, maxAZ).muli(JetScale)
  private val gVect = hstack(zeros(2), g.neg())

  /**
   * Returns a random initial position
   *
   * @param random the random generator
   */
  def initialPos(random: Random): INDArray = {
    val plan = random.nextDouble(Array(1, 2)).muli(h0Range).muli(2).subi(h0Range)
    val pos = hstack(plan, z0)
    pos
  }

  /**
   * Returns the reward from movement
   *
   * @param pos0 the shuttle position (x,y,z) before movement
   * @param pos1 the shuttle position (x,y,z) after movement
   */
  def rewardFromMovement(pos0: INDArray, pos1: INDArray): INDArray = {
    val dist0 = pos0.norm1()
    val dist1 = pos1.norm1()
    val reward = dist0.sub(dist1).muli(rewardDistanceScale)
    reward
  }

  /**
   * Returns the reward from distance
   *
   * @param pos the shuttle position (x,y,z)
   */
  def rewardFromPosition(pos: INDArray): INDArray = {
    val dist = pos.norm1().negi()
    val reward = dist.mul(rewardDistanceScale)
    reward
  }

  /**
   * Returns the status of lander
   *
   * @param pos   the position
   * @param speed the speed
   * @param fuel  the fuel
   */
  def status1(pos: INDArray, speed: INDArray, fuel: INDArray): StatusCode.Value = {
    if (pos.getDouble(2L) <= 0) {
      // has touched ground
      val vhSquare = speed.getColumns(0, 1).norm2()
      if (not(lessThanOrEqual(vhSquare, landingVH)).getInt(0) > 0) {
        Crashed
      } else if (not(greaterThanOrEqual(speed.getColumn(2), landingVZ.neg())).getInt(0) > 0) {
        Crashed
      } else {
        val landPosition = lessThanOrEqual(pos.getColumns(0, 1).norm2(), landingRadius).getInt(0) > 0
        if (landPosition) {
          Landed
        } else {
          OutOfPlatform
        }
      }
    } else if (greaterThanOrEqual(pos, maxRange).sumNumber().intValue() > 0) {
      OutOfRange
    } else if (lessThanOrEqual(pos, minRange).sumNumber().intValue() > 0) {
      OutOfRange
    } else if (fuel.getDouble(0L) <= 0) {
      OutOfFuel
    } else {
      Flying
    }
  }

  def status(pos: INDArray, speed: INDArray, fuel: INDArray): StatusCode.Value = {
    if (pos.getDouble(2L) <= 0) {
      val landPosition = lessThanOrEqual(pos.getColumns(0, 1).norm2(), landingRadius).getInt(0) > 0
      if (landPosition) {
        Landed
      } else {
        OutOfPlatform
      }
    } else if (greaterThanOrEqual(pos, maxRange).sumNumber().intValue() > 0) {
      OutOfRange
    } else if (lessThanOrEqual(pos, minRange).sumNumber().intValue() > 0) {
      OutOfRange
    } else if (fuel.getDouble(0L) <= 0) {
      OutOfFuel
    } else {
      Flying
    }
  }

  /**
   * Returns the new status by driving with action
   *
   * @param action the action
   * @param pos    the shuttle position (x,y,z)
   * @param speed  the speed the shuttle speed (x,y,z)
   */
  def drive(action: INDArray, pos: INDArray, speed: INDArray): (INDArray, INDArray) = {
    val a = acceleration(action).addi(gVect)
    val dv = a.mul(dt)
    val dp = speed.mul(dt)
    val pos1 = pos.add(dp)
    val speed1 = speed.add(dv)
    (pos1, speed1)
  }

  /**
   * Returns the jet accelerations
   *
   * @param actions the actions
   */
  private def acceleration(actions: INDArray): INDArray = {
    val acc = Utils.clip(actions, 0, NumDiscreteJet - 1).add(JetOffset).muli(jetScale)
    acc
  }
}

/** Factory for [[LanderConf]] instances */
object LanderConf {
  val NumDiscreteJet = 5
  val JetScale: INDArray = create(Array(0.5, 0.5, 0.25))
  val JetOffset: INDArray = create(Array(-(NumDiscreteJet - 1) / 2.0, -(NumDiscreteJet - 1) / 2.0, 0.0))

  /**
   *
   * @param conf the json configuration
   */
  def fromJson(conf: ACursor): LanderConf = new LanderConf(
    dt = conf.get[Double]("dt").toTry.map(ones(1).mul(_)).get,
    h0Range = conf.get[Double]("h0Range").toTry.map(ones(1).mul(_)).get,
    z0 = conf.get[Double]("z0").toTry.map(ones(1).mul(_)).get,
    hRange = conf.get[Double]("hRange").toTry.map(ones(1).mul(_)).get,
    zMax = conf.get[Double]("zMax").toTry.map(ones(1).mul(_)).get,
    landingRadius = conf.get[Double]("landingRadius").toTry.map(ones(1).mul(_)).get,
    landingVH = conf.get[Double]("landingVH").toTry.map(ones(1).mul(_)).get,
    landingVZ = conf.get[Double]("landingVZ").toTry.map(ones(1).mul(_)).get,
    g = conf.get[Double]("g").toTry.map(ones(1).mul(_)).get,
    maxAH = conf.get[Double]("maxAH").toTry.map(ones(1).mul(_)).get,
    maxAZ = conf.get[Double]("maxAZ").toTry.map(ones(1).mul(_)).get,
    landedReward = conf.get[Double]("landedReward").toTry.map(ones(1).mul(_)).get,
    crashReward = conf.get[Double]("crashReward").toTry.map(ones(1).mul(_)).get,
    outOfRangeReward = conf.get[Double]("outOfRangeReward").toTry.map(ones(1).mul(_)).get,
    outOfFuelReward = conf.get[Double]("outOfFuelReward").toTry.map(ones(1).mul(_)).get,
    rewardDistanceScale = conf.get[Double]("rewardDistanceScale").toTry.map(ones(1).mul(_)).get,
    fuel = conf.get[Int]("fuel").toTry.map(ones(1).mul(_)).get)
}

/**
 * Lander status
 */
object StatusCode extends Enumeration {
  val Flying, Landed, OutOfPlatform, Crashed, OutOfRange, OutOfFuel = Value
}