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
import org.mmarini.scalarl.v4.Utils._
import org.mmarini.scalarl.v4.envs.Configuration._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms._

import scala.math.Pi
import scala.util.Try

/**
 *
 * @param dt                          the simulation interval
 * @param initialLocationTrans        the transformation for initial location
 * @param fuel                        the initial available fuel
 * @param g                           the acceleration m / (s**2)
 * @param spaceRange                  the range of positions
 * @param landingRadius               the radius of landing area m
 * @param landingSpeed                the landing speed range (horizontal, vertical) m/s
 * @param optimalSpeed                the optimal speed m/s
 * @param jetAccRange                 the acceleration reactor range
 * @param landedReward                the reward when landed
 * @param landedOutOfPlatformReward   the reward when landed out of platform
 * @param hCrashedOnPlatformReward    the reward when horizontal crash on platform
 * @param vCrashedOnPlatformReward    the reward when verticaly crasj on platform
 * @param outOfRangeReward            the reward when out of range
 * @param outOfFuelReward             the rewaord when out of fuel
 * @param hCrashedOutOfPlatformReward the reward when horizontal crash out of platform
 * @param vCrashedOutOfPlatformReward the reward when vertical crash out of platform
 * @param flyingReward                the reward when flying
 * @param directionReward             the reward by direction
 * @param hSpeedReward                the reward for horizontal speed
 * @param vSpeedReward                the reward for vertical speed
 * @param encoder                     the input signal encoder function
 */
class LanderConf(val dt: INDArray,
                 val initialLocationTrans: INDArray => INDArray,
                 val spaceRange: INDArray,
                 val fuel: INDArray,
                 val g: INDArray,
                 val landingRadius: INDArray,
                 val landingSpeed: INDArray,
                 val optimalSpeed: INDArray,
                 val jetAccRange: INDArray,
                 val landedReward: INDArray,
                 val landedOutOfPlatformReward: INDArray,
                 val hCrashedOnPlatformReward: INDArray,
                 val vCrashedOnPlatformReward: INDArray,
                 val outOfRangeReward: INDArray,
                 val outOfFuelReward: INDArray,
                 val hCrashedOutOfPlatformReward: INDArray,
                 val vCrashedOutOfPlatformReward: INDArray,
                 val flyingReward: INDArray,
                 val directionReward: INDArray,
                 val hSpeedReward: INDArray,
                 val vSpeedReward: INDArray,
                 val encoder: LanderEncoder) {

  import StatusCode._

  private val EPS = 1e-6
  private val gVect = hstack(zeros(2), g.neg())
  private val ActionRange = create(Array(
    Array(-Pi, 0.0, -landingSpeed.getDouble(1L)),
    Array(Pi, landingSpeed.getDouble(0L), landingSpeed.getDouble(1L))
  ))

  /**
   * Returns a random initial position
   *
   * @param random the random generator
   */
  def initialPos(random: Random): INDArray = {
    val pos = initialLocationTrans(random.nextDouble(Array(1, 3)))
    pos
  }

  /**
   * Returns the reward from direction
   *
   * @param pos   the shuttle position (x,y,z)
   * @param speed the shuttle speed (x,y,z)
   */
  def rewardFromDirection(pos: INDArray, speed: INDArray): INDArray = {
    val posxy = pos.getColumns(0, 1).neg()
    val speedxy = speed.getColumns(0, 1)
    val dirNorm = posxy.norm2().mul(speedxy.norm2())
    if (dirNorm.getDouble(0L) > EPS) {
      val prod = posxy.mmul(speedxy.transpose())
      val cos = prod.divi(dirNorm)
      val reward = cos.mul(directionReward)
      reward
    } else {
      directionReward.neg()
    }
  }

  /**
   * Returns the reward from horizontal speed
   *
   * @param speed the speed
   * @return
   */
  def rewardFromHSpeed(speed: INDArray): INDArray = {
    val hSpeed = speed.getColumns(0, 1).norm2()
    val dh = hSpeed.sub(optimalSpeed.getColumn(0))
    val reward = Transforms.max(dh, 0.0).muli(hSpeedReward)
    reward
  }

  /**
   * Returns the reward from vertical speed
   *
   * @param speed the speed
   * @return
   */
  def rewardFromVSpeed(speed: INDArray): INDArray = {
    val vSpeed = speed.getColumn(2)
    val dv = vSpeed.sub(optimalSpeed.getColumn(1)).norm2()
    val reward = dv.mul(vSpeedReward)
    reward
  }

  /**
   * Returns the status of lander
   *
   * @param pos   the position
   * @param speed the speed
   * @param fuel  the fuel
   */
  def status(pos: INDArray, speed: INDArray, fuel: INDArray): StatusCode.Value = {
    if (pos.getDouble(2L) <= 0) {
      // has touched ground
      val vh = speed.getColumns(0, 1).norm2()
      val landPosition = lessThanOrEqual(pos.getColumns(0, 1).norm2(), landingRadius).getInt(0) > 0
      if (speed.getDouble(2L) < landingSpeed.getDouble(1L)) {
        if (landPosition) {
          VCrashedOnPlatform
        } else {
          VCrashedOutOfPlatform
        }
      } else if (vh.getDouble(0L) > landingSpeed.getDouble(0L)) {
        if (landPosition) {
          HCrashedOnPlatform
        } else {
          HCrashedOutOfPlatform
        }
      } else if (landPosition) {
        Landed
      } else {
        LandedOutOfPlatform
      }
    } else if (greaterThanOrEqual(pos, spaceRange.getRow(1)).sumNumber().intValue() > 0) {
      OutOfRange
    } else if (lessThanOrEqual(pos, spaceRange.getRow(0)).sumNumber().intValue() > 0) {
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
    val a = acceleration(action, speed).add(gVect)
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
   * @param speed   the current speed
   */
  private def acceleration(actions: INDArray, speed: INDArray): INDArray = {
    val dir = actions.getColumn(0)
    val speedHVersor = hstack(cos(dir), sin(dir))
    val targetHSpeed = speedHVersor.mul(actions.getColumn(1))
    val targetSpeed = hstack(targetHSpeed, actions.getColumn(2))
    val targetJetAcc = targetSpeed.sub(speed).div(dt).sub(gVect)
    val jetAcc = Utils.clip(targetJetAcc, jetAccRange, copy = true)
    jetAcc
  }

  /**
   * Returns the input base signals
   */
  def signals(status: LanderStatus): INDArray = {
    encoder.signals(baseSignals(status))
  }

  /**
   * Returns the input base signals
   * The signals are in range -1, 1 and is composed by:
   *  - platform direction
   *  - speed direction
   *  - platfrom distance
   *  - height
   *  - horizontal speed
   *  - vertical speed
   *
   * @param status the status
   */
  def baseSignals(status: LanderStatus): INDArray = {
    val signals = hstack(status.direction, status.speedDirection, status.distance, status.height, status.hSpeed, status.vSpeed)
    signals
  }

  def noSignals: Int = encoder.noSignals
}

/** Factory for [[LanderConf]] instances */
object LanderConf {
  val NumActors = 3
  val NumDims = 6
  val NumHDirections = 8
  val NumHJet = 3
  val NumDiscreteJet = 5
  val JetScale: INDArray = create(Array(0.5, 0.5, 0.25))
  val JetOffset: INDArray = create(Array(-(NumDiscreteJet - 1) / 2.0, -(NumDiscreteJet - 1) / 2.0, 0.0))
  val DiscreteChoices: Array[Int] = Array(NumHDirections, NumHJet, NumDiscreteJet)

  /**
   * Returns the LanderConf from json
   *
   * @param conf the json configuration
   */
  def fromJson(conf: ACursor): Try[LanderConf] = {
    val result = for {
      dt <- scalarFromJson(conf.downField("dt"))
      g <- scalarFromJson(conf.downField("g"))
      fuel <- scalarFromJson(conf.downField("fuel"))
      initialLocationRange <- rangesFromJson(conf.downField("initialLocationRange"))(3)
      initialLocationTrans = denormalize01(initialLocationRange)
      spaceRanges <- rangesFromJson(conf.downField("spaceRanges"))(3)
      landingRadius <- scalarFromJson(conf.downField("landingRadius"))
      landingSpeedLimits <- vectorFromJson(conf.downField("landingSpeedLimits"))(2)
      optimalSpeed <- vectorFromJson(conf.downField("optimalSpeed"))(2)
      jetAccRange <- rangesFromJson(conf.downField("jetAccRange"))(NumActors)
      landedReward <- scalarFromJson(conf.downField("landedReward"))
      landedOutOfPlatformReward <- scalarFromJson(conf.downField("landedOutOfPlatformReward"))
      vCrashedOnPlatformReward <- scalarFromJson(conf.downField("vCrashedOnPlatformReward"))
      hCrashedOnPlatformReward <- scalarFromJson(conf.downField("hCrashedOnPlatformReward"))
      outOfRangeReward <- scalarFromJson(conf.downField("outOfRangeReward"))
      outOfFuelReward <- scalarFromJson(conf.downField("outOfFuelReward"))
      hCrashedOutOfPlatformReward <- scalarFromJson(conf.downField("hCrashedOutOfPlatformReward"))
      vCrashedOutOfPlatformReward <- scalarFromJson(conf.downField("vCrashedOutOfPlatformReward"))
      flyingReward <- scalarFromJson(conf.downField("flyingReward"))
      directionReward <- scalarFromJson(conf.downField("directionReward"))
      hSpeedReward <- scalarFromJson(conf.downField("hSpeedReward"))
      vSpeedReward <- scalarFromJson(conf.downField("vSpeedReward"))
      encoder <- encoderFromJson(conf)
    } yield new LanderConf(
      dt = dt,
      g = g,
      fuel = fuel,
      initialLocationTrans = initialLocationTrans,
      spaceRange = spaceRanges,
      landingRadius = landingRadius,
      landingSpeed = landingSpeedLimits,
      optimalSpeed = optimalSpeed,
      jetAccRange = jetAccRange,
      landedReward = landedReward,
      landedOutOfPlatformReward = landedOutOfPlatformReward,
      hCrashedOnPlatformReward = hCrashedOnPlatformReward,
      vCrashedOnPlatformReward = vCrashedOnPlatformReward,
      outOfRangeReward = outOfRangeReward,
      outOfFuelReward = outOfFuelReward,
      hCrashedOutOfPlatformReward = hCrashedOutOfPlatformReward,
      vCrashedOutOfPlatformReward = vCrashedOutOfPlatformReward,
      flyingReward = flyingReward,
      directionReward = directionReward,
      hSpeedReward = hSpeedReward,
      vSpeedReward = vSpeedReward,
      encoder = encoder
    )
    result
  }

  def encoderFromJson(conf: ACursor): Try[LanderEncoder] = {
    val result = for {
      range <- rangesFromJson(conf.downField("signalRanges"))(NumDims)
      typ <- conf.get[String]("encoder").toTry
      encoder <- Try {
        typ match {
          case "LanderTiles" => LanderTilesEncoder.fromJson(conf, range)
          case "LanderContinuous" => new LanderContinuousEncoder(Utils.normalize(ranges = range))
          case typ => throw new IllegalArgumentException(s"Unreconginzed coder type '$typ'")
        }
      }
    } yield encoder
    result
  }
}

/**
 * Lander status
 */
object StatusCode extends Enumeration {
  val Flying, Landed, LandedOutOfPlatform, VCrashedOutOfPlatform, HCrashedOutOfPlatform, VCrashedOnPlatform, HCrashedOnPlatform, OutOfRange, OutOfFuel = Value
}