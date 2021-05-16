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

import io.circe.ACursor
import org.mmarini.scalarl.v6.Configuration._
import org.mmarini.scalarl.v6.Utils._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._

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
 * @param landedReward                the reward function when landed
 * @param landedOutOfPlatformReward   the reward function when landed out of platform
 * @param hCrashedOnPlatformReward    the reward function when horizontal crash on platform
 * @param vCrashedOnPlatformReward    the reward function when vertically crash on platform
 * @param outOfRangeReward            the reward function when out of range
 * @param outOfFuelReward             the reward function when out of fuel
 * @param hCrashedOutOfPlatformReward the reward function when horizontal crash out of platform
 * @param vCrashedOutOfPlatformReward the reward function when vertical crash out of platform
 * @param flyingReward                the reward function when flying
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
                 val landedReward: LanderStatus => INDArray,
                 val landedOutOfPlatformReward: LanderStatus => INDArray,
                 val hCrashedOnPlatformReward: LanderStatus => INDArray,
                 val vCrashedOnPlatformReward: LanderStatus => INDArray,
                 val outOfRangeReward: LanderStatus => INDArray,
                 val outOfFuelReward: LanderStatus => INDArray,
                 val hCrashedOutOfPlatformReward: LanderStatus => INDArray,
                 val vCrashedOutOfPlatformReward: LanderStatus => INDArray,
                 val flyingReward: LanderStatus => INDArray) {

  import StatusCode._

  val gVector: INDArray = hstack(zeros(2), g.neg())
  val v0: INDArray = mean(optimalSpeed, 0)
  val dv: INDArray = optimalSpeed.getRow(1).sub(optimalSpeed.getRow(0)).divi(2)

  val rewardFunctionTable = Map(
    Landed -> landedReward,
    LandedOutOfPlatform -> landedOutOfPlatformReward,
    HCrashedOnPlatform -> hCrashedOnPlatformReward,
    VCrashedOnPlatform -> vCrashedOnPlatformReward,
    OutOfRange -> outOfRangeReward,
    OutOfFuel -> outOfFuelReward,
    HCrashedOutOfPlatform -> hCrashedOutOfPlatformReward,
    VCrashedOutOfPlatform -> vCrashedOutOfPlatformReward,
    Flying -> flyingReward
  )

  //  private val ActionRange = create(Array(
  //    Array(-Pi, 0.0, -landingSpeed.getDouble(1L)),
  //    Array(Pi, landingSpeed.getDouble(0L), landingSpeed.getDouble(1L))
  //  ))

  /**
   *
   * @param random the random generator
   */
  def initial(random: Random): INDArray = {
    val pos = initialLocationTrans(random.nextDouble(Array(1, 3)).muli(2).subi(1))
    pos
  }
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
    val rewardsConf = conf.downField("rewards")
    val result = for {
      dt <- scalarFromJson(conf.downField("dt"))
      g <- scalarFromJson(conf.downField("g"))
      fuel <- scalarFromJson(conf.downField("fuel"))
      initialLocationRange <- rangesFromJson(conf.downField("initialLocationRange"))(3)
      initialLocationTrans = clipAndDenormalize(initialLocationRange)
      spaceRanges <- rangesFromJson(conf.downField("spaceRanges"))(3)
      landingRadius <- scalarFromJson(conf.downField("landingRadius"))
      landingSpeedLimits <- vectorFromJson(conf.downField("landingSpeedLimits"))(2)
      optimalSpeedRanges <- rangesFromJson(conf.downField("optimalSpeedRanges"))(2)
      jetAccRange <- rangesFromJson(conf.downField("jetAccRange"))(NumActors)
      landed <- LanderRewards.fromJson(rewardsConf.downField("landed"))
      landedOutOfPlatform <- LanderRewards.fromJson(rewardsConf.downField("landedOutOfPlatform"))
      vCrashedOnPlatform <- LanderRewards.fromJson(rewardsConf.downField("vCrashedOnPlatform"))
      hCrashedOnPlatform <- LanderRewards.fromJson(rewardsConf.downField("hCrashedOnPlatform"))
      outOfRange <- LanderRewards.fromJson(rewardsConf.downField("outOfRange"))
      outOfFuel <- LanderRewards.fromJson(rewardsConf.downField("outOfFuel"))
      hCrashedOutOfPlatform <- LanderRewards.fromJson(rewardsConf.downField("hCrashedOutOfPlatform"))
      vCrashedOutOfPlatform <- LanderRewards.fromJson(rewardsConf.downField("vCrashedOutOfPlatform"))
      flying <- LanderRewards.fromJson(rewardsConf.downField("flying"))
    } yield new LanderConf(
      dt = dt,
      g = g,
      fuel = fuel,
      initialLocationTrans = initialLocationTrans,
      spaceRange = spaceRanges,
      landingRadius = landingRadius,
      landingSpeed = landingSpeedLimits,
      optimalSpeed = optimalSpeedRanges,
      jetAccRange = jetAccRange,
      landedReward = landed,
      landedOutOfPlatformReward = landedOutOfPlatform,
      hCrashedOnPlatformReward = hCrashedOnPlatform,
      vCrashedOnPlatformReward = vCrashedOnPlatform,
      outOfRangeReward = outOfRange,
      outOfFuelReward = outOfFuel,
      hCrashedOutOfPlatformReward = hCrashedOutOfPlatform,
      vCrashedOutOfPlatformReward = vCrashedOutOfPlatform,
      flyingReward = flying)
    result
  }
}

/**
 * Lander status
 */
object StatusCode extends Enumeration {
  val Flying, Landed, LandedOutOfPlatform, VCrashedOnPlatform, VCrashedOutOfPlatform, HCrashedOnPlatform, HCrashedOutOfPlatform, OutOfRange, OutOfFuel = Value
}