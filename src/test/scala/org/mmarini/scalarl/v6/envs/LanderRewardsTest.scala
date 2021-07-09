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

import org.mmarini.scalarl.v6.Utils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

import scala.math._

class LanderRewardsTest extends FunSpec with Matchers {
  private val Ds = 0.1
  private val Dh = 0.1
  private val Dt = 0.25
  private val DefaultFuel = 10.0
  private val H0Range = 5.0
  private val Z0 = 1.0
  private val ZMax = 100.0
  private val HRange = 500.0
  private val LandingRadius = 10.0
  private val LandingVH = 0.5
  private val LandingVZ = 4.0
  private val G = 1.6
  private val MaxAZ = 3.2
  private val LandedReward = 1.0
  private val VCrashReward = -1.0
  private val HCrashReward = -2.0
  private val OutOfPlatformReward = -3.0
  private val OutOfRangeReward = -4.0
  private val OutOfFuelReward = -5.0
  private val FlyingReward = -6.0
  private val DirectionReward = 2.0
  private val DistanceReward = 3.0
  private val HeightReward = 4.0
  private val HSpeedReward = 5.0
  private val VSpeedReward = 6.0
  private val DEG0 = 0
  private val DEG45 = 45
  private val DEG90 = 90
  private val DEG135 = 135
  private val DEG180 = 180
  private val DEG225 = 225
  private val DEG270 = 270
  private val DEG315 = 315
  private val M50 = 50.0
  private val Epsilon = 0.01

  private val AllAngles = Seq(DEG0, DEG45, DEG90, DEG135, DEG180, DEG225, DEG270, DEG315)

  create()
  private val defaultFuel: INDArray = ones(1).mul(DefaultFuel)
  private val landerReward = LanderRewards(vector(LandedReward, DirectionReward, DistanceReward, HeightReward, HSpeedReward, VSpeedReward))
  private val hCrashReward = LanderRewards(vector(HCrashReward, DirectionReward, DistanceReward, HeightReward, HSpeedReward, VSpeedReward))
  private val vCrashReward = LanderRewards(vector(VCrashReward, DirectionReward, DistanceReward, HeightReward, HSpeedReward, VSpeedReward))
  private val outOfPlatformReward = LanderRewards(vector(OutOfPlatformReward, DirectionReward, DistanceReward, HeightReward, HSpeedReward, VSpeedReward))
  private val outOfRangeReward = LanderRewards(vector(OutOfRangeReward, DirectionReward, DistanceReward, HeightReward, HSpeedReward, VSpeedReward))
  private val outOfFuelReward = LanderRewards(vector(OutOfFuelReward, DirectionReward, DistanceReward, HeightReward, HSpeedReward, VSpeedReward))
  private val flyingReward = LanderRewards(vector(FlyingReward, DirectionReward, DistanceReward, HeightReward, HSpeedReward, VSpeedReward))
  private val conf: LanderConf = new LanderConf(
    dt = ones(1).mul(Dt),
    fuel = defaultFuel,
    landingRadius = ones(1).mul(LandingRadius),
    g = ones(1).mul(G),
    initialLocationTrans = Utils.clipAndNormalize01(create(Array(
      Array(-H0Range, -H0Range, Z0),
      Array(H0Range, H0Range, Z0),
    ))),
    spaceRange = create(Array(
      Array(-HRange, -HRange, 0),
      Array(HRange, HRange, ZMax)
    )),
    landingSpeed = vector(LandingVH, -LandingVZ),
    jetAccRange = create(Array(
      Array(-1.0, -1.0, -MaxAZ),
      Array(1.0, 1.0, MaxAZ)
    )),
    optimalSpeed = create(Array(
      Array(0.0, -LandingVZ),
      Array(LandingVH, 0.0)
    )),
    landedReward = landerReward,
    landedOutOfPlatformReward = outOfPlatformReward,
    hCrashedOnPlatformReward = hCrashReward,
    vCrashedOnPlatformReward = vCrashReward,
    hCrashedOutOfPlatformReward = hCrashReward,
    vCrashedOutOfPlatformReward = vCrashReward,
    outOfRangeReward = outOfRangeReward,
    outOfFuelReward = outOfFuelReward,
    flyingReward = flyingReward
  )

  def status(pos: INDArray, speed: INDArray = zeros(3), time: Double = 0.0, fuel: Double = DefaultFuel): LanderStatus =
    LanderStatus(
      pos = pos,
      speed = speed,
      time = ones(1).muli(time),
      fuel = ones(1).muli(fuel),
      conf
    )

  private def cyl(angle: Double, radius: Double, z: Double) = {
    val rad = toRadians(angle)
    vector(radius * cos(rad), radius * sin(rad), z)
  }

  private def vector(data: Double*): INDArray = create(data.toArray)

  private def expectedRewardVector(dDis: Double = 0,
                                   distance: Double = 0,
                                   height: Double = 0,
                                   dvh: Double = 0,
                                   dvz: Double = 0): INDArray =
    vector(1, dDis, distance, height, dvh, dvz)

  private def expectedReward(base: Double,
                             dDis: Double = 0,
                             distance: Double = 0,
                             height: Double = 0,
                             dvh: Double = 0,
                             dvz: Double = 0): Double =
    base + dDis * DirectionReward + distance * DistanceReward + height * HeightReward + dvh * HSpeedReward + dvz * VSpeedReward

  private def transitionCase(distance: Double = 0, direction: Double = 0, height: Double = 0,
                             dDist: Double = 0, dDir: Double = 0, dh: Double = 0,
                             vh: Double = 0, vDir: Double = 0, vz: Double = 0): (LanderStatus, LanderStatus) = {
    val pos = cyl(direction, distance, height)
    val speed = cyl(vDir, vh, vz)
    val s0 = status(pos = pos, speed = speed)
    val s1 = status(pos = pos.add(cyl(dDir, dDist, dh)),
      speed = speed)
    (s0, s1)
  }

  describe(s"LanderConf Landed") {
    describe("at(0,0), v=(0,0,-1)") {
      val (s0, s1) = transitionCase(dh = -0.1, vz = -1)
      it("should compute rewardVector") {
        val rv = conf.rewardVector(s0, s1)
        rv shouldBe expectedRewardVector(dDis = 0.1, height = -0.1)
      }
      it("should compute reward") {
        val r = conf.reward(s0, s1)
        r.getDouble(0l) shouldBe expectedReward(LandedReward, dDis = 0.1, height = -0.1) +- Epsilon
      }
    }

    describe("at (0,0) speed (0,0,-4)") {
      val (s0, s1) = transitionCase(dh = -0.1, vz = -4)
      it("should compute rewardVector") {
        val rv = conf.rewardVector(s0, s1)
        rv shouldBe expectedRewardVector(dDis = 0.1, height = -0.1)
      }
      it("should compute reward") {
        val r = conf.reward(s0, s1)
        r.getDouble(0l) shouldBe expectedReward(LandedReward, dDis = 0.1, height = -0.1) +- Epsilon
      }
    }

    describe(s"at (0,0,-0.1)") {
      AllAngles.foreach(angle => {
        describe(s"speed(0.5,R${angle},-4)") {
          val (s0, s1) = transitionCase(dh = -0.1, vh = 0.5, vDir = angle, vz = -4)

          it("should compute rewardVector") {
            val rv = conf.rewardVector(s0, s1)
            rv shouldBe expectedRewardVector(dDis = 0.1, height = -0.1)
          }
          it("should compute reward") {
            val r = conf.reward(s0, s1)
            r.getDouble(0l) shouldBe expectedReward(LandedReward, dDis = 0.1, height = -0.1) +- Epsilon
          }
        }
      })
    }

    describe(s"speed (0,0,-4)") {
      AllAngles.foreach(angle => {
        describe(s"at(10,R${angle})") {
          val (s0, s1) = transitionCase(distance = 10, direction = angle,
            dh = -0.1,
            vz = -4)
          it("should compute rewardVector") {
            val rv = conf.rewardVector(s0, s1)
            rv shouldBe expectedRewardVector(dDis = 0.5e-3, height = -0.1, distance = 10)
          }
          it("should compute reward") {
            val r = conf.reward(s0, s1)
            r.getDouble(0l) shouldBe expectedReward(LandedReward, dDis = 0.5e-3, height = -0.1, distance = 10) +- Epsilon
          }
        }
      })
    }

    describe(s"at(0,0,-0.1)") {
      AllAngles.foreach(angle => {
        val Speed = 0.5
        describe(s"speed (0.5,R${angle},-4)") {
          val (s0, s1) = transitionCase(dh = -0.1,
            vh = Speed, vDir = angle,
            vz = -4)
          it("should compute rewardVector") {
            val rv = conf.rewardVector(s0, s1)
            rv shouldBe expectedRewardVector(dDis = 0.1, height = -0.1)
          }
          it("should compute reward") {
            val r = conf.reward(s0, s1)
            r.getDouble(0l) shouldBe expectedReward(LandedReward, dDis = 0.1, height = -0.1) +- Epsilon
          }
        }
      })
    }
  }

  describe(s"LanderConf OutOfPlatform") {
    val Distance = 10.1
    AllAngles.foreach(angle => {
      describe(s"at (D$Distance, R$angle,-0.1) speed(0,0,0)") {
        val (s0, s1) = transitionCase(distance = Distance, direction = angle, dh = -0.1)
        it("should compute rewardVector") {
          val rv = conf.rewardVector(s0, s1)
          rv shouldBe expectedRewardVector(dDis = 500e-6, distance = Distance, height = -0.1)
        }
        it("should compute reward") {
          val r = conf.reward(s0, s1)
          r.getDouble(0l) shouldBe expectedReward(OutOfPlatformReward, dDis = 500e-6, distance = Distance, height = -0.1) +- Epsilon
        }
      }
    })
  }

  describe("LanderConf out of range") {
    describe("at (0,0,100.1) speed (0,0,0)") {
      val H = 100
      val (s0, s1) = transitionCase(height = H, dh = Dh)

      /**
       * val s0 = status(
       * pos = vector(0 - Ds, 0 - Ds, H))
       * val s1 = status(
       * pos = vector(0, 0, H))
       */
      it("should compute rewardVector") {
        val rv = conf.rewardVector(s0, s1)
        rv shouldBe expectedRewardVector(dDis = Dh, height = H + Dh)
      }
      it("should compute reward") {
        val r = conf.reward(s0, s1)
        r.getDouble(0L) shouldBe expectedReward(OutOfRangeReward, dDis = Dh, height = H + Dh) +- Epsilon
      }
    }

    //          describe("at (600.1,0,10) speed (0,0,0)") {
    val Distance = 707.2
    describe(s"at D${Distance} speed (0,0,0)") {
      AllAngles.foreach(angle => {
        val Z = 10.0
        describe(s"at (D${Distance},R${angle},100.1)") {
          val (s0, s1) = transitionCase(distance = Distance, direction = angle, height = Z)
          it("should compute rewardVector") {
            val rv = conf.rewardVector(s0, s1)
            rv shouldBe expectedRewardVector(distance = Distance, height = Z)
          }
          it("should compute reward") {
            val r = conf.reward(s0, s1)
            r.getDouble(0L) shouldBe expectedReward(OutOfRangeReward, distance = Distance, height = Z) +- Epsilon
          }
        }
      })
    }
  }
  describe("LanderConf flying") {
    describe("at (0,0,10.0) speed (0,0,0)") {
      val H = 10.0
      val (s0, s1) = transitionCase(height = H)

      it("should compute rewardVector") {
        val rv = conf.rewardVector(s0, s1)
        rv shouldBe expectedRewardVector(height = H)
      }
      it("should compute reward") {
        val r = conf.reward(s0, s1)
        r.getDouble(0L) shouldBe expectedReward(FlyingReward, height = H) +- Epsilon
      }
    }

    describe("at D5") {
      val Distance = 5
      val Ds = 0.1
      val Z = 10.0
      AllAngles.foreach(direction => {
        val r = sqrt(Z * Z + Distance * Distance)
        describe(s"at (D$Distance,R$direction,10.0) ds (D$Ds,R$direction)") {
          val (s0, s1) = transitionCase(distance = Distance, direction = direction, height = Z,
            dDist = Ds, dDir = direction)
          val r0 = sqrt(pow(Distance + Ds, 2) + Z * Z)
          val ds = r0 - r
          it("should compute rewardVector") {
            val rv = conf.rewardVector(s0, s1)
            rv shouldBe expectedRewardVector(dDis = ds, distance = Distance + Ds, height = Z)
          }
          it("should compute reward") {
            val r = conf.reward(s0, s1)
            r.getDouble(0L) shouldBe expectedReward(FlyingReward, dDis = ds, distance = Distance + Ds, height = Z) +- Epsilon
          }
        }

        describe(s"at (D$Distance,R$direction,10.0) ds (D$Ds,R${direction + 180})") {
          val (s0, s1) = transitionCase(distance = Distance, direction = direction, height = Z,
            dDist = Ds, dDir = direction + 180)
          val r180 = sqrt(pow(Distance - Ds, 2) + Z * Z)
          val ds = r180 - r
          it("should compute rewardVector") {
            val rv = conf.rewardVector(s0, s1)
            rv shouldBe expectedRewardVector(dDis = ds, distance = Distance - Ds, height = Z)
          }
          it("should compute reward") {
            val r = conf.reward(s0, s1)
            r.getDouble(0L) shouldBe expectedReward(FlyingReward, dDis = ds, distance = Distance - Ds, height = Z) +- Epsilon
          }
        }

        describe(s"at (D$Distance,R$direction,10.0) ds (D$Ds,R${direction + 90})") {
          val (s0, s1) = transitionCase(distance = Distance, direction = direction, height = Z,
            dDist = Ds, dDir = direction + 90)
          val r180 = sqrt(pow(Distance, 2) + pow(Ds, 2) + Z * Z)
          val dh = sqrt(Distance * Distance + Ds * Ds)
          val ds = r180 - r
          it("should compute rewardVector") {
            val rv = conf.rewardVector(s0, s1)
            rv shouldBe expectedRewardVector(dDis = ds, distance = dh, height = Z)
          }
          it("should compute reward") {
            val r = conf.reward(s0, s1)
            r.getDouble(0L) shouldBe expectedReward(FlyingReward, dDis = ds, distance = dh, height = Z) +- Epsilon
          }
        }

        describe(s"at (D$Distance,R$direction,10.0) ds (D$Ds,R${direction - 90})") {
          val (s0, s1) = transitionCase(distance = Distance, direction = direction, height = Z,
            dDist = Ds, dDir = direction - 90)
          val r180 = sqrt(pow(Distance, 2) + pow(Ds, 2) + Z * Z)
          val dh = sqrt(Distance * Distance + Ds * Ds)
          val ds = r180 - r
          it("should compute rewardVector") {
            val rv = conf.rewardVector(s0, s1)
            rv shouldBe expectedRewardVector(dDis = ds, distance = dh, height = Z)
          }
          it("should compute reward") {
            val r = conf.reward(s0, s1)
            r.getDouble(0L) shouldBe expectedReward(FlyingReward, dDis = ds, distance = dh, height = Z) +- Epsilon
          }
        }
      })
    }
  }
}
