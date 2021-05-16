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
import org.mmarini.scalarl.v6.envs.StatusCode._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

import scala.math.{cos, sin, toRadians}

class LanderRewardsTest extends FunSpec with Matchers {
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
  private val LandedReward = 100.0
  private val VCrashReward = -100.0
  private val HCrashReward = -100.0
  private val OutOfPlatformReward = -100.0
  private val OutOfRangeReward = -100.0
  private val OutOfFuelReward = -100.0
  private val FlyingReward = -1.0
  private val DirectionReward = 2.0
  private val DistanceReward = 3.0
  private val HeightReward = 4.0
  private val HSpeedReward = 5.0
  private val VSpeedReward = 6.0
  private val DEG0 = 0.0
  private val DEG45 = toRadians(45)
  private val DEG90 = toRadians(90)
  private val DEG135 = toRadians(135)
  private val DEG180 = toRadians(180)
  private val DEG225 = toRadians(225)
  private val DEG270 = toRadians(270)
  private val DEG315 = toRadians(315)
  private val M50 = 50.0

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

  private def vector(data: Double*): INDArray = create(data.toArray)

  private def cyl(angle: Double, radius: Double, z: Double) = vector(radius * cos(angle), radius * sin(angle), z)

  describe("LanderConf at land point") {
    describe("at (0,0,-0.1), (0,0,1)") {
      val s = status(
        pos = vector(0.0, 0.0, -0.1),
        speed = vector(0, 0, -1))

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 0, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward - 0.1 * HeightReward)
      }
    }

    describe("at (0,0,-0.1) speed (0,0,-4)") {
      val s = status(
        pos = vector(0, 0, -0.1),
        speed = vector(0, 0, -4.0))
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 0, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward - 0.1 * HeightReward)
      }
    }

    describe("a at (0,0,-0.1) speed (0.5,0,-4)") {
      val s = status(
        pos = vector(0, 0, -0.1),
        speed = vector(0.5, 0, -4.0))

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 0, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward - 0.1 * HeightReward)
      }
    }

    describe("at (0,0,-0.1) speed (-0.5,0,-4)") {
      val s = status(
        pos = vector(0, 0, -0.1),
        speed = vector(-0.5, 0, -4.0))

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 0, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward - 0.1 * HeightReward)
      }
    }

    describe("at(0,0,0) speed (0,0.5,-4)") {
      val s = status(
        pos = vector(0, 0, -0.1),
        speed = vector(0, 0.5, -4.0))

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 0, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward - 0.1 * HeightReward)
      }
    }

    describe("at(0,0,-0.1) speed (0,-0.5,-4)") {
      val s = status(
        pos = vector(0, 0, -0.1),
        speed = vector(0, -0.5, -4.0))

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 0, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward - 0.1 * HeightReward)
      }
    }

    describe("at(10,0,-0.1) speed (0,0,-4)") {
      val s = status(
        pos = vector(10.0, 0, -0.1),
        speed = vector(0, 0, -4.0))

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 10, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward + DistanceReward * 10 - 0.1 * HeightReward)
      }
    }

    describe("at(-10,0,-0.1) speed (0,0,-4)") {
      val s = status(
        pos = vector(-10.0, 0, -0.1),
        speed = vector(0, 0, -4.0))

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 10, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward + DistanceReward * 10 - 0.1 * HeightReward)
      }
    }

    describe("at(0,10,-0.1) speed (0,0,-4)") {
      val s = status(
        pos = vector(0, 10.0, -0.1),
        speed = vector(0, 0, -4.0))

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 10, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward + DistanceReward * 10 - 0.1 * HeightReward)
      }
    }

    describe("at(0,-10,-0.1) speed (0,0,-4)") {
      val s = status(
        pos = vector(0, 10.0, -0.1),
        speed = vector(0, 0, -4.0))
      it("should be landed") {
        s.status shouldBe Landed
      }
    }

    describe("at(7.07,7.07,-0.1) speed (0,0,-4)") {
      val Pos = 7.07
      val s = status(
        pos = vector(Pos, Pos, -0.1),
        speed = vector(0, 0, -4.0)
      )
      val distance = Pos * Math.sqrt(2)

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, distance, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward + DistanceReward * distance - 0.1 * HeightReward)
      }
    }

    describe("at(0,0,-0.1) speed (0.353,0.353,-4)") {
      val Speed = 0.353
      val s = status(
        pos = vector(0, 0, -0.1),
        speed = vector(Speed, Speed, -4.0))
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 0, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(LandedReward - 0.1 * HeightReward)
      }
    }
  }

  describe("LanderConf at crash point") {
    //    describe("at (0,0,-0.1) speed (0,0,-4.1)") {
    //      val pos = create(vector(0, 0, -0.1))
    //      val speed = create(vector(0, 0, -4.1))
    //      it("should be crashed") {
    //        conf.status(pos, speed, DefaultFuel) shouldBe Crashed
    //      }
    //    }

    //    describe("at (0,0,-0.1) speed (0.354,0.354,-4.1)") {
    //      val pos = create(vector(0, 0, -0.1))
    //      val speed = create(vector(0.354, 0.354, -4.1))
    //      it("should be crashed") {
    //        conf.status(pos, speed, DefaultFuel) shouldBe Crashed
    //      }
    //    }

    describe("at (7.08,7-08,-0.1) speed (0, 0, 0)") {
      val Pos = 7.08
      val s = status(
        pos = vector(Pos, Pos, -0.1))
      val dist = Math.sqrt(2) * Pos
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, dist, -0.1, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(OutOfPlatformReward - 0.1 * HeightReward + dist * DistanceReward)
      }
    }
  }

  describe("LanderConf at out of range point") {
    describe("at (0,0,100.1) speed (0,0,0)") {
      val H = 100.1
      val s = status(
        pos = vector(0, 0, H))
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 0, H, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r shouldBe vector(OutOfPlatformReward + H * HeightReward)
      }
    }

    describe("at (600.1,0,10) speed (0,0,0)") {
      val X = 600.1
      val Z = 10.0
      val s = status(
        pos = vector(X, 0, Z))
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, X, Z, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r.getDouble(0L) shouldBe
          (OutOfRangeReward + X * DistanceReward + Z * HeightReward) +- 1e-4
      }
    }

    describe("at (-600.1,0,10) speed (0,0,0)") {
      val X = 600.1
      val Z = 10.0
      val s = status(
        pos = vector(-X, 0, Z))
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, X, Z, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r.getDouble(0L) shouldBe
          (OutOfRangeReward + X * DistanceReward + Z * HeightReward) +- 1e-4
      }
    }

    describe("at (0, 600.1,10) speed (0,0,0)") {
      val Y = 600.1
      val Z = 10.0
      val s = status(
        pos = vector(0, Y, Z))
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, Y, Z, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r.getDouble(0L) shouldBe
          (OutOfRangeReward + Y * DistanceReward + Z * HeightReward) +- 1e-4
      }
    }

    describe("a lander status at (0, -600.1,10) speed (0,0,0)") {
      val Y = 600.1
      val Z = 10.0
      val s = status(
        pos = vector(0, -Y, Z))
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, Y, Z, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r.getDouble(0L) shouldBe
          (OutOfRangeReward + Y * DistanceReward + Z * HeightReward) +- 1e-4
      }
    }
  }
  describe("LanderConf flying") {
    describe("at (0,0,10.0) speed (0,0,0)") {
      val H = 10.0
      val s = status(
        pos = vector(0, 0, H))

      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 0, 0, H, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r.getDouble(0L) shouldBe
          (FlyingReward + H * HeightReward) +- 1e-4
      }
    }
    describe("at (1,2,10.0) speed (-0.1,-0.2,-1)") {
      val X = 1.0
      val Y = 2.0
      val Z = 10.0
      val VX = -0.1
      val VY = -0.2
      val VZ = -1.0
      val s = status(
        pos = vector(X, Y, Z),
        speed = vector(VX, VY, VZ))

      val dist = Math.sqrt(X * X + Y * Y)
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, 1, dist, Z, 0, 0)
      }
      it("should compute reward") {
        val r = s.reward
        r.getDouble(0L) shouldBe
          (FlyingReward + DirectionReward + Z * HeightReward + dist * DistanceReward) +- 1e-4
      }
    }

    describe("at (10,20,10.0) speed (1,2,1)") {
      val X = 10.0
      val Y = 20.0
      val Z = 10.0
      val VX = 1.0
      val VY = 2.0
      val VZ = 1.0
      val s = status(
        pos = vector(X, Y, Z),
        speed = vector(VX, VY, VZ))

      val dist = Math.sqrt(X * X + Y * Y)
      val hv = Math.sqrt(VX * VX + VY * VY)
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, -1, dist, Z, hv - LandingVH, VZ)
      }
      it("should compute reward") {
        val r = s.reward
        r.getDouble(0L) shouldBe
          (FlyingReward - DirectionReward + Z * HeightReward + (hv - LandingVH) * HSpeedReward + dist * DistanceReward + VZ * VSpeedReward) +- 1e-4
      }
    }

    describe("at (1,2,10.0) speed (0.1,0.2,1)") {
      val X = 1.0
      val Y = 2.0
      val Z = 10.0
      val VX = 0.1
      val VY = 0.2
      val VZ = 1.0
      val s = status(
        pos = vector(X, Y, Z),
        speed = vector(VX, VY, VZ))

      val dist = Math.sqrt(X * X + Y * Y)
      it("should compute rewardVector") {
        val rv = s.rewardVector
        rv shouldBe vector(1, -1, dist, Z, 0, VZ)
      }
      it("should compute reward") {
        val r = s.reward
        r.getDouble(0L) shouldBe
          (FlyingReward - DirectionReward + Z * HeightReward + dist * DistanceReward + VZ * VSpeedReward) +- 1e-4
      }
    }

    describe("at (100 m, 0 DEG)") {
      val pos = cyl(DEG0, M50, M50)
      describe("v=(0.5, 0 DEG)") {
        val s = status(pos = pos, speed = cyl(DEG0, LandingVH, -LandingVZ))

        it("should compute rewardVector") {
          val rv = s.rewardVector
          rv shouldBe vector(1, -1, M50, M50, 0, 0)
        }
        it("should compute reward") {
          val r = s.reward
          r.getDouble(0L) shouldBe
            (FlyingReward - DirectionReward + M50 * HeightReward + M50 * DistanceReward) +- 1e-4
        }
      }

      describe("v=(0.5, 90 DEG)") {
        val s = status(pos = pos, speed = cyl(DEG90, LandingVH, -LandingVZ))

        it("should compute rewardVector") {
          val rv = s.rewardVector
          rv shouldBe vector(1, 0, M50, M50, 0, 0)
        }
        it("should compute reward") {
          val r = s.reward
          r.getDouble(0L) shouldBe
            (FlyingReward + M50 * HeightReward + M50 * DistanceReward) +- 1e-4
        }
      }

      describe("v=(0.5, 180 DEG)") {
        val s = status(pos, speed = cyl(DEG180, LandingVH, -LandingVZ))

        it("should compute rewardVector") {
          val rv = s.rewardVector
          rv shouldBe vector(1, 1, M50, M50, 0, 0)
        }
        it("should compute reward") {
          val r = s.reward
          r.getDouble(0L) shouldBe
            (FlyingReward + DirectionReward + M50 * HeightReward + M50 * DistanceReward) +- 1e-4
        }
      }

      describe("v=(0.5, 270 DEG)") {
        val s = status(pos, speed = cyl(DEG270, LandingVH, -LandingVZ))

        it("should compute rewardVector") {
          val rv = s.rewardVector
          rv shouldBe vector(1, 0, M50, M50, 0, 0)
        }
        it("should compute reward") {
          val r = s.reward
          r.getDouble(0L) shouldBe
            (FlyingReward + M50 * HeightReward + M50 * DistanceReward) +- 1e-4
        }
      }
    }

    describe("at (100 m, 45 DEG)") {
      val pos = cyl(DEG45, M50, M50)

      describe("v=(0.5, 45 DEG)") {
        val s = status(pos, speed = cyl(DEG45, LandingVH, -LandingVZ))

        it("should compute rewardVector") {
          val rv = s.rewardVector
          rv shouldBe vector(1, -1, M50, M50, 0, 0)
        }
        it("should compute reward") {
          val r = s.reward
          r.getDouble(0L) shouldBe
            (FlyingReward - DirectionReward + M50 * HeightReward + M50 * DistanceReward) +- 1e-4
        }
      }

      describe("v=(0.5, 135 DEG)") {
        val s = status(pos, speed = cyl(DEG135, LandingVH, -LandingVZ))

        it("should compute rewardVector") {
          val rv = s.rewardVector
          rv shouldBe vector(1, 0, M50, M50, 0, 0)
        }
        it("should compute reward") {
          val r = s.reward
          r.getDouble(0L) shouldBe
            (FlyingReward + M50 * HeightReward + M50 * DistanceReward) +- 1e-4
        }
      }

      describe("v=(0.5, 225 DEG)") {
        val s = status(pos, speed = cyl(DEG225, LandingVH, -LandingVZ))

        it("should compute rewardVector") {
          val rv = s.rewardVector
          rv shouldBe vector(1, 1, M50, M50, 0, 0)
        }
        it("should compute reward") {
          val r = s.reward
          r.getDouble(0L) shouldBe
            (FlyingReward + DirectionReward + M50 * HeightReward + M50 * DistanceReward) +- 1e-4
        }
      }

      describe("v=(0.5, 315 DEG)") {
        val s = status(pos, speed = cyl(DEG315, LandingVH, -LandingVZ))

        it("should compute rewardVector") {
          val rv = s.rewardVector
          rv shouldBe vector(1, 0, M50, M50, 0, 0)
        }
        it("should compute reward") {
          val r = s.reward
          r.getDouble(0L) shouldBe
            (FlyingReward + M50 * HeightReward + M50 * DistanceReward) +- 1e-4
        }
      }
    }
  }
}