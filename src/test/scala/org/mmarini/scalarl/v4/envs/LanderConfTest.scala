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

import org.mmarini.scalarl.v4.Utils
import org.mmarini.scalarl.v4.envs.StatusCode._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class LanderConfTest extends FunSpec with Matchers {
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
  private val DirectionReward = 0.1
  private val HSpeedReward = 0.1
  private val VSpeedReward = 0.1

  create()

  private val defaultFuel: INDArray = ones(1).mul(DefaultFuel)

  private def encoder = new LanderContinuousEncoder(
    Utils.normalize(
      create(Array(
        Array(-1.0, -1.0, 0.0, 0.0, 0.0, -12.0),
        Array(1.0, 1.0, 10.0, 10.0, 24.0, 12.0)
      ))))

  private val conf: LanderConf = new LanderConf(
    dt = ones(1).mul(Dt),
    fuel = defaultFuel,
    landingRadius = ones(1).mul(LandingRadius),
    g = ones(1).mul(G),
    initialLocationTrans = Utils.normalize01(create(Array(
      Array(-H0Range, -H0Range, Z0),
      Array(H0Range, H0Range, Z0),
    ))),
    spaceRange = create(Array(
      Array(-HRange, -HRange, 0),
      Array(HRange, HRange, ZMax)
    )),
    landingSpeed = create(
      Array(LandingVH, -LandingVZ),
    ),
    jetAccRange = create(Array(
      Array(-1.0, -1.0, -MaxAZ),
      Array(1.0, 1.0, MaxAZ)
    )),
    optimalSpeed = create(Array(LandingVH, LandingVZ)),
    landedReward = ones(1).mul(LandedReward),
    landedOutOfPlatformReward = ones(1).mul(VCrashReward),
    vCrashedOnPlatformReward = ones(1).mul(VCrashReward),
    hCrashedOnPlatformReward = ones(1).mul(HCrashReward),
    hCrashedOutOfPlatformReward = ones(1).mul(OutOfPlatformReward),
    vCrashedOutOfPlatformReward = ones(1).mul(OutOfPlatformReward),
    outOfRangeReward = ones(1).mul(OutOfRangeReward),
    outOfFuelReward = ones(1).mul(OutOfFuelReward),
    flyingReward = ones(1).mul(FlyingReward),
    directionReward = ones(1).mul(DirectionReward),
    hSpeedReward = ones(1).mul(HSpeedReward),
    vSpeedReward = ones(1).mul(VSpeedReward),
    encoder = encoder
  )
  /*
  maxVH = ones(1).mul(LandingVH),
  maxVZ = ones(1).mul(LandingVZ),
  maxAH = ones(1),
  maxAZ = ones(1).mul(MaxAZ)
*/

  describe("LanderConf at land point") {
    describe("at (0,0,-0.1), (0,0,1)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0, 0, 1))

      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }

    describe("at (0,0,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }

    describe("a at (0,0,-0.1) speed (0.5,0,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0.5, 0, -4.0))

      it("should be landed") {
        val x = conf.status(pos, speed, defaultFuel)
        x shouldBe Landed
      }
    }

    describe("at (0,0,-0.1) speed (-0.5,0,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](-0.5, 0, -4.0))
      it("should be landed") {
        val s = conf.status(pos, speed, defaultFuel)
        s shouldBe Landed
      }
    }

    describe("at(0,0,0) speed (0,0.5,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0, 0.5, -4.0))
      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }

    describe("at(0,0,-0.1) speed (0,-0.5,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0, -0.5, -4.0))
      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }

    describe("at(10,0,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](10.0, 0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }

    describe("at(-10,0,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](-10.0, 0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }

    describe("at(0,10,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](0, 10.0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }

    describe("at(0,-10,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](0, 10.0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }

    describe("at(7.07,7.07,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](7.07, 7.07, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }

    describe("at(0,0,-0.1) speed (0.353,0.353,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0.353, 0.353, -4.0))
      it("should be landed") {
        conf.status(pos, speed, defaultFuel) shouldBe Landed
      }
    }
  }

  describe("LanderConf at crash point") {
    //    describe("at (0,0,-0.1) speed (0,0,-4.1)") {
    //      val pos = create(Array[Double](0, 0, -0.1))
    //      val speed = create(Array[Double](0, 0, -4.1))
    //      it("should be crashed") {
    //        conf.status(pos, speed, DefaultFuel) shouldBe Crashed
    //      }
    //    }

    //    describe("at (0,0,-0.1) speed (0.354,0.354,-4.1)") {
    //      val pos = create(Array[Double](0, 0, -0.1))
    //      val speed = create(Array[Double](0.354, 0.354, -4.1))
    //      it("should be crashed") {
    //        conf.status(pos, speed, DefaultFuel) shouldBe Crashed
    //      }
    //    }

    describe("at (7.08,7-08,-0.1) speed (0, 0, 0)") {
      val pos = create(Array[Double](7.08, 7.08, -0.1))
      val speed = create(Array[Double](0, 0, 0))
      it("should be crashed") {
        conf.status(pos, speed, defaultFuel) shouldBe LandedOutOfPlatform
      }
    }
  }

  describe("LanderConf at out of range point") {
    describe("at (0,0,100.1) speed (0,0,0)") {
      val pos = create(Array[Double](0, 0, 100.1))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, defaultFuel) shouldBe OutOfRange
      }
    }

    describe("at (600.1,0,10) speed (0,0,0)") {
      val pos = create(Array[Double](600.1, 0, 10.0))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, defaultFuel) shouldBe OutOfRange
      }
    }

    describe("at (-600.1,0,10) speed (0,0,0)") {
      val pos = create(Array[Double](-600.1, 0, 10.0))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, defaultFuel) shouldBe OutOfRange
      }
    }

    describe("at (0, 600.1,10) speed (0,0,0)") {
      val pos = create(Array[Double](0, 600.1, 10.0))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, defaultFuel) shouldBe OutOfRange
      }
    }

    describe("a lander status at (0, -600.1,10) speed (0,0,0)") {
      val pos = create(Array[Double](0, -600.1, 10.0))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, defaultFuel) shouldBe OutOfRange
      }
    }
  }
  describe("LanderConf flying") {
    describe("at (0,0,10.0) speed (0,0,0)") {
      val pos = create(Array[Double](0, 0, 10.0))
      val speed = zeros(3)
      it("should be Flying") {
        conf.status(pos, speed, defaultFuel) shouldBe Flying
        conf.rewardFromDirection(pos, speed) shouldBe ones(1).muli(-DirectionReward)
      }
    }
    describe("at (1,2,10.0) speed (-0.1,-0.2,-1)") {
      val pos = create(Array[Double](1, 2, 10.0))
      val speed = create(Array[Double](-0.1, -0.2, -1.0))
      it("should be Flying") {
        conf.status(pos, speed, defaultFuel) shouldBe Flying
        conf.rewardFromDirection(pos, speed) shouldBe ones(1).muli(DirectionReward)
      }
    }
    describe("at (1,2,10.0) speed (0.1,0.2,1)") {
      val pos = create(Array[Double](1, 2, 10.0))
      val speed = create(Array[Double](0.1, 0.2, 1.0))
      it("should be Flying") {
        conf.status(pos, speed, defaultFuel) shouldBe Flying
        conf.rewardFromDirection(pos, speed) shouldBe ones(1).muli(-DirectionReward)
      }
    }
  }
}
