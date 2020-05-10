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

package org.mmarini.scalarl.v3.envs

import org.mmarini.scalarl.v3.envs.StatusCode._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class LanderConfTest extends FunSpec with Matchers {

  create()

  val DefaultFuel: INDArray = ones(1).mul(10.0)
  val conf: LanderConf = new LanderConf(
    dt = ones(1).mul(0.25),
    h0Range = ones(1).mul(5.0),
    z0 = ones(1).mul(1),
    fuel = DefaultFuel,
    zMax = ones(1).mul(100.0),
    hRange = ones(1).mul(500.0),
    landingRadius = ones(1).mul(10.0),
    landingVH = ones(1).mul(0.5),
    landingVZ = ones(1).mul(4.0),
    g = ones(1).mul(1.6),
    maxAH = ones(1),
    maxAZ = ones(1).mul(3.2),
    landedReward = ones(1).mul(100.0),
    crashReward = ones(1).mul(-100.0),
    outOfRangeReward = ones(1).mul(-100.0),
    outOfFuelReward = ones(1).mul(-100.0),
    rewardDistanceScale = ones(1).mul(0.01))

  describe("LanderConf at land point") {
    describe("at (0,0,-0.1), (0,0,1)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0, 0, 1))

      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("at (0,0,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("a at (0,0,-0.1) speed (0.5,0,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0.5, 0, -4.0))

      it("should be landed") {
        val x = conf.status(pos, speed, DefaultFuel)
        x shouldBe Landed
      }
    }

    describe("at (0,0,-0.1) speed (-0.5,0,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](-0.5, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("at(0,0,0) speed (0,0.5,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0, 0.5, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("at(0,0,-0.1) speed (0,-0.5,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0, -0.5, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("at(10,0,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](10.0, 0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("at(-10,0,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](-10.0, 0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("at(0,10,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](0, 10.0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("at(0,-10,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](0, 10.0, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("at(7.07,7.07,-0.1) speed (0,0,-4)") {
      val pos = create(Array[Double](7.07, 7.07, -0.1))
      val speed = create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
      }
    }

    describe("at(0,0,-0.1) speed (0.353,0.353,-4)") {
      val pos = create(Array[Double](0, 0, -0.1))
      val speed = create(Array[Double](0.353, 0.353, -4.0))
      it("should be landed") {
        conf.status(pos, speed, DefaultFuel) shouldBe Landed
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
        conf.status(pos, speed, DefaultFuel) shouldBe OutOfPlatform
      }
    }
  }

  describe("LanderConf at out of range point") {
    describe("at (0,0,100.1) speed (0,0,0)") {
      val pos = create(Array[Double](0, 0, 100.1))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, DefaultFuel) shouldBe OutOfRange
      }
    }

    describe("at (600.1,0,10) speed (0,0,0)") {
      val pos = create(Array[Double](600.1, 0, 10.0))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, DefaultFuel) shouldBe OutOfRange
      }
    }

    describe("at (-600.1,0,10) speed (0,0,0)") {
      val pos = create(Array[Double](-600.1, 0, 10.0))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, DefaultFuel) shouldBe OutOfRange
      }
    }

    describe("at (0, 600.1,10) speed (0,0,0)") {
      val pos = create(Array[Double](0, 600.1, 10.0))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, DefaultFuel) shouldBe OutOfRange
      }
    }

    describe("a lander status at (0, -600.1,10) speed (0,0,0)") {
      val pos = create(Array[Double](0, -600.1, 10.0))
      val speed = zeros(3)
      it("should be out of range") {
        conf.status(pos, speed, DefaultFuel) shouldBe OutOfRange
      }
    }
  }
}
