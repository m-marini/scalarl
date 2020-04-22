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

package org.mmarini.scalarl.v2.envs

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FunSpec, Matchers}

class LanderConfTest extends FunSpec with Matchers {
  Nd4j.create()
  val DefaultFuel = 10
  val conf: LanderConf = new LanderConf(
    dt = 0.25,
    h0Range = 5.0,
    z0 = 1.0,
    fuel = DefaultFuel,
    zMax = 100.0,
    hRange = 500.0,
    landingRadius = 10.0,
    landingVH = 0.5,
    landingVZ = 4.0,
    g = 1.6,
    maxAH = 1,
    maxAZ = 3.2,
    landedReward = 100.0,
    crashReward = -100.0,
    outOfRangeReward = -100.0,
    outOfFuelReward = -100.0,
    rewardDistanceScale = 0.01)

  describe("LanderConf at land point") {
    describe("at (0,0,-0.1), (0,0,1)") {
      val pos = Nd4j.create(Array[Double](0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0, 1))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at (0,0,-0.1) speed (0,0,-4)") {
      val pos = Nd4j.create(Array[Double](0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("a at (0,0,-0.1) speed (0.5,0,-4)") {
      val pos = Nd4j.create(Array[Double](0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0.5, 0, -4.0))

      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at (0,0,-0.1) speed (-0.5,0,-4)") {
      val pos = Nd4j.create(Array[Double](0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](-0.5, 0, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at(0,0,0) speed (0,0.5,-4)") {
      val pos = Nd4j.create(Array[Double](0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0.5, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at(0,0,-0.1) speed (0,-0.5,-4)") {
      val pos = Nd4j.create(Array[Double](0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0, -0.5, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at(10,0,-0.1) speed (0,0,-4)") {
      val pos = Nd4j.create(Array[Double](10.0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at(-10,0,-0.1) speed (0,0,-4)") {
      val pos = Nd4j.create(Array[Double](-10.0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at(0,10,-0.1) speed (0,0,-4)") {
      val pos = Nd4j.create(Array[Double](0, 10.0, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at(0,-10,-0.1) speed (0,0,-4)") {
      val pos = Nd4j.create(Array[Double](0, 10.0, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at(7.07,7.07,-0.1) speed (0,0,-4)") {
      val pos = Nd4j.create(Array[Double](7.07, 7.07, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }

    describe("at(0,0,-0.1) speed (0.353,0.353,-4)") {
      val pos = Nd4j.create(Array[Double](0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0.353, 0.353, -4.0))
      it("should be landed") {
        conf.isLanded(pos, speed) shouldBe true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
    }
  }

  describe("LanderConf at crash point") {
    describe("at (0,0,-0.1) speed (0,0,-4.1)") {
      val pos = Nd4j.create(Array[Double](0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0, -4.1))
      it("should be crashed") {
        conf.isCrashed(pos, speed) shouldBe true
      }
      it("should not be landed") {
        conf.isLanded(pos, speed) should not be true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
    }

    describe("at (0,0,-0.1) speed (0.354,0.354,-4.1)") {
      val pos = Nd4j.create(Array[Double](0, 0, -0.1))
      val speed = Nd4j.create(Array[Double](0.354, 0.354, -4.1))
      it("should be crashed") {
        conf.isCrashed(pos, speed) shouldBe true
      }
      it("should not be landed") {
        conf.isLanded(pos, speed) should not be true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
    }

    describe("at (7.08,7-08,-0.1) speed (0, 0, 0)") {
      val pos = Nd4j.create(Array[Double](7.08, 7.08, -0.1))
      val speed = Nd4j.create(Array[Double](0, 0, 0))
      it("should be crashed") {
        conf.isCrashed(pos, speed) shouldBe true
      }
      it("should not be landed") {
        conf.isLanded(pos, speed) should not be true
      }
      it("should not be out of range") {
        conf.isOutOfRange(pos) should not be true
      }
    }
  }

  describe("LanderConf at out of range point") {
    describe("at (0,0,100.1) speed (0,0,0)") {
      val pos = Nd4j.create(Array[Double](0, 0, 100.1))
      val speed = Nd4j.zeros(3)
      it("should be out of range") {
        conf.isOutOfRange(pos) shouldBe true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
      it("should not be landed") {
        conf.isLanded(pos, speed) should not be true
      }
    }

    describe("at (600.1,0,10) speed (0,0,0)") {
      val pos = Nd4j.create(Array[Double](600.1, 0, 10.0))
      val speed = Nd4j.zeros(3)
      it("should be out of range") {
        conf.isOutOfRange(pos) shouldBe true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
      it("should not be landed") {
        conf.isLanded(pos, speed) should not be true
      }
    }

    describe("at (-600.1,0,10) speed (0,0,0)") {
      val pos = Nd4j.create(Array[Double](-600.1, 0, 10.0))
      val speed = Nd4j.zeros(3)
      it("should be out of range") {
        conf.isOutOfRange(pos) shouldBe true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
      it("should not be landed") {
        conf.isLanded(pos, speed) should not be true
      }
    }

    describe("at (0, 600.1,10) speed (0,0,0)") {
      val pos = Nd4j.create(Array[Double](0, 600.1, 10.0))
      val speed = Nd4j.zeros(3)
      it("should be out of range") {
        conf.isOutOfRange(pos) shouldBe true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
      it("should not be landed") {
        conf.isLanded(pos, speed) should not be true
      }
    }

    describe("a lander status at (0, -600.1,10) speed (0,0,0)") {
      val pos = Nd4j.create(Array[Double](0, -600.1, 10.0))
      val speed = Nd4j.zeros(3)
      it("should be out of range") {
        conf.isOutOfRange(pos) shouldBe true
      }
      it("should not be crashed") {
        conf.isCrashed(pos, speed) should not be true
      }
      it("should not be landed") {
        conf.isLanded(pos, speed) should not be true
      }
    }
  }
}
