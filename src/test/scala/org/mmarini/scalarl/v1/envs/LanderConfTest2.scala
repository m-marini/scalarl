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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.scalatest.{FunSpec, Matchers}

class LanderConfTest2 extends FunSpec with Matchers {
  Nd4j.create()
  val DefaultFuel = 10
  val conf: LanderConf = LanderConf(
    dt = 0.25,
    h0Range = 5.0,
    z0 = 1.0,
    fuel = DefaultFuel,
    z1 = 10.0,
    zMax = 100.0,
    hRange = 500.0,
    zRange = 150.0,
    vhRange = 24.0,
    vzRange = 12.0,
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

  val PosSize = 10
  val HSpeedSize = 8
  val ZSpeedSize = 4

  val HighSW = 4
  val HighNW = 5
  val HighSE = 6
  val HighNE = 7
  val LowCenter = 8
  val HighCenter = 9

  val PosFeatureIdx = 6
  val VHFeatureIdx = 16
  val VZFeatureIdx = 24
  val FeatureSize = 28

  val Pos: INDArrayIndex = NDArrayIndex.interval(0, 3)
  val Speed: INDArrayIndex = NDArrayIndex.interval(3, PosFeatureIdx)
  val PosFeatures: INDArrayIndex = NDArrayIndex.interval(PosFeatureIdx, VHFeatureIdx)
  val HSpeedFeatures: INDArrayIndex = NDArrayIndex.interval(VHFeatureIdx, VZFeatureIdx)
  val ZSpeedFeatures: INDArrayIndex = NDArrayIndex.interval(VZFeatureIdx, FeatureSize)

  def features(idx: Int, size: Int): INDArray = {
    require(idx >= 0 && idx < size)
    val f = Nd4j.zeros(size)
    f.putScalar(idx, 1)
    f
  }

  describe("LanderConf pos signals") {
    val speed = Nd4j.zeros(3)
    describe("at (0,0,0)") {
      val pos = Nd4j.zeros(3)
      val signals = conf.signals(pos, speed)
      it("should return pos signals (0,0,0)") {
        signals.get(Pos) shouldBe pos
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(LowCenter, PosSize)
      }
    }
    describe("at low SW") {
      val pos = Nd4j.create(Array(-250.0, -250.0, 5.0))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (-0.5,-0.5,0)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(-0.5, -0.5, 5.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(0, PosSize)
      }
    }
    describe("at low NW") {
      val pos = Nd4j.create(Array(-250.0, 250.0, 5.0))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (-0.5,0.5,0)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(-0.5, 0.5, 5.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(1, PosSize)
      }
    }
    describe("at low SE") {
      val pos = Nd4j.create(Array(250.0, -250.0, 5.0))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (0.5,-0.5,0)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(0.5, -0.5, 5.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(2, PosSize)
      }
    }
    describe("at low NE") {
      val pos = Nd4j.create(Array(250.0, 250.0, 5.0))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (0.5,0.5,0)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(0.5, 0.5, 5.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(3, PosSize)
      }
    }
    describe("at (0,0,11)") {
      val pos = Nd4j.create(Array(0.0, 0.0, 10.0))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (0,0,1/15)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(0.0, 0.0, 10.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(HighCenter, PosSize)
      }
    }
    describe("at high SW") {
      val pos = Nd4j.create(Array(-250.0, -250.0, 10.0))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (-0.5,-0.5,10/150)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(-0.5, -0.5, 10.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(HighSW, PosSize)
      }
    }
    describe("at high NW") {
      val pos = Nd4j.create(Array(-250.0, 250.0, 10.0))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (-0.5,0.5,1/15)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(-0.5, 0.5, 10.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(HighNW, PosSize)
      }
    }
    describe("at high SE") {
      val pos = Nd4j.create(Array(250.0, -250.0, 10.0))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (0.5,-0.5,1/15)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(0.5, -0.5, 10.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(HighSE, PosSize)
      }
    }
    describe("at high NE") {
      val pos = Nd4j.create(Array(250.0, 250.0, 10.0))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (0.5,0.5,1/15)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(0.5, 0.5, 10.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(HighNE, PosSize)
      }
    }
    describe("border of low center") {
      val pos = Nd4j.create(Array(7.07, 7.07, 9.999))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (7.07/500,7.07/500,1/15)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(7.07 / 500, 7.07 / 500, 10.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(LowCenter, PosSize)
      }
    }
    describe("border out low center") {
      val pos = Nd4j.create(Array(7.072, 7.072, 9.999))
      val signals = conf.signals(pos, speed)
      it("should return pos signals (7.072/500,7.072/500,1/15)") {
        signals.get(Pos) shouldBe Nd4j.create(Array(7.072 / 500, 7.072 / 500, 10.0 / 150.0))
      }
      it("should return pos features") {
        signals.get(PosFeatures) shouldBe features(3, PosSize)
      }
    }
  }

  describe("LanderConf hspeed signals") {
    val pos = Nd4j.zeros(3)
    describe("speed (0,0,0)") {
      val speed = Nd4j.zeros(3)
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0,0,0)") {
        signals.get(Speed) shouldBe speed
      }
      it("should return h speed features") {
        signals.get(HSpeedFeatures) shouldBe features(0, HSpeedSize)
      }
    }
    describe("low speed SW") {
      val speed = Nd4j.create(Array(-0.176, -0.176, 0.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (-0.176,-0.176,0)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(-0.176 / 24, -0.176 / 24, 0.0))
      }
      it("should return h speed features") {
        signals.get(HSpeedFeatures) shouldBe features(0, HSpeedSize)
      }
    }
    describe("low speed centrer NW") {
      val speed = Nd4j.create(Array(-0.176, 0.176, 0.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (-0.176,0.176,0)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(-0.176 / 24, 0.176 / 24, 0.0))
      }
      it("should return h speed features") {
        signals.get(HSpeedFeatures) shouldBe features(1, HSpeedSize)
      }
    }
    describe("low speed SE") {
      val speed = Nd4j.create(Array(0.176, -0.176, 0.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.176,-0.176,0)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.176 / 24, -0.176 / 24, 0.0))
      }
      it("should return h speed features") {
        signals.get(HSpeedFeatures) shouldBe features(2, HSpeedSize)
      }
    }
    describe("low speed NE") {
      val speed = Nd4j.create(Array(0.176, 0.176, 0.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.176,0.176,0)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.176 / 24, 0.176 / 24, 0.0))
      }
      it("should return h speed features") {
        signals.get(HSpeedFeatures) shouldBe features(3, HSpeedSize)
      }
    }
    describe("high speed SW") {
      val speed = Nd4j.create(Array(-0.177, -0.177, 0.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (-0.177,-0.177,0)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(-0.177 / 24, -0.177 / 24, 0.0))
      }
      it("should return h speed features") {
        signals.get(HSpeedFeatures) shouldBe features(HighSW, HSpeedSize)
      }
    }
    describe("high speed NW") {
      val speed = Nd4j.create(Array(-0.177, 0.177, 0.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (-0.177,0.177,0)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(-0.177 / 24, 0.177 / 24, 0.0))
      }
      it("should return h speed features") {
        signals.get(HSpeedFeatures) shouldBe features(HighNW, HSpeedSize)
      }
    }
    describe("high speed SE") {
      val speed = Nd4j.create(Array(0.177, -0.177, 0.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.177,-0.177,0)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.177 / 24, -0.177 / 24, 0.0))
      }
      it("should return h speed features") {
        signals.get(HSpeedFeatures) shouldBe features(HighSE, HSpeedSize)
      }
    }
    describe("high speed NE") {
      val speed = Nd4j.create(Array(0.177, 0.177, 0.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.177,0.177,0)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.177 / 24, 0.177 / 24, 0.0))
      }
      it("should return h speed features") {
        signals.get(HSpeedFeatures) shouldBe features(HighNE, HSpeedSize)
      }
    }
  }

  describe("LanderConf zspeed signals") {
    val pos = Nd4j.zeros(3)
    describe("0 z speed") {
      val speed = Nd4j.zeros(3)
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0,0,0)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.0, 0.0, 0.0))
      }
      it("should return z speed features") {
        signals.get(ZSpeedFeatures) shouldBe features(0, ZSpeedSize)
      }
    }
    describe("very slow z speed") {
      val speed = Nd4j.create(Array(0.0, 0.0, -0.1))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.0,0.0,-0.1/12)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.0, 0.0, -0.1 / 12))
      }
      it("should return z speed features") {
        signals.get(ZSpeedFeatures) shouldBe features(1, ZSpeedSize)
      }
    }
    describe("slow z speed") {
      val speed = Nd4j.create(Array(0.0, 0.0, -1.9))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.0,0.0,-1.9/12)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.0, 0.0, -1.9 / 12))
      }
      it("should return z speed features") {
        signals.get(ZSpeedFeatures) shouldBe features(1, ZSpeedSize)
      }
    }
    describe("slow mid z speed") {
      val speed = Nd4j.create(Array(0.0, 0.0, -2.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.0,0.0,-2/12)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.0, 0.0, -2.0 / 12))
      }
      it("should return z speed features") {
        signals.get(ZSpeedFeatures) shouldBe features(2, ZSpeedSize)
      }
    }
    describe("fast mid z speed") {
      val speed = Nd4j.create(Array(0.0, 0.0, -3.9))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.0,0.0,-3.9/12)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.0, 0.0, -3.9 / 12))
      }
      it("should return z speed features") {
        signals.get(ZSpeedFeatures) shouldBe features(2, ZSpeedSize)
      }
    }
    describe("slow high z speed") {
      val speed = Nd4j.create(Array(0.0, 0.0, -4))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.0,0.0,-4/12)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.0, 0.0, -4.0 / 12))
      }
      it("should return z speed features") {
        signals.get(ZSpeedFeatures) shouldBe features(3, ZSpeedSize)
      }
    }
    describe("fast high z speed") {
      val speed = Nd4j.create(Array(0.0, 0.0, -12.0))
      val signals = conf.signals(pos, speed)
      it("should return speed signals (0.0,0.0,-12/12)") {
        signals.get(Speed) shouldBe Nd4j.create(Array(0.0, 0.0, -12.0 / 12))
      }
      it("should return z speed features") {
        signals.get(ZSpeedFeatures) shouldBe features(3, ZSpeedSize)
      }
    }
  }
}