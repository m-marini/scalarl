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

class LanderStatusTest2 extends FunSpec with Matchers {


  create()
  private val landerReward = LanderRewards(vector(1.0, 0.0, 0.0, 0.0, 0.0))
  private val hCrashReward = LanderRewards(vector(-0.2, 0.0, 0.0, -0.5, -0.25))
  private val vCrashReward = LanderRewards(vector(-0.2, 0.0, 0.0, -0.5, -0.25))
  private val outOfPlatformReward = LanderRewards(vector(-0.219, -0.0111, 0.0, 0.0, 0.0))
  private val outOfRangeReward = LanderRewards(vector(-1.0, -0.0943, 0.0, 0.0, 0.0))
  private val outOfFuelReward = LanderRewards(vector(-0.8, -0.1, -6.67e-3, 0.0, 0.0))
  private val flyingReward = LanderRewards(vector(-0.55, 0.45, 0.0, 0.0, 0.0))

  private lazy val conf: LanderConf = new LanderConf(
    dt = ones(1).mul(0.25),
    fuel = ones(1).muli(244),
    spaceRange = create(Array(
      Array(-500.0, -500, 0),
      Array(500.0, 500, 150)
    )),
    landingRadius = ones(1).muli(10),
    landingSpeed = vector(2, -4),
    optimalSpeed = create(Array(
      Array(0.0, -2.0),
      Array(1, 0.0)
    )),
    g = ones(1).mul(1.6),
    initialLocationTrans = Utils.clipAndNormalize01(create(Array(
      Array(-100.0, -100.0, 80.0),
      Array(100.0, 100.0, 100.0)
    ))),
    jetAccRange = create(Array(
      Array(-1.0, -1.0, 0.0),
      Array(1.0, 1.0, 3.2)
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

  private def status(pos: INDArray, speed: INDArray = zeros(3), time: Double = 0.0, fuel: Double = 244): LanderStatus =
    LanderStatus(
      pos = pos,
      speed = speed,
      time = ones(1).muli(time),
      fuel = ones(1).muli(fuel),
      conf
    )

  private def random = getRandomFactory.getNewRandomInstance(1234)

  private def vector(data: Double*): INDArray = create(data.toArray)

  describe("Lander status changeing on action") {
    describe("speed 1,1,0") {
      val s0 = status(
        pos = vector(100, 100, 50),
        speed = vector(1, 1, 0))
      val random = getRandomFactory.getNewRandomInstance(1234)
      val (s1: LanderStatus, reward) = s0.change(actions = vector(-Math.PI * 3 / 4, 1, 0), random = random)
      it("should change the speed") {
        s1.speed shouldBe vector(0.75, 0.75, 0)
      }
      it("should reward") {
        reward shouldBe ones(1).muli(-1)
      }
      it("should change the pos") {
        s1.pos shouldBe vector(100.25, 100.25, 50)
      }
    }

    describe("speed 0,0,0") {
      val s0 = status(
        pos = vector(100, 100, 50),
        speed = vector(0, 0, 0))
      val (s1: LanderStatus, reward) = s0.change(actions = vector(-Math.PI * 3 / 4, 1, 0), random = random)
      it("should change the speed") {
        s1.speed shouldBe vector(-0.25, -0.25, 0)
      }
      it("should change the pos") {
        s1.pos shouldBe vector(100, 100, 50)
      }
      it("should reward") {
        reward shouldBe ones(1).muli(-0.1)
      }
    }

    describe("at 100,100") {
      val s0 = status(
        pos = vector(100, 100, 50),
        speed = vector(0, 0, 0))

      it("should move to NE") {
        val (s1, r) = s0.change(vector(Math.PI * 1 / 4, 1, 0), random)
        r shouldBe ones(1).muli(-1)
      }
      it("should move to SE") {
        val (s1, r) = s0.change(vector(Math.PI * 3 / 4, 1, 0), random)
        r shouldBe ones(1).muli(-0.55)
      }
      it("should move to NW") {
        val (s1, r) = s0.change(vector(-Math.PI * 1 / 4, 1, 0), random)
        r shouldBe ones(1).muli(-0.55)
      }
      it("should move to SW") {
        val (s1, r) = s0.change(vector(-Math.PI * 3 / 4, 1, 0), random)
        r shouldBe ones(1).muli(-0.1)
      }
    }

    describe("at -100,100") {
      val s0 = status(
        pos = vector(-100, 100, 50),
        speed = vector(0, 0, 0))

      it("should move to NE") {
        val (s1, r) = s0.change(vector(Math.PI * 1 / 4, 1, 0), random)
        r shouldBe ones(1).muli(-0.55)
      }
      it("should move to SE") {
        val (s1, r) = s0.change(vector(Math.PI * 3 / 4, 1, 0), random)
        r shouldBe ones(1).muli(-1)
      }
      it("should move to NW") {
        val (s1, r) = s0.change(vector(-Math.PI * 1 / 4, 1, 0), random)
        r shouldBe ones(1).muli(-0.1)
      }
      it("should move to SW") {
        val (s1, r) = s0.change(vector(-Math.PI * 3 / 4, 1, 0), random)
        r shouldBe ones(1).muli(-0.55)
      }
    }
  }
}