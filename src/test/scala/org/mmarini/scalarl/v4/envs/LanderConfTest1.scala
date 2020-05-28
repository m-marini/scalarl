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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class LanderConfTest1 extends FunSpec with Matchers {
  create()

  val DefaultFuel: INDArray = ones(1).mul(10.0)
  val conf: LanderConf = new LanderConf(
    dt = ones(1).mul(0.25),
    h0Range = ones(1).mul(5.0),
    z0 = ones(1).mul(1.0),
    fuel = DefaultFuel,
    zMax = ones(1).mul(100.0),
    hRange = ones(1).mul(500.0),
    landingRadius = ones(1).mul(10.0),
    landingVH = ones(1).mul(0.5),
    landingVZ = ones(1).mul(4.0),
    g = ones(1).mul(1.6),
    maxAH = ones(1).mul(1),
    maxAZ = ones(1).mul(3.2),
    landedReward = ones(1).mul(100.0),
    crashReward = ones(1).mul(-100.0),
    outOfRangeReward = ones(1).mul(-100.0),
    outOfFuelReward = ones(1).mul(-100.0),
    flyingReward = ones(1).mul(-1.0),
    rewardDistanceScale = ones(1).mul(0.01))
  private val MaxPower = 4
  val Act000: INDArray = zeros(3)
  val Act111: INDArray = ones(3)
  val Act222: INDArray = ones(3).muli(2)
  val Act333: INDArray = ones(3).muli(3)
  val Act444: INDArray = ones(3).muli(MaxPower)

  private def vector(x: Double, y: Double, z: Double) = create(Array(x, y, z))

  describe("LanderConf drive") {
    val pos = vector(0, 0, 10.0)
    val speed = vector(0, 0, 0)
    it("should remain in the position when hover") {
      val (p, v) = conf.drive(Act222, pos, speed)
      p shouldBe pos
      v shouldBe speed
    }
    it("should increase speeds when jetting") {
      val (p, v) = conf.drive(Act333, pos, speed)
      p shouldBe pos
      v shouldBe vector(0.5 * 0.25, 0.5 * 0.25, 0.8 * 0.25)
    }
    it("should increase speeds when jetting max") {
      val (p, v) = conf.drive(Act444, pos, speed)
      p shouldBe pos
      v shouldBe vector(0.25, 0.25, 1.6 * 0.25)
    }
    it("should decrease speeds when jetting z") {
      val act = ones(3)
      val (p, v) = conf.drive(act, pos, speed)
      p shouldBe pos
      v shouldBe vector(-0.5 * 0.25, -0.5 * 0.25, -0.8 * 0.25)
    }
    it("should decrease speeds when jetting max z") {
      val (p, v) = conf.drive(Act000, pos, speed)
      p shouldBe pos
      v shouldBe vector(-0.25, -0.25, -1.6 * 0.25)
    }
    it("should change position when moving") {
      val speed = vector(1, 1, 1)
      val (p, v) = conf.drive(Act222, pos, speed)
      p shouldBe vector(0.25, 0.25, 10.25)
      v shouldBe speed
    }
  }
}
