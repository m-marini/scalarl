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

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FunSpec, Matchers}

class LanderConfTest1 extends FunSpec with Matchers {
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

  private val MaxPower = 4
  private val NumChannels = 15

  private def actions(x: Int, y: Int, z: Int) = {
    require(x >= 0 && x < 5)
    require(y >= 0 && y < 5)
    require(z >= 0 && z < 5)
    val result = Nd4j.zeros(NumChannels)
    result.putScalar(x, 1)
    result.putScalar(y + 5, 1)
    result.putScalar(z + 10, 1)
    result
  }

  private def vector(x: Double, y: Double, z: Double) = Nd4j.create(Array(x, y, z))

  describe("LanderConf drive") {
    val pos = vector(0, 0, 10.0)
    val speed = vector(0, 0, 0)
    val act = 2 + 2 * 5 + 2 * 25
    it("should remain in the position when hover") {
      val (p, v) = conf.drive(act, pos, speed)
      p shouldBe pos
      v shouldBe speed
    }
    it("should increase speeds when jetting") {
      val act = 3 + 3 * 5 + 3 * 25
      val (p, v) = conf.drive(act, pos, speed)
      p shouldBe pos
      v shouldBe vector(0.5 * 0.25, 0.5 * 0.25, 0.8 * 0.25)
    }
    it("should increase speeds when jetting max") {
      val act = MaxPower + MaxPower * 5 + MaxPower * 25
      val (p, v) = conf.drive(act, pos, speed)
      p shouldBe pos
      v shouldBe vector(0.25, 0.25, 1.6 * 0.25)
    }
    it("should decrease speeds when jetting z") {
      val act = 1 + 5 + 25
      val (p, v) = conf.drive(act, pos, speed)
      p shouldBe pos
      v shouldBe vector(-0.5 * 0.25, -0.5 * 0.25, -0.8 * 0.25)
    }
    it("should decrease speeds when jetting max z") {
      val act = 0
      val (p, v) = conf.drive(act, pos, speed)
      p shouldBe pos
      v shouldBe vector(-0.25, -0.25, -1.6 * 0.25)
    }
    it("should change position when moving") {
      val act = 2 + 2 * 5 + 2 * 25
      val speed = vector(1, 1, 1)
      val (p, v) = conf.drive(act, pos, speed)
      p shouldBe vector(0.25, 0.25, 10.25)
      v shouldBe speed
    }
  }
}
