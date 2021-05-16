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

class LanderStatusTest1 extends FunSpec with Matchers {
  private val DefaultFuel = 10.0
  private val Dt = 4
  private val G = 1.6
  private val Z0 = 10.0
  private val VH0 = 0.0
  private val VH1 = 0.5
  private val VH2 = 1.0
  private val VZ_2 = -4.0
  private val VZ_1 = -2.0
  private val VZ0 = 0.0
  private val VZ1 = 2.0
  private val VZ2 = 4.0
  private val R0 = 0.0
  private val R45 = Pi / 4
  private val R225 = Pi * 5 / 4
  private val JetHAcc = 1.0
  private val JetVAcc = 3.2
  private val LandingVH = 1.0
  private val LandingVZ = 4.0
  private val LandedReward = 100.0
  private val VCrashReward = -100.0
  private val HCrashReward = -100.0
  private val OutOfPlatformReward = -100.0
  private val OutOfRangeReward = -100.0
  private val OutOfFuelReward = -100.0
  private val FlyingReward = -1.0

  create()

  private val landerReward = LanderRewards(vector(LandedReward, 0.0, 0.0, 0.0, 0.0))
  private val hCrashReward = LanderRewards(vector(HCrashReward, 0.0, 0.0, 0.0, 0.0))
  private val vCrashReward = LanderRewards(vector(VCrashReward, 0.0, 0.0, 0.0, 0.0))
  private val outOfPlatformReward = LanderRewards(vector(OutOfPlatformReward, 0.0, 0.0, 0.0, 0.0))
  private val outOfRangeReward = LanderRewards(vector(OutOfRangeReward, 0.0, 0.0, 0.0, 0.0))
  private val outOfFuelReward = LanderRewards(vector(OutOfFuelReward, 0.0, 0.0, 0.0, 0.0))
  private val flyingReward = LanderRewards(vector(FlyingReward, 0.0, 0.0, 0.0, 0.0))
  //private val ActionJet: INDArray = create(Array(0.0, 1.0,))
  private val ActionMax: INDArray = vector(1.0, 2.0, 4.0)
  private val ActionSED: INDArray = vector(5.0, 1.0, 1.0)
  private val ActionSEDMax: INDArray = vector(5.0, 2.0, 0.0)

  private lazy val conf: LanderConf = new LanderConf(
    dt = ones(1).mul(Dt),
    fuel = ones(1).muli(DefaultFuel),
    spaceRange = zeros(2L, 6L),
    landingRadius = zeros(1),
    landingSpeed = vector(LandingVH, -LandingVZ),
    optimalSpeed = create(Array(
      Array(0.0, -LandingVZ),
      Array(LandingVH, 0.0)
    )),
    g = ones(1).mul(G),
    initialLocationTrans = Utils.clipAndNormalize01(zeros(2L, 3L)),
    jetAccRange = create(Array(
      Array(-JetHAcc, -JetHAcc, 0.0),
      Array(JetHAcc, JetHAcc, JetVAcc)
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

  private def cyl(angle: Double, radius: Double, z: Double) = vector(radius * cos(angle), radius * sin(angle), z)

  private def vector(data: Double*): INDArray = create(data.toArray)

  describe("LanderConf drive") {
    val s = status(
      pos = vector(0, 0, Z0),
      speed = vector(0, 0, 0))
    it("should remain in the position when hovering") {
      val action = vector(R0, VH0, VZ0)
      val s1 = s.drive(action)
      s1.pos shouldBe s.pos
      s1.speed shouldBe s.speed
    }
    it("should increase speeds when jetting") {
      val action = vector(R45, VH1, VZ1)
      val s1 = s.drive(action)
      s1.pos shouldBe s.pos
      s1.speed shouldBe cyl(R45, VH1, VZ1)
    }
    it("should increase speeds when jetting max") {
      val action = vector(R45, VH2, VZ2)
      val s1 = s.drive(action)
      s1.pos shouldBe s.pos
      s1.speed shouldBe cyl(R45, VH2, VZ2)
    }
    it("should decrease speeds when jetting z") {
      val action = vector(R225, VH1, VZ_1)
      val s1 = s.drive(action)
      s1.pos shouldBe s.pos
      s1.speed shouldBe cyl(R225, VH1, VZ_1)
    }
    it("should decrease speeds when jetting max z") {
      val action = vector(R225, VH1, VZ_2)
      val s1 = s.drive(action)
      s1.pos shouldBe s.pos
      s1.speed shouldBe cyl(R225, VH1, VZ_2)
    }
    it("should change position when moving") {
      val s2 = status(
        pos = vector(0, 0, Z0),
        speed = cyl(R45, VH1, VZ1))
      val action = vector(R45, VH1, VZ1)
      val s1 = s2.drive(action)
      s1.pos shouldBe cyl(R45, VH1 * Dt, Z0 + VZ1 * Dt)
      s1.speed shouldBe s2.speed
    }
  }
}
