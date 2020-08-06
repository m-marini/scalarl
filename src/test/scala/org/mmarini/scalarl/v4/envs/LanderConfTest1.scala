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
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

import scala.math._

class LanderConfTest1 extends FunSpec with Matchers {
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

  create()

  private def encoder = new LanderContinuousEncoder(
    Utils.normalize(
      create(Array(
        Array(-1.0, -1.0, 0.0, 0.0, 0.0, -12.0),
        Array(1.0, 1.0, 10.0, 10.0, 24.0, 12.0)
      ))))

  val conf: LanderConf = new LanderConf(
    dt = ones(1).mul(Dt),
    fuel = ones(1).muli(DefaultFuel),
    spaceRange = zeros(2L, 6L),
    landingRadius = zeros(1),
    landingSpeed = create(Array(LandingVH, -LandingVZ)),
    optimalSpeed = zeros(2),
    g = ones(1).mul(G),
    initialLocationTrans = Utils.normalize01(zeros(2L, 3L)),
    jetAccRange = create(Array(
      Array(-JetHAcc, -JetHAcc, 0.0),
      Array(JetHAcc, JetHAcc, JetVAcc)
    )),
    landedReward = zeros(1),
    landedOutOfPlatformReward = zeros(1),
    vCrashedOnPlatformReward = zeros(1),
    hCrashedOnPlatformReward = zeros(1),
    hCrashedOutOfPlatformReward = zeros(1),
    vCrashedOutOfPlatformReward = zeros(1),
    outOfRangeReward = zeros(1),
    outOfFuelReward = zeros(1),
    flyingReward = zeros(1),
    directionReward = zeros(1),
    hSpeedReward = zeros(1),
    vSpeedReward = zeros(1),
    encoder = encoder
  )

  //private val ActionJet: INDArray = create(Array(0.0, 1.0,))
  private val ActionMax: INDArray = create(Array(1.0, 2.0, 4.0))
  private val ActionSED: INDArray = create(Array(5.0, 1.0, 1.0))
  private val ActionSEDMax: INDArray = create(Array(5.0, 2.0, 0.0))

  private def vector(x: Double, y: Double, z: Double) = create(Array(x, y, z))

  private def cyl(angle: Double, radius: Double, z: Double) = create(
    Array(radius * cos(angle), radius * sin(angle), z))

  describe("LanderConf drive") {
    val pos = vector(0, 0, Z0)
    val speed = vector(0, 0, 0)
    it("should remain in the position when hovering") {
      val action = vector(R0, VH0, VZ0)
      val (p, v) = conf.drive(action, pos, speed)
      p shouldBe pos
      v shouldBe speed
    }
    it("should increase speeds when jetting") {
      val action = vector(R45, VH1, VZ1)
      val (p, v) = conf.drive(action, pos, speed)
      p shouldBe pos
      v shouldBe cyl(R45, VH1, VZ1)
    }
    it("should increase speeds when jetting max") {
      val action = vector(R45, VH2, VZ2)
      val (p, v) = conf.drive(action, pos, speed)
      p shouldBe pos
      v shouldBe cyl(R45, VH2, VZ2)
    }
    it("should decrease speeds when jetting z") {
      val action = vector(R225, VH1, VZ_1)
      val (p, v) = conf.drive(action, pos, speed)
      p shouldBe pos
      v shouldBe cyl(R225, VH1, VZ_1)
    }
    it("should decrease speeds when jetting max z") {
      val action = vector(R225, VH1, VZ_2)
      val (p, v) = conf.drive(action, pos, speed)
      p shouldBe pos
      v shouldBe cyl(R225, VH1, VZ_2)
    }
    it("should change position when moving") {
      val speed = cyl(R45, VH1, VZ1)
      val action = vector(R45, VH1, VZ1)
      val (p, v) = conf.drive(action, pos, speed)
      p shouldBe cyl(R45, VH1 * Dt, Z0 + VZ1 * Dt)
      v shouldBe speed
    }
  }
}
