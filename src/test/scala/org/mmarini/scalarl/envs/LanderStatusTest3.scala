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

package org.mmarini.scalarl.envs

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FunSpec
import org.scalatest.Matchers
import org.scalatest.prop.PropertyChecks
import org.mmarini.scalarl.ChannelAction
import org.scalatest.GivenWhenThen

class LanderStatusTest3 extends FunSpec with Matchers with GivenWhenThen {
  Nd4j.create()

  describe("An LanderStatus stepping") {
    it("should step for max z jet") {
      Given("a lander status at (0, 0, 10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 10)),
        speed = Nd4j.zeros(3))

      And("an action with max z jet")
      val action = Nd4j.zeros(15)
      action.putScalar(2, 1)
      action.putScalar(7, 1)
      action.putScalar(14, 1)

      When("step")
      val (env, obs, reward) = status.step(action)

      Then("newStatus should have speed (0,0,1.6*0.25=0.4)")
      val expSpeed = Nd4j.create(Array[Double](0, 0, 0.4))
      env.asInstanceOf[LanderStatus].speed shouldBe expSpeed

      And("newStatus should be at(0,0,10+0.4*0.25=10.1)")
      val expPos = Nd4j.create(Array[Double](0, 0, 10.1))
      env.asInstanceOf[LanderStatus].pos shouldBe expPos

      And("reward should be -(10.1^2)/10000")
      reward shouldBe -(10.1 * 10.1) / 10000 +- 10e-6
    }

    it("should step for min z jet") {
      Given("a lander status at (0, 0, 10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 10)),
        speed = Nd4j.zeros(3))

      And("an action with max z jet")
      val action = Nd4j.zeros(15)
      action.putScalar(2, 1)
      action.putScalar(7, 1)
      action.putScalar(10, 1)

      When("step")
      val (env, obs, reward) = status.step(action)

      Then("newStatus should have speed (0,0,-1.6*0.25=-0.4)")
      val expSpeed = Nd4j.create(Array[Double](0, 0, -0.4))
      env.asInstanceOf[LanderStatus].speed shouldBe expSpeed

      And("newStatus should be at(0,0,10-0.4*0.25=9.9)")
      val expPos = Nd4j.create(Array[Double](0, 0, 9.9))
      env.asInstanceOf[LanderStatus].pos shouldBe expPos

      And("reward should be -(9.9^2)/10000")
      reward shouldBe -(9.9 * 9.9) / 10000 +- 10e-6
    }

    it("should step for mid z jet") {
      Given("a lander status at (0, 0, 10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 10)),
        speed = Nd4j.zeros(3))

      And("an action with max z jet")
      val action = Nd4j.zeros(15)
      action.putScalar(2, 1)
      action.putScalar(7, 1)
      action.putScalar(12, 1)

      When("step")
      val (env, obs, reward) = status.step(action)

      Then("newStatus should have speed (0,0,0)")
      val expSpeed = Nd4j.create(Array[Double](0, 0, 0))
      env.asInstanceOf[LanderStatus].speed shouldBe expSpeed

      And("newStatus should be at(0,0,10)")
      val expPos = Nd4j.create(Array[Double](0, 0, 10))
      env.asInstanceOf[LanderStatus].pos shouldBe expPos

      And("reward should be -(10^2)/10000")
      reward shouldBe -100.0 / 10000
    }

    it("should step for min x jet") {
      Given("a lander status at (0, 0, 10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 10)),
        speed = Nd4j.zeros(3))

      And("an action with min x jet")
      val action = Nd4j.zeros(15)
      action.putScalar(0, 1)
      action.putScalar(7, 1)
      action.putScalar(12, 1)

      When("step")
      val (env, obs, reward) = status.step(action)

      Then("newStatus should have speed (-1*0.25=-0.25,0,0)")
      val expSpeed = Nd4j.create(Array[Double](-0.25, 0, 0))
      env.asInstanceOf[LanderStatus].speed shouldBe expSpeed

      And("newStatus should be at(-0.25*0.25=-0.0625,0,10)")
      val expPos = Nd4j.create(Array[Double](-0.0625, 0, 10))
      env.asInstanceOf[LanderStatus].pos shouldBe expPos

      And("reward should be -(10^2 + 0.0625^2)/10000")
      reward shouldBe -(100 + 0.0625 * 0.0625) / 10000
    }

    it("should step for max x jet") {
      Given("a lander status at (0, 0, 10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 10)),
        speed = Nd4j.zeros(3))

      And("an action with max x jet")
      val action = Nd4j.zeros(15)
      action.putScalar(4, 1)
      action.putScalar(7, 1)
      action.putScalar(12, 1)

      When("step")
      val (env, obs, reward) = status.step(action)

      Then("newStatus should have speed (1*0.25=0.25,0,0)")
      val expSpeed = Nd4j.create(Array[Double](0.25, 0, 0))
      env.asInstanceOf[LanderStatus].speed shouldBe expSpeed

      And("newStatus should be at(0.25*0.25=0.0625,0,10)")
      val expPos = Nd4j.create(Array[Double](0.0625, 0, 10))
      env.asInstanceOf[LanderStatus].pos shouldBe expPos

      And("reward should be -(10^2 + 0.0625^2)/10000")
      reward shouldBe -(100 + 0.0625 * 0.0625) / 10000
    }

    it("should step for min y jet") {
      Given("a lander status at (0, 0, 10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 10)),
        speed = Nd4j.zeros(3))

      And("an action with min y jet")
      val action = Nd4j.zeros(15)
      action.putScalar(2, 1)
      action.putScalar(5, 1)
      action.putScalar(12, 1)

      When("step")
      val (env, obs, reward) = status.step(action)

      Then("newStatus should have speed (0,-1*0.25=-0.25,0,0)")
      val expSpeed = Nd4j.create(Array[Double](0, -0.25, 0))
      env.asInstanceOf[LanderStatus].speed shouldBe expSpeed

      And("newStatus should be at(0,-0.25*0.25=-0.0625,10)")
      val expPos = Nd4j.create(Array[Double](0, -0.0625, 10))
      env.asInstanceOf[LanderStatus].pos shouldBe expPos

      And("reward should be -(10^2 + 0.0625^2)/10000")
      reward shouldBe -(100 + 0.0625 * 0.0625) / 10000
    }

    it("should step for max y jet") {
      Given("a lander status at (0, 0, 10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 10)),
        speed = Nd4j.zeros(3))

      And("an action with max y jet")
      val action = Nd4j.zeros(15)
      action.putScalar(2, 1)
      action.putScalar(9, 1)
      action.putScalar(12, 1)

      When("step")
      val (env, obs, reward) = status.step(action)

      Then("newStatus should have speed (0,1*0.25=0.25,0)")
      val expSpeed = Nd4j.create(Array[Double](0, 0.25, 0))
      env.asInstanceOf[LanderStatus].speed shouldBe expSpeed

      And("newStatus should be at(0,0.25*0.25=0.0625,10)")
      val expPos = Nd4j.create(Array[Double](0, 0.0625, 10))
      env.asInstanceOf[LanderStatus].pos shouldBe expPos

      And("reward should be -(10^2 + 0.0625^2)/10000")
      reward shouldBe -(100 + 0.0625 * 0.0625) / 10000
    }
  }

  describe("An LanderStatus stepping for rewards") {
    it("should step to landed reward") {
      Given("a lander status at (0, 0, 0) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 0)),
        speed = Nd4j.zeros(3))

      And("an action with min z jet")
      val action = Nd4j.zeros(15)
      action.putScalar(2, 1)
      action.putScalar(7, 1)
      action.putScalar(11, 1)

      When("step")
      val (_, _, reward) = status.step(action)

      Then(s"reward should be ${status.conf.landedReward}")
      reward shouldBe status.conf.landedReward
    }

    it("should step to out of range reward") {
      Given("a lander status at (0, 0, 100) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 100)),
        speed = Nd4j.zeros(3))

      And("an action with max z jet")
      val action = Nd4j.zeros(15)
      action.putScalar(2, 1)
      action.putScalar(7, 1)
      action.putScalar(14, 1)

      When("step")
      val (_, _, reward) = status.step(action)

      Then(s"reward should be ${status.conf.outOfRangeReward}")
      reward shouldBe status.conf.outOfRangeReward
    }

    it("should step to crash reward") {
      Given("a lander status at (10, 10, 0) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](10, 10, 0)),
        speed = Nd4j.zeros(3))

      Then("an action with min z jet")
      val action = Nd4j.zeros(15)
      action.putScalar(2, 1)
      action.putScalar(7, 1)
      action.putScalar(10, 1)

      When("step")
      val (_, _, reward) = status.step(action)

      And(s"reward should be ${status.conf.crashReward}")
      reward shouldBe status.conf.crashReward
    }
  }
}

