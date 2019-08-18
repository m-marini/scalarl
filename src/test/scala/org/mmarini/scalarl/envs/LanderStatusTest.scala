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

class LanderStatusTest extends FunSpec with Matchers with GivenWhenThen {
  Nd4j.create()

  describe("A landed LanderStatus") {
    it("should result landed at (0,0,-0.1) speed (0,0,0)") {
      Given("a lander status at (0,0,-0.1) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (0,0,-0.1) speed (0,0,1)") {
      Given("a lander status at (0,0,-0.1) speed (0,0,1)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, 1)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (0,0,-0.1) speed (0,0,-4)") {
      Given("a lander status at (0,0,-0.1) speed (0,0,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (0,0,-0.1) speed (0.5,0,-4)") {
      Given("a lander status at (0,0,-0.1) speed (0.5,0,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0.5, 0, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (0,0,-0.1) speed (-0.5,0,-4)") {
      Given("a lander status at (0,0,-0.1) speed (-0.5,0,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](-0.5, 0, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (0,0,-0.1) speed (0,0.5,-4)") {
      Given("a lander status at (0,0,0) speed (0,0.5,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0.5, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (0,0,-0.1) speed (0,-0.5,-4)") {
      Given("a lander status at (0,0,-0.1) speed (0,-0.5,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0, -0.5, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (10,0,-0.1) speed (0,0,-4)") {
      Given("a lander status at (10,0,-0.1) speed (0,0,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](10, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (-10,0,-0.1) speed (0,0,-4)") {
      Given("a lander status at (-10,0,-0.1) speed (0,0,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](-10, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (0,10,-0.1) speed (0,0,-4)") {
      Given("a lander status at (0,10,-0.1) speed (0,0,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 10, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (0,-10,-0.1) speed (0,0,-4)") {
      Given("a lander status at (0,-10,-0.1) speed (0,0,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 10, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (7.07,7.07,-0.1) speed (0,0,-4)") {
      Given("a lander status at (7.07,7.07,-0.1) speed (0,0,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](7.07, 7.07, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result landed at (0,0,-0.1) speed (0.353,0.353,-4)") {
      Given("a lander status at (0,0,-0.1) speed (0.353,0.353,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0.353, 0.353, -4)))

      Then("should be landed")
      status shouldBe 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should not be crush")
      status should not be 'crush

      And("should be endUp")
      status.observation shouldBe 'endUp
    }
  }

  describe("A crush LanderStatus") {
    it("should result crush at (0,0,-0.1) speed (0,0,-4.1)") {
      Given("a lander status at (0,0,-0.1) speed (0,0,-4.1)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, -4.1)))

      Then("should be crush")
      status shouldBe 'crush

      And("should not be landed")
      status should not be 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result crush at (0,0,-0.1) speed (0.354,0.354,-4.1)") {
      Given("a lander status at (0,0,-0.1) speed (0.354,0.354,-4.1)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, -0.1)),
        speed = Nd4j.create(Array[Double](0.354, 0.354, -4.1)))

      Then("should be crush")
      status shouldBe 'crush

      And("should not be landed")
      status should not be 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result crush at (7.08,7-08,-0.1) speed (0, 0, 0)") {
      Given("a lander status at (7.08,7-08,-0.1) speed (0, 0, 0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](7.08, 7.08, -0.1)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      Then("should be crush")
      status shouldBe 'crush

      And("should not be landed")
      status should not be 'landed

      And("should not be out of range")
      status should not be 'outOfRange

      And("should be endUp")
      status.observation shouldBe 'endUp
    }
  }

  describe("An out of range LanderStatus") {
    it("should result out of range at (0,0,100.1) speed (0,0,0)") {
      Given("a lander status at (0,0,100.1) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 100.1)),
        speed = Nd4j.zeros(3))

      Then("should be out of range")
      status shouldBe 'outOfRange

      Then("should not be crush")
      status should not be 'crush

      And("should not be landed")
      status should not be 'landed

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result out of range at (600.1,0,10) speed (0,0,0)") {
      Given("a lander status at (600.1,0,10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](600.1, 0, 10)),
        speed = Nd4j.zeros(3))

      Then("should be out of range")
      status shouldBe 'outOfRange

      Then("should not be crush")
      status should not be 'crush

      And("should not be landed")
      status should not be 'landed

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result out of range at (-600.1,0,10) speed (0,0,0)") {
      Given("a lander status at (-600.1,0,10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](-600.1, 0, 10)),
        speed = Nd4j.zeros(3))

      Then("should be out of range")
      status shouldBe 'outOfRange

      Then("should not be crush")
      status should not be 'crush

      And("should not be landed")
      status should not be 'landed

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result out of range at (0, 600.1,10) speed (0,0,0)") {
      Given("a lander status at (0, 600.1,10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 600.1, 10)),
        speed = Nd4j.zeros(3))

      Then("should be out of range")
      status shouldBe 'outOfRange

      Then("should not be crush")
      status should not be 'crush

      And("should not be landed")
      status should not be 'landed

      And("should be endUp")
      status.observation shouldBe 'endUp
    }

    it("should result out of range at (0, -600.1,10) speed (0,0,0)") {
      Given("a lander status at (0, -600.1,10) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, -600.1, 10)),
        speed = Nd4j.zeros(3))

      Then("should be out of range")
      status shouldBe 'outOfRange

      Then("should not be crush")
      status should not be 'crush

      And("should not be landed")
      status should not be 'landed

      And("should be endUp")
      status.observation shouldBe 'endUp
    }
  }
}

