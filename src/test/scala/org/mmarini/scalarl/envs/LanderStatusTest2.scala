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
import org.nd4j.linalg.indexing.NDArrayIndex

class LanderStatusTest2 extends FunSpec with Matchers with GivenWhenThen {
  val PosInterval = NDArrayIndex.interval(0, 3)
  val SpeedInterval = NDArrayIndex.interval(3, 6)
  val SquaredPosInterval = NDArrayIndex.interval(6, 9)
  val SquaredSpeedInterval = NDArrayIndex.interval(9, 12)
  val HPosDirInterval = NDArrayIndex.interval(12, 14)
  val SpeedDirInterval = NDArrayIndex.interval(14, 17)
  val NoLandInterval = NDArrayIndex.point(17)
  val VHNoLandInterval = NDArrayIndex.point(18)
  val VZNoLandInterval = NDArrayIndex.interval(19, 21)

  Nd4j.create()

  describe("The pos signals LanderStatus") {
    it("should result ... at (0,0,0) speed (0,0,0)") {
      Given("a lander status at (0,0,0) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 0)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be 0,0,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("speed signals should be 0,0,0")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared pos signals should be 0,0,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared speed signals should be 0,0,0")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("h pos dir should be 1,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1))

      And("speed dir should be 1,1,1")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("no landing should be 0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vh no landing should be 0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,1")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("should not be endup")
      obs should not be 'endUp
    }

    it("should result ... at (600,600,100) speed (0,0,0)") {
      Given("a lander status at (600,600,100) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](600, 600, 100)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be 1,1,1")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("speed signals should be 0,0,0")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared pos signals should be 1,1,1")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("squared speed signals should be 0,0,0")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("h pos dir should be 1,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1))

      And("speed dir should be 1,1,1")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("no landing should be 1")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](1))

      And("vh no landing should be 0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,1")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("should not be endup")
      obs should not be 'endUp
    }

    it("should result ... at (-600,-600,100) speed (0,0,0)") {
      Given("a lander status at (-600,-600,100) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](-600, -600, 100)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be -1,-1,1")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](-1, -1, 1))

      And("speed signals should be 0,0,0")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared pos signals should be 1,1,1")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("squared speed signals should be 0,0,0")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("h pos dir should be 0,0")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](0, 0))

      And("speed dir should be 1,1,1")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("x no landing should be 1")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](1))

      And("vh no landing should be 0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,1")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("should not be endup")
      obs should not be 'endUp
    }

    it("should result ... at (-300,300,50) speed (0,0,0)") {
      Given("a lander status at (-300,300,50) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](-300, 300, 50)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be -0.5,0.5,0.5")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](-0.5, 0.5, 0.5))

      And("speed signals should be 0,0,0")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared pos signals should be 0.25,0.25,0.25")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0.25, 0.25, 0.25))

      And("squared speed signals should be 0,0,0")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("h pos dir should be 0,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("speed dir should be 1,1,1")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("no landing should be 1")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](1))

      And("vh no landing should be 0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,1")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("should not be endup")
      obs should not be 'endUp
    }
  }

  describe("The speed signals LanderStatus") {
    it("should result ... at (0,0,0) speed (24,24,12)") {
      Given("a lander status at (0,0,0) speed (24,24,24)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 0)),
        speed = Nd4j.create(Array[Double](24, 24, 12)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be 0,0,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("speed signals should be 1,1,1")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("squared pos signals should be 0,0,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared speed signals should be 1,1,1")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("h pos dir should be 1,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1))

      And("speed dir should be 1,1,1")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("no landing should be 0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vh no landing should be 1")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](1))

      And("vz no landing should be 0,1")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("should not be endup")
      obs should not be 'endUp
    }

    it("should result ... at (0,0,0) speed (-24,-24,-12)") {
      Given("a lander status at (0,0,0) speed (-24,-24,-24)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 0)),
        speed = Nd4j.create(Array[Double](-24, -24, -12)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be 0,0,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("speed signals should be -1,-1,-1")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](-1, -1, -1))

      And("squared pos signals should be 0,0,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared speed signals should be 1,1,1")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("h pos dir should be 1,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1))

      And("speed dir should be 0,0,0")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("no landing should be 0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vh no landing should be 1")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](1))

      And("vz no landing should be 1,0")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](1, 0))

      And("should not be endup")
      obs should not be 'endUp
    }

    it("should result ... at (0,0,0) speed (-12,12,-6)") {
      Given("a lander status at (0,0,0) speed (-12,12,-6)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 0)),
        speed = Nd4j.create(Array[Double](-12, 12, -6)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be 0,0,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("speed signals should be -0.5,0.5,-0.5")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](-0.5, 0.5, -0.5))

      And("squared pos signals should be 0,0,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared speed signals should be 0.25,0.25,0.25")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0.25, 0.25, 0.25))

      And("h pos dir should be 1,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1))

      And("speed dir should be 0,1,0")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](0, 1, 0))

      And("no landing should be 0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vh no landing should be 1")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](1))

      And("vz no landing should be 1,0")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](1, 0))

      And("should not be endup")
      obs should not be 'endUp
    }
  }

  describe("The pos landing signals LanderStatus") {
    it("should result ... at (-10,0,0) speed (0,0,0)") {
      Given("a lander status at (-10,0,0) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](-10, 0, 0)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be -0.01667,0,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](-0.01667, 0, 0))

      And("speed signals should be 0,0,0")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared pos signals should be 0.0002778,0,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0.0002778, 0, 0))

      And("squared speed signals should be 0,0,0")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("h pos dir should be 0,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("speed dir should be 1,1,1")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("no landing should be 0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vh no landing should be 0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,1")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("should not be endup")
      obs should not be 'endUp
    }

    it("should result ... at (0,10,0) speed (0,0,0)") {
      Given("a lander status at (0,10,0) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 10, 0)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be 0,0.01667,0,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0.01667, 0))

      And("speed signals should be 0,0,0")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared pos signals should be 0,0.0002778,0,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0.0002778, 0))

      And("squared speed signals should be 0,0,0")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("h pos dir should be 1,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1))

      And("speed dir should be 1,1,1")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("no landing should be 0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vh no landing should be 0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,1")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("should not be endup")
      obs should not be 'endUp
    }

    it("should result ... at (7.07,-7.07,0) speed (0,0,0)") {
      Given("a lander status at (7.07,-7.07,0) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](7.07, -7.07, 0)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be 0.011783,-0.011783,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](0.011783, -0.011783, 0))

      And("speed signals should be 0,0,0")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared pos signals should be 0.000138847,0.000138847,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0.000138847, 0.000138847, 0))

      And("squared speed signals should be 0,0,0")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("h pos dir should be 1,0")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 0))

      And("speed dir should be 1,1,1")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("no landing should be 0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vh no landing should be 0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,1")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("should not be endup")
      obs should not be 'endUp
    }

    it("should result ... at (-12,12,0) speed (0,0,0)") {
      Given("a lander status at (-12,12,0) speed (0,0,0)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](-12, 12, 0)),
        speed = Nd4j.create(Array[Double](0, 0, 0)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be -0.02,0.02,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](-0.02, 0.02, 0))

      And("speed signals should be 0,0,0")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared pos signals should be 0.0004,0.0004,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0.0004, 0.0004, 0))

      And("squared speed signals should be 0,0,0")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("h pos dir should be 0,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("speed dir should be 1,1,1")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 1))

      And("x no landing should be 1,0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](1))

      And("vh no landing should be 0,0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,1")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 1))

      And("should not be endup")
      obs should not be 'endUp
    }
  }

  describe("The speed landing signals LanderStatus") {
    it("should result ... at (0,0,0) speed (-0.5,0,-4)") {
      Given("a lander status at (0,0,0) speed (-0.5,0,-4)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 0)),
        speed = Nd4j.create(Array[Double](-0.5, 0, -4)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be 0,0,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("speed signals should be -0.0208333,0,-0.333333")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](-0.0208333, 0, -0.333333))

      And("squared pos signals should be 0,0,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared speed signals should be 0.00043402,0,0.111111")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0.00043402, 0, 0.111111))

      And("h pos dir should be 1,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1))

      And("speed dir should be 0,1,0")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](0, 1, 0))

      And("no landing should be 0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vh no landing should be 0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,0")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 0))

      And("should not be endup")
      obs should not be 'endUp
    }

    it("should result ... at (0,0,0) speed (0,0.5,-0.12)") {
      Given("a lander status at (0,0,0) speed (0,0.5,-0.12)")
      val status = LanderStatus(
        pos = Nd4j.create(Array[Double](0, 0, 0)),
        speed = Nd4j.create(Array[Double](0, 0.5, -0.12)))

      When("get the observation")
      val obs = status.observation

      Then("pos signals should be 0,0,0")
      obs.signals.get(PosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("speed signals should be 0, 0.0208333,-0.01")
      obs.signals.get(SpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0.0208333, -0.01))

      And("squared pos signals should be 0,0,0")
      obs.signals.get(SquaredPosInterval) shouldBe
        Nd4j.create(Array[Double](0, 0, 0))

      And("squared speed signals should be 0,0.00043402,0.0001")
      obs.signals.get(SquaredSpeedInterval) shouldBe
        Nd4j.create(Array[Double](0, 0.00043402, 0.0001))

      And("h pos dir should be 1,1")
      obs.signals.get(HPosDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1))

      And("speed dir should be 1,1,0")
      obs.signals.get(SpeedDirInterval) shouldBe
        Nd4j.create(Array[Double](1, 1, 0))

      And("no landing should be 0")
      obs.signals.get(NoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vh no landing should be 0")
      obs.signals.get(VHNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0))

      And("vz no landing should be 0,0")
      obs.signals.get(VZNoLandInterval) shouldBe
        Nd4j.create(Array[Double](0, 0))

      And("should not be endup")
      obs should not be 'endUp
    }
  }
}

