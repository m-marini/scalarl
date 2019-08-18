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

class MazeEnvTest2 extends FunSpec with Matchers with GivenWhenThen {

  def actionFormIndex(idx: Int): ChannelAction = {
    val action = Nd4j.zeros(8)
    action.putScalar(idx, 1)
    action
  }

  describe("A MazaEnv") {
    it("should change the subject position when step for N action") {
      Given("A maze environment with subject at (1,1)")
      val env = MazeEnv.fromStrings(
        List(
          "|   |",
          "| * |",
          "|   |",
          "|  O|"))
      And("a N action")
      val action = actionFormIndex(0)

      When("step the envinronment")
      val (result, _, _) = env.step(action)

      Then("the subject shouls be (0,1)")
      result.asInstanceOf[MazeEnv].subject shouldBe (MazePos(0, 1))
    }

    it("should change the subject position when step for NE action") {
      Given("A maze environment with subject at (1,1)")
      val env = MazeEnv.fromStrings(
        List(
          "|   |",
          "| * |",
          "|   |",
          "|  O|"))
      And("a NE action")
      val action = actionFormIndex(1)

      When("step the envinronment")
      val (result, _, _) = env.step(action)

      Then("the subject shouls be (0,2)")
      result.asInstanceOf[MazeEnv].subject shouldBe (MazePos(0, 2))
    }

    it("should change the subject position when step for E action") {
      Given("A maze environment with subject at (1,1)")
      val env = MazeEnv.fromStrings(
        List(
          "|   |",
          "| * |",
          "|   |",
          "|  O|"))
      And("a E action")
      val action = actionFormIndex(2)

      When("step the envinronment")
      val (result, _, _) = env.step(action)

      Then("the subject shouls be (1,2)")
      result.asInstanceOf[MazeEnv].subject shouldBe (MazePos(1, 2))
    }

    it("should change the subject position when step for SE action") {
      Given("A maze environment with subject at (1,1)")
      val env = MazeEnv.fromStrings(
        List(
          "|   |",
          "| * |",
          "|   |",
          "|  O|"))
      And("a SE action")
      val action = actionFormIndex(3)

      When("step the envinronment")
      val (result, _, _) = env.step(action)

      Then("the subject shouls be (2,2)")
      result.asInstanceOf[MazeEnv].subject shouldBe (MazePos(2, 2))
    }

    it("should change the subject position when step for S action") {
      Given("A maze environment with subject at (1,1)")
      val env = MazeEnv.fromStrings(
        List(
          "|   |",
          "| * |",
          "|   |",
          "|  O|"))
      And("a S action")
      val action = actionFormIndex(4)

      When("step the envinronment")
      val (result, _, _) = env.step(action)

      Then("the subject shouls be (2,1)")
      result.asInstanceOf[MazeEnv].subject shouldBe (MazePos(2, 1))
    }

    it("should change the subject position when step for SW action") {
      Given("A maze environment with subject at (1,1)")
      val env = MazeEnv.fromStrings(
        List(
          "|   |",
          "| * |",
          "|   |",
          "|  O|"))
      And("a SW action")
      val action = actionFormIndex(5)

      When("step the envinronment")
      val (result, _, _) = env.step(action)

      Then("the subject shouls be (2,0)")
      result.asInstanceOf[MazeEnv].subject shouldBe (MazePos(2, 0))
    }

    it("should change the subject position when step for W action") {
      Given("A maze environment with subject at (1,1)")
      val env = MazeEnv.fromStrings(
        List(
          "|   |",
          "| * |",
          "|   |",
          "|  O|"))
      And("a W action")
      val action = actionFormIndex(6)

      When("step the envinronment")
      val (result, _, _) = env.step(action)

      Then("the subject shouls be (1,0)")
      result.asInstanceOf[MazeEnv].subject shouldBe (MazePos(1, 0))
    }

    it("should change the subject position when step for NW action") {
      Given("A maze environment with subject at (1,1)")
      val env = MazeEnv.fromStrings(
        List(
          "|   |",
          "| * |",
          "|   |",
          "|  O|"))
      And("a NW action")
      val action = actionFormIndex(7)

      When("step the envinronment")
      val (result, _, _) = env.step(action)

      Then("the subject shouls be (0,0)")
      result.asInstanceOf[MazeEnv].subject shouldBe (MazePos(0, 0))
    }
  }
}
