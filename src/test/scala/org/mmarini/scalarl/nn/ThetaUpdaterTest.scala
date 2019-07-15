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

package org.mmarini.scalarl.nn

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FunSpec
import org.scalatest.GivenWhenThen
import org.scalatest.Matchers

class ThetaUpdaterTest extends FunSpec with GivenWhenThen with Matchers {
  val Epsilon = 1e-3

  Nd4j.create()

  describe("ThetaUpdaterTest") {
    it("should generate updated theta") {
      Given("an theta updater")
      val updater = OperationBuilder.thetaBuilder("l").build

      And("a layer data with feedback and theta")
      val feedback = Nd4j.create(Array(0.1, 0.2))
      val theta = Nd4j.create(Array(0.3, 0.4))
      val thetaDelta = Nd4j.create(Array(0.5, 0.5))
      val inputsData = Map(
        "l.feedback" -> feedback,
        "l.thetaDelta" -> thetaDelta,
        "l.theta" -> theta)

      When("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the theta updated")
      val expetcedTheta = Nd4j.create(Array(0.35, 0.5))
      newData.get("l.theta") should contain(expetcedTheta)
    }

    it("should generate nothing if no feedback provided") {
      Given("an theta updater")
      val updater = OperationBuilder.thetaBuilder("l").build

      And("a layer data without feedback")
      val inputsData: NetworkData = Map()

      When("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the feedback updated")
      newData should be theSameInstanceAs (inputsData)
    }
  }
}
