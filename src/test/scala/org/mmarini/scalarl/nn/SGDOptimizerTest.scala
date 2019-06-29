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

import org.scalatest.FunSpec
import org.scalatest.Matchers
import org.scalatest.prop.PropertyChecks

import io.circe.Json
import io.circe.yaml
import io.circe.yaml.syntax.AsYaml
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.PropSpec
import org.scalacheck.Gen
import org.scalatest.GivenWhenThen

class SGDOptimizerTest extends FunSpec with GivenWhenThen with Matchers {
  val Epsilon = 1e-6
  val Alpha = 0.1

  Nd4j.create()

  describe("SGDOptimizer") {
    it("should generate optimizer updater") {

      Given("a sgd optimizer")
      val opt = SGDOptimizer(Alpha)

      And("a layer data with gradients")
      val gradient = Nd4j.create(Array(1.0, 2.0))
      val inputsData = Map("l.gradient" -> gradient)

      When("build a optimizer updater")
      val updater = opt.buildOptimizer("l")

      And("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the feedback")
      val feedback = Nd4j.create(Array(0.1, 0.2))
      newData.get("l.feedback") should contain(feedback)
    }

    it("should generate null optimizer updater for no parametered layer") {

      Given("a sgd optimizer")
      val opt = SGDOptimizer(Alpha)

      And("a layer data without gradients")
      val gradient = Nd4j.create(Array(1.0, 2.0))
      val inputsData: NetworkData = Map()

      When("build a optimizer updater")
      val updater = opt.buildOptimizer("l")

      And("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the feedback")
      newData should be theSameInstanceAs (inputsData)
    }
  }
}
