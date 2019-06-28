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

class ActivationFunctionTest extends FunSpec with Matchers with GivenWhenThen {
  val Epsilon = 1e-6

  describe("TanhActivationFunction") {
    it("should compute the activation value") {
      Given("an Activation Function")
      val tanh = TanhActivationFunction

      And("an initial layer data with 3 inputs")
      val inputs = Nd4j.create(Array(-1.0, 0.0, 1.0))
      val inputData = Map("inputs" -> inputs)

      When("build a activation updater")
      val updater = tanh.buildActivation

      And("apply it to initial layer")
      val newData = updater(inputData)

      Then("should result the layer with activated outputs")
      val expected = Nd4j.create(Array(math.tanh(-1), math.tanh(0), math.tanh(1)))
      newData.get("outputs") should contain(expected)
    }
  }

  describe("TanhActivationFunction") {
    it("should backward delta") {
      Given("an Activation Function")
      val tanh = TanhActivationFunction

      And("an input layer data with 3 outputs and delta")
      val outputs = Nd4j.create(Array(math.tanh(-1), math.tanh(0), math.tanh(1)))
      val delta = Nd4j.create(Array(0.1, -0.1, 0.2))
      val inputData = Map(
        "outputs" -> outputs,
        "delta" -> delta)

      When("build a delta updater")
      val updater = tanh.buildDelta

      And("apply it to initial layer")
      val newData = updater(inputData)

      Then("should result the layer with inputDelta")
      val expected = Nd4j.create(Array(
        (1 - math.tanh(-1) * math.tanh(-1)) * 0.1,
        -0.1,
        (1 - math.tanh(1) * math.tanh(1)) * 0.2))
      newData.get("inputDelta") should contain(expected)
    }
  }
}
