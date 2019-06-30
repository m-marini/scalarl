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

import io.circe.Json

class MSELossFunctionTest extends FunSpec with GivenWhenThen with Matchers {
  val Epsilon = 1e-6

  describe("MSELossFunctionTest") {
    it("should compute the delta for loss function with mask") {
      Given("a MSE loss function")
      val loss = MSELossFunction
      And("a initial data with 2 random output, mask, label")
      val outputs = Nd4j.create(Array(0.1, 0.9))
      val labels = Nd4j.create(Array(0.5, 0.3))
      val mask = Nd4j.create(Array(0.0, 1.0))
      val initialData = Map(
        "outputs" -> outputs,
        "mask" -> mask,
        "labels" -> labels)

      When("build a delta updater")
      val updater = loss.deltaBuilder.build
      And("appling to initial data")
      val result = updater(initialData)

      Then("should result the delta value")
      val delta = Nd4j.create(Array(0.0, 0.3 - 0.9))
      result.get("delta") should contain(delta)
    }

    it("should compute the loss for loss function with mask") {
      Given("a MSE loss function")
      val loss = MSELossFunction
      And("a initial data with 2 random output, mask, label")
      val outputs = Nd4j.create(Array(0.1, 0.9))
      val labels = Nd4j.create(Array(0.5, 0.3))
      val mask = Nd4j.create(Array(0.0, 1.0))
      val initialData = Map(
        "outputs" -> outputs,
        "mask" -> mask,
        "labels" -> labels)

      When("build a loss updater")
      val updater = loss.lossBuilder.build
      And("appling to initial data")
      val result = updater(initialData)

      Then("should result the delta value")
      val delta = Nd4j.create(Array(0.6 * 0.6))
      result.get("loss") should contain(delta)
    }

    it("should compute the delta for loss function without mask") {
      Given("a MSE loss function")
      val loss = MSELossFunction
      And("a initial data with 2 random output, mask, label")
      val outputs = Nd4j.create(Array(0.1, 0.9))
      val labels = Nd4j.create(Array(0.5, 0.3))
      val mask = Nd4j.ones(2)
      val initialData = Map(
        "outputs" -> outputs,
        "mask" -> mask,
        "labels" -> labels)

      When("build a delta updater")
      val updater = loss.deltaBuilder.build
      And("appling to initial data")
      val result = updater(initialData)

      Then("should result the delta value")
      val delta = Nd4j.create(Array(0.5 - 0.1, 0.3 - 0.9))
      result.get("delta") should contain(delta)
    }

    it("should compute the loss for loss function without mask") {
      Given("a MSE loss function")
      val loss = MSELossFunction
      And("a initial data with 2 random output, mask, label")
      val outputs = Nd4j.create(Array(0.1, 0.9))
      val labels = Nd4j.create(Array(0.5, 0.3))
      val mask = Nd4j.ones(2)
      val initialData = Map(
        "outputs" -> outputs,
        "mask" -> mask,
        "labels" -> labels)

      When("build a loss updater")
      val updater = loss.lossBuilder.build
      And("appling to initial data")
      val result = updater(initialData)

      Then("should result the delta value")
      val delta = Nd4j.create(Array(0.4 * 0.4 + 0.6 * 0.6))
      result.get("loss") should contain(delta)
    }

    it("should generate json") {
      Given("a MSE loss function")
      val loss = MSELossFunction

      When("create json")
      val json = loss.toJson

      Then("should be a string")
      json.isString shouldBe true

      And("should contain MSE")
      json.asString should contain("MSE")
    }

    it("should generate from json") {
      Given("a json string")
      val json = Json.fromString("MSE")

      When("create loss")
      val loss = LossFunction.fromJson(json)

      Then("should be a MSELossFunction")
      loss should be theSameInstanceAs (MSELossFunction)
    }
  }
}
