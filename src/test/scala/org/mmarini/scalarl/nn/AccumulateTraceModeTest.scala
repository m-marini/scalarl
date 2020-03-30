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

import io.circe.yaml.parser
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FunSpec, GivenWhenThen, Matchers}

class AccumulateTraceModeTest extends FunSpec with GivenWhenThen with Matchers {
  val Gamma = 0.9
  val Lambda = 0.8
  val Epsilon = 1e-3

  Nd4j.create()

  val yamlDoc =
    """---
mode: ACCUMULATE
gamma: 0.9
lambda: 0.8
"""

  describe("AccumulateTraceMode") {
    it("should generate trace updater with no clear trace") {
      Given("an accumulate trace mode")
      val traceMode = AccumulateTraceMode(Lambda, Gamma)

      And("a layer data with feedback, trace and no clear trace")
      val feedback = Nd4j.create(Array(0.1, 0.2))
      val trace = Nd4j.create(Array(0.0, 0.3))
      val clearTrace = Nd4j.create(Array(1.0))
      val inputsData = Map(
        "l.feedback" -> feedback,
        "l.trace" -> trace,
        "noClearTrace" -> clearTrace)

      When("build a trace updater")
      val updater = traceMode.traceBuilder("l").build

      And("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the feedback updated")
      val expetcedFeedback = Nd4j.create(Array(0.1, 0.3 * 0.9 * 0.8 + 0.2))
      newData.get("l.feedback") should contain(expetcedFeedback)

      And("trace updated")
      newData.get("l.trace") should contain(expetcedFeedback)
    }

    it("should generate trace updater with clear trace") {
      Given("an accumulate trace mode")
      val traceMode = AccumulateTraceMode(Lambda, Gamma)

      And("a layer data with feedback, trace and clear trace")
      val feedback = Nd4j.create(Array(0.1, 0.2))
      val trace = Nd4j.create(Array(0.0, 0.3))
      val clearTrace = Nd4j.create(Array(0.0))
      val inputsData = Map(
        "l.feedback" -> feedback,
        "l.trace" -> trace,
        "noClearTrace" -> clearTrace)

      When("build a trace updater")
      val updater = traceMode.traceBuilder("l").build

      And("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the feedback updated")
      val expetcedFeedback = Nd4j.create(Array(0.1, 0.2))
      newData.get("l.feedback") should contain(expetcedFeedback)

      And("trace updated")
      newData.get("l.trace") should contain(expetcedFeedback)
    }

    it("should generate trace updater for no trace layer") {
      Given("an accumulate trace mode")
      val traceMode = AccumulateTraceMode(Lambda, Gamma)

      And("a layer data without trace")
      val inputsData: NetworkData = Map()

      When("build a trace updater")
      val updater = traceMode.traceBuilder("l").build

      And("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the initial layer")
      newData should be theSameInstanceAs (inputsData)
    }

    it("should generate an yaml document") {
      Given("an accumulate trace mode")
      val traceMode = AccumulateTraceMode(Lambda, Gamma)

      When("convert to json")
      val json = traceMode.toJson

      Then("json should be object")
      json.isObject shouldBe true

      And("should contain mode ACCUMULATE")
      json.asObject.flatMap(_ ("mode")).flatMap(_.asString) should contain("ACCUMULATE")

      And("should contain gamma Gamma")
      json.asObject.flatMap(_ ("gamma")).flatMap(_.asNumber).map(_.toDouble) should contain(Gamma)

      And("should contain lambda Lambda")
      json.asObject.flatMap(_ ("lambda")).flatMap(_.asNumber).map(_.toDouble) should contain(Lambda)
    }

    it("should generate trace mode from yaml") {
      Given("an yaml doc")
      val doc = yamlDoc
      And("parsing it")
      val json = parser.parse(doc).right.get

      When("convert to trace mode")
      val mode = TraceMode.fromJson(json)

      Then("mode should be a Accumulate trace")
      mode shouldBe a[AccumulateTraceMode]

      And("should have gamma Gamma")
      mode.asInstanceOf[AccumulateTraceMode].gamma shouldBe Gamma

      And("should have lambda Lambda")
      mode.asInstanceOf[AccumulateTraceMode].lambda shouldBe Lambda
    }
  }
}
