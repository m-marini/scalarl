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

import io.circe.yaml.parser

class NoneTraceModeTest extends FunSpec with GivenWhenThen with Matchers {
  val Epsilon = 1e-3

  Nd4j.create()

  val yamlDoc = """---
mode: NONE
"""

  describe("NoneTraceMode") {
    it("should generate an yaml document") {
      Given("a none trace mode")
      val traceMode = NoneTraceMode

      When("convert to json")
      val json = traceMode.toJson

      Then("json should be object")
      json.isObject shouldBe true

      And("should contain mode NONE")
      json.asObject.flatMap(_("mode")).flatMap(_.asString) should contain("NONE")
    }

    it("should generate trace mode from yaml") {
      Given("an yaml doc")
      val doc = yamlDoc
      And("parsing it")
      val json = parser.parse(doc).right.get

      When("convert to trace mode")
      val mode = TraceMode.fromJson(json)

      Then("mode should be a none trace")
      mode should be theSameInstanceAs (NoneTraceMode)
    }
  }
}
