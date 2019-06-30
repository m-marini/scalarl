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
import org.scalacheck.Gen
import org.scalatest.GivenWhenThen
import io.circe.yaml.parser
import io.circe.ParsingFailure

class CirceTest extends FunSpec with GivenWhenThen with Matchers {
  val document = """---
a:
- 1
- 2
i: 1
s: aaaa
"""

  describe("Circe") {
    it("should parse yaml") {
      Given("a yaml document")
      val doc = document
      When("parse it")
      val json = parser.parse(doc).toTry.get

      Then("It should return an object")
      json.isObject shouldBe true
    }

    it("should traverse yaml") {
      Given("a yaml document")
      val doc = document

      And("parse it")
      val json = parser.parse(doc).toTry.get.asObject
      json should not be empty
      val obj = json.get

      When("traversing for array")
      val a = obj("a")

      And("traversing for integer")
      val i = obj("i")

      And("traversing for String")
      val s = obj("s")

      Then("array should contain 2 element")
      a should not be empty
      val aAsVector = a.flatMap(_.asArray)
      aAsVector should not be empty
      val v = aAsVector.get
      v should have length (2)

      And("first element should be 1")
      val e = v(0).asNumber
      e should not be empty
      e.flatMap(_.toInt) should contain(1)

      And("number should contain 1")
      i should not be empty
      val iAsInt = i.flatMap(_.asNumber)
      iAsInt should not be empty
      val iint = iAsInt.flatMap(_.toInt)
      iint should contain(1)

      And("string should contain aaaa")
      s should not be empty
      val sAsString = s.flatMap(_.asString)
      sAsString should not be empty
      sAsString should contain("aaaa")
    }

    it("should traverse yaml with cursor") {
      Given("a yaml document")
      val doc = document

      And("parse it")
      val jsonOpt = parser.parse(doc).toOption
      jsonOpt should not be empty
      val json = jsonOpt.get

      When("traversing for array")
      val a = json.hcursor.downField("a")

      And("traversing for integer")
      val i = json.hcursor.get[Int]("i")

      And("traversing for String")
      val s = json.hcursor.get[String]("s")

      Then("array should contain 2 element")
       a.as[Seq[Json]].right.get should have size(2)

      And("first element should be 1")
      a.downN(0).as[Double].toOption should contain(1)

      And("number should contain 1")
      i.toOption should contain(1)

      And("string should contain aaaa")
      s.toOption should contain("aaaa")
    }
  }
}