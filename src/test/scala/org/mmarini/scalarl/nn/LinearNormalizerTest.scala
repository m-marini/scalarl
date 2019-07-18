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
import io.circe.yaml.parser

class LinearNormalizerTest extends FunSpec with Matchers with GivenWhenThen {
  val Epsilon = 1e-6

  val JsonDoc = """---
type: LINEAR
offset:
- 0.5
- 0.5
scale:
- 2.0
- 2.0
"""

  describe("LinearNormalizer") {
    it("should normalize") {
      Given("an Normalizer")
      val normalizer = LinearNormalizer(
        offset = Nd4j.create(Array(-0.5, -0.5, -0.5)),
        scale = Nd4j.create(Array(2.0, 2.0, 2.0)))

      And("an input vector")
      val inputs = Nd4j.create(Array(0.0, 0.5, 1.0))

      When("normalizer")
      val newData = normalizer.normalize(inputs)

      Then("should result the normalized vector")
      val expected = Nd4j.create(Array(-1.0, 0.0, 1.0))
      newData shouldBe expected
    }

    it("should create a normalize") {
      Given("a Normalizer builder")
      val normalizer = Normalizer.minMax(
        min = Nd4j.create(Array(0.0, 0.0, 0.0)),
        max = Nd4j.create(Array(1.0, 1.0, 1.0)))

      Then("should result the offset")
      val offset = Nd4j.create(Array(-0.5, -0.5, -0.5))
      normalizer.offset shouldBe offset

      Then("should result the scala")
      val scale = Nd4j.create(Array(2.0, 2.0, 2.0))
      normalizer.scale shouldBe scale
    }

    it("should create a normalize 2") {
      Given("a Normalizer builder")
      val normalizer = Normalizer.minMax(3, 0.0, 1.0)

      Then("should result the offset")
      val offset = Nd4j.create(Array(-0.5, -0.5, -0.5))
      normalizer.offset shouldBe offset

      Then("should result the scala")
      val scale = Nd4j.create(Array(2.0, 2.0, 2.0))
      normalizer.scale shouldBe scale
    }

    it("should generate json") {
      Given("a Normalizer")
      val normalizer = Normalizer.minMax(
        min = Nd4j.create(Array(0.0, 0.0, 0.0)),
        max = Nd4j.create(Array(1.0, 1.0, 1.0)))

      When("create json")
      val json = normalizer.toJson

      Then("json should be object")
      json shouldBe 'isObject

      And("should have type LINEAR")
      json.hcursor.get[String]("type").toOption should contain("LINEAR")

      And("should have offset")
      json.hcursor.get[Array[Double]]("offset").toOption should contain(Array(-0.5, -0.5, -0.5))

      And("should have scale")
      json.hcursor.get[Array[Double]]("scale").toOption should contain(Array(2.0, 2.0, 2.0))
    }

    it("should create normalizer from json") {
      Given("a json doc")
      val doc = JsonDoc

      And("parsing it")
      val json = parser.parse(doc).right.get

      When("create Normalizer")
      val norm = Normalizer.fromJson(json)

      And("should be a linear normalizer")
      norm shouldBe a[LinearNormalizer]
      
      And("offset")
      norm.asInstanceOf[LinearNormalizer].offset shouldBe Nd4j.create(Array(0.5, 0.5))

      And("offset")
      norm.asInstanceOf[LinearNormalizer].offset shouldBe Nd4j.create(Array(0.5, 0.5))

      And("scale")
      norm.asInstanceOf[LinearNormalizer].scale shouldBe Nd4j.create(Array(2.0, 2.0))
    }
  }
}