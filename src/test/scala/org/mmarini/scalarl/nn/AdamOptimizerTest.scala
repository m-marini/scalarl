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

class AdamOptimizerTest extends FunSpec with GivenWhenThen with Matchers {
  val Alpha = 0.1
  val Beta1 = 0.9
  val Beta2 = 0.999
  val Epsilon = 1e-3

  Nd4j.create()

  val yamlDoc = """---
mode: ADAM
alpha: 0.1
beta1: 0.9
beta2: 0.999
epsilon: 0.001
"""
  describe("AdamOptimizer") {
    it("should generate optimizer updater") {

      Given("a adam optimizer")
      val opt = AdamOptimizer(Alpha, Beta1, Beta2, Epsilon)

      And("a layer data with gradients, m1, m2")
      val gradient = Nd4j.create(Array(-0.1, 0.2))
      val m1 = Nd4j.create(Array(-0.2, 0.4))
      val m2 = Nd4j.create(Array(0.3, 0.5))
      val inputsData = Map(
        "l.gradient" -> gradient,
        "l.m1" -> m1,
        "l.m2" -> m2)

      When("build a optimizer updater")
      val updater = opt.optimizeBuilder("l").build

      And("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the feedback")
      // g = | -0.1 0.2|
      // g2 = | 0.01 0.04 |
      // m1 = | -0.2 0.4 |
      // m2 = | 0.3 0.4 |
      // m1' = M1 beta1 + g (1 - beta1) = | -0.19   0.38 |
      // m2' = M2 beta2 + g2 (1 - bet2) = | 0.29971   0.49954 |

      val feedback = Nd4j.create(Array(-0.010974, 0.017001))
      newData.get("l.feedback") should contain(feedback)

      val newM1 = Nd4j.create(Array(-0.19, 0.38))
      newData.get("l.m1") should contain(newM1)

      val newM2 = Nd4j.create(Array(0.29971, 0.49954))
      newData.get("l.m2") should contain(newM2)
    }

    it("should generate null optimizer updater for no parametered layer") {

      Given("a adam optimizer")
      val opt = AdamOptimizer(Alpha, Beta1, Beta2, Epsilon)

      And("a layer data without gradients")
      val gradient = Nd4j.create(Array(1.0, 2.0))
      val inputsData: NetworkData = Map()

      When("build a optimizer updater")
      val updater = opt.optimizeBuilder("l").build

      And("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the feedback")
      newData should be theSameInstanceAs (inputsData)
    }

    it("should generate json") {
      Given("a adam optimizer")
      val opt = AdamOptimizer(Alpha, Beta1, Beta2, Epsilon)

      When("build json")
      val json = opt.toJson

      Then("json should be on object")
      json shouldBe 'isObject

      And("should contain mode ADAM")
      json.asObject.flatMap(_("mode")).flatMap(_.asString) should contain("ADAM")

      And(s"should contain alpha ${Alpha}")
      json.asObject.flatMap(_("alpha")).flatMap(_.asNumber).map(_.toDouble) should contain(Alpha)

      And(s"should contain beta1 ${Beta1}")
      json.asObject.flatMap(_("beta1")).flatMap(_.asNumber).map(_.toDouble) should contain(Beta1)

      And(s"should contain beta2 ${Beta2}")
      json.asObject.flatMap(_("beta2")).flatMap(_.asNumber).map(_.toDouble) should contain(Beta2)

      And(s"should contain epsilon ${Epsilon}")
      json.asObject.flatMap(_("epsilon")).flatMap(_.asNumber).map(_.toDouble) should contain(Epsilon)
    }

    it("should generate from json") {
      Given("a yaml doc")
      val doc = yamlDoc

      And("parse it")
      val json = parser.parse(doc).right.get

      When("build from json")
      val opt = Optimizer.fromJson(json)

      Then("should be a SGD optimizer")
      opt shouldBe a[AdamOptimizer]

      And(s"alpha should be ${Alpha}")
      opt.asInstanceOf[AdamOptimizer].alpha shouldBe Alpha

      And(s"beta1 should be ${Beta1}")
      opt.asInstanceOf[AdamOptimizer].beta1 shouldBe Beta1

      And(s"beta2 should be ${Beta2}")
      opt.asInstanceOf[AdamOptimizer].beta2 shouldBe Beta2

      And(s"epsilon should be ${Epsilon}")
      opt.asInstanceOf[AdamOptimizer].epsilon shouldBe Epsilon
    }
  }
}
