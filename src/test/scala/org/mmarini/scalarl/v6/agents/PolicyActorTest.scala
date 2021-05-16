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

package org.mmarini.scalarl.v6.agents

import org.mmarini.scalarl.v6.Configuration
import org.mmarini.scalarl.v6.Utils._
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class PolicyActorTest extends FunSpec with Matchers {
  create()

  private val PrefRange = create(Array(-3.0, 3.0)).transpose().broadcast(2, 5)
  private val ActionRange = create(Array(-1.0, 1.0)).transpose()
  private val Alpha = ones(1)

  def actor(): PolicyActor = {
    PolicyActor.apply(dimension = 0,
      noOutputs = 5,
      normalize = clipAndNormalize(PrefRange),
      denormalize = clipDenormalizeAndCenter(PrefRange),
      decode = encode(5, ActionRange),
      encode = decode(5, ActionRange),
      alpha = Alpha)
  }

  describe("PolicyActor") {
    it("should computeLabels") {
      val act = actor()

      val outputs = Array(
        zeros(1),
        zeros(5)
      )
      val actions = ones(1).muli(0.5)
      val delta = ones(1).muli(0.1)

      val map = act.computeLabels(outputs, actions, delta)

      map("h(0)") shouldBe create(Array(0.0, 0.0, 0.0, 0.0, 0.0))
      map("h*(0)") shouldBe create(Array(-0.02, -0.02, -0.02, 0.08, -0.02))
      map("labels(0)") shouldBe create(Array(-2.0 / 300, -2.0 / 300, -2.0 / 300, 8.0 / 300, -2.0 / 300))

      val map1 = act.computeLabels(outputs, actions, delta.neg())

      map1("h(0)") shouldBe create(Array(0.0, 0.0, 0.0, 0.0, 0.0))
      map1("h*(0)") shouldBe create(Array(0.02, 0.02, 0.02, -0.08, 0.02))
      map1("labels(0)") shouldBe create(Array(2.0 / 300, 2.0 / 300, 2.0 / 300, -8.0 / 300, 2.0 / 300))
    }
  }

  describe("PolicyActor") {
    it("should load from config") {
      val conf = Configuration.jsonFormString(
        """
          |---
          |actor:
          |    alpha: 0.1
          |    noValues: 5
          |    range:
          |        - [-1, 1]
          |    prefRange:
          |        - [-3, 3]
          |""".stripMargin)
      val actor = PolicyActor.fromJson(conf.hcursor.downField("actor"))(0).get
      actor.alpha shouldBe ones(1).muli(0.1)
    }
  }
}
