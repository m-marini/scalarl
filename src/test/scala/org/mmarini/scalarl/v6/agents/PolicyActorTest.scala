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
import org.nd4j.linalg.ops.transforms.Transforms._
import org.scalatest.{FunSpec, Matchers}

class PolicyActorTest extends FunSpec with Matchers {
  create()

  private val PrefRange = create(Array(-3.0, 3.0)).transpose().broadcast(2, 5)
  private val ActionRange = create(Array(-1.0, 1.0)).transpose()
  private val Alpha = ones(1)
  private val EpsilonH = 0.6
  private val AlphaDecay = 0.9

  def actor(): PolicyActor = {
    PolicyActor.apply(dimension = 0,
      noOutputs = 5,
      alphaDecay = AlphaDecay,
      epsilonH = ones(1).muli(EpsilonH),
      normalize = clipAndNormalize(PrefRange),
      denormalize = clipDenormalizeAndCenter(PrefRange),
      decode = encode(5, ActionRange),
      encode = decode(5, ActionRange))
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

      val map = act.computeLabels(outputs, actions, delta, Alpha)

      val H0 = create(Array(0.0, 0.0, 0.0, 0.0, 0.0))
      val H0Star = create(Array(-0.02, -0.02, -0.02, 0.08, -0.02))
      val DeltaH0 = H0Star.sub(H0)
      val DeltaHAvg = sqrt(pow(DeltaH0, 2).mean())
      val A1 = ones(1).muli(EpsilonH).divi(DeltaHAvg)
      val AS = Alpha.mul(AlphaDecay).addi(A1.mul(1 - AlphaDecay))

      map("h(0)") shouldBe H0
      map("h*(0)") shouldBe H0Star
      map("deltaH(0)") shouldBe DeltaH0
      map("labels(0)") shouldBe create(Array(-2.0 / 300, -2.0 / 300, -2.0 / 300, 8.0 / 300, -2.0 / 300))
      map("alpha*(0)") shouldBe AS

      val map1 = act.computeLabels(outputs, actions, delta.neg(), Alpha)

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
      val (actor, alpha) = PolicyActor.fromJson(conf.hcursor.downField("actor"))(0).get
      alpha shouldBe ones(1).muli(0.1)
      actor.noOutputs shouldBe 5
      actor.alphaDecay shouldBe 1.0
      actor.epsilonH shouldBe ones(1).muli(0.6)
    }

    it("should load from config with dynamic") {
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
          |    alphaDecay: 0.9
          |    epsilon: 0.01
          |""".stripMargin)
      val (actor, alpha) = PolicyActor.fromJson(conf.hcursor.downField("actor"))(0).get
      alpha shouldBe ones(1).muli(0.1)
      actor.noOutputs shouldBe 5
      actor.alphaDecay shouldBe 0.9
      actor.epsilonH shouldBe ones(1).muli(0.06)
    }
  }
}
