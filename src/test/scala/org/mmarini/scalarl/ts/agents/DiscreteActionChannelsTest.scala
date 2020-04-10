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

package org.mmarini.scalarl.ts.agents

import org.mmarini.scalarl.ts.DiscreteActionChannels
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FunSpec, Matchers}

class DiscreteActionChannelsTest extends FunSpec with Matchers {
  val Index4 = 4
  val Size = 6

  Nd4j.create()

  describe("DiscreteActionChannels") {
    val channels = DiscreteActionChannels(Array(1, 2, 3))

    it("should create indices") {
      val inter = channels.indices
      inter should have size 3
      inter(0) shouldBe(0, 0)
      inter(1) shouldBe(1, 2)
      inter(2) shouldBe(3, 5)
    }

    it("should create channel intervals") {
      val inter = channels.intervals
      inter should have size 3
      val data = Nd4j.create(Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
      data.get(inter(0)) shouldBe Nd4j.create(Array(0.0))
      data.get(inter(1)) shouldBe Nd4j.create(Array(1.0, 2.0))
      data.get(inter(2)) shouldBe Nd4j.create(Array(3.0, 4.0, 5.0))
    }

    it("should create action") {
      val data = channels.action(0, 0, 1)
      data shouldBe Nd4j.create(Array(1.0, 1.0, 0.0, 0.0, 1.0, 0.0))
    }

    it("should get indices") {
      val data = channels.notZeroIndices(Nd4j.create(Array(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)))
      data shouldBe Seq(0, 1, Index4)
    }

    it("should get action indices") {
      val data = channels.actions(Nd4j.create(Array(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)))
      data shouldBe Seq(0, 0, 1)
    }

    it("should generate error for wrong features") {
      val thrown = the[IllegalArgumentException] thrownBy {
        channels.actions(Nd4j.create(Array(1.0, 0.0, 0.0, 0.0, 1.0, 1.0)))
      }
      thrown.getMessage shouldBe "requirement failed: action value 4 should be between 1 to 2"
    }

    it("should generate error for wrong number") {
      val thrown = the[IllegalArgumentException] thrownBy {
        channels.actions(Nd4j.create(Array(1.0, 1.0, 0.0, 0.0, 1.0, 1.0)))
      }
      thrown.getMessage shouldBe "requirement failed: number of actions [4] should be 3"
    }

    it("should generate error for wrong size") {
      val thrown = the[IllegalArgumentException] thrownBy {
        channels.actions(Nd4j.create(Array(1.0, 1.0, 0.0, 0.0, 1.0)))
      }
      thrown.getMessage shouldBe "requirement failed: action length [5] should be 6"
    }

    it("should generate a random action") {
      val mask = Nd4j.ones(Size)
      val random = Nd4j.getRandomFactory.getNewRandomInstance(1L)
      val data = channels.random(mask, random)
      data shouldBe Nd4j.create(Array(1.0, 0.0, 1.0, 0.0, 0.0, 1.0))
    }

    it("should generate a random action with mask") {
      val mask = Nd4j.create(Array(1.0, 1.0, 0.0, 1.0, 1.0, 0.0))
      val random = Nd4j.getRandomFactory.getNewRandomInstance(1L)
      val data = channels.random(mask, random)
      data shouldBe Nd4j.create(Array(1.0, 1.0, 0.0, 1.0, 0.0, 0.0))
    }

    it("should get status value for channels") {
      val mask = Nd4j.ones(Size)
      val policy = Nd4j.create(Array(1.5, 1.0, 2.0, -1.0, 2.5, 1.0))
      val (data, idx) = channels.statusValue(policy, mask)

      data shouldBe Nd4j.create(Array(1.5, 2.0, 2.5))
      idx shouldBe Array(0, 1, 1)
    }

    it("should get status value for channels with mask") {
      val mask = Nd4j.create(Array(1.0, 1.0, 0.0, 1.0, 0.0, 1.0))
      val policy = Nd4j.create(Array(1.5, 1.0, 2.0, -1.0, 2.5, 1.0))
      val (data, idx) = channels.statusValue(policy, mask)

      data shouldBe Nd4j.create(Array(1.5, 1.0, 1.0))
      idx shouldBe Array(0, 0, 2)
    }

    it("should get greedyAction") {
      val mask = Nd4j.ones(Size)
      val policy = Nd4j.create(Array(1.5, 1.0, 2.0, -1.0, 2.5, 1.0))
      val data = channels.greedyAction(policy, mask)
      data shouldBe Nd4j.create(Array(1.0, 0.0, 1.0, 0.0, 1.0, 0.0))
    }

    it("should get greedyAction with mask") {
      val mask = Nd4j.create(Array(1.0, 1.0, 0.0, 1.0, 0.0, 1.0))
      val policy = Nd4j.create(Array(1.5, 1.0, 2.0, -1.0, 2.5, 1.0))
      val data = channels.greedyAction(policy, mask)
      data shouldBe Nd4j.create(Array(1.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    }

    it("should get actionValues") {
      val mask = Nd4j.create(Array(1.0, 0.0, 1.0, 1.0, 0.0, 0.0))
      val policy = Nd4j.create(Array(1.5, 1.0, 2.0, -1.0, 2.5, 1.0))
      val data = channels.actionValues(policy, mask)
      data shouldBe Nd4j.create(Array(1.5, 2.0, -1.0))
    }
  }
}