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

package org.mmarini.scalarl.v4.agents

import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class ActorCriticAgentTest extends FunSpec with Matchers {

  create()

  describe("ActorCriticAgent") {
    it("should compute delta") {
      val v0 = create(Array[Double](1))
      val v1 = create(Array[Double](3))
      val reward = create(Array[Double](2))
      val avg = create(Array[Double](1))
      val valueDecay = create(Array[Double](0.9))
      val rewardDecay = create(Array[Double](0.8))

      val (delta, newv0, newAvg) = ActorCriticAgent.computeDelta(v0, v1, reward, avg, valueDecay, rewardDecay)

      val ExpectedV0 = 0.9 * (5 - 1) + 0.1 * 1
      val ExpectedDelta = 5 - 1 - 1
      newv0 shouldBe create(Array[Double](ExpectedV0))
      delta shouldBe create(Array[Double](ExpectedDelta))
      newAvg shouldBe create(Array[Double](0.8 * 1 + 0.2 * 2))
    }
  }
}
