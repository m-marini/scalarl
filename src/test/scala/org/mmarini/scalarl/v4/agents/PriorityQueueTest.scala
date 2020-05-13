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

class PriorityQueueTest extends FunSpec with Matchers {

  create()

  describe("PriorityQueue") {
    val p = PriorityQueue[String](threshold = 0.1, Map())

    it("should enqueues values with priority higer than threshold") {
      val p1 = p + ("a", 1)
      p1.queue should contain("a" -> 1.0)
    }

    it("should not enqueues values with priority lower  than threshold") {
      val p1 = p + ("a" -> 0)
      p1.queue shouldBe empty
    }

    it("should not enqueues values with priority equal to threshold") {
      val p1 = p + ("a" -> 0.1)
      p1.queue shouldBe empty
    }

    describe("with enqueued values") {
      val p1 = p + ("a" -> 1.0) + ("b" -> 2.0)
      it("should dequeue higer element") {
        val (value, p2) = p1.dequeue()
        value should contain("b")
        p2.queue.size shouldBe 1
        p2.queue should contain("a" -> 1.0)
      }
    }
  }
}
