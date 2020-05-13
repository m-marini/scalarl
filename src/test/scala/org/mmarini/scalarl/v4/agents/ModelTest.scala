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

import org.mmarini.scalarl.v4.{Feedback, INDArrayObservation}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class ModelTest extends FunSpec with Matchers {

  create()

  val FeedbackOrdering: Ordering[(INDArray, Feedback)] = Ordering.by((t: (INDArray, Feedback)) => t._2.s0.time.getDouble(0L)).reverse

  def key(value: Double): INDArray = ones(1).muli(value)

  def feedback(value: Double, time: Double): Feedback = Feedback(
    s0 = obs(value, time),
    actions = zeros(0),
    reward = zeros(0),
    s1 = obs(10.0, 10.0)
  )

  def obs(value: Double, time: Double): INDArrayObservation =
    INDArrayObservation(time = ones(1).muli(time), ones(1).muli(value))

  describe("Model") {
    val p = Model[INDArray, Feedback](minModelSize = 1,
      maxModelSize = 3,
      data = Map(),
      ordering = FeedbackOrdering)

    it("should add data to model") {
      val k = key(0)
      val value = feedback(1, 10.0)
      val p1 = p + (k -> value)

      p1.data should contain(k -> value)
    }

    it("should update data to model") {
      val k = key(0)
      val v0 = feedback(1, 10.0)
      val v1 = feedback(2, 10.0)
      val p1 = p + (k -> v0)

      val p2 = p1 + (k -> v1)

      p2.data should contain(k -> v1)
      p2.data should have size 1
    }

    it("should add max data to model") {
      val k1 = key(0)
      val k2 = key(1)
      val k3 = key(2)
      val v0 = feedback(1, 10.0)
      val p1 = p + (k1 -> v0) + (k2 -> v0) + (k3 -> v0)

      p1.data should have size 3
      p1.data should contain(k1 -> v0)
      p1.data should contain(k2 -> v0)
      p1.data should contain(k3 -> v0)
    }

    describe("when feed to full size") {
      val k0 = key(0)
      val k1 = key(1)
      val k2 = key(2)
      val v0 = feedback(1, 2)
      val v1 = feedback(1, 1)
      val v2 = feedback(1, 3)
      val p1 = p + (k0 -> v0) + (k1 -> v1) + (k2 -> v2)

      it("should get existing value") {
        p1.get(k1) should contain(v1)
      }

      it("should get none for invalid key") {
        val k3 = key(3)
        p1.get(k3) shouldBe empty
      }

      it("should shrink data when add") {
        val k3 = key(3)
        val v3 = feedback(1, 1)
        val p2 = p1 + (k3 -> v3)

        p2.data should have size 2
        p2.data should contain(k3 -> v3)
        p2.data should contain(k2 -> v2)
      }

      it("should filter data") {
        val v = p1.filterValues(f => f.s0.time.getDouble(0L) > 1.0)

        v should have size 2
        v should contain(k0 -> v0)
        v should contain(k2 -> v2)
      }
    }
  }
}
