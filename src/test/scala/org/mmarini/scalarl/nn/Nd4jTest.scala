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
import org.scalatest.{FunSpec, GivenWhenThen, Matchers}

class Nd4jTest extends FunSpec with GivenWhenThen with Matchers {
  val Epsilon = 1e-6

  Nd4j.create()

  describe("Tests on vectors") {
    it("should be created by 6 values as row vector") {
      Given("6 values")
      val ary = Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

      When("create a row vector")
      val m = Nd4j.create(ary)

      Then("It should have shape 1 x 6")
      m.shape shouldBe Array(1, 6)

      And("It should have expected values by one column index")
      m.getDouble(0L) shouldBe 1.0
      m.getDouble(1L) shouldBe 2.0
      m.getDouble(2L) shouldBe 3.0
      m.getDouble(3L) shouldBe 4.0
      m.getDouble(4L) shouldBe 5.0
      m.getDouble(5L) shouldBe 6.0

      And("It should have expected values accessed by two cell index by column")
      m.getDouble(0L, 0L) shouldBe 1.0
      m.getDouble(0L, 1L) shouldBe 2.0
      m.getDouble(0L, 2L) shouldBe 3.0
      m.getDouble(0L, 3L) shouldBe 4.0
      m.getDouble(0L, 4L) shouldBe 5.0
      m.getDouble(0L, 5L) shouldBe 6.0

      And("It should have expected values accessed by two cell index by row (broadcast)")
      m.getDouble(0L, 0L) shouldBe 1.0
      m.getDouble(1L, 1L) shouldBe 2.0
      m.getDouble(2L, 2L) shouldBe 3.0
      m.getDouble(3L, 3L) shouldBe 4.0
      m.getDouble(4L, 4L) shouldBe 5.0
      m.getDouble(5L, 5L) shouldBe 6.0
    }

    it("should be created by 6 values as col vector") {
      Given("6 values")
      val ary = Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).map(x => Array(x))

      When("create a col vector")
      val m = Nd4j.create(ary)

      Then("It should have shape 6 x 1")
      m.shape shouldBe Array(6, 1)

      And("It should have expected values by two row index")
      m.getDouble(0L, 0L) shouldBe 1.0
      m.getDouble(1L, 0L) shouldBe 2.0
      m.getDouble(2L, 0L) shouldBe 3.0
      m.getDouble(3L, 0L) shouldBe 4.0
      m.getDouble(4L, 0L) shouldBe 5.0
      m.getDouble(5L, 0L) shouldBe 6.0

      And("It should have expected values accessed by two cell index by row (broadcast)")
      m.getDouble(0L, 0L) shouldBe 1.0
      m.getDouble(1L, 1L) shouldBe 2.0
      m.getDouble(2L, 2L) shouldBe 3.0
      m.getDouble(3L, 3L) shouldBe 4.0
      m.getDouble(4L, 4L) shouldBe 5.0
      m.getDouble(5L, 5L) shouldBe 6.0
    }
  }

  describe("Tests on matrix") {
    it("should be created") {
      Given("6 values")
      val ary = Array(
        Array(1.0, 2.0, 3.0),
        Array(4.0, 5.0, 6.0))

      When("create a matrix with 2 x 3")
      val m = Nd4j.create(ary)

      Then("It should have shape 2 x 3")
      m.shape shouldBe Array(2, 3)

      And("It should have expected values")
      m.getDouble(0L, 0L) shouldBe 1.0
      m.getDouble(0L, 1L) shouldBe 2.0
      m.getDouble(0L, 2L) shouldBe 3.0
      m.getDouble(1L, 0L) shouldBe 4.0
      m.getDouble(1L, 1L) shouldBe 5.0
      m.getDouble(1L, 2L) shouldBe 6.0
    }

    it("should be ravel") {
      Given("a matrix 2 x 3")
      val m = Nd4j.create(Array(
        Array(1.0, 2.0, 3.0),
        Array(4.0, 5.0, 6.0)))

      When("ravel")
      val r = m.ravel()

      Then("It should have shape 1 x 6")
      r.shape shouldBe Array(1, 6)

      And("It should have expected values")
      r.getDouble(0L) shouldBe 1.0
      r.getDouble(1L) shouldBe 2.0
      r.getDouble(2L) shouldBe 3.0
      r.getDouble(3L) shouldBe 4.0
      r.getDouble(4L) shouldBe 5.0
      r.getDouble(5L) shouldBe 6.0
    }

    it("should be reshaped") {
      Given("a vector 6")
      val m = Nd4j.create(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

      When("reshape")
      val r = m.reshape(2, 3)

      Then("It should have shape 2 x 3")
      r.shape shouldBe Array(2, 3)

      And("It should have expected values")
      r.getDouble(0L, 0L) shouldBe 1.0
      r.getDouble(0L, 1L) shouldBe 2.0
      r.getDouble(0L, 2L) shouldBe 3.0
      r.getDouble(1L, 0L) shouldBe 4.0
      r.getDouble(1L, 1L) shouldBe 5.0
      r.getDouble(1L, 2L) shouldBe 6.0
    }
  }
}
