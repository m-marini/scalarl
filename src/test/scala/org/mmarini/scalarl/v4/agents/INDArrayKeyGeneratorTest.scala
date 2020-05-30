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

class INDArrayKeyGeneratorTest extends FunSpec with Matchers {

  create()

  describe("INDArraySAKeyGeneratorTest") {
    val sMin = create(Array[Double](0, -10.0, 0, -10.0))
    val sMax = create(Array[Double](1, 10.0, 10.0, 5.0))
    val sNo = create(Array[Double](1, 4.0, 2, 3))
    val kg = INDArrayKeyGenerator.tiles(sMin, sMax, sNo)

    it("should generate key for (0,0,0,0)") {
      val s = zeros(4)
      val r = kg(s)
      r.data shouldBe Seq(0, 2, 0, 2)
    }

    it("should generate key for (1,-10, 10,-10)") {
      val s = create(Array[Double](1, -10.0, 10.0, -10.0))
      val r = kg(s)
      r.data shouldBe Seq(1, 0, 2, 0)
    }

    it("should generate key for (0.50,-7.51,7.50,-7.51)") {
      val s = create(Array[Double](0.50, -7.51, 7.50, -7.51))
      val r = kg(s)
      r.data shouldBe Seq(1, 0, 2, 0)
    }

    it("should generate key for (0.49,-7.51,7.50,-7.51)") {
      val s = create(Array[Double](0.49, -7.50, 7.49, -7.50))
      val r = kg(s)
      r.data shouldBe Seq(0, 1, 1, 1)
    }

    it("should generate key for (-20,-20,-20,-20)") {
      val s = ones(4).muli(-20.0)
      val r = kg(s)
      r.data shouldBe Seq(0, 0, 0, 0)
    }

    it("should generate key for (20,20,20,20)") {
      val s = ones(4).muli(20.0)
      val r = kg(s)

      r.data shouldBe Seq(1, 4, 2, 3)
    }
  }
}
