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
    val sNo = create(Array[Double](1, 4.0, 2, 3))
    val range = create(Array(
      Array[Double](-1.0, -1.0, -1.0, -1.0),
      Array[Double](1.0, 1.0, 1.0, 1.0)
    ))
    val kg = INDArrayKeyGenerator.tiles(sNo, range)

    it("should generate key for (-1,-1,-1,-1)") {
      val v = ones(4).muli(-1)
      val r = kg(v)
      r.data shouldBe Seq(0, 0, 0, 0)
    }

    it("should generate key for (1,1,1,1)") {
      val r = kg(ones(4))
      r.data shouldBe Seq(0, 3, 1, 2)
    }

    it("should generate key for (-0.99,-0.51,-0.1,-0.34)") {
      val v = create(Array(-0.99, -0.51, -0.1, -0.34))
      val r = kg(v)
      r.data shouldBe Seq(0, 0, 0, 0)
    }

    it("should generate key for (0.99,-0.49,0.1,-0.33)") {
      val v = create(Array(0.99, -0.49, 0.1, -0.33))
      val r = kg(v)
      r.data shouldBe Seq(0, 1, 1, 1)
    }

    it("should generate key for (0.99,-0.1,0.99,0.33)") {
      val v = create(Array(0.99, -0.1, 0.99, 0.33))
      val r = kg(v)
      r.data shouldBe Seq(0, 1, 1, 1)
    }

    it("should generate key for (0.99,0.1,0.99,0.334)") {
      val v = create(Array(0.99, 0.1, 0.99, 0.334))
      val r = kg(v)
      r.data shouldBe Seq(0, 2, 1, 2)
    }

    it("should generate key for (0.99,0.49,0.99,0.99)") {
      val v = create(Array(0.99, 0.49, 0.99, 0.99))
      val r = kg(v)
      r.data shouldBe Seq(0, 2, 1, 2)
    }

    it("should generate key for (0.99,0.51,0.99,0.99)") {
      val v = create(Array(0.99, 0.51, 0.99, 0.99))
      val r = kg(v)
      r.data shouldBe Seq(0, 3, 1, 2)
    }

    it("should generate key for (0.99,0.99,0.99,0.99)") {
      val v = create(Array(0.99, 0.99, 0.99, 0.99))
      val r = kg(v)
      r.data shouldBe Seq(0, 3, 1, 2)
    }

    it("should generate key for (-20,-20,-20,-20)") {
      val s = ones(4).muli(-20.0)
      val r = kg(s)
      r.data shouldBe Seq(0, 0, 0, 0)
    }

    it("should generate key for (20,20,20,20)") {
      val s = ones(4).muli(20.0)
      val r = kg(s)

      r.data shouldBe Seq(0, 3, 1, 2)
    }
  }
}
