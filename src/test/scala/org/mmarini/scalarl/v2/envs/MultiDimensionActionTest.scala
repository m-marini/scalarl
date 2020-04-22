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

package org.mmarini.scalarl.v2.envs

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FunSpec, Matchers}

import scala.math._

class MultiDimensionActionTest extends FunSpec with Matchers {

  Nd4j.create()

  describe("MultiDimensionAction") {
    val pr = Nd4j.create(Array(0.125, 0.25, 0.125, 0.5))
    val cdf = Nd4j.create(Array(0.125, 0.375, 0.5, 1.0))
    val softMax = Nd4j.create(Array(-log(2), 0.0, -log(2), log(2)))

    val cfg = new MultiDimensionAction(5, 5, 5)

    it("should get actions") {
      cfg.actions shouldBe 125
    }

    it("should get strides") {
      cfg.strides shouldBe Seq(1, 5, 25)
    }

    it("should get vector(0)") {
      cfg.vector(0) shouldBe Nd4j.create(Array(0.0, 0.0, 0.0))
    }

    it("should get vector(1)") {
      cfg.vector(1) shouldBe Nd4j.create(Array(1.0, 0.0, 0.0))
    }

    it("should get vector(5)") {
      cfg.vector(5) shouldBe Nd4j.create(Array(0.0, 1.0, 0.0))
    }

    it("should get vector(25)") {
      cfg.vector(25) shouldBe Nd4j.create(Array(0.0, 0.0, 1.0))
    }

    it("should get vector(124)") {
      cfg.vector(124) shouldBe Nd4j.create(Array(4.0, 4.0, 4.0))
    }
  }
}