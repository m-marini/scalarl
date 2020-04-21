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

package org.mmarini.scalarl.v2

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FunSpec, Matchers}

import scala.math._

class UtilsTest extends FunSpec with Matchers {

  Nd4j.create()

  describe("Utils") {
    val pr = Nd4j.create(Array(0.125, 0.25, 0.125, 0.5))
    val cdf = Nd4j.create(Array(0.125, 0.375, 0.5, 1.0))
    val softMax = Nd4j.create(Array(-log(2), 0.0, -log(2), log(2)))

    it("should get cdf") {
      Utils.cdf(pr) shouldBe cdf
    }

    it("should get df") {
      val x = Nd4j.create(Array(1.0, 2.0, 1.0, 4.0))
      Utils.df(x) shouldBe pr
    }

    it("should get softMax") {
      Utils.softMax(softMax) shouldBe pr
    }

    it("should get softMax with mask") {
      Utils.softMax(softMax, Seq(0L, 3L)) shouldBe Nd4j.create(Array(1.0 / 5, 0.0, 0.0, 4.0 / 5))
    }

    it("should generate random with cdf") {
      val random = Nd4j.randomFactory.getNewRandomInstance(100)
      val samples = for {i <- 1 to 800} yield {
        Utils.cdfRandomInt(cdf)(random)
      }
      val result = samples.groupBy(k => k).mapValues[Int](v => v.length)

      result(0) should (be >= 80 and be <= 120)
      result(1) should (be >= 180 and be <= 220)
      result(2) should (be >= 80 and be <= 120)
      result(3) should (be >= 380 and be <= 420)
    }

    it("should generate random with df") {
      val random = Nd4j.randomFactory.getNewRandomInstance(100)
      val samples = for {i <- 1 to 800} yield {
        Utils.randomInt(pr)(random)
      }
      val result = samples.groupBy(k => k).mapValues[Int](v => v.length)

      result(0) should (be >= 80 and be <= 120)
      result(1) should (be >= 180 and be <= 220)
      result(2) should (be >= 80 and be <= 120)
      result(3) should (be >= 380 and be <= 420)
    }

    it("should create egreedy policy for a single action") {
      Utils.egreedy(Nd4j.zeros(1), 0.1) shouldBe Nd4j.ones(1)
    }

    it("should create egreedy policy for a multiple actions") {
      Utils.egreedy(Nd4j.create(Array(1.0, 2.0, 0.0)), 0.1) shouldBe Nd4j.create(Array(0.05, 0.9, 0.05))
    }

    it("should find indices") {
      Utils.find(Nd4j.create(Array(1.0, 0.0, 0.0, 1.0, 0.0))) shouldBe Seq(0L, 3L)
    }

    it("should indexed a vector") {
      Utils.indexed(Nd4j.create(Array(1.0, 2.0, 3.0, 4.0, 5.0)), Seq(0L, 3L)) shouldBe Nd4j.create(Array(1.0, 4.0))
    }

    it("should create features vector") {
      Utils.features(Seq(0L, 3L), 5) shouldBe Nd4j.create(Array(1.0, 0.0, 0.0, 1.0, 0.0))
    }
  }
}