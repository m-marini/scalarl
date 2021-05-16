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

package org.mmarini.scalarl.v6

import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

import scala.math._

class UtilsTest extends FunSpec with Matchers {

  create()

  describe("Utils") {
    val pr = create(Array(0.125, 0.25, 0.125, 0.5))
    val cdf = create(Array(0.125, 0.375, 0.5, 1.0))
    val softMax = create(Array(-log(2), 0.0, -log(2), log(2)))

    it("should get cdf") {
      Utils.cdf(pr) shouldBe cdf
    }

    it("should get df") {
      val x = create(Array(1.0, 2.0, 1.0, 4.0))
      Utils.df(x) shouldBe pr
    }

    it("should generate random with cdf") {
      val random = randomFactory.getNewRandomInstance(100)
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
      val random = randomFactory.getNewRandomInstance(100)
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
      Utils.eGreedy(zeros(1), ones(1).mul(0.1)) shouldBe ones(1)
    }

    it("should create egreedy policy for a multiple actions") {
      Utils.eGreedy(create(Array(1.0, 2.0, 0.0)), ones(1).mul(0.1)) shouldBe create(Array(0.05, 0.9, 0.05))
    }

    it("should find indices") {
      Utils.find(create(Array(1.0, 0.0, 0.0, 1.0, 0.0))) shouldBe Seq(0L, 3L)
    }

    it("should indexed a vector") {
      Utils.indexed(create(Array(1.0, 2.0, 3.0, 4.0, 5.0)), Seq(0L, 3L)) shouldBe create(Array(1.0, 4.0))
    }

    it("should create features vector") {
      Utils.features(Seq(0L, 3L), 5) shouldBe create(Array(1.0, 0.0, 0.0, 1.0, 0.0))
    }

    it("should create linear transformation ") {
      val f = Utils.clipAndDenormalize(create(Array(
        Array(0.0, -2.0),
        Array(5.0, 4.0))))

      f(zeros(2)) shouldBe create(Array(2.5, 1.0))
      f(ones(2)) shouldBe create(Array(5.0, 4.0))
      f(ones(2).negi()) shouldBe create(Array(0.0, -2.0))
    }

    it("should create linear inverse ") {
      val f = Utils.clipAndNormalize(create(Array(
        Array(0.0, -2.0),
        Array(5.0, 4.0))))

      f(create(Array(2.5, 1.0))) shouldBe zeros(2)
      f(create(Array(5.0, 4.0))) shouldBe ones(2)
      f(create(Array(0.0, -2.0))) shouldBe ones(2).negi()
    }
  }

  it("should create normalize01") {
    val f = Utils.clipAndNormalize01(create(Array(
      Array(0.0, -2.0),
      Array(5.0, 4.0))))

    f(create(Array(2.5, 1.0))) shouldBe ones(2).muli(0.5)
    f(create(Array(5.0, 4.0))) shouldBe ones(2)
    f(create(Array(0.0, -2.0))) shouldBe zeros(2)
  }

  it("should create clipAndNormalize") {
    val f = Utils.clipAndNormalize(create(Array(
      Array(0.0, -2.0),
      Array(5.0, 4.0))))

    f(create(Array(-1.0, -3.0))) shouldBe ones(2).negi()
    f(create(Array(0.0, -2.0))) shouldBe ones(2).negi()
    f(create(Array(2.5, 1.0))) shouldBe zeros(2)
    f(create(Array(5.0, 4.0))) shouldBe ones(2)
    f(create(Array(6.0, 5.0))) shouldBe ones(2)
  }

  it("should create clipAndNormalize01") {
    val f = Utils.clipAndNormalize01(create(Array(
      Array(0.0, -2.0),
      Array(5.0, 4.0))))

    f(create(Array(-1.0, -3.0))) shouldBe zeros(2).negi()
    f(create(Array(0.0, -2.0))) shouldBe zeros(2).negi()
    f(create(Array(2.5, 1.0))) shouldBe ones(2).muli(0.5)
    f(create(Array(5.0, 4.0))) shouldBe ones(2)
    f(create(Array(6.0, 5.0))) shouldBe ones(2)
  }

  it("should create clipAndDenormalize") {
    val f = Utils.clipAndDenormalize(create(Array(
      Array(0.0, -2.0),
      Array(5.0, 4.0))))

    f(ones(2).muli(-1.1)) shouldBe (create(Array(0.0, -2.0)))
    f(ones(2).negi()) shouldBe (create(Array(0.0, -2.0)))
    f(zeros(2)) shouldBe (create(Array(2.5, 1.0)))
    f(ones(2)) shouldBe (create(Array(5.0, 4.0)))
    f(ones(2).muli(1.1)) shouldBe (create(Array(5.0, 4.0)))
  }

  it("should create clipAndTransform") {
    val from = create(Array(
      Array(0.0, -2.0),
      Array(5.0, 4.0)))
    val to = create(Array(
      Array(1.0, -4.0),
      Array(11.0, 8.0)))
    val f = Utils.clipAndTransform(from , to)

    f(create(Array(-0.1, -2.1))) shouldBe (create(Array(1.0, -4.0)))
    f(create(Array(0.0, -2.0))) shouldBe (create(Array(1.0, -4.0)))
    f(create(Array(2.5, 1.0))) shouldBe (create(Array(6.0, 2.0)))
    f(create(Array(5.0, 4.0))) shouldBe (create(Array(11.0, 8.0)))
    f(create(Array(5.1, 4.1))) shouldBe (create(Array(11.0, 8.0)))
  }

  it("should create clipDenormalizeAndCenter") {
    val f = Utils.clipDenormalizeAndCenter(create(Array(
      Array(0.0, -2.0),
      Array(5.0, 4.0))))

    f(ones(2).muli(-1.1)) shouldBe (create(Array(1.0, -1.0)))
    f(ones(2).negi()) shouldBe (create(Array(1.0, -1.0)))
    f(zeros(2)) shouldBe (create(Array(0.75, -0.75)))
    f(ones(2)) shouldBe (create(Array(0.5, -0.5)))
    f(ones(2).muli(1.1)) shouldBe (create(Array(0.5, -0.5)))
  }

  it("should create encode") {
    val f = Utils.encode(5, create(Array(-2.0, 2.0)))

    f(-1) shouldBe (ones(1).muli(-2))
    f(0) shouldBe (ones(1).muli(-2))
    f(1) shouldBe (ones(1).muli(-1))
    f(2) shouldBe (zeros(1))
    f(3) shouldBe (ones(1).muli(1))
    f(4) shouldBe (ones(1).muli(2))
    f(5) shouldBe (ones(1).muli(2))
  }

  it("should create decode") {
    val f = Utils.decode(5, create(Array(-2.0, 2.0)).transpose())

    f(ones(1).muli(-3)) shouldBe create(Array(1.0, 0.0, 0.0, 0.0, 0.0))
    f(ones(1).muli(-2)) shouldBe create(Array(1.0, 0.0, 0.0, 0.0, 0.0))
    f(ones(1).muli(-1.51)) shouldBe create(Array(1.0, 0.0, 0.0, 0.0, 0.0))
    f(ones(1).muli(-1.5)) shouldBe create(Array(0.0, 1.0, 0.0, 0.0, 0.0))
    f(ones(1).muli(-1)) shouldBe create(Array(0.0, 1.0, 0.0, 0.0, 0.0))
    f(ones(1).muli(-0.51)) shouldBe create(Array(0.0, 1.0, 0.0, 0.0, 0.0))
    f(ones(1).muli(-0.5)) shouldBe create(Array(0.0, 0.0, 1.0, 0.0, 0.0))
    f(zeros(1)) shouldBe create(Array(0.0, 0.0, 1.0, 0.0, 0.0))
    f(ones(1).muli(0.49)) shouldBe create(Array(0.0, 0.0, 1.0, 0.0, 0.0))
    f(ones(1).muli(0.5)) shouldBe create(Array(0.0, 0.0, 0.0, 1.0, 0.0))
    f(ones(1).muli(1)) shouldBe create(Array(0.0, 0.0, 0.0, 1.0, 0.0))
    f(ones(1).muli(1.49)) shouldBe create(Array(0.0, 0.0, 0.0, 1.0, 0.0))
    f(ones(1).muli(1.5)) shouldBe create(Array(0.0, 0.0, 0.0, 0.0, 1.0))
    f(ones(1).muli(2)) shouldBe create(Array(0.0, 0.0, 0.0, 0.0, 1.0))
    f(ones(1).muli(3)) shouldBe create(Array(0.0, 0.0, 0.0, 0.0, 1.0))
  }
}