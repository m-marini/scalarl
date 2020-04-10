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

package org.mmarini.scalarl.ts.envs

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FunSpec, Matchers}

class TestEnvConfBuilderTest extends FunSpec with Matchers {
  Nd4j.create()

  describe("TestEnvConfBuilder build") {

    val b = TestEnvConfBuilder().numState(3).
      add(0, 0, 0, 7.0, 0.0).
      add(0, 0, 1, 1.0, 0.0).
      add(0, 1, 0, 1.0, 0.0).
      add(0, 1, 1, 7.0, 0.0).
      add(1, 0, 1, 1.0, 0.0).
      add(1, 0, 2, 7.0, 1.0).
      add(1, 1, 1, 7.0, 0.0).
      add(1, 1, 2, 1.0, 1.0).
      add(2, 0, 0, 1.0, 0.0).
      add(2, 1, 0, 1.0, 0.0)

    val result = b.build

    it("should have 3 state ") {
      result.numState shouldBe 3
    }

    val pi = result.pi

    it("should have pi(0,0)") {
      pi(0, 0) should have size 3
      pi(0, 0)(0) shouldBe(7.0 / 8, 0.0)
      pi(0, 0)(1) shouldBe(1.0, 0.0)
      pi(0, 0)(2) shouldBe(1.0, 0.0)
    }

    it("should have pi(0,1)") {
      pi(0, 1) should have size 3
      pi(0, 1)(0) shouldBe(1.0 / 8, 0.0)
      pi(0, 1)(1) shouldBe(1.0, 0.0)
      pi(0, 1)(2) shouldBe(1.0, 0.0)
    }

    it("should have pi(1,0)") {
      pi(1, 0) should have size 3
      pi(1, 0)(0) shouldBe(0.0, 0.0)
      pi(1, 0)(1) shouldBe(1.0 / 8, 0.0)
      pi(1, 0)(2) shouldBe(1.0, 1.0)
    }

    it("should have pi(1,1)") {
      pi(1, 1) should have size 3
      pi(1, 1)(0) shouldBe(0.0, 0.0)
      pi(1, 1)(1) shouldBe(7.0 / 8, 0.0)
      pi(1, 1)(2) shouldBe(1.0, 1.0)
    }

    it("should have pi(2,0)") {
      pi(2, 0) should have size 3
      pi(2, 0)(0) shouldBe(1.0, 0.0)
      pi(2, 0)(1) shouldBe(1.0, 0.0)
      pi(2, 0)(2) shouldBe(1.0, 0.0)
    }

    it("should have pi(2,1)") {
      pi(2, 1) should have size 3
      pi(2, 1)(0) shouldBe(1.0, 0.0)
      pi(2, 1)(1) shouldBe(1.0, 0.0)
      pi(2, 1)(2) shouldBe(1.0, 0.0)
    }
  }
}