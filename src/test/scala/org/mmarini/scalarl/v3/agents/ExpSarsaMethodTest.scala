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

package org.mmarini.scalarl.v3.agents

import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class ExpSarsaMethodTest extends FunSpec with Matchers {

  create()

  describe("ExpSarsaMethod") {
    describe("when create data") {
      val q0 = create(Array[Double](1, 2))
      val q1 = create(Array[Double](1.5, 2.0))
      val reward = ones(1).mul(2)
      val avg = ones(1)
      val action = 0
      val beta = ones(1).muli(0.9)
      val epsilon = ones(1).muli(0.1)

      val (labels, newAvg, score) = ExpSarsaMethod.createData(q0, q1, action, reward, beta, epsilon, avg)

      val v0 = 1.0
      val v1 = 1.5 * 0.1 + 2.0 * 0.9
      val newv0 = v1 + 2 - 1
      val delta = newv0 - v0
      val ExpScore = delta * delta
      val ExpAvg = 0.9 * 2 + 0.1 * 1
      val ExpLabels = q0.dup()
      ExpLabels.putScalar(action, newv0)

      it("should return labels") {
        labels shouldBe ExpLabels
      }
      it("should return new average") {
        newAvg shouldBe ones(1).muli(ExpAvg)
      }
      it("should return score") {
        score shouldBe ones(1).muli(ExpScore)
      }
    }
  }
}