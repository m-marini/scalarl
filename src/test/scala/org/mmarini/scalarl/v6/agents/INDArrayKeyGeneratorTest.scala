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

package org.mmarini.scalarl.v6.agents

import org.mmarini.scalarl.v6.Configuration
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class INDArrayKeyGeneratorTest extends FunSpec with Matchers {

  create()

  private val jsonText1 =
    """
      |---
      |key:
      |    type: Binary
      |""".stripMargin
  describe(s"INDArrayKeyGenerator for $jsonText1") {
    val json = Configuration.jsonFormString(jsonText1)
    val keyGen = INDArrayKeyGenerator.fromJson(json.hcursor.downField("key"))(2).get
    it("should generate a key for 0,0") {
      keyGen(zeros(2)) shouldBe ModelKey(Nil)
    }
    it("should generate a key for 1,0") {
      keyGen(create(Array(1.0, 0.0))) shouldBe ModelKey(Seq(0))
    }
    it("should generate a key for 0,1") {
      keyGen(create(Array(0.0, 1.0))) shouldBe ModelKey(Seq(1))
    }
    it("should generate a key for 1,1") {
      keyGen(create(Array(1.0, 1.0))) shouldBe ModelKey(Seq(0, 1))
    }
  }

  private val jsonText2 =
    """
      |---
      |key:
      |    type: Discrete
      |    ranges:
      |    - [-0.5, 2.5 ]
      |    - [-1, 1]
      |    noValues: [ 3, 3]
      |""".stripMargin
  describe(s"INDArrayKeyGenerator for $jsonText2") {
    val json = Configuration.jsonFormString(jsonText2)
    val keyGen = INDArrayKeyGenerator.fromJson(json.hcursor.downField("key"))(2).get
    it("should generate a key for 0,0") {
      keyGen(zeros(2)) shouldBe ModelKey(Seq(0, 1))
    }
    it("should generate a key for 0,-1") {
      keyGen(create(Array(0.0, -1.0))) shouldBe ModelKey(Seq(0, 0))
    }
    it("should generate a key for -1,-2") {
      keyGen(create(Array(0.0, -1.0))) shouldBe ModelKey(Seq(0, 0))
    }
    it("should generate a key for 1,-0.334") {
      keyGen(create(Array(1.0, -0.334))) shouldBe ModelKey(Seq(1, 0))
    }
    it("should generate a key for 2,-0.333") {
      keyGen(create(Array(2.0, -0.333))) shouldBe ModelKey(Seq(2, 1))
    }
    it("should generate a key for 3,0.333") {
      keyGen(create(Array(3.0, 0.333))) shouldBe ModelKey(Seq(2, 1))
    }
    it("should generate a key for 3,0.334") {
      keyGen(create(Array(3.0, 0.334))) shouldBe ModelKey(Seq(2, 2))
    }
    it("should generate a key for 3,2") {
      keyGen(create(Array(3.0, 2.0))) shouldBe ModelKey(Seq(2, 2))
    }
  }
}
