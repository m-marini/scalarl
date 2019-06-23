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

import org.scalatest.FunSpec
import org.scalatest.Matchers
import org.scalatest.prop.PropertyChecks

import io.circe.Json
import io.circe.yaml
import io.circe.yaml.syntax.AsYaml

class NetworkBuilderTest extends FunSpec with PropertyChecks with Matchers {

  describe("Given an network builder") {
    val builder = NetworkBuilder()
    describe("When setNoInputs") {
      val newBuilder = builder.setNoInputs(10)
      it("Then should create a new builder") {
        newBuilder should not be (builder)
        newBuilder.noInputs shouldBe 10
      }
    }
  }

  describe("Given an network builder") {
    val builder = NetworkBuilder().
      setNoInputs(10).
      addLayer(DenseLayerBuilder(2)).
      addLayer(ActivationLayerBuilder(TanhActivationFunction)).
      setTraceMode(AccumulateTraceMode(0.8, 0.9)).
      setOptimizer(AdamOptimizer(0.1, 0.8, 0.9, 0.5))

    describe("When toJson") {
      val json = builder.toJson
      val txt = json.asYaml.spaces2
      it("Then should create a json object") {
        txt shouldBe """optimizer:
  beta2: 0.9
  mode: ADAM
  beta1: 0.8
  epsilon: 0.5
  alpha: 0.1
lossFunction: MSE
noInputs: 10
layers:
- type: DENSE
  noOutputs: 2
- type: ACTIVATION
  activation: TANH
traceMode:
  mode: ACCUMULATE
  lambda: 0.8
  gamma: 0.9
initializer: XAVIER
"""
      }
    }
  }

  describe("Given circe") {
    val json = yaml.parser.parse("""
foo: Hello, World
bar:
    one: One Third
    two: 33.333333
baz:
    - Hello
    - World
""")
    val v = json.right.get
    it("") {
      "" shouldBe ""
    }
  }

  describe("gen  yaml") {
    val x = (1 to 3).map(Json.fromInt).toArray
    val doc = Json.obj(
      "a" -> Json.fromInt(69),
      "b" -> Json.fromString("aa"),
      "c" -> Json.arr(x: _*))
    val txt = doc.asYaml.spaces2 // 2 spaces for each indent level
    it("") {
      txt shouldBe """a: 69
b: aa
c:
- 1
- 2
- 3
"""
    }
  }
}
