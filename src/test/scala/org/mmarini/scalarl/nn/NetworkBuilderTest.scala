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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FunSpec
import org.scalatest.GivenWhenThen
import org.scalatest.Matchers

class NetworkBuilderTest extends FunSpec with GivenWhenThen with Matchers {

  Nd4j.create()

  def loss(a: INDArray, b: INDArray): Double = a.squaredDistance(b)

  describe("A NetworkBuilder") {
    it("should create a new builder") {
      Given("a netwok builder")
      val builder = NetworkBuilder()

      When("setNoInputs")
      val newBuilder = builder.setNoInputs(10)

      Then("Then should create a new builder")
      newBuilder should not be (builder)
      newBuilder.noInputs shouldBe 10
    }

    it("should build network data") {
      Given("a netwok builder")
      val builder = NetworkBuilder().
        setNoInputs(2).
        addLayers(
          DenseLayerBuilder("0", 2),
          ActivationLayerBuilder("1", TanhActivationFunction))
      And("a random generator")
      val random = Nd4j.getRandomFactory().getNewRandomInstance(1234)

      When("initialize data")
      val data = builder.buildData(random)

      Then("should return the expected value")
      data.keySet should contain("0.theta")
      data.keySet should contain("0.trace")
      data.keySet should contain("0.m1")
      data.keySet should contain("0.m2")
    }

    it("should predict output") {
      Given("a netwok builder")
      val builder = NetworkBuilder().
        setNoInputs(2).
        addLayers(
          DenseLayerBuilder("0", 2),
          ActivationLayerBuilder("1", TanhActivationFunction))
      And("a random generator")
      val random = Nd4j.getRandomFactory().getNewRandomInstance(1234)
      And("a initialized network data")
      val data = builder.buildData(random)
      And("inputs data")
      val inputs = Nd4j.create(Array(0.5, -0.5))

      When("predict data")
      val outData = builder.buildProcessor.forward(data, inputs)

      Then("should return the expected value")
      outData.shape shouldBe Array(1, 2)
    }

    it("should improve the prediction") {
      Given("a NetworkBuilder with 2 input")
      val builder = NetworkBuilder().
        setNoInputs(2).
        setOptimizer(SGDOptimizer(0.1)).
        setTraceMode(AccumulateTraceMode(0, 0)).
        addLayers(
          DenseLayerBuilder("0", 2),
          ActivationLayerBuilder("1", TanhActivationFunction))
      And("a random generator")
      val random = Nd4j.getRandomFactory().getNewRandomInstance(1234)
      And("data network")
      val data = builder.buildData(random)
      And("inputs")
      val inputs = Nd4j.create(Array(0.5, 0.5))
      And("noClearTrace")
      val noClearTrace = Nd4j.zeros(1)
      And("labels")
      val labels = Nd4j.create(Array(0.2, 0.8))
      And("mask")
      val mask = Nd4j.create(Array(0.0, 1.0))

      When("create processor")
      val proc = builder.buildProcessor
      And("fit data")
      val fitted = proc.fit(data, inputs, labels, mask, noClearTrace)
      And("fit again")
      val fitted2 = proc.fit(fitted, inputs, labels, mask, noClearTrace)

      Then("should return a smaller loss value")

      val loss1 = fitted("loss").getDouble(0L)
      val loss2 = fitted2("loss").getDouble(0L)
      loss2 should be < (loss1)

      val err1 = loss(labels, fitted("outputs"))
      val err2 = loss(labels, fitted2("outputs"))
      err2 should be < (err1)
    }

    //    val builder = NetworkBuilder().
    //      setNoInputs(10).
    //      addLayer(DenseLayerBuilder(2)).
    //      addLayer(ActivationLayerBuilder(TanhActivationFunction)).
    //      setTraceMode(AccumulateTraceMode(0.8, 0.9)).
    //      setOptimizer(AdamOptimizer(0.1, 0.8, 0.9, 0.5))
    //
    //    describe("When toJson") {
    //      val json = builder.toJson
    //      val txt = json.asYaml.spaces2
    //      it("Then should create a json object") {
    //        txt shouldBe """optimizer:
    //  beta2: 0.9
    //  mode: ADAM
    //  beta1: 0.8
    //  epsilon: 0.5
    //  alpha: 0.1
    //lossFunction: MSE
    //noInputs: 10
    //layers:
    //- type: DENSE
    //  noOutputs: 2
    //- type: ACTIVATION
    //  activation: TANH
    //traceMode:
    //  mode: ACCUMULATE
    //  lambda: 0.8
    //  gamma: 0.9
    //initializer: XAVIER
    //"""
    //      }
    //    }
    //  }
    //
    //  describe("Given circe") {
    //    val json = yaml.parser.parse("""
    //foo: Hello, World
    //bar:
    //    one: One Third
    //    two: 33.333333
    //baz:
    //    - Hello
    //    - World
    //""")
    //    val v = json.right.get
    //    it("") {
    //      "" shouldBe ""
    //    }
    //  }
    //
    //  describe("gen  yaml") {
    //    val x = (1 to 3).map(Json.fromInt).toArray
    //    val doc = Json.obj(
    //      "a" -> Json.fromInt(69),
    //      "b" -> Json.fromString("aa"),
    //      "c" -> Json.arr(x: _*))
    //    val txt = doc.asYaml.spaces2 // 2 spaces for each indent level
    //    it("") {
    //      txt shouldBe """a: 69
    //b: aa
    //c:
    //- 1
    //- 2
    //- 3
    //"""
    //    }
  }
}
