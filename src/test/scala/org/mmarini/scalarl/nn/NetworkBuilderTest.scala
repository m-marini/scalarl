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

    it("should write and read json") {
      Given("a NetworkBuilder with 2 input")
      val builder = NetworkBuilder().
        setNoInputs(2).
        setOptimizer(SGDOptimizer(0.1)).
        setTraceMode(AccumulateTraceMode(0, 0)).
        addLayers(
          DenseLayerBuilder("0", 2),
          ActivationLayerBuilder("1", TanhActivationFunction))

      When("create json")
      val json = builder.toJson

      And("create back the builder")
      val builder2 = NetworkBuilder.fromJson(json)

      Then("should be the same builder")
      builder2 shouldBe builder
    }

    it("should fit a linear regressio") {
      Given("a NetworkBuilder with 3 input and 2 output")
      val builder = NetworkBuilder().
        setNoInputs(3).
        setOptimizer(SGDOptimizer(0.1)).
        setTraceMode(NoneTraceMode).
        addLayers(
          DenseLayerBuilder("0", 2))

      And("inputs")
      val inputs = Nd4j.create(Array(1.0, 0.0, 0.0))

      And("labels")
      val labels = Nd4j.create(Array(1.0, -1.0))

      And("mask")
      val mask = Nd4j.create(Array(1.0, 0.0))

      And("no clear trace")
      val noClearTrace = Nd4j.zeros(1)

      And("theta")
      val theta = Nd4j.create(Array(
        0.2, -0.03,
        -1.0, -0.45,
        0.03, -0.02,
        0.0, 0.0))

      And("a random generator")
      val random = Nd4j.getRandomFactory().getNewRandomInstance(1234)

      And("initial data with theta")
      val data = builder.buildData(random) + ("0.theta" -> theta)

      When("fit")
      val fit = builder.buildProcessor.fit(data, inputs, labels, mask, noClearTrace);

      Then("should result output")
      val outputs = Nd4j.create(Array(0.2, -0.03))
      fit.get("outputs") should contain(outputs)

      And("should result delta and 0.delta")
      val delta = Nd4j.create(Array(0.8, 0.0))
      fit.get("delta") should contain(delta)
      fit.get("0.delta") should contain(delta)

      And("should result 0.inputDelta")
      val inputDelta = Nd4j.create(Array(0.2 * 0.8, -1.0 * 0.8, 0.03 * 0.8))
      fit.get("delta") should contain(delta)
      fit.get("0.delta") should contain(delta)

      And("should result gradient")
      val gradient = Nd4j.create(Array(
        1.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
        1.0, 1.0))
      fit.get("0.gradient") should contain(gradient)

      And("should result feedback")
      val feedback = Nd4j.create(Array(
        0.1, 0.1,
        0.0, 0.0,
        0.0, 0.0,
        0.1, 0.1))
      fit.get("0.feedback") should contain(feedback)

      And("should result thetaDelta")
      val thetaDelta = Nd4j.create(Array(
        0.8, 0.0,
        0.8, 0.0,
        0.8, 0.0,
        0.8, 0.0))
      fit.get("0.thetaDelta") should contain(thetaDelta)

      And("should result new theta")
      val newTheta = Nd4j.create(Array(
        0.2 + 0.8 * 0.1, -0.03,
        -1.0, -0.45,
        0.03, -0.02,
        0.8 * 0.1, 0.0))
      fit.get("0.theta") should contain(newTheta)
    }
  }
}