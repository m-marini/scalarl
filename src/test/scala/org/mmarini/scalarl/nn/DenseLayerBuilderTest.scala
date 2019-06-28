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
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.PropSpec
import org.scalacheck.Gen
import org.scalatest.GivenWhenThen

class DenseLayerBuilderTest extends FunSpec with GivenWhenThen with Matchers {
  val Epsilon = 1e-6

  Nd4j.create()

  def mockTopology = new MockTopology() {
    override def prevLayer(layer: LayerBuilder): Option[LayerBuilder] = Some(new MockLayerBuilder() {
      override def noOutputs(topology: NetworkTopology): Int = 3
    })
  }

  def initialLayerData = {
    val inputs = (1 to 3).map(x => x / 10.0).toArray
    val theta = (1 to 8).map(x => x / 10.0).toArray
    Map(
      "inputs" -> Nd4j.create(inputs),
      "theta" -> Nd4j.create(theta))
  }

  /**
   * {{{
   * | 0.1 0.2 0.3 | x | 0.1 0.2 | + | 0.7 0.8 | = | 0.92 1.08 |
   *                   | 0.3 0.4 |
   *                   | 0.5 0.6 |
   * }}}
   */
  def expectedOutputs = Nd4j.create(Array(0.92, 1.08))

  def expectedWeights = Nd4j.create(Array(Array(0.1, 0.2), Array(0.3, 0.4), Array(0.5, 0.6)))

  def expectedGradient = Nd4j.create(Array(
    0.1, 0.1,
    0.2, 0.2,
    0.3, 0.3,
    1.0, 1.0))

  describe("DenseLayer") {
    it("should generate trace updater") {

      Given("a dense layer builder with 2 outputs")
      val layer = DenseLayerBuilder(noOutputs = 2)

      And("a inputs layer data with 3 random inputs")
      val inputsData = initialLayerData

      When("build a clear trace updater")
      val updater = layer.buildClearTrace(mockTopology)

      And("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the trace parameter with 8 zeros")
      newData.get("trace") should contain(Nd4j.zeros(8L))
    }

    it("should generate forward updater") {

      Given("a dense layer builder with 2 outputs")
      val layer = DenseLayerBuilder(noOutputs = 2)

      And("a inputs layer data with 3 random inputs")
      val inputsData = initialLayerData

      When("build a forward updater")
      val updater = layer.buildForward(mockTopology)

      And("apply to initial layer")
      val newData = updater(inputsData)

      Then("should result the output from linear combination of inputs")
      newData.get("outputs") should contain(expectedOutputs)
    }

    it("should generate a weights converter") {

      Given("a dense layer builder with 2 outputs")
      val layer = DenseLayerBuilder(noOutputs = 2)

      And("a theta parameters")
      val theta = initialLayerData("theta")

      When("build a weights converter")
      val converter = layer.weights(mockTopology)

      And("apply to theta")
      val weights = converter(theta)

      Then("should result the weights")
      weights shouldBe expectedWeights
    }

    it("should generate a bias converter") {

      Given("a dense layer builder with 2 outputs")
      val layer = DenseLayerBuilder(noOutputs = 2)

      And("a theta parameters")
      val theta = initialLayerData("theta")

      When("build a weights converter")
      val converter = layer.bias(mockTopology)

      And("apply to theta")
      val bias = converter(theta)

      Then("should result bias")
      bias shouldBe Nd4j.create(Array(0.7, 0.8))
    }

    it("should generate backward gradient") {

      Given("a dense layer builder with 2 outputs")
      val layer = DenseLayerBuilder(noOutputs = 2)

      And("a initial layer with outpus")
      val withMask = initialLayerData + ("outputs" -> expectedOutputs)

      When("build a gradient updater")
      val converter = layer.buildGradient(mockTopology)

      And("apply to initial layer")
      val outData = converter(withMask)

      Then("should result bias")
      outData.get("gradient") should contain(expectedGradient)
    }

    it("should generate backward delta") {
      Given("a dense layer builder with 2 outputs")
      val layer = DenseLayerBuilder(noOutputs = 2)

      And("a initial layer with outpus, labels, mask")
      val delta = Nd4j.create(Array(0.1, 0.2))
      val withDelta = initialLayerData +
        ("delta" -> delta)

      When("build a gradient updater")
      val converter = layer.buildDelta(mockTopology)

      And("apply to initial layer")
      val outData = converter(withDelta)

      // | 0.1 0.2 | x | 0.1 0.3 0.5 | = | 0.05 0.11 0.17 |
      //               | 0.2 0.4 0.6 |
      Then("should result bias")
      val expectedDelta = Nd4j.create(Array(0.05, 0.11, 0.17))
      outData.get("inputDelta") should contain(expectedDelta)
    }

    it("should generate initial theta") {
      Given("a dense layer builder with 2 outputs")
      val layer = DenseLayerBuilder(noOutputs = 2)
      And("a random generator")
      val random = Nd4j.getRandomFactory().getNewRandomInstance(1234)
      And("a topology with previous layer")
      val topology = mockTopology

      When("build initial data")
      val data = layer.buildData(topology, XavierInitializer, random)

      Then("should result trace data")
      data.get("trace") should contain(Nd4j.zeros(8))

      And("should result m1 data")
      data.get("m1") should contain(Nd4j.zeros(8))

      And("should result m2 data")
      data.get("m2") should contain(Nd4j.zeros(8))

      And("should result theta data")
      data.get("theta").map(_.shape()) should contain(Array(1, 8))
    }
  }
}
